"""
triangle_id.py — Star identification via pair-based matching + RANSAC voting.

Algorithm (Liebe 1995 / Mortari Pyramid-inspired):

  Offline (cached):
    1. Load Hipparcos catalog, filter to mag < MAG_LIMIT
    2. For each star, find all neighbors within FOV
    3. Store all star pairs (a, b, angular_distance) sorted by angle

  Online:
    1. Detected body vectors (top-N brightest preferred)
    2. RANSAC loop:
       a. Pick 2 detected stars at random
       b. Look up catalog pairs with same angular distance (within tol)
       c. For each candidate pair (both orderings):
          - Pick a 3rd detected star
          - Find a catalog star matching both angles to the pair members
          - Compute candidate rotation R via Wahba (3 pairs)
          - Verify: project all detections and count catalog matches
       d. Keep best rotation by verification count
    3. Return best mapping if score >= min_verified

Usage:
    from triangle_id import StarIdentifier
    sid = StarIdentifier('/path/to/catalog.csv', mag_limit=6.0)
    result = sid.identify(body_vecs)
    # result.matches: {det_idx: cat_idx}
    # result.R:       3×3 rotation matrix
"""

from __future__ import annotations

import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TESS_FOV_DEG = 14.0
MAG_LIMIT    = 7.5              # ~80 catalog stars per TESS FOV (vs ~17 at mag<6)
# Tolerances are sized to the residual budget of SIP-corrected body vectors on
# real TESS imagery — Wahba-fit residuals against the catalog sit at ~30" at
# the centre and ~300" at the corners, so tighter tolerances will miss valid
# matches even with perfect ground-truth centroids.
ANGLE_TOL_ARCSEC      = 120.0   # pair matching tolerance
VERIFY_TOL_ARCSEC     = 200.0   # verification tolerance (covers edge stars)
MAX_RESIDUAL_ARCSEC   = 150.0   # accept triangles whose self-fit residual is consistent
REFINE_DRIFT_DEG_MAX  = 2.0     # allow more drift since pair tol is wider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def radec_to_unit(ra_deg, dec_deg):
    ra  = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    return np.stack([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec),
    ], axis=-1)


# ---------------------------------------------------------------------------
# Pattern DB
# ---------------------------------------------------------------------------
@dataclass
class PatternDB:
    star_ids:    np.ndarray   # (N,)  hipparcos_id
    star_vecs:   np.ndarray   # (N, 3) unit vectors
    pair_idx_a:  np.ndarray   # (M,)
    pair_idx_b:  np.ndarray   # (M,)
    pair_angles: np.ndarray   # (M,) sorted

    def lookup(self, angle_rad: float, tol_rad: float):
        lo = np.searchsorted(self.pair_angles, angle_rad - tol_rad)
        hi = np.searchsorted(self.pair_angles, angle_rad + tol_rad)
        return slice(lo, hi)


def build_pattern_db(catalog_csv: Path,
                     mag_limit: float = MAG_LIMIT,
                     fov_deg: float = TESS_FOV_DEG,
                     cache_path: Optional[Path] = None) -> PatternDB:
    catalog_csv = Path(catalog_csv)
    if cache_path is None:
        cache_path = catalog_csv.parent / f"pattern_db_mag{mag_limit:.1f}.pkl"

    if cache_path.exists():
        print(f"Loading cached pattern DB from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Building pattern DB from {catalog_csv} (mag < {mag_limit})...")
    cat = pd.read_csv(catalog_csv)
    cat = cat[cat["mag"] < mag_limit].reset_index(drop=True)
    print(f"  {len(cat):,} stars after magnitude filter")

    star_ids  = cat["hipparcos_id"].astype(int).to_numpy()
    star_vecs = radec_to_unit(cat["ra_deg"].to_numpy(), cat["dec_deg"].to_numpy())

    cos_fov = np.cos(np.radians(fov_deg))
    pair_a, pair_b, pair_ang = [], [], []

    CHUNK = 500
    N = len(star_vecs)
    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)
        dots = star_vecs[start:end] @ star_vecs.T
        for li, ai in enumerate(range(start, end)):
            row = dots[li]
            mask = (row >= cos_fov) & (np.arange(N) > ai)
            js = np.where(mask)[0]
            for bj in js:
                pair_a.append(ai)
                pair_b.append(bj)
                pair_ang.append(np.arccos(np.clip(row[bj], -1, 1)))

    pair_a   = np.array(pair_a,   dtype=np.int32)
    pair_b   = np.array(pair_b,   dtype=np.int32)
    pair_ang = np.array(pair_ang, dtype=np.float64)
    order    = np.argsort(pair_ang)
    pair_a, pair_b, pair_ang = pair_a[order], pair_b[order], pair_ang[order]

    db = PatternDB(star_ids, star_vecs, pair_a, pair_b, pair_ang)
    print(f"  total pairs: {len(pair_ang):,}")
    print(f"  angle range: {np.degrees(pair_ang.min()):.3f}° "
          f"to {np.degrees(pair_ang.max()):.3f}°")
    with open(cache_path, "wb") as f:
        pickle.dump(db, f)
    return db


# ---------------------------------------------------------------------------
# Wahba SVD (3-point)
# ---------------------------------------------------------------------------
def wahba_3(body_list, ref_list):
    B = sum(np.outer(r, b) for b, r in zip(body_list, ref_list))
    U, _, Vt = np.linalg.svd(B)
    ds = np.linalg.det(U) * np.linalg.det(Vt)
    return U @ np.diag([1.0, 1.0, ds]) @ Vt


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass
class IdentificationResult:
    matches: dict            # {det_idx: cat_idx}
    R: np.ndarray            # 3×3 rotation (body → ICRS)
    score: int               # number of verified matches
    triangle: tuple          # (det_i, det_j, det_k, cat_i, cat_j, cat_k)
    iterations: int          # how many RANSAC iterations used


# ---------------------------------------------------------------------------
# Star Identifier (RANSAC pair-based)
# ---------------------------------------------------------------------------
class StarIdentifier:
    def __init__(self, catalog_csv,
                 mag_limit: float = MAG_LIMIT,
                 fov_deg:   float = TESS_FOV_DEG):
        self.db = build_pattern_db(Path(catalog_csv), mag_limit, fov_deg)

    def identify(self, body_vecs: np.ndarray,
                 verify_body_vecs: Optional[np.ndarray] = None,
                 tol_arcsec:       float = ANGLE_TOL_ARCSEC,
                 verify_tol_arcsec: float = VERIFY_TOL_ARCSEC,
                 min_verified:     int   = 10,
                 max_iter:         int   = 1500,
                 seed:             int   = 0,
                 verbose:          bool  = False) -> Optional[IdentificationResult]:
        """
        Identify stars via 3-point triangle search + Pyramid-style strict verification.

        body_vecs:        Top-N brightest body vectors for triangle search (~30-60).
        verify_body_vecs: If provided, used for verification step (typically ALL detections).
                          Required for proper Pyramid-style disambiguation.
        min_verified:     ≥8 → mathematically disambiguates from random R.
                          With 10" tolerance, random R can't align 8+ stars.
        """
        if len(body_vecs) < 4:
            return None
        body_vecs = body_vecs / np.linalg.norm(body_vecs, axis=1, keepdims=True)

        # Use top-N for triangle search, but ALL detections for verification
        if verify_body_vecs is None:
            verify_body_vecs = body_vecs
        else:
            verify_body_vecs = verify_body_vecs / np.linalg.norm(
                verify_body_vecs, axis=1, keepdims=True)

        N = len(body_vecs)
        tol_rad        = np.radians(tol_arcsec / 3600.0)
        verify_tol_rad = np.radians(verify_tol_arcsec / 3600.0)

        # Precompute all pair angles among detections (top-N only)
        det_angles = np.arccos(np.clip(body_vecs @ body_vecs.T, -1, 1))

        rng = random.Random(seed)
        best = None
        iterations_used = 0

        for it in range(max_iter):
            iterations_used = it + 1

            # 1. Pick 2 random distinct detections
            i, j = rng.sample(range(N), 2)
            θ_ij = det_angles[i, j]
            if θ_ij < np.radians(0.1 / 60):   # skip duplicates
                continue

            # 2. Find catalog pairs matching this angle
            sl = self.db.lookup(θ_ij, tol_rad)
            if sl.stop - sl.start == 0:
                continue

            cand_a = self.db.pair_idx_a[sl]
            cand_b = self.db.pair_idx_b[sl]

            # 3. For each candidate catalog pair, try both orderings
            for ca, cb in zip(cand_a, cand_b):
                for c_i, c_j in [(ca, cb), (cb, ca)]:
                    # 4. Pick a 3rd detected star and find a 3rd catalog star
                    # Random 3rd detected:
                    others = [k for k in range(N) if k != i and k != j]
                    if not others:
                        continue
                    k = rng.choice(others)
                    θ_ik = det_angles[i, k]
                    θ_jk = det_angles[j, k]

                    # Chirality (handedness) of the body triangle — must match catalog
                    det_sign = np.sign(np.dot(
                        np.cross(body_vecs[i], body_vecs[j]), body_vecs[k]))
                    if det_sign == 0:
                        continue

                    # Find catalog star c_k with right angles to (c_i, c_j)
                    # AND matching triple-product sign (rejects mirror triangles)
                    c_k = self._find_third(c_i, c_j, θ_ik, θ_jk, tol_rad, det_sign)
                    if c_k is None:
                        continue

                    # 5. Compute candidate R
                    try:
                        R = wahba_3(
                            [body_vecs[i],     body_vecs[j],     body_vecs[k]],
                            [self.db.star_vecs[c_i], self.db.star_vecs[c_j], self.db.star_vecs[c_k]],
                        )
                    except Exception:
                        continue

                    # 5b. Triangle self-fit gate: if Wahba can't even align the
                    #     three seed stars, this is a bad triple — skip cheaply.
                    seed_residuals = []
                    for b_seed, r_seed in [
                        (body_vecs[i], self.db.star_vecs[c_i]),
                        (body_vecs[j], self.db.star_vecs[c_j]),
                        (body_vecs[k], self.db.star_vecs[c_k]),
                    ]:
                        ang = np.arccos(np.clip(np.dot(R @ b_seed, r_seed), -1, 1))
                        seed_residuals.append(np.degrees(ang) * 3600)
                    if max(seed_residuals) > MAX_RESIDUAL_ARCSEC:
                        continue

                    # 6. TWO-PASS VERIFICATION
                    #    Pass 1: wide tolerance (30") to gather candidate matches.
                    #    3-point R has ~10-30" noise, so strict 10" misses real matches.
                    #    After refit on those candidates, R becomes accurate.
                    R_seed  = R
                    INITIAL_VERIFY_TOL = np.radians(30.0 / 3600.0)
                    matches = self._verify_full(verify_body_vecs, R, INITIAL_VERIFY_TOL)

                    # 7. Iterative refinement (capped at 2 steps to limit drift)
                    if len(matches) >= 4:
                        for refine_step in range(2):
                            body_pairs = [verify_body_vecs[di]      for di in matches.keys()]
                            ref_pairs  = [self.db.star_vecs[ci]     for ci in matches.values()]
                            R_new = wahba_3(body_pairs, ref_pairs)
                            # Use TIGHT tolerance after first refit (R is now precise)
                            new_matches = self._verify_full(verify_body_vecs, R_new, verify_tol_rad)
                            if len(new_matches) <= len(matches):
                                R = R_new
                                break
                            R = R_new
                            matches = new_matches

                    # Final tight verification with refined R
                    if len(matches) >= 4:
                        matches = self._verify_full(verify_body_vecs, R, verify_tol_rad)

                    # 7b. R-drift gate: refinement must not slide off to a
                    #     different sky region. Reject if R rotated > REFINE_DRIFT_DEG_MAX
                    #     from the initial 3-point Wahba — that's a sign refinement
                    #     leaked into a different self-consistent constellation.
                    trace_rel = np.trace(R_seed.T @ R)
                    drift_rad = np.arccos(np.clip((trace_rel - 1) / 2, -1, 1))
                    if np.degrees(drift_rad) > REFINE_DRIFT_DEG_MAX:
                        continue

                    # Consistency check: tight residual filter
                    if len(matches) >= 4:
                        residuals = []
                        for di, ci in matches.items():
                            r_pred = R @ verify_body_vecs[di]
                            r_cat  = self.db.star_vecs[ci]
                            a = np.arccos(np.clip(np.dot(r_pred, r_cat), -1, 1))
                            residuals.append(np.degrees(a) * 3600)
                        median_res = np.median(residuals)
                        if median_res > MAX_RESIDUAL_ARCSEC:
                            continue

                    if best is None or len(matches) > best.score:
                        best = IdentificationResult(
                            matches=matches, R=R, score=len(matches),
                            triangle=(i, j, k, int(c_i), int(c_j), int(c_k)),
                            iterations=iterations_used,
                        )
                        if verbose:
                            med_str = f"med_res={median_res:.1f}\"" if len(matches) >= 4 else ""
                            print(f"  iter {iterations_used}: "
                                  f"triangle ({i},{j},{k})↔({c_i},{c_j},{c_k}) → "
                                  f"{len(matches)} verified  {med_str}")

                        if len(matches) >= min_verified * 2:
                            # Excellent — likely correct identification, early exit
                            return best

            # Early stop if we have a decent result and no improvement for many iters
            if best is not None and best.score >= min_verified and iterations_used >= 50:
                if not hasattr(self, "_last_best_score"):
                    self._last_best_score = best.score
                    self._stagnation_iters = 0
                elif best.score > self._last_best_score:
                    self._last_best_score = best.score
                    self._stagnation_iters = 0
                else:
                    self._stagnation_iters += 1
                    if self._stagnation_iters > 80:
                        # 80 iter без улучшения — выходим
                        if verbose:
                            print(f"  early exit: stagnated at score={best.score} for 80 iters")
                        delattr(self, "_last_best_score")
                        delattr(self, "_stagnation_iters")
                        return best

        if best is not None and best.score >= min_verified:
            return best
        return None

    def _find_third(self, c_i, c_j, θ_ik, θ_jk, tol_rad, det_sign=None):
        """
        Find catalog star c_k such that:
          angle(c_i, c_k) ≈ θ_ik   AND   angle(c_j, c_k) ≈ θ_jk

        If det_sign is given, also require:
          sign((v_i × v_j) · v_k) == det_sign
        which rejects mirror-image triangles (handedness check).
        """
        v_i = self.db.star_vecs[c_i]
        v_j = self.db.star_vecs[c_j]

        # Stars at angle ~θ_ik from v_i
        dots_i = self.db.star_vecs @ v_i
        cos_i_lo = np.cos(θ_ik + tol_rad)
        cos_i_hi = np.cos(θ_ik - tol_rad)
        mask_i = (dots_i >= cos_i_lo) & (dots_i <= cos_i_hi)

        # Stars at angle ~θ_jk from v_j
        dots_j = self.db.star_vecs @ v_j
        cos_j_lo = np.cos(θ_jk + tol_rad)
        cos_j_hi = np.cos(θ_jk - tol_rad)
        mask_j = (dots_j >= cos_j_lo) & (dots_j <= cos_j_hi)

        # Intersection — stars satisfying BOTH
        candidates = np.where(mask_i & mask_j)[0]
        candidates = [c for c in candidates if c != c_i and c != c_j]
        if not candidates:
            return None

        # Chirality filter — keep only candidates with matching triple-product sign
        if det_sign is not None:
            cross_ij = np.cross(v_i, v_j)
            candidates = [
                c for c in candidates
                if np.sign(np.dot(cross_ij, self.db.star_vecs[c])) == det_sign
            ]
            if not candidates:
                return None

        # Pick the one closest to both targets (least error)
        best_err = float("inf")
        best_c = None
        for c in candidates:
            e_i = abs(np.arccos(np.clip(dots_i[c], -1, 1)) - θ_ik)
            e_j = abs(np.arccos(np.clip(dots_j[c], -1, 1)) - θ_jk)
            err = e_i + e_j
            if err < best_err:
                best_err = err
                best_c   = int(c)
        return best_c

    def _verify_full(self, body_vecs, R, tol_rad):
        """
        Project body vectors through R into ICRS.
        For each, find nearest catalog star — if within tol, match.
        Returns {det_idx: cat_idx}.
        """
        body_in_icrs = body_vecs @ R.T
        dots = body_in_icrs @ self.db.star_vecs.T
        cos_tol = np.cos(tol_rad)

        matches = {}
        used_cat = set()
        # Sort detected indices by best score, take in greedy order
        best_per_det = np.argmax(dots, axis=1)
        best_score_per_det = dots[np.arange(len(body_vecs)), best_per_det]
        order = np.argsort(-best_score_per_det)
        for di in order:
            ci = int(best_per_det[di])
            if best_score_per_det[di] >= cos_tol and ci not in used_cat:
                matches[int(di)] = ci
                used_cat.add(ci)
        return matches


# ---------------------------------------------------------------------------
# CLI for DB pre-build
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--mag",     type=float, default=MAG_LIMIT)
    ap.add_argument("--fov",     type=float, default=TESS_FOV_DEG)
    args = ap.parse_args()

    db = build_pattern_db(Path(args.catalog), args.mag, args.fov)
    print(f"Ready: {len(db.star_ids):,} stars, {len(db.pair_angles):,} pairs")
