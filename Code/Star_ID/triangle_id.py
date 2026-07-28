"""
Triangle-based star identification with RANSAC voting.

Algorithm — inspired by Liebe (1995) and Mortari's Pyramid:

  Offline (cached on disk):
    1. Filter Hipparcos catalog to mag < MAG_LIMIT.
    2. For each star, find all neighbours within the field of view.
    3. Store all (a, b, angular_distance) pairs, sorted by angle for binary search.

  Online (per image):
    1. Receive body vectors (top-N brightest for triangle search).
    2. RANSAC loop:
       a. Pick 2 detections at random.
       b. Look up catalog pairs with matching angular distance.
       c. For each candidate pair (both orderings):
          - Pick a 3rd detection.
          - Find a catalog star whose angles to the pair match, with consistent chirality.
          - Solve Wahba's problem on the 3 correspondences → candidate rotation R.
          - Verify by projecting all detections through R and counting catalog matches.
       d. Keep the rotation with the highest verification count.
    3. Return the best mapping if score ≥ min_verified.

Example:
    sid = StarIdentifier("catalog_hipparcos_full.csv", mag_limit=7.5)
    result = sid.identify(body_vecs)
    # result.matches:    {det_idx: cat_idx}
    # result.R:          3×3 body → ICRS rotation matrix
    # result.score:      verified-match count
    # result.iterations: RANSAC iterations consumed
"""

from __future__ import annotations

import hashlib
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
MAG_LIMIT    = 7.5
# Tolerances sized to the residual budget of SIP-corrected body vectors on real
# TESS imagery: Wahba-fit residuals are ~30″ at the centre, ~300″ at the corners.
# Tighter tolerances reject otherwise valid matches even with perfect centroids.
ANGLE_TOL_ARCSEC     = 120.0
VERIFY_TOL_ARCSEC    = 200.0
MAX_RESIDUAL_ARCSEC  = 150.0
REFINE_DRIFT_DEG_MAX = 2.0


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
        digest = hashlib.sha256()
        with open(catalog_csv, "rb") as catalog_file:
            for block in iter(lambda: catalog_file.read(1024 * 1024), b""):
                digest.update(block)
        catalog_hash = digest.hexdigest()[:10]
        cache_path = catalog_csv.parent / (
            f"pattern_db_{catalog_csv.stem}_{catalog_hash}_"
            f"mag{mag_limit:.1f}_fov{fov_deg:.1f}.pkl"
        )

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
# RANSAC identifier
# ---------------------------------------------------------------------------
class StarIdentifier:
    def __init__(self, catalog_csv,
                 mag_limit: float = MAG_LIMIT,
                 fov_deg:   float = TESS_FOV_DEG):
        self.db = build_pattern_db(Path(catalog_csv), mag_limit, fov_deg)

    def identify(self, body_vecs: np.ndarray,
                 verify_body_vecs:  Optional[np.ndarray] = None,
                 tol_arcsec:        float = ANGLE_TOL_ARCSEC,
                 verify_tol_arcsec: float = VERIFY_TOL_ARCSEC,
                 min_verified:      int   = 10,
                 max_iter:          int   = 1500,
                 seed:              int   = 0,
                 verbose:           bool  = False) -> Optional[IdentificationResult]:
        """
        Run the RANSAC triangle search.

        Args:
            body_vecs:        Body vectors used for triangle sampling (typically
                              the ~30–60 brightest detections).
            verify_body_vecs: Body vectors used for verification — pass ALL
                              detections here for Pyramid-style disambiguation.
            min_verified:     Minimum verified matches required to return a
                              result. ≥8 mathematically rules out a chance lock
                              at the tolerances used.
        """
        if len(body_vecs) < 4:
            return None
        body_vecs = body_vecs / np.linalg.norm(body_vecs, axis=1, keepdims=True)

        if verify_body_vecs is None:
            verify_body_vecs = body_vecs
        else:
            verify_body_vecs = verify_body_vecs / np.linalg.norm(
                verify_body_vecs, axis=1, keepdims=True)

        N = len(body_vecs)
        tol_rad        = np.radians(tol_arcsec / 3600.0)
        verify_tol_rad = np.radians(verify_tol_arcsec / 3600.0)
        det_angles = np.arccos(np.clip(body_vecs @ body_vecs.T, -1, 1))

        rng = random.Random(seed)
        best = None
        iterations_used = 0
        last_best_score = None
        stagnation_iters = 0

        for it in range(max_iter):
            iterations_used = it + 1

            # Sample two distinct detections.
            i, j = rng.sample(range(N), 2)
            θ_ij = det_angles[i, j]
            if θ_ij < np.radians(0.1 / 60):
                continue

            sl = self.db.lookup(θ_ij, tol_rad)
            if sl.stop - sl.start == 0:
                continue

            cand_a = self.db.pair_idx_a[sl]
            cand_b = self.db.pair_idx_b[sl]

            for ca, cb in zip(cand_a, cand_b):
                for c_i, c_j in [(ca, cb), (cb, ca)]:
                    others = [k for k in range(N) if k != i and k != j]
                    if not others:
                        continue
                    k = rng.choice(others)
                    θ_ik = det_angles[i, k]
                    θ_jk = det_angles[j, k]

                    # Chirality must match — rejects mirror constellations.
                    det_sign = np.sign(np.dot(
                        np.cross(body_vecs[i], body_vecs[j]), body_vecs[k]))
                    if det_sign == 0:
                        continue

                    c_k = self._find_third(c_i, c_j, θ_ik, θ_jk, tol_rad, det_sign)
                    if c_k is None:
                        continue

                    # 5. Compute candidate R
                    try:
                        R = wahba_3(
                            [body_vecs[i],     body_vecs[j],     body_vecs[k]],
                            [self.db.star_vecs[c_i], self.db.star_vecs[c_j], self.db.star_vecs[c_k]],
                        )
                    except np.linalg.LinAlgError:
                        continue

                    # Self-fit gate: a bad triplet can't even align its own three stars.
                    seed_residuals = [
                        np.arccos(np.clip(np.dot(R @ b, r), -1, 1)) * 3600 * 180 / np.pi
                        for b, r in [(body_vecs[i], self.db.star_vecs[c_i]),
                                     (body_vecs[j], self.db.star_vecs[c_j]),
                                     (body_vecs[k], self.db.star_vecs[c_k])]
                    ]
                    if max(seed_residuals) > MAX_RESIDUAL_ARCSEC:
                        continue

                    # Two-pass verification: first collect a small set of
                    # high-confidence 30″ matches. Refit R on those matches, then
                    # expand to the 200″ tolerance needed at distorted frame edges.
                    R_seed  = R
                    INITIAL_VERIFY_TOL = np.radians(30.0 / 3600.0)
                    matches = self._verify_full(verify_body_vecs, R, INITIAL_VERIFY_TOL)

                    if len(matches) >= 4:
                        for _ in range(2):
                            body_pairs = [verify_body_vecs[di]    for di in matches.keys()]
                            ref_pairs  = [self.db.star_vecs[ci]   for ci in matches.values()]
                            R_new = wahba_3(body_pairs, ref_pairs)
                            new_matches = self._verify_full(verify_body_vecs, R_new, verify_tol_rad)
                            if len(new_matches) <= len(matches):
                                R = R_new
                                break
                            R = R_new
                            matches = new_matches

                    if len(matches) >= 4:
                        matches = self._verify_full(verify_body_vecs, R, verify_tol_rad)

                    # Drift gate: refinement must not slide far from the seed R,
                    # otherwise we leaked into a different self-consistent constellation.
                    trace_rel = np.trace(R_seed.T @ R)
                    drift_rad = np.arccos(np.clip((trace_rel - 1) / 2, -1, 1))
                    if np.degrees(drift_rad) > REFINE_DRIFT_DEG_MAX:
                        continue

                    median_res = float("nan")
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

                        # Early exit if we already have a clear, high-confidence lock.
                        if len(matches) >= min_verified * 2:
                            return best

            # Early stop after sustained no-improvement past min_verified threshold.
            if best is not None and best.score >= min_verified and iterations_used >= 50:
                if last_best_score is None:
                    last_best_score = best.score
                    stagnation_iters = 0
                elif best.score > last_best_score:
                    last_best_score = best.score
                    stagnation_iters = 0
                else:
                    stagnation_iters += 1
                    if stagnation_iters > 80:
                        if verbose:
                            print(f"  early exit: stagnated at score={best.score} for 80 iters")
                        return best

        if best is not None and best.score >= min_verified:
            return best
        return None

    def _find_third(self, c_i, c_j, θ_ik, θ_jk, tol_rad, det_sign=None):
        """
        Return a catalog star whose angles to c_i and c_j match (θ_ik, θ_jk)
        within `tol_rad`. If `det_sign` is provided, also require matching
        triangle handedness to reject mirror-image solutions.
        """
        sv = self.db.star_vecs
        v_i = sv[c_i]
        v_j = sv[c_j]

        # Apply the first constraint against the full catalog once. The second
        # constraint only touches the small survivor set.
        dots_i = sv @ v_i
        cand = np.where(
            (dots_i >= np.cos(θ_ik + tol_rad))
            & (dots_i <= np.cos(θ_ik - tol_rad))
        )[0]
        if cand.size == 0:
            return None

        dots_j = sv[cand] @ v_j
        keep = (
            (dots_j >= np.cos(θ_jk + tol_rad))
            & (dots_j <= np.cos(θ_jk - tol_rad))
            & (cand != c_i)
            & (cand != c_j)
        )
        cand = cand[keep]
        dots_j = dots_j[keep]
        if cand.size == 0:
            return None

        if det_sign is not None:
            cross_ij = np.cross(v_i, v_j)
            keep = np.sign(sv[cand] @ cross_ij) == det_sign
            cand = cand[keep]
            dots_j = dots_j[keep]
            if cand.size == 0:
                return None

        e_i = np.abs(np.arccos(np.clip(dots_i[cand], -1, 1)) - θ_ik)
        e_j = np.abs(np.arccos(np.clip(dots_j, -1, 1)) - θ_jk)
        return int(cand[np.argmin(e_i + e_j)])

    def _verify_full(self, body_vecs, R, tol_rad):
        """
        Project body vectors through R into ICRS.
        For each, find nearest catalog star — if within tol, match.
        Returns {det_idx: cat_idx}.

        Only catalog stars inside a 12° cone around the candidate boresight can
        match an in-frame TESS detection. Culling to that cone reduces the
        nearest-neighbour score matrix from roughly 25,000 columns to ~150
        without changing the returned match.
        """
        body_in_icrs = body_vecs @ R.T

        boresight = R @ np.array([1.0, 0.0, 0.0])
        cone_cos = np.cos(np.radians(12.0))
        cat_idx = np.where(self.db.star_vecs @ boresight >= cone_cos)[0]
        if cat_idx.size == 0:
            return {}
        dots = body_in_icrs @ self.db.star_vecs[cat_idx].T
        cos_tol = np.cos(tol_rad)

        matches = {}
        used_cat = set()
        # Sort detected indices by best score, take in greedy order
        best_per_det = np.argmax(dots, axis=1)
        best_score_per_det = dots[np.arange(len(body_vecs)), best_per_det]
        order = np.argsort(-best_score_per_det)
        for di in order:
            if best_score_per_det[di] < cos_tol:
                continue
            ci = int(cat_idx[best_per_det[di]])
            if ci not in used_cat:
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
