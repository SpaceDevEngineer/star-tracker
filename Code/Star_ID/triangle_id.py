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

An optional Mortari-style 4th-star ("Pyramid") gate can be inserted before the
Wahba + verification step via `pyramid="hard"|"soft"`. It is OFF by default:
on this pipeline verification is cone-culled to ~150 catalog stars and costs
~0.3 ms, so the gate costs more than the check it is supposed to save, while
its veto discards true triangles whose 4th star simply is not in the mag<7.5
catalog. `pyramid_ablation.py` measures this rather than asserting it.

Usage:
    from triangle_id import StarIdentifier
    sid = StarIdentifier('/path/to/catalog.csv', mag_limit=6.0)
    result = sid.identify(body_vecs)
    # result.matches: {det_idx: cat_idx}
    # result.R:       3×3 rotation matrix
    # result.gate:    GateStats (populated when pyramid/gate_audit are used)
"""

from __future__ import annotations

import hashlib
import pickle
import random
import time
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

# --- 4th-star (Pyramid) gate ------------------------------------------------
# The gate is OFF by default: on this pipeline it has never paid for itself (see
# `pyramid_ablation.py`). It is kept, fully instrumented, so the negative result
# can be measured rather than asserted.
PYRAMID_K_CANDIDATES  = 10      # how many 4th detections to try per triangle
PYRAMID_TOL_SCALE     = 2.0     # gate tol = PYRAMID_TOL_SCALE × pair tol
# SIP-corrected body-vec residual grows from ~30" on-axis to ~300" at the field
# corner, so a single flat tolerance is simultaneously too loose on-axis (admits
# chance matches) and too tight off-axis (vetoes true stars). The soft gate
# scales its tolerance with field radius: tol(r) = tol × (1 + SLOPE·r/r_corner).
PYRAMID_TOL_SLOPE     = 3.0
PYRAMID_FIELD_CORNER_DEG = 8.5  # TESS FOV corner half-angle

# --- Neighbour-restricted third-star search ---------------------------------
# The third catalogue star of a triangle lies within theta_ik + tol of c_i, and
# theta_ik (an angle between two detections) cannot exceed the field diagonal
# (~17 deg for TESS). Restricting the search to the precomputed neighbours of
# c_i within this radius returns the same candidates in the same order as a
# scan of the whole catalogue, at ~3 % of the cost.
NEIGHBOUR_RADIUS_DEG = 20.0


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
class GateStats:
    """
    Bookkeeping for the 4th-star gate.

    The `winner_*` counters are the important ones: they are filled by the
    *counterfactual audit*, which evaluates the gate on every triangle that
    becomes a new best WITHOUT letting it veto anything. Running the baseline
    (`pyramid="off"`) with `gate_audit=True` therefore answers the decisive
    question directly — "how often would the gate have thrown away the triangle
    that actually won?" — without needing the gate to be enabled at all.
    """
    calls:                int = 0   # gate evaluations performed
    passed:               int = 0   # a consistent 4th catalog star was found
    veto_absent:          int = 0   # no plausible counterpart at all → no evidence
    veto_contradiction:   int = 0   # plausible counterpart existed, but inconsistent
    candidates_tried:     int = 0   # total 4th-detection candidates examined
    seconds:            float = 0.0 # wall-clock spent inside the gate

    # Counterfactual audit on triangles that became a new best. BOTH gate
    # variants are evaluated on every audited triangle, so a single baseline
    # (pyramid="off") run reports what each variant would have thrown away.
    winner_calls:              int = 0
    winner_passed:             int = 0   # hard gate
    winner_veto_absent:        int = 0
    winner_veto_contradiction: int = 0
    winner_soft_passed:        int = 0   # soft gate
    winner_soft_veto:          int = 0

    def as_dict(self):
        d = {k: v for k, v in self.__dict__.items()}
        d["veto_rate"] = (
            (self.veto_absent + self.veto_contradiction) / self.calls
            if self.calls else None)
        d["winner_veto_rate"] = (
            (self.winner_veto_absent + self.winner_veto_contradiction) / self.winner_calls
            if self.winner_calls else None)
        d["winner_soft_veto_rate"] = (
            self.winner_soft_veto / self.winner_calls if self.winner_calls else None)
        return d


@dataclass
class IdentificationResult:
    matches: dict            # {det_idx: cat_idx}
    R: np.ndarray            # 3×3 rotation (body → ICRS)
    score: int               # number of verified matches
    triangle: tuple          # (det_i, det_j, det_k, cat_i, cat_j, cat_k)
    iterations: int          # how many RANSAC iterations used
    history: object = None   # if recorded: list of per-attempt dicts
                             # {iter, det:(i,j,k), cat:(ci,cj,ck), n_verified}
    gate: object = None      # GateStats for this identify() call
    significance: float = None  # score / expected chance matches (see identify)


# ---------------------------------------------------------------------------
# Star Identifier (RANSAC pair-based)
# ---------------------------------------------------------------------------
class StarIdentifier:
    def __init__(self, catalog_csv,
                 mag_limit: float = MAG_LIMIT,
                 fov_deg:   float = TESS_FOV_DEG,
                 fast_third="batch"):
        self.db = build_pattern_db(Path(catalog_csv), mag_limit, fov_deg)
        self._nbr_cache = (Path(catalog_csv).parent /
                           f"neighbours_mag{mag_limit:.1f}_r{NEIGHBOUR_RADIUS_DEG:.0f}.npz")
        # Third-star search: False = scan the whole catalogue (original);
        # "nbr" = per-candidate scan of c_i's neighbours; "batch" = all
        # candidates of a RANSAC iteration at once over angle-sorted
        # neighbour lists. All three return the same third star.
        self.fast_third = fast_third
        self._nbr_ptr = self._nbr_idx = None
        self._nbr_sorted = None

    def _neighbours_by_angle(self):
        """Neighbour lists re-sorted by angle to their centre star, plus a
        globally monotone key  centre*4 + angle  for vectorised range search."""
        if self._nbr_sorted is None:
            ptr, idx = self._neighbours()
            sv = self.db.star_vecs
            centre = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
            ang = np.arccos(np.clip(np.einsum("ij,ij->i", sv[idx], sv[centre]), -1, 1))
            order = np.lexsort((ang, centre))
            self._nbr_sorted = (idx[order], centre[order] * 4.0 + ang[order])
        return self._nbr_sorted

    def _third_batch(self, ci, cj, th_ik, th_jk, dsign, tol_rad):
        """Vectorised `_find_third` for many (c_i, c_j, theta_ik, theta_jk, sign).
        Returns an int array, -1 where no third star exists."""
        sv = self.db.star_vecs
        n = len(ci)
        out = np.full(n, -1, dtype=np.int64)
        R = np.radians(NEIGHBOUR_RADIUS_DEG)
        ok = dsign != 0
        far = ok & (th_ik + tol_rad > R)          # outside neighbour radius: exact fallback
        for o in np.nonzero(far)[0]:
            r = self._find_third(int(ci[o]), int(cj[o]), th_ik[o], th_jk[o], tol_rad, dsign[o])
            out[o] = -1 if r is None else r
        ok &= ~far
        if not ok.any():
            return out
        nbr, key = self._neighbours_by_angle()
        eps = 1e-9                                # widen the range; exact cos test below
        lo = np.searchsorted(key, ci * 4.0 + (th_ik - tol_rad - eps))
        hi = np.searchsorted(key, ci * 4.0 + (th_ik + tol_rad + eps))
        cnt = np.where(ok, hi - lo, 0)
        tot = int(cnt.sum())
        if tot == 0:
            return out
        o_of = np.repeat(np.arange(n), cnt)
        start = np.repeat(np.cumsum(cnt) - cnt, cnt)
        cand = nbr[np.repeat(lo, cnt) + (np.arange(tot) - start)]
        vi, vj, vc = sv[ci[o_of]], sv[cj[o_of]], sv[cand]
        d_i = np.einsum("ij,ij->i", vc, vi)
        d_j = np.einsum("ij,ij->i", vc, vj)
        ti, tj = th_ik[o_of], th_jk[o_of]
        keep = ((d_i >= np.cos(ti + tol_rad)) & (d_i <= np.cos(ti - tol_rad)) &
                (d_j >= np.cos(tj + tol_rad)) & (d_j <= np.cos(tj - tol_rad)) &
                (cand != ci[o_of]) & (cand != cj[o_of]))
        keep &= np.sign(np.einsum("ij,ij->i", vc, np.cross(vi, vj))) == dsign[o_of]
        if not keep.any():
            return out
        o_k, c_k = o_of[keep], cand[keep]
        e = (np.abs(np.arccos(np.clip(d_i[keep], -1, 1)) - ti[keep]) +
             np.abs(np.arccos(np.clip(d_j[keep], -1, 1)) - tj[keep]))
        order = np.lexsort((c_k, e, o_k))        # per ordering: min error, ties -> lowest index
        o_s, c_s = o_k[order], c_k[order]
        first = np.ones(len(o_s), bool)
        first[1:] = o_s[1:] != o_s[:-1]
        out[o_s[first]] = c_s[first]
        return out

    def _neighbours(self):
        """CSR neighbour lists (ascending catalogue index) within NEIGHBOUR_RADIUS_DEG."""
        if self._nbr_ptr is None:
            if self._nbr_cache.exists():
                z = np.load(self._nbr_cache)
                self._nbr_ptr, self._nbr_idx = z["ptr"], z["idx"]
            else:
                sv = self.db.star_vecs
                cos_r = np.cos(np.radians(NEIGHBOUR_RADIUS_DEG))
                lens, parts = [], []
                for s0 in range(0, len(sv), 500):
                    d = sv[s0:s0 + 500] @ sv.T
                    for row in d:
                        nz = np.nonzero(row >= cos_r)[0].astype(np.int32)
                        lens.append(nz.size)
                        parts.append(nz)
                ptr = np.zeros(len(sv) + 1, dtype=np.int64)
                ptr[1:] = np.cumsum(lens)
                self._nbr_ptr, self._nbr_idx = ptr, np.concatenate(parts)
                try:
                    np.savez(self._nbr_cache, ptr=self._nbr_ptr, idx=self._nbr_idx)
                except OSError:
                    pass
        return self._nbr_ptr, self._nbr_idx

    def identify(self, body_vecs: np.ndarray,
                 verify_body_vecs: Optional[np.ndarray] = None,
                 tol_arcsec:       float = ANGLE_TOL_ARCSEC,
                 verify_tol_arcsec: float = VERIFY_TOL_ARCSEC,
                 min_verified:     int   = 10,
                 max_iter:         int   = 1500,
                 seed:             int   = 0,
                 record_history:   bool  = False,
                 pyramid:          str   = "off",
                 gate_audit:       bool  = False,
                 exclude_boresights: Optional[list] = None,
                 exclude_radius_deg: float = 6.0,
                 min_significance: float = 0.0,
                 verbose:          bool  = False) -> Optional[IdentificationResult]:
        """
        Identify stars via 3-point triangle search + strict verification.

        body_vecs:        Top-N brightest body vectors for triangle search (~30-60).
        verify_body_vecs: If provided, used for verification step (typically ALL detections).
                          Required for proper Pyramid-style disambiguation.
        min_verified:     ≥8 → mathematically disambiguates from random R.
                          With 10" tolerance, random R can't align 8+ stars.
        pyramid:          4th-star gate mode, applied BEFORE Wahba + verification:
                            "off"  — no gate (default; the published baseline).
                            "hard" — classic Mortari veto: no consistent 4th
                                     catalog star ⇒ discard the triangle.
                            "soft" — veto only on *contradiction*. If the 4th
                                     detection has no plausible catalog
                                     counterpart at all it is treated as missing
                                     evidence, not as a refutation. Candidates
                                     are tried centre-of-field first and the
                                     tolerance scales with field radius.
        gate_audit:       Evaluate the gate on every new-best triangle and record
                          the verdict WITHOUT acting on it. Costs a couple of
                          gate evaluations per new best (a few per frame), not
                          per iteration.

                          ONLY MEANINGFUL WITH pyramid="off". When the gate is
                          live it has already discarded the vetoed triangles
                          before they can become a new best, so the audited
                          population is pre-filtered and the measured veto rate
                          collapses to ~0 by construction. Read the false-veto
                          rate off the baseline arm, never off the gated arms.
        exclude_boresights: Optional list of ICRS unit vectors. Any candidate
                          whose boresight (R applied to camera +x) lies within
                          `exclude_radius_deg` of one of them is skipped. Used
                          by the refusal-retry in inference_full: after the
                          quality gate rejects a lock, the search is re-run
                          with that sky region forbidden, so a dense-sky
                          attractor cannot win twice. None (default) leaves
                          the search bit-identical to the published baseline.
        min_significance: 0 (default) keeps the published stopping rules.
                          When > 0, the two early-stopping rules (score >= 20,
                          and 80 iterations without improvement) may only fire
                          on a hypothesis whose verified-match count is at
                          least `min_significance` times the number of matches
                          expected BY CHANCE for its sky region:
                              chance = N_det * n_cone * cap(tol) / cap(12 deg)
                          In a dense Milky-Way field a wrong constellation
                          collects 16-21 matches against a chance level of ~6
                          (2.5-3.3x), while true locks sit at 11-16x; raw-count
                          stopping therefore ends the search on the impostor
                          before RANSAC has drawn the true triangle.
        """
        if pyramid not in ("off", "hard", "soft"):
            raise ValueError(f"pyramid must be off|hard|soft, got {pyramid!r}")
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

        # Field radius of each detection (angle off boresight = camera +x).
        # Drives the soft gate: on-axis stars carry ~30" of SIP residual,
        # corner stars ~300", so both the ordering and the tolerance use it.
        field_rad_deg = np.degrees(np.arccos(np.clip(body_vecs[:, 0], -1, 1)))

        excl = (np.asarray(exclude_boresights, dtype=np.float64).reshape(-1, 3)
                if exclude_boresights else None)
        excl_cos = np.cos(np.radians(exclude_radius_deg))
        # Probability that one detection lands within verify tol of a given
        # catalogue star in the 12-degree verification cone, per cone star.
        _cap = lambda a: 2.0 * np.pi * (1.0 - np.cos(a))      # noqa: E731
        chance_per_star = (len(verify_body_vecs) * _cap(verify_tol_rad)
                           / _cap(np.radians(12.0)))

        gate = GateStats()
        gate_on = pyramid != "off"

        # Reset the stagnation tracker. It lives on `self` and the early-exit
        # paths below do not always clear it, so a StarIdentifier reused across
        # frames (as the parallel driver does — one identifier per worker) would
        # otherwise inherit the previous frame's counters and stop early. That
        # made results depend on image order; the A/B needs them not to.
        self._last_best_score = None
        self._stagnation_iters = 0

        rng = random.Random(seed)
        best = None
        iterations_used = 0
        history = [] if record_history else None

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

            # Per-(i, j) quantities, identical for every candidate below;
            # computed once instead of once per candidate (same values).
            others = [k for k in range(N) if k != i and k != j]
            cross_ij_body = np.cross(body_vecs[i], body_vecs[j])
            find_third = (self._find_third_nbr if self.fast_third
                          else self._find_third)
            batched = self.fast_third == "batch" and len(others) > 0
            if batched:
                # Same RNG consumption as the per-candidate loop: one choice
                # per ordering, in the same order (the loop below draws exactly
                # one before any `continue`, and returns early only for good).
                n_ord = 2 * len(cand_a)
                ks = np.fromiter((rng.choice(others) for _ in range(n_ord)),
                                 dtype=np.int64, count=n_ord)
                ci_all = np.empty(n_ord, dtype=np.int64)
                cj_all = np.empty(n_ord, dtype=np.int64)
                ci_all[0::2], cj_all[0::2] = cand_a, cand_b
                ci_all[1::2], cj_all[1::2] = cand_b, cand_a
                th_ik_all = det_angles[i, ks]
                th_jk_all = det_angles[j, ks]
                dsign_all = np.sign(body_vecs[ks] @ cross_ij_body)
                ck_all = self._third_batch(ci_all, cj_all, th_ik_all, th_jk_all,
                                           dsign_all, tol_rad)

            # 3. For each candidate catalog pair, try both orderings
            for pair_n, (ca, cb) in enumerate(zip(cand_a, cand_b)):
                for side, (c_i, c_j) in enumerate(((ca, cb), (cb, ca))):
                    # 4. Pick a 3rd detected star and find a 3rd catalog star
                    # Random 3rd detected:
                    if not others:
                        continue
                    if batched:
                        o = 2 * pair_n + side
                        k = int(ks[o])
                        θ_ik = th_ik_all[o]
                        θ_jk = th_jk_all[o]
                        det_sign = dsign_all[o]
                        if det_sign == 0:
                            continue
                        c_k = None if ck_all[o] < 0 else int(ck_all[o])
                    else:
                        k = rng.choice(others)
                        θ_ik = det_angles[i, k]
                        θ_jk = det_angles[j, k]

                        # Chirality (handedness) of the body triangle — must match catalog
                        det_sign = np.sign(np.dot(cross_ij_body, body_vecs[k]))
                        if det_sign == 0:
                            continue

                        # Find catalog star c_k with right angles to (c_i, c_j)
                        # AND matching triple-product sign (rejects mirror triangles)
                        c_k = find_third(c_i, c_j, θ_ik, θ_jk, tol_rad, det_sign)
                    if c_k is None:
                        continue
                    
                    # 4b. PYRAMID: try to confirm the triangle with a 4th star
                    #     BEFORE the expensive Wahba + verification.
                    #
                    #     Note the premise this rests on: that verification is
                    #     the expensive step worth pre-filtering for. It is not.
                    #     The gate runs BEFORE R is known, so it must scan the
                    #     full 25k-star catalog (0.242 ms measured);
                    #     `_verify_full` runs AFTER R is known, culls to the
                    #     ~150 stars in the FOV cone, and checks all ~400
                    #     detections in 0.156 ms. The filter costs 1.55x what it
                    #     filters — so it defaults to "off". See
                    #     Results/PYRAMID_ANALYSIS.md and pyramid_ablation.py.
                    gate_verdict, det_l, c_l = None, None, None
                    if gate_on:
                        gate_verdict, det_l, c_l = self._gate_fourth(
                            i, j, k, c_i, c_j, c_k,
                            body_vecs, det_angles, field_rad_deg,
                            tol_rad, pyramid, gate)
                        verdict = gate_verdict
                        if verdict != "pass":
                            # Record the veto so a visualisation can show WHICH
                            # triangles the gate throws away, not only the ones
                            # that survived it.
                            if record_history and len(history) < 500:
                                history.append({
                                    "iter": iterations_used,
                                    "det": (int(i), int(j), int(k)),
                                    "cat": (int(c_i), int(c_j), int(c_k)),
                                    "det4": None if det_l is None else int(det_l),
                                    "cat4": None if c_l is None else int(c_l),
                                    "gate": verdict,
                                    "n_verified": 0,
                                    "vetoed": True,
                                })
                            # "hard" vetoes on anything short of a confirmation;
                            # "soft" only on an actual contradiction.
                            if pyramid == "hard" or verdict == "contradiction":
                                continue

                    # 5. Compute candidate R
                    try:
                        R = wahba_3(
                            [body_vecs[i],     body_vecs[j],     body_vecs[k]],
                            [self.db.star_vecs[c_i], self.db.star_vecs[c_j], self.db.star_vecs[c_k]],
                        )
                    except Exception:
                        continue

                    # 5a. Refusal-retry exclusion: skip hypotheses pointing
                    #     into a sky region whose lock was already refused.
                    if excl is not None and np.any(
                            excl @ (R @ np.array([1.0, 0.0, 0.0])) >= excl_cos):
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

                    # Re-check after refinement (the refit may drift <= 2 deg).
                    if excl is not None and np.any(
                            excl @ (R @ np.array([1.0, 0.0, 0.0])) >= excl_cos):
                        continue

                    significance = len(matches) / max(
                        chance_per_star * getattr(self, "_last_cone_n", 0), 0.5)
                    is_new_best = best is None or len(matches) > best.score

                    # 8b. COUNTERFACTUAL GATE AUDIT — the decisive diagnostic.
                    #     This triangle just verified better than anything seen
                    #     so far, i.e. it is almost certainly a TRUE triangle.
                    #     Ask what the 4th-star gate would have done to it, and
                    #     record the verdict without acting on it. Run in the
                    #     "off" arm this measures the gate's false-veto rate on
                    #     true triangles directly.
                    if gate_audit and is_new_best:
                        gate.winner_calls += 1
                        v_hard, _, _ = self._gate_fourth(
                            i, j, k, c_i, c_j, c_k, body_vecs, det_angles,
                            field_rad_deg, tol_rad, "hard", None)
                        if v_hard == "pass":
                            gate.winner_passed += 1
                        elif v_hard == "contradiction":
                            gate.winner_veto_contradiction += 1
                        else:
                            gate.winner_veto_absent += 1
                        # The soft gate only vetoes on contradiction, so its
                        # verdict is not derivable from the hard one — evaluate it.
                        v_soft, _, _ = self._gate_fourth(
                            i, j, k, c_i, c_j, c_k, body_vecs, det_angles,
                            field_rad_deg, tol_rad, "soft", None)
                        if v_soft == "contradiction":
                            gate.winner_soft_veto += 1
                        else:
                            gate.winner_soft_passed += 1

                    # Record attempts for visualisation: keep up to 500 ordinary
                    # attempts, but ALWAYS record a new best (incl. the winner) so
                    # the escalating scores are never truncated by the cap.
                    if record_history and (is_new_best or len(history) < 500):
                        history.append({
                            "iter": iterations_used,
                            "det": (int(i), int(j), int(k)),
                            "cat": (int(c_i), int(c_j), int(c_k)),
                            "det4": None if det_l is None else int(det_l),
                            "cat4": None if c_l is None else int(c_l),
                            "gate": gate_verdict,
                            "n_verified": int(len(matches)),
                            "vetoed": False,
                        })

                    if is_new_best:
                        best = IdentificationResult(
                            matches=matches, R=R, score=len(matches),
                            triangle=(i, j, k, int(c_i), int(c_j), int(c_k)),
                            iterations=iterations_used,
                            gate=gate,          # same object → keeps accumulating
                            significance=float(significance),
                        )
                        if record_history:
                            best.history = history   # same list object → grows
                        if verbose:
                            med_str = f"med_res={median_res:.1f}\"" if len(matches) >= 4 else ""
                            print(f"  iter {iterations_used}: "
                                  f"triangle ({i},{j},{k})↔({c_i},{c_j},{c_k}) → "
                                  f"{len(matches)} verified  {med_str}")

                        if len(matches) >= min_verified * 2 and (
                                min_significance <= 0
                                or significance >= min_significance):
                            # Excellent — likely correct identification, early exit
                            return best

            # Early stop if we have a decent result and no improvement for many iters
            if (best is not None and best.score >= min_verified
                    and iterations_used >= 50
                    and (min_significance <= 0
                         or best.significance >= min_significance)):
                if self._last_best_score is None or best.score > self._last_best_score:
                    self._last_best_score = best.score
                    self._stagnation_iters = 0
                else:
                    self._stagnation_iters += 1
                    if self._stagnation_iters > 80:
                        # 80 iter без улучшения — выходим
                        if verbose:
                            print(f"  early exit: stagnated at score={best.score} for 80 iters")
                        return best

        if best is not None and best.score >= min_verified:
            return best
        return None

    def _find_third_nbr(self, c_i, c_j, θ_ik, θ_jk, tol_rad, det_sign=None):
        """Same result as `_find_third`, searching only the neighbours of c_i."""
        if θ_ik + tol_rad > np.radians(NEIGHBOUR_RADIUS_DEG):
            return self._find_third(c_i, c_j, θ_ik, θ_jk, tol_rad, det_sign)
        ptr, nbr = self._neighbours()
        idx = nbr[ptr[c_i]:ptr[c_i + 1]]
        sv  = self.db.star_vecs
        v_i = sv[c_i]
        v_j = sv[c_j]
        dots_i = sv[idx] @ v_i
        m = (dots_i >= np.cos(θ_ik + tol_rad)) & (dots_i <= np.cos(θ_ik - tol_rad))
        cand = idx[m]
        if cand.size == 0:
            return None
        d_i = dots_i[m]
        dots_j = sv[cand] @ v_j
        keep = (dots_j >= np.cos(θ_jk + tol_rad)) & (dots_j <= np.cos(θ_jk - tol_rad))
        keep &= (cand != c_i) & (cand != c_j)
        cand, d_i, dots_j = cand[keep], d_i[keep], dots_j[keep]
        if cand.size == 0:
            return None
        if det_sign is not None:
            keep2 = np.sign(sv[cand] @ np.cross(v_i, v_j)) == det_sign
            cand, d_i, dots_j = cand[keep2], d_i[keep2], dots_j[keep2]
            if cand.size == 0:
                return None
        e_i = np.abs(np.arccos(np.clip(d_i,    -1, 1)) - θ_ik)
        e_j = np.abs(np.arccos(np.clip(dots_j, -1, 1)) - θ_jk)
        return int(cand[np.argmin(e_i + e_j)])

    def _find_third(self, c_i, c_j, θ_ik, θ_jk, tol_rad, det_sign=None):
        """
        Find catalog star c_k such that:
          angle(c_i, c_k) ≈ θ_ik   AND   angle(c_j, c_k) ≈ θ_jk

        If det_sign is given, also require:
          sign((v_i × v_j) · v_k) == det_sign
        which rejects mirror-image triangles (handedness check).

        Optimised: only one full O(N) dot product (against c_i). The second
        angle constraint and the chirality/selection steps run only over the
        few stars that already pass the first constraint — fully vectorised,
        no per-candidate Python loop. Result is identical to the naive form.
        """
        sv  = self.db.star_vecs
        v_i = sv[c_i]
        v_j = sv[c_j]

        # First angle constraint: stars at angle ~θ_ik from v_i (one O(N) dot)
        dots_i = sv @ v_i
        cand = np.where((dots_i >= np.cos(θ_ik + tol_rad)) &
                        (dots_i <= np.cos(θ_ik - tol_rad)))[0]
        if cand.size == 0:
            return None

        # Second angle constraint — evaluated ONLY on first-constraint survivors
        dots_j = sv[cand] @ v_j
        keep = (dots_j >= np.cos(θ_jk + tol_rad)) & (dots_j <= np.cos(θ_jk - tol_rad))
        # Drop the two seed stars themselves
        keep &= (cand != c_i) & (cand != c_j)
        cand   = cand[keep]
        dots_j = dots_j[keep]
        if cand.size == 0:
            return None

        # Chirality filter — keep only matching triple-product sign (vectorised)
        if det_sign is not None:
            cross_ij = np.cross(v_i, v_j)
            keep2 = np.sign(sv[cand] @ cross_ij) == det_sign
            cand   = cand[keep2]
            dots_j = dots_j[keep2]
            if cand.size == 0:
                return None

        # Pick the candidate closest to both target angles (vectorised argmin)
        e_i = np.abs(np.arccos(np.clip(dots_i[cand], -1, 1)) - θ_ik)
        e_j = np.abs(np.arccos(np.clip(dots_j,       -1, 1)) - θ_jk)
        return int(cand[np.argmin(e_i + e_j)])

    # -----------------------------------------------------------------------
    # 4th-star (Pyramid) gate
    # -----------------------------------------------------------------------
    def _gate_fourth(self, i, j, k, c_i, c_j, c_k,
                     body_vecs, det_angles, field_rad_deg,
                     tol_rad, mode, gate):
        """
        Try to confirm triangle (i,j,k)↔(c_i,c_j,c_k) with a 4th star.

        Returns one of:
          "pass"          — a catalog star consistent with all three new angles
                            (and the handedness) was found.
          "contradiction" — for at least one 4th detection a plausible catalog
                            counterpart existed (it satisfied the distances to
                            *two* triangle vertices) yet failed the third
                            constraint. That is evidence AGAINST the triangle.
          "absent"        — no tried detection had any plausible counterpart.
                            This is missing evidence, NOT a refutation: only
                            ~1 in 5 bright TESS detections is a V-band catalog
                            star at mag<7.5, so a true triangle routinely lands
                            here. Conflating it with "contradiction" is what
                            makes the classic hard gate veto true triangles.

        Candidate selection and tolerance differ by mode:
          "hard" — brightest-first (body_vecs are brightness-ordered), flat
                   tolerance PYRAMID_TOL_SCALE × pair tol. This reproduces the
                   textbook implementation exactly.
          "soft" — centre-of-field first, tolerance growing with field radius.
                   Both changes target the SIP residual gradient, which is the
                   reason a flat off-axis tolerance misses true 4th stars.

        Returns (verdict, det_l, c_l) — det_l is the detection index that
        produced the verdict (for visualisation), c_l the confirming catalog star
        on "pass". Both are None when nothing was examined.

        `gate` may be None (counterfactual audit — the caller records its own
        counters and must not pollute the live ones).
        """
        t_gate = time.perf_counter()
        N = len(body_vecs)
        others = [m for m in range(N) if m not in (i, j, k)]
        if not others:
            return "absent", None, None

        if mode == "soft":
            # Centre-first: on-axis detections carry ~10× less SIP residual,
            # so they are the most informative confirmations available.
            others.sort(key=lambda m: field_rad_deg[m])
        others = others[:PYRAMID_K_CANDIDATES]

        cross_ij = np.cross(body_vecs[i], body_vecs[j])
        r_tri = max(field_rad_deg[i], field_rad_deg[j], field_rad_deg[k])

        # The first constraint's O(N_catalog) dot product depends only on c_i,
        # which is fixed for the whole candidate loop — hoist it out. Without
        # this the gate would do up to PYRAMID_K_CANDIDATES full catalog scans
        # per triangle, and the ablation would be measuring a bad implementation
        # rather than the method.
        dots_i = self.db.star_vecs @ self.db.star_vecs[c_i]

        saw_contradiction = False
        n_tried = 0
        verdict, c_l, det_l = "absent", None, None
        contra_det = None

        for l in others:
            n_tried += 1
            if mode == "soft":
                r_max = max(r_tri, field_rad_deg[l])
                tol_l = tol_rad * (1.0 + PYRAMID_TOL_SLOPE *
                                   (r_max / PYRAMID_FIELD_CORNER_DEG))
            else:
                tol_l = PYRAMID_TOL_SCALE * tol_rad

            det_sign_4 = np.sign(np.dot(cross_ij, body_vecs[l]))
            status, cand = self._probe_fourth(
                c_i, c_j, c_k,
                det_angles[i, l], det_angles[j, l], det_angles[k, l],
                tol_l, det_sign_4 if det_sign_4 != 0 else None,
                dots_i=dots_i)

            if status == "pass":
                verdict, c_l, det_l = "pass", cand, l
                break
            if status == "contradiction":
                saw_contradiction = True
                if contra_det is None:
                    contra_det = l

        if verdict != "pass" and saw_contradiction:
            verdict, det_l = "contradiction", contra_det

        if gate is not None:
            gate.calls += 1
            gate.candidates_tried += n_tried
            gate.seconds += time.perf_counter() - t_gate
            if verdict == "pass":
                gate.passed += 1
            elif verdict == "contradiction":
                gate.veto_contradiction += 1
            else:
                gate.veto_absent += 1
        return verdict, det_l, c_l

    def _probe_fourth(self, c_i, c_j, c_k, θ_il, θ_jl, θ_kl, tol_rad,
                      det_sign=None, dots_i=None):
        """
        Single 4th-star probe. Find catalog star c_l with
          angle(c_i,c_l) ≈ θ_il  AND  angle(c_j,c_l) ≈ θ_jl  AND  angle(c_k,c_l) ≈ θ_kl
        and, if det_sign is given, sign((v_i × v_j) · v_l) == det_sign
        (rejects the mirror-image solution).

        Returns (status, c_l) with status in {"pass", "contradiction", "absent"}.
        The two-constraint survivor set is the pivot: a star at the right
        distance from BOTH c_i and c_j is a plausible counterpart for this
        detection, so failing the remaining checks is a genuine contradiction.
        Having nothing survive those first two constraints just means the
        detection is not in the catalog.

        Optimised: at most one full O(N) dot product (against c_i), and the
        caller can supply it precomputed via `dots_i` since it is the same for
        every 4th-star candidate of a given triangle. Every later constraint
        runs on the handful of survivors, fully vectorised.
        """
        sv  = self.db.star_vecs
        v_i = sv[c_i]
        v_j = sv[c_j]
        v_k = sv[c_k]

        # First angle constraint: stars at angle ~θ_il from v_i (one O(N) dot)
        if dots_i is None:
            dots_i = sv @ v_i
        cand = np.where((dots_i >= np.cos(θ_il + tol_rad)) &
                        (dots_i <= np.cos(θ_il - tol_rad)))[0]
        if cand.size == 0:
            return "absent", None

        # Second angle constraint — evaluated ONLY on first-constraint survivors
        dots_j = sv[cand] @ v_j
        keep = (dots_j >= np.cos(θ_jl + tol_rad)) & (dots_j <= np.cos(θ_jl - tol_rad))
        # Drop the three seed stars themselves
        keep &= (cand != c_i) & (cand != c_j) & (cand != c_k)
        cand   = cand[keep]
        dots_j = dots_j[keep]
        if cand.size == 0:
            # Nothing is simultaneously at the right distance from two vertices
            # → this detection has no plausible catalog counterpart.
            return "absent", None

        # From here on a plausible counterpart DOES exist, so any further
        # elimination is evidence against the triangle, not missing data.
        dots_k = sv[cand] @ v_k
        keep3 = (dots_k >= np.cos(θ_kl + tol_rad)) & (dots_k <= np.cos(θ_kl - tol_rad))
        cand   = cand[keep3]
        dots_j = dots_j[keep3]
        dots_k = dots_k[keep3]
        if cand.size == 0:
            return "contradiction", None

        # Chirality filter — keep only matching triple-product sign (vectorised)
        if det_sign is not None:
            cross_ij = np.cross(v_i, v_j)
            keep2 = np.sign(sv[cand] @ cross_ij) == det_sign
            cand   = cand[keep2]
            dots_j = dots_j[keep2]
            dots_k = dots_k[keep2]
            if cand.size == 0:
                return "contradiction", None

        # Pick the candidate closest to all three target angles (vectorised argmin)
        e_i = np.abs(np.arccos(np.clip(dots_i[cand], -1, 1)) - θ_il)
        e_j = np.abs(np.arccos(np.clip(dots_j,       -1, 1)) - θ_jl)
        e_k = np.abs(np.arccos(np.clip(dots_k,       -1, 1)) - θ_kl)
        return "pass", int(cand[np.argmin(e_i + e_j + e_k)])

    def _verify_full(self, body_vecs, R, tol_rad):
        """
        Project body vectors through R into ICRS.
        For each, find nearest catalog star — if within tol, match.
        Returns {det_idx: cat_idx} (cat_idx are GLOBAL catalog indices).

        Optimised with an FOV-cone cull: only catalog stars within 12° of the
        candidate boresight (R applied to the camera +x axis) can possibly
        match an in-frame detection (TESS FOV corner is ~8.5° off boresight,
        and tol is ≤200"). Restricting the nearest-neighbour search to that
        cone shrinks the (det × cat) score matrix from 25k to ~150 columns —
        a ~150× speedup — without changing which matches are returned.
        """
        body_in_icrs = body_vecs @ R.T

        # FOV-cone cull around the boresight (camera +x axis → ICRS)
        bs = R @ np.array([1.0, 0.0, 0.0])
        cone_cos = np.cos(np.radians(12.0))          # >> TESS 8.5° corner
        cat_idx = np.where(self.db.star_vecs @ bs >= cone_cos)[0]
        self._last_cone_n = int(cat_idx.size)   # read by identify(): chance level
        if cat_idx.size == 0:
            return {}
        sub_vecs = self.db.star_vecs[cat_idx]

        dots = body_in_icrs @ sub_vecs.T             # (N_body × N_cone)
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
            ci = int(cat_idx[best_per_det[di]])      # map cone-local → global
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
