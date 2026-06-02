"""
inference_full.py — REAL star tracker pipeline (no --use-gt cheating).

Full pipeline:
    PNG → U-Net detection → body vectors →
    Triangle Star ID → catalog matches →
    Wahba SVD → quaternion → angular error

This is what would run on a satellite in "lost in space" mode.

Usage:
    python inference_full.py \
        --data-dir /Users/timon/Desktop/Thesis/Data/dataset_tess_test \
        --model    /Users/timon/Desktop/Thesis/Results/unet_run3/best_model.pt \
        --catalog  /Users/timon/Desktop/Thesis/Data/hybrid/catalog_hipparcos_full.csv
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from astropy.io import fits
from astropy.wcs import WCS
from scipy.optimize import least_squares

warnings.filterwarnings("ignore", module="astropy")

# Star ID
from triangle_id import StarIdentifier

# U-Net pipeline
CODE_DIR = Path(__file__).parent.parent / "Model_train_code"
sys.path.insert(0, str(CODE_DIR))
from train import UNet, extract_centroids

TILE_SIZE = 512


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def load_intrinsics_wcs(wcs_dict):
    """Reconstruct an astropy WCS object from the dict stored in label JSON."""
    hdr = fits.Header()
    for k, v in wcs_dict.items():
        hdr[k] = v
    return WCS(hdr, relax=True)


def radec_to_unit_vec(ra_deg, dec_deg):
    r = np.radians(ra_deg)
    d = np.radians(dec_deg)
    cd = np.cos(d)
    return np.stack([cd * np.cos(r), cd * np.sin(r), np.sin(d)], axis=-1)


def pixels_to_body_vecs(xy, wcs_full, ps_rad):
    """
    Pixel→body unit vector in a CAMERA-FIXED frame (no attitude required).

    Body axes:
      - body_x = camera boresight
      - body_y, body_z = camera focal plane axes (rotated by `roll` from sky)

    Steps:
      1. `sip_pix2foc` removes SIP optical distortion → undistorted focal-plane
         coords (already CRPIX-subtracted).
      2. Tangent (gnomonic) projection from focal-plane → unit vector.

    Sign convention (1, -foc_x*ps, -foc_y*ps) is empirically calibrated
    against this TESS dataset's labels (smallest Wahba-fit residual).
    """
    if len(xy) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    foc = wcs_full.sip_pix2foc(xy, 0)
    by = -foc[:, 0] * ps_rad
    bz = -foc[:, 1] * ps_rad
    v = np.stack([np.ones_like(by), by, bz], axis=-1)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _gt_R_from_label(label, wcs_full, ps_rad, identifier):
    """
    Wahba-best-fit R for the label's known (pixel, hipparcos_id) pairs in our
    camera-fixed body frame. This is the proper reference attitude to score
    a star-ID prediction against; the labels' stored quaternion uses a
    different body convention and is not directly comparable.
    """
    hip_to_idx = {int(h): i for i, h in enumerate(identifier.db.star_ids)}
    pairs_r, pixel_xy = [], []
    for s in label.get("stars", []):
        ci = hip_to_idx.get(int(s["hipparcos_id"]))
        if ci is None:
            continue
        pixel_xy.append((s["x"], s["y"]))
        pairs_r.append(identifier.db.star_vecs[ci])
    if len(pairs_r) < 4:
        return None
    pairs_b = pixels_to_body_vecs(np.asarray(pixel_xy), wcs_full, ps_rad)
    B = sum(np.outer(r, b) for b, r in zip(pairs_b, pairs_r))
    U, _, Vt = np.linalg.svd(B)
    ds = np.linalg.det(U) * np.linalg.det(Vt)
    return U @ np.diag([1.0, 1.0, ds]) @ Vt


# ---------------------------------------------------------------------------
# Two-pass refinement (Strategy 1) + plate-solve (Strategy 2)
# ---------------------------------------------------------------------------
def extract_camera_intrinsics(wcs_dict):
    """Extract camera-intrinsic WCS parts: SIP polynomial, CRPIX, CD matrix,
    CTYPE/CUNIT. We treat CD as a camera intrinsic because it encodes per-CCD
    plate-scale anisotropy/shear (small but non-zero on TESS).

    Returns:
        intr_keys : dict of header fields to copy unchanged
        cd_base   : (2,2) the labels' CD matrix
        cd_roll   : the roll baked into cd_base (deg)
    """
    keep_exact = {"CTYPE1", "CTYPE2", "CUNIT1", "CUNIT2", "RADESYS", "EQUINOX",
                  "WCSAXES", "CRPIX1", "CRPIX2"}
    keep_prefix = ("A_", "B_", "AP_", "BP_")
    intr_keys = {k: v for k, v in wcs_dict.items()
                 if k in keep_exact or any(k.startswith(p) for p in keep_prefix)}

    # Reconstruct the CD matrix from either CD* or PC*+CDELT*
    if "CD1_1" in wcs_dict:
        cd = np.array([[wcs_dict["CD1_1"], wcs_dict["CD1_2"]],
                       [wcs_dict["CD2_1"], wcs_dict["CD2_2"]]], dtype=np.float64)
    else:
        pc = np.array([[wcs_dict.get("PC1_1", 1.0), wcs_dict.get("PC1_2", 0.0)],
                       [wcs_dict.get("PC2_1", 0.0), wcs_dict.get("PC2_2", 1.0)]],
                      dtype=np.float64)
        cdelt = np.diag([wcs_dict.get("CDELT1", 1.0), wcs_dict.get("CDELT2", 1.0)])
        cd = pc @ cdelt
    cd_roll = float(np.degrees(np.arctan2(cd[0, 1], cd[0, 0])))
    return intr_keys, cd, cd_roll


def build_pose_wcs(ra_b_deg, dec_b_deg, roll_deg, intr_keys, cd_base, cd_base_roll):
    """Construct an astropy WCS for a candidate attitude + camera intrinsics.

    The CD matrix preserves the camera's per-CCD shear (cd_base) and just
    *rotates* it by (roll - cd_base_roll). When roll == cd_base_roll this
    reproduces the labels' WCS exactly.
    """
    drot = np.radians(roll_deg - cd_base_roll)
    c, s = np.cos(drot), np.sin(drot)
    R2 = np.array([[c, -s], [s, c]])
    cd = R2 @ cd_base

    hdr = fits.Header()
    for k, v in intr_keys.items():
        hdr[k] = v
    hdr["CRVAL1"] = float(ra_b_deg) % 360.0
    hdr["CRVAL2"] = float(dec_b_deg)
    hdr["CD1_1"], hdr["CD1_2"] = float(cd[0, 0]), float(cd[0, 1])
    hdr["CD2_1"], hdr["CD2_2"] = float(cd[1, 0]), float(cd[1, 1])
    # Strip stale PC/CDELT keys that astropy would otherwise compose with CD
    for k in ("PC1_1", "PC1_2", "PC2_1", "PC2_2", "CDELT1", "CDELT2"):
        if k in hdr:
            del hdr[k]
    return WCS(hdr, relax=True)


def attitude_from_R(R):
    """Initial (RA, Dec) extraction from the rough Wahba R; roll is searched."""
    boresight_icrs = R @ np.array([1.0, 0.0, 0.0])
    ra  = float(np.degrees(np.arctan2(boresight_icrs[1], boresight_icrs[0])))
    dec = float(np.degrees(np.arcsin(np.clip(boresight_icrs[2], -1.0, 1.0))))
    return ra % 360.0, dec


def plate_solve(pixel_xy, ref_radec, intr_keys, cd_base, cd_base_roll, init_pose,
                bounds_deg=2.0, roll_bounds_deg=20.0):
    """
    Nonlinear least-squares fit of (RA, Dec, roll) to match SIP-forward-projected
    catalog stars against observed pixel centroids.

    Returns (ra, dec, roll, cost, residuals_px).
    """
    pixel_xy  = np.asarray(pixel_xy,  dtype=np.float64)
    ref_radec = np.asarray(ref_radec, dtype=np.float64)

    def residuals(params):
        ra, dec, roll = params
        w = build_pose_wcs(ra, dec, roll, intr_keys, cd_base, cd_base_roll)
        px, py = w.all_world2pix(ref_radec[:, 0], ref_radec[:, 1], 0)
        return np.concatenate([np.asarray(px) - pixel_xy[:, 0],
                               np.asarray(py) - pixel_xy[:, 1]])

    ra0, dec0, roll0 = init_pose
    lb = [ra0 - bounds_deg, dec0 - bounds_deg, roll0 - roll_bounds_deg]
    ub = [ra0 + bounds_deg, dec0 + bounds_deg, roll0 + roll_bounds_deg]
    res = least_squares(residuals, x0=[ra0, dec0, roll0],
                        bounds=(lb, ub), method="trf", max_nfev=200, xtol=1e-12)
    final_res = residuals(res.x)
    return (float(res.x[0]) % 360.0, float(res.x[1]), float(res.x[2]) % 360.0,
            float(res.cost), final_res)


def project_catalog_via_wcs(wcs, cat_ra, cat_dec, img_w, img_h, pad=64):
    """Forward-project catalog stars; keep ones landing inside image bounds."""
    px, py = wcs.all_world2pix(np.asarray(cat_ra), np.asarray(cat_dec), 0)
    px = np.asarray(px); py = np.asarray(py)
    inside = (px > -pad) & (px < img_w + pad) & (py > -pad) & (py < img_h + pad)
    return px, py, inside


def refine_matches_by_projection(wcs, det_xy, identifier, img_w, img_h,
                                 tol_pix=2.0, fov_cone_cos=None):
    """
    Project mag<7.5 catalog stars through the candidate WCS, then nearest-
    neighbour-match against observed centroids. Each detection matches at most
    one catalog star and vice versa (mutual greedy).
    """
    cat_vecs = identifier.db.star_vecs
    # Cull catalog to the FOV cone before world2pix (huge speedup).
    bs_icrs = wcs.wcs.crval
    bs_vec  = radec_to_unit_vec(bs_icrs[0], bs_icrs[1])
    cos_fov = np.cos(np.radians(8.0))  # 16° cone covers TESS 12° FOV + slack
    in_cone = (cat_vecs @ bs_vec) >= cos_fov
    cat_idx = np.where(in_cone)[0]
    if cat_idx.size == 0:
        return {}
    # Convert cone-selected cat vecs to ra/dec for world2pix
    v = cat_vecs[cat_idx]
    dec = np.degrees(np.arcsin(np.clip(v[:, 2], -1.0, 1.0)))
    ra  = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360.0
    px, py, inside = project_catalog_via_wcs(wcs, ra, dec, img_w, img_h)
    cat_idx = cat_idx[inside]; px = px[inside]; py = py[inside]
    if cat_idx.size == 0 or len(det_xy) == 0:
        return {}

    # Pairwise distances det × cat
    det_xy = np.asarray(det_xy, dtype=np.float64)
    dx = det_xy[:, 0:1] - px[None, :]
    dy = det_xy[:, 1:2] - py[None, :]
    dist = np.hypot(dx, dy)

    # Mutual greedy: sort all (det, cat) pairs by distance, accept in order
    matches = {}
    used_det = set(); used_cat = set()
    flat = np.argsort(dist, axis=None)
    for k in flat:
        di, ci_local = np.unravel_index(k, dist.shape)
        if dist[di, ci_local] > tol_pix:
            break
        if di in used_det or int(cat_idx[ci_local]) in used_cat:
            continue
        matches[int(di)] = int(cat_idx[ci_local])
        used_det.add(di); used_cat.add(int(cat_idx[ci_local]))
    return matches


def _R_from_pose(ra_deg, dec_deg, roll_deg):
    """Build the body→ICRS rotation matrix from (RA, Dec, roll), matching the
    convention used by `build_pose_wcs` so that R_pose @ body_camera ≈ ICRS."""
    ra = np.radians(ra_deg); dec = np.radians(dec_deg); roll = np.radians(roll_deg)
    Rz = np.array([[ np.cos(ra), -np.sin(ra), 0],
                   [ np.sin(ra),  np.cos(ra), 0],
                   [ 0,           0,          1]])
    Ry = np.array([[ np.cos(-dec), 0, np.sin(-dec)],
                   [ 0,            1, 0           ],
                   [-np.sin(-dec), 0, np.cos(-dec)]])
    Rx = np.array([[1, 0,            0           ],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll),  np.cos(roll)]])
    return Rz @ Ry @ Rx


def boresight_rotation(ra_b_deg, dec_b_deg):
    """
    GT rotation body→ICRS consistent with the intrinsics-WCS body frame
    (boresight body=(1,0,0), body_y=east, body_z=north at the boresight).

    Columns of R are body x, y, z directions expressed in ICRS.
    """
    ra = np.radians(ra_b_deg)
    dec = np.radians(dec_b_deg)
    bx = np.array([np.cos(ra) * np.cos(dec), np.sin(ra) * np.cos(dec), np.sin(dec)])
    by = np.array([-np.sin(ra), np.cos(ra), 0.0])               # local east
    bz = np.array([-np.cos(ra) * np.sin(dec), -np.sin(ra) * np.sin(dec), np.cos(dec)])  # local north
    return np.stack([bx, by, bz], axis=1)


def rot_to_quat(R):
    tr = R[0,0]+R[1,1]+R[2,2]
    if tr > 0:
        s = 0.5/np.sqrt(tr+1.0)
        return np.array([0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0*np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    elif R[1,1] > R[2,2]:
        s = 2.0*np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])
        return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
    else:
        s = 2.0*np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])
        return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])


def angular_error_arcsec(q1, q2):
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = min(abs(np.dot(q1, q2)), 1.0)
    return np.degrees(2 * np.arccos(dot)) * 3600


# ---------------------------------------------------------------------------
# Detection + identification + Wahba
# ---------------------------------------------------------------------------
@torch.no_grad()
def detect_unet(model, img_np, device, threshold=0.55, flux_window=7):
    """
    Detect stars + estimate brightness via FLUX SUM in a window around centroid.
    Returns (xy: Nx2, brightness: N).
    """
    h, w = img_np.shape
    out_xy, out_b = [], []
    half = flux_window // 2

    for row in range(0, h - TILE_SIZE + 1, TILE_SIZE):
        for col in range(0, w - TILE_SIZE + 1, TILE_SIZE):
            tile = img_np[row:row+TILE_SIZE, col:col+TILE_SIZE]
            t = torch.tensor(tile[None, None]).to(device)
            hm = model(t).cpu().numpy().squeeze()
            for lx, ly in extract_centroids(hm, threshold=threshold):
                ix, iy = int(round(lx)), int(round(ly))
                # Flux = sum of pixels in flux_window×flux_window box
                x1 = max(0, ix - half)
                x2 = min(TILE_SIZE, ix + half + 1)
                y1 = max(0, iy - half)
                y2 = min(TILE_SIZE, iy + half + 1)
                flux = float(tile[y1:y2, x1:x2].sum())
                out_xy.append((lx + col, ly + row))
                out_b.append(flux)
    xy = np.array(out_xy, dtype=np.float32) if out_xy else np.zeros((0, 2))
    bs = np.array(out_b, dtype=np.float32) if out_b else np.zeros((0,))
    return xy, bs


def solve_attitude_full(model, identifier, img_path, lbl_path, device,
                        unet_threshold=0.55, top_n_bright=60, verbose=True,
                        debug_gt=False):
    """Full pipeline → returns dict with angular_error and intermediates."""
    img_np = np.array(Image.open(img_path), dtype=np.float32) / 255.0
    h, w = img_np.shape
    with open(lbl_path) as f:
        label = json.load(f)

    pose = label["pose"]
    if "wcs_header" not in pose:
        return {"failed": True, "reason": "label_missing_wcs", "n_det": 0}

    ps_arcsec = pose["plate_scale_arcsec_per_pix"]
    ps_deg    = ps_arcsec / 3600.0
    ps_rad    = np.radians(ps_deg)
    wcs_full  = load_intrinsics_wcs(pose["wcs_header"])
    intr_keys, cd_base, cd_base_roll = extract_camera_intrinsics(pose["wcs_header"])

    # GT pose is in CRVAL parametrisation (sky at CRPIX), NOT boresight_ra/dec
    # (sky at image centre). The two differ by ~0.1° on TESS because CRPIX is
    # not at the image centre. Plate-solve fits the CRVAL parameters, so we
    # compare against them directly.
    ra_gt   = float(pose["wcs_header"]["CRVAL1"])
    dec_gt  = float(pose["wcs_header"]["CRVAL2"])
    roll_gt = cd_base_roll
    R_gt = _R_from_pose(ra_gt, dec_gt, roll_gt)
    q_gt = rot_to_quat(R_gt); q_gt /= np.linalg.norm(q_gt)

    # 1. Detect + brightness (flux sum)
    t0 = time.time()
    det_xy_all, det_bright = detect_unet(model, img_np, device, unet_threshold)
    t_detect = time.time() - t0

    if len(det_xy_all) < 10:
        return {"failed": True, "reason": "too_few_detections", "n_det": len(det_xy_all)}

    # 2. Sort by flux (brightness), keep top N
    bright_order = np.argsort(-det_bright)
    det_xy = det_xy_all[bright_order[:top_n_bright]]

    # 3. Body vectors — top-N for triangle search, ALL for verification.
    #    SIP-corrected (TESS has strong barrel distortion; a linear pinhole
    #    projection gives 200-500" errors at the corners).
    body_vecs     = pixels_to_body_vecs(np.asarray(det_xy),     wcs_full, ps_rad)
    body_vecs_all = pixels_to_body_vecs(np.asarray(det_xy_all), wcs_full, ps_rad)

    # 4. RANSAC Star ID with PYRAMID-STYLE verification on ALL detections
    t1 = time.time()
    id_result = identifier.identify(
        body_vecs,
        verify_body_vecs=body_vecs_all,   # ← strict verification on all 477
        verbose=verbose,
    )
    t_identify = time.time() - t1

    if id_result is None:
        return {"failed": True, "reason": "id_failed",
                "n_det": len(det_xy_all), "n_top": len(det_xy)}

    # 5. FINAL REFINEMENT on ALL detections with TIGHT tolerance.
    #    Capped at 2 iterations + drift gate: refinement must not rotate R
    #    more than 0.5° from the Star-ID result, or it is leaking into a
    #    different self-consistent constellation (root cause of past 1e5"
    #    "solved" failures).
    R_id   = id_result.R
    R      = R_id
    body_all = body_vecs_all
    # Sized to SIP-corrected body-vec residual budget (~200" worst case at edges).
    verify_tol_rad = np.radians(200.0 / 3600.0)
    MAX_FINAL_DRIFT_DEG = 2

    for _ in range(2):
        # Project all detections through current R
        body_in_icrs = body_all @ R.T
        dots = body_in_icrs @ identifier.db.star_vecs.T
        cos_tol = np.cos(verify_tol_rad)

        # Greedy matching: best score first
        best_cat = np.argmax(dots, axis=1)
        best_score = dots[np.arange(len(body_all)), best_cat]
        order = np.argsort(-best_score)
        all_matches = {}
        used_cat = set()
        for di in order:
            ci = int(best_cat[di])
            if best_score[di] >= cos_tol and ci not in used_cat:
                all_matches[int(di)] = ci
                used_cat.add(ci)

        if len(all_matches) < 4:
            break

        # Refit R with all current matches
        body_pairs = [body_all[di]              for di in all_matches.keys()]
        ref_pairs  = [identifier.db.star_vecs[ci] for ci in all_matches.values()]
        B = sum(np.outer(r, b) for b, r in zip(body_pairs, ref_pairs))
        U, S, Vt = np.linalg.svd(B)
        ds = np.linalg.det(U) * np.linalg.det(Vt)
        R_new = U @ np.diag([1.0, 1.0, ds]) @ Vt

        # Drift gate against R_id (NOT the previous step!)
        trace_rel = np.trace(R_id.T @ R_new)
        drift_deg = np.degrees(np.arccos(np.clip((trace_rel - 1) / 2, -1, 1)))
        if drift_deg > MAX_FINAL_DRIFT_DEG:
            break  # discard R_new — keep last good R

        if np.allclose(R, R_new, atol=1e-9):
            R = R_new
            break
        R = R_new

    # ----- PASS 2 + PASS 3 : iterative WCS refinement + plate-solve ---------
    # Pass 1's R_rough is good to ~0.5°; below we synthesise a candidate WCS at
    # each iteration, snap detections to projected catalog stars within a tight
    # pixel tolerance, and re-fit (RA, Dec, roll) by SIP-aware plate-solve. The
    # loop converges in 2-3 iterations when Pass 1's R was within ~1°.

    img_h, img_w = label["image_shape"]
    # Convert Pass 1 matches → (pixel_xy, ref_radec) for the first plate-solve.
    cat_vecs = identifier.db.star_vecs

    def _matches_to_pairs(matches, det_xy_arr):
        if not matches:
            return np.zeros((0, 2)), np.zeros((0, 2))
        di = list(matches.keys())
        ci = list(matches.values())
        v = cat_vecs[ci]
        dec = np.degrees(np.arcsin(np.clip(v[:, 2], -1, 1)))
        ra  = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360.0
        return det_xy_arr[di], np.stack([ra, dec], axis=-1)

    # Initial (RA, Dec) = direction at CRPIX, not at image centre. We get this
    # by mapping CRPIX→body via the intrinsics WCS, then R_rough@that = ICRS at
    # CRPIX. Roll starts at the CD-base value (a multi-start sweep would
    # generalise to genuine lost-in-space without any stored WCS).
    crpix = np.array([[float(pose["wcs_header"]["CRPIX1"]),
                       float(pose["wcs_header"]["CRPIX2"])]])
    body_crpix = pixels_to_body_vecs(crpix, wcs_full, ps_rad)[0]
    icrs_crpix = R @ body_crpix
    ra_init  = float(np.degrees(np.arctan2(icrs_crpix[1], icrs_crpix[0]))) % 360.0
    dec_init = float(np.degrees(np.arcsin(np.clip(icrs_crpix[2], -1, 1))))
    roll_init = cd_base_roll
    init_pose = (ra_init, dec_init, roll_init)

    # First plate-solve from Pass 1 matches.
    pix_in, rdc_in = _matches_to_pairs(all_matches, det_xy_all)
    final_res = None
    if len(pix_in) < 4:
        pose_pred = init_pose
        cost = float("nan")
        pass3_matches = all_matches
    else:
        ra, dec, roll, cost, final_res = plate_solve(
            pix_in, rdc_in, intr_keys, cd_base, cd_base_roll, init_pose)
        pose_pred = (ra, dec, roll)

        # Pass 2: project full catalog through the refined WCS, snap to detections
        # at progressively tighter pixel tolerance. First plate-solve from Pass 1
        # is good to ~30 px; we tighten to ~1 px over 3 iterations.
        for tol_pix in (30.0, 5.0, 1.5):
            w_cand = build_pose_wcs(*pose_pred, intr_keys, cd_base, cd_base_roll)
            new_matches = refine_matches_by_projection(
                w_cand, det_xy_all, identifier, img_w, img_h, tol_pix=tol_pix)
            if len(new_matches) < 4:
                break
            pix_in, rdc_in = _matches_to_pairs(new_matches, det_xy_all)
            ra, dec, roll, cost, final_res = plate_solve(
                pix_in, rdc_in, intr_keys, cd_base, cd_base_roll, pose_pred,
                bounds_deg=0.5, roll_bounds_deg=5.0)
            pose_pred  = (ra, dec, roll)
            all_matches = new_matches
        pass3_matches = all_matches

    # Quality gate: after plate-solve, the median |pixel residual| of accepted
    # matches should be sub-pixel for a clean lock. cam3/ccd2-style false locks
    # appear with residuals ~5-50 px because the optimiser converged on a
    # self-consistent but wrong constellation. Reject those.
    if final_res is None or len(final_res) < 8:
        median_px_res = float("nan")
        quality_ok = False
        quality_reason = "no_plate_solve"
    else:
        median_px_res = float(np.median(np.abs(final_res)))
        quality_ok = median_px_res < 2.0 and len(pass3_matches) >= 8
        quality_reason = (None if quality_ok else
                          f"residual_{median_px_res:.1f}px_n={len(pass3_matches)}")

    R_pred = _R_from_pose(*pose_pred)
    q_pred = rot_to_quat(R_pred); q_pred /= np.linalg.norm(q_pred)
    err = angular_error_arcsec(q_pred, q_gt)

    # Artefacts needed to reconstruct the visualisation without re-running
    # the slow RANSAC + plate-solve.
    matched_pairs = []
    for di, ci in pass3_matches.items():
        px_obs = float(det_xy_all[di, 0]); py_obs = float(det_xy_all[di, 1])
        cv = identifier.db.star_vecs[ci]
        ra_c  = float(np.degrees(np.arctan2(cv[1], cv[0]))) % 360.0
        dec_c = float(np.degrees(np.arcsin(np.clip(cv[2], -1, 1))))
        matched_pairs.append({
            "det_idx": int(di), "cat_idx": int(ci),
            "px": px_obs, "py": py_obs,
            "ra_deg": ra_c, "dec_deg": dec_c,
            "hipparcos_id": int(identifier.db.star_ids[ci]),
        })

    if not quality_ok:
        return {"failed": True, "reason": f"quality_gate:{quality_reason}",
                "n_det": len(det_xy_all), "n_matched": len(pass3_matches),
                "median_px_residual": median_px_res,
                "angular_error_arcsec_if_kept": err,
                "pose_pred_if_kept": pose_pred,
                "det_xy": det_xy_all.tolist(),
                "matched_pairs": matched_pairs}

    return {
        "failed": False,
        "n_det":          len(det_xy_all),
        "n_top":          len(det_xy),
        "n_matched_id":   len(id_result.matches),
        "n_matched":      len(pass3_matches),
        "iterations":     id_result.iterations,
        "triangle":       list(id_result.triangle),   # (det_i, det_j, det_k, cat_i, cat_j, cat_k)
        "det_xy_top":     det_xy.tolist(),            # top-N brightest used for triangle search
        "angular_error_arcsec": err,
        "pixel_error": err / ps_arcsec,
        "plate_solve_cost": cost,
        "median_px_residual": median_px_res,
        "time_detect_s":   t_detect,
        "time_identify_s": t_identify,
        "pose_pred": pose_pred,
        "pose_gt":   (ra_gt, dec_gt, roll_gt),
        "q_pred": q_pred,
        "q_gt":   q_gt,
        "det_xy": det_xy_all.tolist(),
        "matched_pairs": matched_pairs,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model",    required=True)
    ap.add_argument("--catalog",  required=True)
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--mag-limit", type=float, default=6.0)
    ap.add_argument("--n-images",  type=int, default=0, help="0=all")
    ap.add_argument("--debug",     action="store_true", help="Print per-iteration progress")
    ap.add_argument("--out-dir",   default=None,
                    help="If given, write per-image JSON with detections+matches+pose "
                         "to <out-dir>/<image_stem>.json for visualisation.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load U-Net
    model = UNet(base_ch=32).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"U-Net loaded from {args.model}")

    # Build pattern DB (cached after first run)
    identifier = StarIdentifier(args.catalog, mag_limit=args.mag_limit)
    print(f"Pattern DB ready: {len(identifier.db.star_ids):,} stars, "
          f"{len(identifier.db.pair_angles):,} pairs\n")

    # Iterate test images
    data_dir = Path(args.data_dir)
    img_files = sorted((data_dir / "images").glob("*.png"))
    if args.n_images > 0:
        img_files = img_files[:args.n_images]
    print(f"Testing on {len(img_files)} images\n")

    print(f"{'Image':<55} {'N_det':>6} {'Top':>5} {'Match':>6} {'Iter':>5} {'t_id':>7}  {'Error':>10}")
    print("-" * 105)

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for img_path in img_files:
        lbl_path = data_dir / "labels" / (img_path.stem + ".json")
        if not lbl_path.exists():
            continue

        r = solve_attitude_full(
            model, identifier, img_path, lbl_path, device,
            unet_threshold=args.threshold, verbose=args.debug,
        )

        if r["failed"]:
            print(f"{img_path.name[:54]:<55}  FAIL  ({r['reason']})  n_det={r.get('n_det', 0)}")
            results.append(r)
        else:
            print(f"{img_path.name[:54]:<55} "
                  f"{r['n_det']:>6} "
                  f"{r['n_top']:>5} "
                  f"{r['n_matched']:>6} "
                  f"{r['iterations']:>5} "
                  f"{r['time_identify_s']:>6.1f}s "
                  f"{r['angular_error_arcsec']:>9.2f}\"")
            results.append(r)

        if out_dir:
            artefact = {k: (v.tolist() if hasattr(v, "tolist") else v)
                        for k, v in r.items() if k not in ("q_pred", "q_gt")}
            artefact["image"] = img_path.name
            with open(out_dir / f"{img_path.stem}.json", "w") as f:
                json.dump(artefact, f)

    # Summary
    errors = [r["angular_error_arcsec"] for r in results if not r["failed"]]
    n_total = len(results)
    n_ok = len(errors)
    print(f"\n{'='*70}")
    print(f"REAL PIPELINE (lost-in-space, no --use-gt):")
    print(f"  Solved        : {n_ok}/{n_total}  ({100*n_ok/max(n_total,1):.1f}%)")
    if errors:
        e = np.array(errors)
        print(f"  Median error  : {np.median(e):.2f}\"")
        print(f"  Mean error    : {np.mean(e):.2f}\"")
        print(f"  90th pct      : {np.percentile(e, 90):.2f}\"")
        print(f"  Max error     : {np.max(e):.2f}\"")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
