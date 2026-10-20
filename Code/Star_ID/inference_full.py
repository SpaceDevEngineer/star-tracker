"""
End-to-end lost-in-space attitude pipeline.

Stages:
    PNG → U-Net detection → SIP-corrected body vectors → Triangle Star ID
    → Wahba SVD → plate-solve refinement → quality gate → attitude quaternion
    (a quality-gate refusal triggers a significance-aware refusal-retry)

Usage:
    python inference_full.py \
        --data-dir <dataset>/dataset_tess_test \
        --model    <results>/unet_run3/best_model.pt \
        --catalog  <data>/hybrid/catalog_hipparcos_full.csv \
        --out-dir  <results>/star_id_run
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

from triangle_id import StarIdentifier

sys.path.insert(0, str(Path(__file__).parent.parent / "Model_train_code"))
from train import UNet, extract_centroids, split_pairs

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
    Pixel → body unit vector in a camera-fixed frame.

    Removes SIP optical distortion via `sip_pix2foc`, then applies a tangent
    (gnomonic) projection from the focal plane to a unit vector. The resulting
    body frame is independent of attitude — only camera intrinsics are used.
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
    Wahba-best-fit reference attitude from the label's (pixel, hipparcos_id)
    pairs, expressed in the same body frame the predictor uses. Needed because
    the labels' stored quaternion is in a different body convention.
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
# WCS synthesis + plate-solve
# ---------------------------------------------------------------------------
def extract_camera_intrinsics(wcs_dict):
    """
    Separate camera intrinsics (SIP, CRPIX, CD scale/shear) from attitude
    (CRVAL and the rotational part of CD).

    The raw FITS CD matrix mixes two physically different quantities:
      * a frame-dependent image rotation (attitude), and
      * a camera-dependent scale/shear calibration (intrinsic).

    We factor the rotation out before returning ``cd_intrinsic``. This prevents
    the plate solver from receiving the ground-truth roll through the label WCS.
    The rotation is the angle of the closest similarity transform,
    atan2(CD21 - CD12, CD11 + CD22); TESS CD matrices have det > 0, so the
    focal plane maps onto the sky without a reflection.

    Returns:
        intr_keys : header fields copied verbatim into every synthesized WCS
        cd_intrinsic : (2, 2) attitude-free CD scale/shear matrix
        physical_roll : frame roll about body +x (deg), retained only for
                        ground-truth scoring
    """
    keep_exact = {"CTYPE1", "CTYPE2", "CUNIT1", "CUNIT2", "RADESYS", "EQUINOX",
                  "WCSAXES", "CRPIX1", "CRPIX2"}
    keep_prefix = ("A_", "B_", "AP_", "BP_")
    intr_keys = {k: v for k, v in wcs_dict.items()
                 if k in keep_exact or any(k.startswith(p) for p in keep_prefix)}

    if "CD1_1" in wcs_dict:
        cd = np.array([[wcs_dict["CD1_1"], wcs_dict["CD1_2"]],
                       [wcs_dict["CD2_1"], wcs_dict["CD2_2"]]], dtype=np.float64)
    else:
        pc = np.array([[wcs_dict.get("PC1_1", 1.0), wcs_dict.get("PC1_2", 0.0)],
                       [wcs_dict.get("PC2_1", 0.0), wcs_dict.get("PC2_2", 1.0)]],
                      dtype=np.float64)
        cdelt = np.diag([wcs_dict.get("CDELT1", 1.0), wcs_dict.get("CDELT2", 1.0)])
        cd = pc @ cdelt
    cd_angle = float(np.degrees(np.arctan2(cd[1, 0] - cd[0, 1], cd[0, 0] + cd[1, 1])))
    a = np.radians(-cd_angle)
    unrotate = np.array([[np.cos(a), -np.sin(a)],
                         [np.sin(a),  np.cos(a)]])
    cd_intrinsic = unrotate @ cd
    # Camera convention: body (y, z) = -(pixel offset), i.e. a 180° turn of the
    # focal plane, so a zero physical body roll corresponds to a 180° CD angle.
    physical_roll = (cd_angle + 180.0) % 360.0
    return intr_keys, cd_intrinsic, physical_roll


def build_pose_wcs(ra_b_deg, dec_b_deg, roll_deg, intr_keys, cd_intrinsic):
    """Construct an astropy WCS for a candidate attitude + camera intrinsics.

    ``cd_intrinsic`` contains only the per-CCD scale/shear calibration. The
    candidate roll is applied explicitly, so no frame attitude is inherited
    from the ground-truth WCS.
    """
    cd_angle = np.radians(roll_deg - 180.0)
    c, s = np.cos(cd_angle), np.sin(cd_angle)
    R2 = np.array([[c, -s], [s, c]])
    cd = R2 @ cd_intrinsic

    hdr = fits.Header()
    for k, v in intr_keys.items():
        hdr[k] = v
    hdr["CRVAL1"] = float(ra_b_deg) % 360.0
    hdr["CRVAL2"] = float(dec_b_deg)
    hdr["CD1_1"], hdr["CD1_2"] = float(cd[0, 0]), float(cd[0, 1])
    hdr["CD2_1"], hdr["CD2_2"] = float(cd[1, 0]), float(cd[1, 1])
    # Astropy composes PC*CDELT *with* CD if both are present; drop the former.
    for k in ("PC1_1", "PC1_2", "PC2_1", "PC2_2", "CDELT1", "CDELT2"):
        if k in hdr:
            del hdr[k]
    return WCS(hdr, relax=True)


def attitude_from_R(R):
    """Extract (RA, Dec) of the boresight from a body→ICRS rotation."""
    boresight_icrs = R @ np.array([1.0, 0.0, 0.0])
    ra  = float(np.degrees(np.arctan2(boresight_icrs[1], boresight_icrs[0])))
    dec = float(np.degrees(np.arcsin(np.clip(boresight_icrs[2], -1.0, 1.0))))
    return ra % 360.0, dec


def plate_solve(pixel_xy, ref_radec, intr_keys, cd_intrinsic, init_pose,
                bounds_deg=2.0, roll_bounds_deg=20.0):
    """
    Refine (RA, Dec, roll) by minimising pixel-space residuals between
    observed centroids and catalog stars projected through the SIP forward map.

    Returns (ra, dec, roll, cost, residuals_px).
    """
    pixel_xy  = np.asarray(pixel_xy,  dtype=np.float64)
    ref_radec = np.asarray(ref_radec, dtype=np.float64)

    def residuals(params):
        ra, dec, roll = params
        w = build_pose_wcs(ra, dec, roll, intr_keys, cd_intrinsic)
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
    Project catalog stars through the candidate WCS, then mutual-greedy
    nearest-neighbour match against observed centroids inside `tol_pix`.
    """
    cat_vecs = identifier.db.star_vecs
    bs_icrs = wcs.wcs.crval
    bs_vec  = radec_to_unit_vec(bs_icrs[0], bs_icrs[1])
    cos_fov = np.cos(np.radians(8.0))   # ~16° cone — comfortably covers TESS FOV
    in_cone = (cat_vecs @ bs_vec) >= cos_fov
    cat_idx = np.where(in_cone)[0]
    if cat_idx.size == 0:
        return {}
    v = cat_vecs[cat_idx]
    dec = np.degrees(np.arcsin(np.clip(v[:, 2], -1.0, 1.0)))
    ra  = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360.0
    px, py, inside = project_catalog_via_wcs(wcs, ra, dec, img_w, img_h)
    cat_idx = cat_idx[inside]; px = px[inside]; py = py[inside]
    if cat_idx.size == 0 or len(det_xy) == 0:
        return {}

    det_xy = np.asarray(det_xy, dtype=np.float64)
    dx = det_xy[:, 0:1] - px[None, :]
    dy = det_xy[:, 1:2] - py[None, :]
    dist = np.hypot(dx, dy)

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
    """Body→ICRS rotation R = Rz(RA)·Ry(-Dec)·Rx(roll), matching `build_pose_wcs`."""
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


def pose_from_R(R):
    """Recover (RA, Dec, roll) from a body→ICRS rotation matrix.

    This is the inverse of :func:`_R_from_pose` away from the usual Euler-angle
    singularity at |Dec|=90°. TESS pointings are comfortably away from it.
    """
    R = np.asarray(R, dtype=np.float64)
    boresight = R[:, 0]
    ra = float(np.degrees(np.arctan2(boresight[1], boresight[0]))) % 360.0
    dec = float(np.degrees(np.arcsin(np.clip(boresight[2], -1.0, 1.0))))

    R_zero_roll = _R_from_pose(ra, dec, 0.0)
    R_roll = R_zero_roll.T @ R
    roll = float(np.degrees(np.arctan2(R_roll[2, 1], R_roll[1, 1]))) % 360.0
    return ra, dec, roll


def pixel_residual_norms(final_res):
    """Convert least-squares [dx..., dy...] output to per-star radial pixels."""
    final_res = np.asarray(final_res, dtype=np.float64)
    if final_res.size == 0 or final_res.size % 2:
        return np.zeros(0, dtype=np.float64)
    n = final_res.size // 2
    return np.hypot(final_res[:n], final_res[n:])


def boresight_rotation(ra_b_deg, dec_b_deg):
    """Body→ICRS rotation with body_y=east, body_z=north (zero roll, sky-aligned)."""
    ra = np.radians(ra_b_deg)
    dec = np.radians(dec_b_deg)
    bx = np.array([np.cos(ra) * np.cos(dec), np.sin(ra) * np.cos(dec), np.sin(dec)])
    by = np.array([-np.sin(ra), np.cos(ra), 0.0])
    bz = np.array([-np.cos(ra) * np.sin(dec), -np.sin(ra) * np.sin(dec), np.cos(dec)])
    return np.stack([bx, by, bz], axis=1)


def rot_to_quat(R):
    """3×3 rotation → unit quaternion [w, x, y, z] (Shepperd's method)."""
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
# Detection + RANSAC ID + plate-solve
# ---------------------------------------------------------------------------
@torch.no_grad()
def detect_unet(model, img_np, device, threshold=0.55, flux_window=7):
    """
    Run the U-Net on every 512×512 non-overlapping tile of the full image.
    Returns (xy: Nx2 sub-pixel centroids, flux: N — sum within `flux_window`).
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
                x1 = max(0, ix - half); x2 = min(TILE_SIZE, ix + half + 1)
                y1 = max(0, iy - half); y2 = min(TILE_SIZE, iy + half + 1)
                out_xy.append((lx + col, ly + row))
                out_b.append(float(tile[y1:y2, x1:x2].sum()))
    xy = np.array(out_xy, dtype=np.float32) if out_xy else np.zeros((0, 2))
    bs = np.array(out_b, dtype=np.float32) if out_b else np.zeros((0,))
    return xy, bs


def inject_false_stars(det_xy, det_bright, img_shape, frac, seed):
    """
    Stress-test hook: add `frac` × N spurious detections at uniformly random
    pixel positions, with brightnesses drawn from the real detections so the
    fakes mix into the top-N used for the triangle search rather than sitting
    harmlessly at the faint end.

    This is what moves the pipeline into the regime the Pyramid gate was
    designed for — high false-star contamination, where triangle-only search
    starts producing false locks. Deterministic given `seed`.
    """
    if frac <= 0:
        return det_xy, det_bright, 0
    h, w = img_shape
    rng = np.random.default_rng(seed)
    n_fake = int(round(frac * len(det_xy)))
    if n_fake <= 0:
        return det_xy, det_bright, 0
    fake_xy = np.stack([rng.uniform(0, w, n_fake),
                        rng.uniform(0, h, n_fake)], axis=-1).astype(np.float32)
    # Resample real brightnesses → fakes are indistinguishable by magnitude
    fake_b = rng.choice(det_bright, size=n_fake, replace=True).astype(np.float32)
    return (np.concatenate([det_xy, fake_xy], axis=0),
            np.concatenate([det_bright, fake_b], axis=0),
            n_fake)


def solve_attitude_full(model, identifier, img_path, lbl_path, device,
                        unet_threshold=0.55, top_n_bright=60, verbose=True,
                        debug_gt=False, pyramid="off", gate_audit=False,
                        false_star_frac=0.0, stress_seed=0, max_iter=1500,
                        exclude_boresights=None, exclude_radius_deg=6.0,
                        min_significance=0.0):
    """
    Full lost-in-space pipeline for one image. Returns a dict with the
    predicted attitude, ground-truth comparison, match table, and timing.

    pyramid / gate_audit: forwarded to StarIdentifier.identify() — see there.
                          Default "off" reproduces the published baseline.
    false_star_frac:      stress-test contamination level (see inject_false_stars).
    max_iter:             RANSAC cap. Matters: a frame that never locks burns
                          every iteration (~4.4 s each), so an unsolvable frame
                          costs ~1.8 h at the default 1500. Lower it for the
                          high-contamination stress arms, where refusals are the
                          expected outcome rather than an anomaly.
    """
    img_np = np.array(Image.open(img_path), dtype=np.float32) / 255.0
    with open(lbl_path) as f:
        label = json.load(f)

    pose = label["pose"]
    if "wcs_header" not in pose:
        return {"failed": True, "reason": "label_missing_wcs", "n_det": 0}

    ps_arcsec = pose["plate_scale_arcsec_per_pix"]
    ps_rad    = np.radians(ps_arcsec / 3600.0)
    wcs_full  = load_intrinsics_wcs(pose["wcs_header"])
    intr_keys, cd_intrinsic, physical_roll_gt = extract_camera_intrinsics(
        pose["wcs_header"])

    # GT attitude uses CRVAL (sky at CRPIX), which differs from `boresight_ra_deg`
    # (sky at image centre) by ~0.1″ on TESS. Plate-solve fits CRVAL, compare against it.
    ra_gt, dec_gt, roll_gt = (float(pose["wcs_header"]["CRVAL1"]),
                              float(pose["wcs_header"]["CRVAL2"]),
                              physical_roll_gt)
    R_gt = _R_from_pose(ra_gt, dec_gt, roll_gt)
    q_gt = rot_to_quat(R_gt); q_gt /= np.linalg.norm(q_gt)

    # --- Stage 1: U-Net detection ------------------------------------------
    t0 = time.time()
    det_xy_all, det_bright = detect_unet(model, img_np, device, unet_threshold)
    t_detect = time.time() - t0

    if len(det_xy_all) < 10:
        return {"failed": True, "reason": "too_few_detections", "n_det": len(det_xy_all)}

    # --- Stage 1b: optional false-star contamination (stress test) ---------
    det_xy_all, det_bright, n_fake = inject_false_stars(
        det_xy_all, det_bright, label["image_shape"], false_star_frac, stress_seed)

    # --- Stage 2: body vectors (SIP-corrected) -----------------------------
    bright_order = np.argsort(-det_bright)
    det_xy = det_xy_all[bright_order[:top_n_bright]]
    body_vecs     = pixels_to_body_vecs(np.asarray(det_xy),     wcs_full, ps_rad)
    body_vecs_all = pixels_to_body_vecs(np.asarray(det_xy_all), wcs_full, ps_rad)

    # --- Stage 3: RANSAC triangle ID + Wahba SVD ---------------------------
    t1 = time.time()
    id_result = identifier.identify(body_vecs,
                                    verify_body_vecs=body_vecs_all,
                                    pyramid=pyramid,
                                    gate_audit=gate_audit,
                                    max_iter=max_iter,
                                    exclude_boresights=exclude_boresights,
                                    exclude_radius_deg=exclude_radius_deg,
                                    min_significance=min_significance,
                                    verbose=verbose)
    t_identify = time.time() - t1

    gate_dict = (id_result.gate.as_dict()
                 if id_result is not None and id_result.gate is not None else None)

    if id_result is None:
        return {"failed": True, "reason": "id_failed",
                "n_det": len(det_xy_all), "n_top": len(det_xy),
                "n_false_injected": n_fake,
                "time_identify_s": t_identify,
                "pyramid": pyramid, "gate": gate_dict}

    # --- Stage 4: refine R against all detections, capped by drift gate ----
    t_refine0 = time.time()      # timing only: stages 4-6 (refine, plate solve, gate)
    # Tolerance is sized to the SIP-corrected body-vec residual at the edges (~200″).
    # The drift gate prevents the refit from sliding off to a different self-
    # consistent constellation when the initial set of matches contains outliers.
    R_id     = id_result.R
    R        = R_id
    body_all = body_vecs_all
    verify_tol_rad      = np.radians(200.0 / 3600.0)
    MAX_FINAL_DRIFT_DEG = 2

    for _ in range(2):
        body_in_icrs = body_all @ R.T
        dots = body_in_icrs @ identifier.db.star_vecs.T
        cos_tol = np.cos(verify_tol_rad)

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

        body_pairs = [body_all[di]               for di in all_matches.keys()]
        ref_pairs  = [identifier.db.star_vecs[ci] for ci in all_matches.values()]
        B = sum(np.outer(r, b) for b, r in zip(body_pairs, ref_pairs))
        U, _, Vt = np.linalg.svd(B)
        ds = np.linalg.det(U) * np.linalg.det(Vt)
        R_new = U @ np.diag([1.0, 1.0, ds]) @ Vt

        trace_rel = np.trace(R_id.T @ R_new)
        drift_deg = np.degrees(np.arccos(np.clip((trace_rel - 1) / 2, -1, 1)))
        if drift_deg > MAX_FINAL_DRIFT_DEG:
            break

        if np.allclose(R, R_new, atol=1e-9):
            R = R_new
            break
        R = R_new

    # --- Stage 5: WCS projection refinement + plate-solve ------------------
    # Synthesise a candidate WCS from the rough R, project the catalog through
    # it, snap detections at tightening pixel tolerance, and re-fit (RA, Dec, roll)
    # via SIP-aware nonlinear least-squares. Converges in 2-3 iterations.
    img_h, img_w = label["image_shape"]
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

    # Seed all three attitude degrees of freedom from the blind RANSAC/Wahba
    # rotation. Body +x is the SIP reference pixel (CRPIX), so the boresight
    # of R is the CRVAL seed; roll does NOT come from the label WCS.
    init_pose = pose_from_R(R)

    pix_in, rdc_in = _matches_to_pairs(all_matches, det_xy_all)
    final_res = None
    if len(pix_in) < 4:
        pose_pred = init_pose
        cost = float("nan")
        pass3_matches = all_matches
    else:
        ra, dec, roll, cost, final_res = plate_solve(
            pix_in, rdc_in, intr_keys, cd_intrinsic, init_pose)
        pose_pred = (ra, dec, roll)

        for tol_pix in (30.0, 5.0, 1.5):
            w_cand = build_pose_wcs(*pose_pred, intr_keys, cd_intrinsic)
            new_matches = refine_matches_by_projection(
                w_cand, det_xy_all, identifier, img_w, img_h, tol_pix=tol_pix)
            if len(new_matches) < 4:
                break
            pix_in, rdc_in = _matches_to_pairs(new_matches, det_xy_all)
            ra, dec, roll, cost, final_res = plate_solve(
                pix_in, rdc_in, intr_keys, cd_intrinsic, pose_pred,
                bounds_deg=0.5, roll_bounds_deg=5.0)
            pose_pred  = (ra, dec, roll)
            all_matches = new_matches
        pass3_matches = all_matches

    # --- Stage 6: quality gate ---------------------------------------------
    # A clean attitude lock has sub-pixel post-fit residual. Larger residuals
    # mean the optimiser settled on a self-consistent but wrong constellation;
    # refuse the answer rather than report a silent garbage attitude.
    if final_res is None or len(final_res) < 8:
        median_px_res = float("nan")
        median_coordinate_res = float("nan")
        quality_ok = False
        quality_reason = "no_plate_solve"
    else:
        radial_res = pixel_residual_norms(final_res)
        median_px_res = float(np.median(radial_res))
        median_coordinate_res = float(np.median(np.abs(final_res)))
        quality_ok = median_px_res < 2.0 and len(pass3_matches) >= 8
        quality_reason = (None if quality_ok else
                          f"residual_{median_px_res:.1f}px_n={len(pass3_matches)}")

    R_pred = _R_from_pose(*pose_pred)
    q_pred = rot_to_quat(R_pred); q_pred /= np.linalg.norm(q_pred)
    err = angular_error_arcsec(q_pred, q_gt)

    matched_pairs = []
    for di, ci in pass3_matches.items():
        cv = identifier.db.star_vecs[ci]
        matched_pairs.append({
            "det_idx": int(di), "cat_idx": int(ci),
            "px": float(det_xy_all[di, 0]),
            "py": float(det_xy_all[di, 1]),
            "ra_deg":  float(np.degrees(np.arctan2(cv[1], cv[0]))) % 360.0,
            "dec_deg": float(np.degrees(np.arcsin(np.clip(cv[2], -1, 1)))),
            "hipparcos_id": int(identifier.db.star_ids[ci]),
        })

    if not quality_ok:
        return {"failed": True, "reason": f"quality_gate:{quality_reason}",
                "n_det": len(det_xy_all), "n_matched": len(pass3_matches),
                "median_px_residual": median_px_res,
                "median_coordinate_residual_px": median_coordinate_res,
                "residual_metric": "median_euclidean_px",
                "angular_error_arcsec_if_kept": err,
                "pose_pred_if_kept": pose_pred,
                "refused_boresight_icrs": (R_id @ np.array([1.0, 0.0, 0.0])).tolist(),
                "iterations":       id_result.iterations,
                "time_identify_s":  t_identify,
                "time_refine_s":    time.time() - t_refine0,
                "n_false_injected": n_fake,
                "pyramid": pyramid, "gate": gate_dict,
                "det_xy": det_xy_all.tolist(),
                "matched_pairs": matched_pairs}

    return {
        "failed": False,
        "n_det":          len(det_xy_all),
        "n_top":          len(det_xy),
        "n_false_injected": n_fake,
        "pyramid":        pyramid,
        "gate":           gate_dict,
        "n_matched_id":   len(id_result.matches),
        "id_significance": id_result.significance,
        "n_matched":      len(pass3_matches),
        "iterations":     id_result.iterations,
        "triangle":       list(id_result.triangle),
        "det_xy_top":     det_xy.tolist(),
        "angular_error_arcsec": err,
        "pixel_error":          err / ps_arcsec,
        "plate_solve_cost":     cost,
        "median_px_residual":   median_px_res,
        "median_coordinate_residual_px": median_coordinate_res,
        "residual_metric":      "median_euclidean_px",
        "attitude_convention":  "physical_body_roll_deg",
        "attitude_seed":        "ransac_wahba_ra_dec_roll",
        "calibration":          "sip_crpix_derotated_cd_scale_shear",
        "time_detect_s":        t_detect,
        "time_identify_s":      t_identify,
        "time_refine_s":        time.time() - t_refine0,
        "pose_pred":            pose_pred,
        "pose_gt":              (ra_gt, dec_gt, roll_gt),
        "q_pred":               q_pred,
        "q_gt":                 q_gt,
        "det_xy":               det_xy_all.tolist(),
        "matched_pairs":        matched_pairs,
    }


def solve_attitude_with_retry(model, identifier, img_path, lbl_path, device,
                              max_retries=2, exclude_radius_deg=6.0,
                              retry_min_significance=5.0, **kw):
    """
    Refusal-retry wrapper around `solve_attitude_full`.

    The only failure mode observed on real TESS frames is a false lock onto a
    dense-sky attractor that the post-fit quality gate refuses. When that
    happens the search is re-run with the refused boresight region forbidden
    (`exclude_boresights`), up to `max_retries` times. A frame that solves on
    the first attempt is returned untouched, so this cannot change any solved
    frame; every retried answer must pass the same quality gate, so it cannot
    let through a false lock the gate would have caught.

    Retries also switch on significance-aware stopping
    (`retry_min_significance`, see StarIdentifier.identify): the search may
    only stop early on a hypothesis that beats the chance-match level of its
    own sky region by that factor, so a dense-field impostor with 16-21 raw
    matches can no longer end the search before the true triangle is drawn.
    """
    excluded, attempts = [], []
    r = solve_attitude_full(model, identifier, img_path, lbl_path, device, **kw)
    for _ in range(max_retries):
        reason = str(r.get("reason") or "")
        if not (r.get("failed") and reason.startswith("quality_gate")
                and r.get("refused_boresight_icrs") is not None):
            break
        attempts.append({"reason": reason,
                         "pose_if_kept": r.get("pose_pred_if_kept"),
                         "error_if_kept": r.get("angular_error_arcsec_if_kept"),
                         "time_identify_s": r.get("time_identify_s")})
        excluded.append(r["refused_boresight_icrs"])
        r = solve_attitude_full(model, identifier, img_path, lbl_path, device,
                                exclude_boresights=excluded,
                                exclude_radius_deg=exclude_radius_deg,
                                min_significance=retry_min_significance, **kw)
    r["retry_attempts"] = attempts
    r["n_attempts"] = len(attempts) + 1
    return r


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True,
                    help="Dataset directory containing images/ and labels/.")
    ap.add_argument("--model",    required=True,
                    help="Path to the trained U-Net checkpoint (best_model.pt).")
    ap.add_argument("--catalog",  required=True,
                    help="Hipparcos catalog CSV (hipparcos_id, ra_deg, dec_deg, mag).")
    ap.add_argument("--threshold", type=float, default=0.55,
                    help="U-Net heatmap detection threshold (default 0.55).")
    ap.add_argument("--mag-limit", type=float, default=7.5,
                    help="Magnitude cutoff for catalog stars (default 7.5).")
    ap.add_argument("--retries", type=int, default=2,
                    help="Refusal-retries after a quality-gate refusal: the "
                         "refused sky region is excluded and significance-"
                         "aware stopping is switched on (default 2; 0 = "
                         "first attempt only).")
    ap.add_argument("--n-images",  type=int, default=0,
                    help="Stop after this many images (0 = all).")
    ap.add_argument("--split",     choices=("all", "train", "val", "test"),
                    default="all",
                    help="Restrict to a deterministic split (uses split_pairs "
                         "from the U-Net training code, seed=42, by image to "
                         "avoid tile-level leakage). 'all' (default) scans "
                         "every PNG in images/.")
    ap.add_argument("--debug",     action="store_true",
                    help="Print per-iteration RANSAC progress.")
    ap.add_argument("--out-dir",   default=None,
                    help="If set, write per-image JSON artefacts here for "
                         "downstream visualisation.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = UNet(base_ch=32).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"Model loaded from {args.model}")

    identifier = StarIdentifier(args.catalog, mag_limit=args.mag_limit)
    print(f"Pattern DB ready: {len(identifier.db.star_ids):,} stars, "
          f"{len(identifier.db.pair_angles):,} pairs\n")

    # Iterate test images
    data_dir = Path(args.data_dir)
    if args.split == "all":
        img_files = sorted((data_dir / "images").glob("*.png"))
        pair_index = {p: data_dir / "labels" / (p.stem + ".json") for p in img_files}
    else:
        train_pairs, val_pairs, test_pairs = split_pairs(str(data_dir))
        split_map = {"train": train_pairs, "val": val_pairs, "test": test_pairs}
        chosen = split_map[args.split]
        img_files  = [img for img, _ in chosen]
        pair_index = {img: lbl for img, lbl in chosen}
        print(f"Using split='{args.split}' "
              f"({len(train_pairs)}/{len(val_pairs)}/{len(test_pairs)} "
              f"train/val/test by image, seed=42)")

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
        lbl_path = pair_index[img_path]
        if not lbl_path.exists():
            continue

        r = solve_attitude_with_retry(
            model, identifier, img_path, lbl_path, device,
            max_retries=args.retries,
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
                  f"{r['angular_error_arcsec']:>9.2f}\""
                  + (f"  (retry {r['n_attempts'] - 1})" if r["n_attempts"] > 1 else ""))
            results.append(r)

        if out_dir:
            artefact = {k: (v.tolist() if hasattr(v, "tolist") else v)
                        for k, v in r.items() if k not in ("q_pred", "q_gt")}
            artefact["image"] = img_path.name
            with open(out_dir / f"{img_path.stem}.json", "w") as f:
                json.dump(artefact, f)

    errors  = [r["angular_error_arcsec"] for r in results if not r["failed"]]
    n_total = len(results)
    n_ok    = len(errors)
    print(f"\n{'='*70}")
    print(f"Lost-in-space pipeline summary:")
    print(f"  Solved       : {n_ok}/{n_total}  ({100 * n_ok / max(n_total, 1):.1f}%)")
    print(f"  Retried      : {sum(r.get('n_attempts', 1) > 1 for r in results)}")
    if errors:
        e = np.array(errors)
        print(f"  Median error : {np.median(e):.2f}\"")
        print(f"  Mean error   : {np.mean(e):.2f}\"")
        print(f"  90th pct     : {np.percentile(e, 90):.2f}\"")
        print(f"  Max error    : {np.max(e):.2f}\"")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
