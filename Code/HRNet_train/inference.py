"""
inference.py — End-to-end HR-Net star tracker evaluation pipeline.

Full pipeline (per test image):
    PNG → HR-Net tiles → centroids → match to Hipparcos catalog →
    Wahba SVD → quaternion → angular error vs ground truth

Usage:
    python inference.py \
        --data-dir ~/star_tracker/Data/dataset_tess \
        --model    ~/star_tracker/Results/hrnet_run1/best_model.pt \
        --catalog  ~/star_tracker/Data/hybrid/catalog_hipparcos_full.csv \
        --use-gt --threshold 0.55

Notes:
    --use-gt synthesises reference vectors from the ground-truth quaternion,
    so this script isolates centroid-detection quality (apples-to-apples
    with the U-Net inference under the same flag).  Without --use-gt the
    linear pixel_to_body_vec leaves 200-1000" SIP residuals at the edges
    of the field — for true lost-in-space attitude use Code/Star_ID/inference_full.py.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.spatial.distance import cdist

# Reuse U-Net's data-loading helpers (split_pairs) and centroid extractor —
# these are IDENTICAL to what HR-Net training used (see Code/HRNet_train/train.py).
sys.path.insert(0, str(Path(__file__).parent.parent / "Model_train_code"))
from train import extract_centroids, split_pairs

# HR-Net model definition
sys.path.insert(0, str(Path(__file__).parent))
from hrnet_model import HRNet

ARCSEC_PER_RAD = 206264.806


# ---------------------------------------------------------------------------
# 1. Camera model
# ---------------------------------------------------------------------------
def pixel_to_body_vec(x, y, cx, cy, plate_scale_rad):
    """
    Pixel (x,y) → unit vector in camera body frame.
    Convention from compute_pose() in process_tess.py:
      camera x = boresight, camera y = east, camera z = north
    In PNG coords (y↓): east = -(x-cx), north = -(y-cy)
    """
    vec = np.array([1.0,
                    -(x - cx) * plate_scale_rad,
                    -(y - cy) * plate_scale_rad])
    return vec / np.linalg.norm(vec)


def radec_to_unit(ra_deg, dec_deg):
    """RA/Dec (degrees) → unit vector in ICRS frame."""
    ra  = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    return np.array([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec),
    ])


def quat_wxyz_to_rot(q_wxyz):
    """Quaternion [w,x,y,z] → 3×3 rotation matrix."""
    w, x, y, z = q_wxyz
    return np.array([
        [1-2*(y*y+z*z),  2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),    1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),    2*(y*z+w*x),   1-2*(x*x+y*y)],
    ])


# ---------------------------------------------------------------------------
# 2. Wahba SVD solver
# ---------------------------------------------------------------------------
def wahba_svd(body_vecs, ref_vecs):
    """
    Solve Wahba's problem via SVD.
    Finds rotation R such that R @ body_i ≈ ref_i for each pair.
    Returns (3×3) rotation matrix.
    """
    B = sum(np.outer(r, b) for b, r in zip(body_vecs, ref_vecs))
    U, _, Vt = np.linalg.svd(B)
    # Ensure proper rotation (det = +1, not reflection)
    det_sign = np.linalg.det(U) * np.linalg.det(Vt)
    R = U @ np.diag([1.0, 1.0, det_sign]) @ Vt
    return R


def rot_to_quat(R):
    """3×3 rotation matrix → unit quaternion [w, x, y, z]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array([0.25 / s,
                         (R[2, 1] - R[1, 2]) * s,
                         (R[0, 2] - R[2, 0]) * s,
                         (R[1, 0] - R[0, 1]) * s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                         (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                         0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                         (R[1, 2] + R[2, 1]) / s, 0.25 * s])


def angular_error_arcsec(q_pred, q_gt):
    """
    Angular difference between two attitude quaternions, in arcseconds.
    Both quaternions must be [w, x, y, z] and unit-normalised.
    The angle is the magnitude of the relative rotation.
    """
    q_pred = q_pred / np.linalg.norm(q_pred)
    q_gt   = q_gt   / np.linalg.norm(q_gt)
    dot = min(abs(np.dot(q_pred, q_gt)), 1.0)
    return np.degrees(2.0 * np.arccos(dot)) * 3600.0


# ---------------------------------------------------------------------------
# 3. Full-image HR-Net inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def detect_stars(model, img_path, device, tile_size=512, threshold=0.55):
    """
    Run HR-Net on all non-overlapping tiles of a full TESS image.
    Returns (N,2) array of (x, y) centroids in full-image pixel coordinates.
    """
    img = np.array(Image.open(img_path), dtype=np.float32) / 255.0
    h, w = img.shape
    all_centroids = []

    for row in range(0, h - tile_size + 1, tile_size):
        for col in range(0, w - tile_size + 1, tile_size):
            tile = img[row:row + tile_size, col:col + tile_size]
            t = torch.tensor(tile[None, None]).to(device)
            pred = model(t).cpu().numpy().squeeze()
            for lx, ly in extract_centroids(pred, threshold=threshold):
                all_centroids.append((lx + col, ly + row))

    return np.array(all_centroids, dtype=np.float32) if all_centroids else np.zeros((0, 2))


# ---------------------------------------------------------------------------
# 4. Star ID + attitude solver
# ---------------------------------------------------------------------------
def solve_attitude(detected_xy, catalog_stars, cx, cy, plate_scale_rad,
                   match_px=5.0, min_matches=4):
    """
    Match detected centroids to catalog stars (by pixel proximity),
    then solve Wahba's problem to get a quaternion.

    catalog_stars: list of {x, y, ra_deg, dec_deg}
    Returns (q_wxyz, n_matched) or (None, n_matched) if too few matches.
    """
    if len(detected_xy) == 0 or len(catalog_stars) == 0:
        return None, 0

    cat_xy = np.array([[s["x"], s["y"]] for s in catalog_stars])
    dist   = cdist(detected_xy, cat_xy)

    body_vecs, ref_vecs = [], []
    used_pred, used_cat = set(), set()

    for flat_idx in np.argsort(dist.ravel()):
        pi, ci = np.unravel_index(flat_idx, dist.shape)
        if dist[pi, ci] > match_px:
            break
        if pi in used_pred or ci in used_cat:
            continue
        used_pred.add(pi)
        used_cat.add(ci)

        bv = pixel_to_body_vec(detected_xy[pi, 0], detected_xy[pi, 1],
                               cx, cy, plate_scale_rad)
        if "_rv" in catalog_stars[ci]:
            rv = catalog_stars[ci]["_rv"]
        else:
            rv = radec_to_unit(catalog_stars[ci]["ra_deg"], catalog_stars[ci]["dec_deg"])
        body_vecs.append(bv)
        ref_vecs.append(rv)

    n_matched = len(body_vecs)
    if n_matched < min_matches:
        return None, n_matched

    R = wahba_svd(body_vecs, ref_vecs)
    q = rot_to_quat(R)
    return q / np.linalg.norm(q), n_matched


# ---------------------------------------------------------------------------
# 5. Main evaluation loop
# ---------------------------------------------------------------------------
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Hipparcos catalog (not needed in --use-gt mode)
    hip_lookup = {}
    if not args.use_gt:
        import pandas as pd
        cat_df = pd.read_csv(args.catalog)
        hip_lookup = {
            int(row["hipparcos_id"]): (float(row["ra_deg"]), float(row["dec_deg"]))
            for _, row in cat_df.iterrows()
        }
        print(f"Hipparcos catalog: {len(hip_lookup):,} stars")

    # Load model
    model = HRNet(base_ch=32).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"Model: {args.model}\n")

    _, _, test_pairs = split_pairs(args.data_dir)
    print(f"Test images: {len(test_pairs)}\n")
    
    errors_arcsec = []
    n_total = n_solved = n_failed_match = 0

    for img_path, lbl_path in test_pairs:
        with open(lbl_path) as f:
            label = json.load(f)

        # --- ground truth ---
        pose = label["pose"]
        # JSON stores quaternion as [x,y,z,w]; convert to [w,x,y,z]
        xyzw = pose["quaternion_xyzw"]
        q_gt = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])

        img_h, img_w = label["image_shape"]
        cx, cy = img_w / 2.0, img_h / 2.0
        plate_scale_rad = np.radians(pose["plate_scale_arcsec_per_pix"] / 3600.0)

        # --- build catalog star list with reference vectors ---
        R_gt = quat_wxyz_to_rot(q_gt)
        catalog_stars = []
        if args.use_gt:
            # Compute exact ICRS direction for each catalog star from ground truth rotation.
            # Bypasses catalog CSV lookup — isolates centroid detection error only.
            for s in label["stars"]:
                bv = pixel_to_body_vec(s["x"], s["y"], cx, cy, plate_scale_rad)
                catalog_stars.append({"x": s["x"], "y": s["y"], "_rv": R_gt @ bv})
        else:
            for s in label["stars"]:
                hip_id = int(s["hipparcos_id"])
                if hip_id in hip_lookup:
                    ra, dec = hip_lookup[hip_id]
                    catalog_stars.append({"x": s["x"], "y": s["y"],
                                          "ra_deg": ra, "dec_deg": dec})

        if len(catalog_stars) < 4:
            print(f"SKIP {img_path.name}: only {len(catalog_stars)} catalog stars")
            continue

        # --- detect stars with HR-Net ---
        detected = detect_stars(model, img_path, device,
                                threshold=args.threshold)

        # --- solve attitude ---
        n_total += 1
        q_pred, n_matched = solve_attitude(
            detected, catalog_stars, cx, cy, plate_scale_rad,
            match_px=args.match_px, min_matches=args.min_matches,
        )

        if q_pred is None:
            n_failed_match += 1
            print(f"FAIL {img_path.name}  detected={len(detected)}  "
                  f"matched={n_matched}  (need ≥{args.min_matches})")
        else:
            err = angular_error_arcsec(q_pred, q_gt)
            errors_arcsec.append(err)
            n_solved += 1
            print(f"OK   {img_path.name}  detected={len(detected)}  "
                  f"matched={n_matched}  error={err:.1f} arcsec")

    # --- summary ---
    print(f"\n{'='*55}")
    print(f"Images evaluated : {n_total}")
    print(f"Solved           : {n_solved}  ({100*n_solved/max(n_total,1):.1f}%)")
    print(f"Failed (matches) : {n_failed_match}")

    if errors_arcsec:
        e = np.array(errors_arcsec)
        print(f"\nAngular error (arcsec) — {n_solved} solved images:")
        print(f"  Median    : {np.median(e):.1f}\"")
        print(f"  Mean      : {np.mean(e):.1f}\"")
        print(f"  Std       : {np.std(e):.1f}\"")
        print(f"  90th pct  : {np.percentile(e, 90):.1f}\"")
        print(f"  95th pct  : {np.percentile(e, 95):.1f}\"")
        print(f"  Max       : {np.max(e):.1f}\"")

        # Convert arcsec → degrees for context
        print(f"\n  Median    : {np.median(e)/3600:.4f}°")
        print(f"  90th pct  : {np.percentile(e, 90)/3600:.4f}°")

        _save_plots(e, Path(args.model).parent)

    return errors_arcsec


def _save_plots(errors, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].hist(errors, bins=30, edgecolor="black", color="steelblue")
    axes[0].set_xlabel('Angular error (arcsec)')
    axes[0].set_ylabel('Images')
    axes[0].set_title('Angular error distribution')
    axes[0].axvline(np.median(errors), color="red", linestyle="--",
                    label=f'Median {np.median(errors):.0f}"')
    axes[0].axvline(np.percentile(errors, 90), color="orange", linestyle="--",
                    label=f'90th pct {np.percentile(errors, 90):.0f}"')
    axes[0].legend()

    # CDF
    sorted_e = np.sort(errors)
    cdf = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
    axes[1].plot(sorted_e, cdf, "steelblue")
    axes[1].set_xlabel('Angular error (arcsec)')
    axes[1].set_ylabel('Fraction of images')
    axes[1].set_title('CDF of angular error')
    axes[1].axhline(0.5, color="red",    linestyle="--", alpha=0.5, label="50%")
    axes[1].axhline(0.9, color="orange", linestyle="--", alpha=0.5, label="90%")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    out = out_dir / "angular_error.png"
    plt.savefig(out, dpi=120)
    print(f"\nPlot saved → {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",    required=True,
                   help="Dataset folder with images/ and labels/")
    p.add_argument("--model",       required=True,
                   help="Path to best_model.pt")
    p.add_argument("--catalog",     default=None,
                   help="Hipparcos CSV with columns: hipparcos_id, ra_deg, dec_deg")
    p.add_argument("--use-gt",      action="store_true",
                   help="Use ground truth quaternion for reference vectors. "
                        "Bypasses catalog lookup — measures centroid error only.")
    p.add_argument("--threshold",   type=float, default=0.55,
                   help="HR-Net detection threshold (default 0.55)")
    p.add_argument("--match-px",    type=float, default=5.0,
                   help="Max pixel distance for centroid→catalog matching (default 5px)")
    p.add_argument("--min-matches", type=int,   default=4,
                   help="Min matched stars to attempt attitude solve (default 4)")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
