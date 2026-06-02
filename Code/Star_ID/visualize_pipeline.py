"""
visualize_pipeline.py — show every stage of the star-tracker pipeline on one
image, side-by-side. Useful for understanding how the U-Net → body vec →
triangle ID → plate-solve cascade actually works.

Output: a single multi-panel PNG with:
  Panel 1: PNG image + ALL U-Net detections (red)
  Panel 2: One U-Net heatmap tile (raw model output)
  Panel 3: Top-60 brightest detections + the RANSAC-chosen TRIANGLE (yellow)
  Panel 4: Triangle + matched catalog stars (cyan) and the rough Pass 1 R
  Panel 5: All Pass 1 verified matches (green lines from det → catalog)
  Panel 6: Final Pass 3 matches after plate-solve (blue lines, much tighter)

Usage:
    python visualize_pipeline.py \
        --data-dir Data/dataset_tess_test \
        --image    hlsp_tica_tess_ffi_s1751-o1-01222079-cam1-ccd1_tess_v01_img \
        --model    Results/unet_run3/best_model.pt \
        --catalog  Data/hybrid/catalog_hipparcos_full.csv \
        --out      Results/star_id_run/pipeline_cam1-ccd1.png
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image

CODE_DIR = Path(__file__).parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR.parent / "Model_train_code"))

from train import UNet, extract_centroids
from triangle_id import StarIdentifier
from inference_full import (
    detect_unet, load_intrinsics_wcs, pixels_to_body_vecs,
    extract_camera_intrinsics, build_pose_wcs, solve_attitude_full,
)

warnings.filterwarnings("ignore")
TILE_SIZE = 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--image",    required=True, help="image stem (no extension)")
    ap.add_argument("--model",    required=True)
    ap.add_argument("--catalog",  required=True)
    ap.add_argument("--out",      required=True)
    ap.add_argument("--mag-limit", type=float, default=7.5)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    img_path = data_dir / "images" / f"{args.image}.png"
    lbl_path = data_dir / "labels" / f"{args.image}.json"

    print("Loading model + catalog...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(base_ch=32).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    identifier = StarIdentifier(args.catalog, mag_limit=args.mag_limit)

    print("Running full pipeline (this takes 1-15 min depending on RANSAC)...")
    r = solve_attitude_full(model, identifier, img_path, lbl_path, device, verbose=False)
    if r["failed"]:
        print(f"  FAILED ({r['reason']}); some panels will be empty")

    # Load image + extra info
    img = np.array(Image.open(img_path), dtype=np.float32) / 255.0
    label = json.load(open(lbl_path))
    intr_keys, cd_base, cd_roll = extract_camera_intrinsics(label["pose"]["wcs_header"])

    det_xy     = np.array(r["det_xy"])
    det_xy_top = np.array(r.get("det_xy_top", []))
    triangle   = r.get("triangle", None)         # (di, dj, dk, ci, cj, ck) — top-N indices
    mp         = r.get("matched_pairs", [])

    # Re-run U-Net on the centre tile to show the heatmap output
    print("Re-running U-Net on one tile for heatmap visualisation...")
    h, w = img.shape
    row0 = (h - TILE_SIZE) // 2
    col0 = (w - TILE_SIZE) // 2
    tile = img[row0:row0+TILE_SIZE, col0:col0+TILE_SIZE]
    with torch.no_grad():
        t = torch.tensor(tile[None, None]).to(device)
        heatmap = model(t).cpu().numpy().squeeze()

    # Build composite figure -------------------------------------------------
    print("Rendering 6-panel composite...")
    fig = plt.figure(figsize=(20, 12))
    vmin, vmax = np.percentile(img, [1, 99.7])

    def img_panel(ax, title):
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
        ax.set_xlim(0, w); ax.set_ylim(h, 0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=11)

    # Panel 1: all detections
    ax1 = fig.add_subplot(2, 3, 1)
    img_panel(ax1, f"Step 1: U-Net detections (n={len(det_xy)})")
    ax1.scatter(det_xy[:, 0], det_xy[:, 1], facecolors="none", edgecolors="red",
                s=18, linewidths=0.6)

    # Panel 2: U-Net heatmap of centre tile
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(tile, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
    # Overlay heatmap with alpha
    ax2.imshow(heatmap, cmap="hot", alpha=0.55, vmin=0, vmax=1, origin="upper")
    centroids_local = np.asarray(extract_centroids(heatmap, threshold=0.55))
    if len(centroids_local) > 0:
        ax2.scatter(centroids_local[:, 0], centroids_local[:, 1],
                    facecolors="none", edgecolors="cyan", s=20, linewidths=0.7)
    ax2.set_title(f"Step 1 detail: U-Net heatmap (centre tile, {len(centroids_local)} peaks)",
                  fontsize=11)
    ax2.set_xticks([]); ax2.set_yticks([])

    # Panel 3: top-60 brightest + RANSAC triangle
    ax3 = fig.add_subplot(2, 3, 3)
    img_panel(ax3, f"Step 3: RANSAC triangle picked (iter {r.get('iterations', '?')}, n_top=60)")
    if len(det_xy_top):
        ax3.scatter(det_xy_top[:, 0], det_xy_top[:, 1], facecolors="none",
                    edgecolors="yellow", s=22, linewidths=0.7,
                    label="top-60 brightest")
    if triangle is not None and len(det_xy_top):
        di, dj, dk = triangle[:3]
        tri_pix = det_xy_top[[di, dj, dk]]
        # Draw the triangle
        tri_poly = Polygon(tri_pix, closed=True, fill=False, edgecolor="lime",
                           linewidth=2.0, label="RANSAC triangle")
        ax3.add_patch(tri_poly)
        ax3.scatter(tri_pix[:, 0], tri_pix[:, 1], c="lime", s=80, marker="*",
                    edgecolors="black", linewidths=0.8, zorder=5)
        for idx, lbl in zip([di, dj, dk], "ABC"):
            ax3.annotate(lbl, det_xy_top[idx], xytext=(8, -8), textcoords="offset points",
                         color="lime", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper right", fontsize=9, framealpha=0.85)

    # Panel 4: Triangle + matched catalog (3 cat stars from triangle)
    ax4 = fig.add_subplot(2, 3, 4)
    img_panel(ax4, "Step 4: Catalog stars matching the triangle  →  Wahba rotation")
    if triangle is not None and not r["failed"]:
        ci, cj, ck = triangle[3:6]
        cat_vecs = identifier.db.star_vecs
        wcs_pred = build_pose_wcs(*r["pose_pred"], intr_keys, cd_base, cd_roll)
        cat_idx = [ci, cj, ck]
        ra  = np.degrees(np.arctan2(cat_vecs[cat_idx, 1], cat_vecs[cat_idx, 0])) % 360
        dec = np.degrees(np.arcsin(np.clip(cat_vecs[cat_idx, 2], -1, 1)))
        px_cat, py_cat = wcs_pred.all_world2pix(ra, dec, 0)
        px_cat = np.asarray(px_cat); py_cat = np.asarray(py_cat)
        ax4.scatter(px_cat, py_cat, marker="+", c="cyan", s=200, linewidths=2.2,
                    label="catalog (3 matched)")
        # Draw the triangle for context
        di, dj, dk = triangle[:3]
        tri_pix = det_xy_top[[di, dj, dk]]
        tri_poly = Polygon(tri_pix, closed=True, fill=False, edgecolor="lime",
                           linewidth=1.5, alpha=0.7, label="detection triangle")
        ax4.add_patch(tri_poly)
        # Connect each triangle vertex to its catalog match
        for tp, pxc, pyc in zip(tri_pix, px_cat, py_cat):
            ax4.plot([tp[0], pxc], [tp[1], pyc], "-", c="white", linewidth=1.0,
                     alpha=0.7)
    ax4.legend(loc="upper right", fontsize=9, framealpha=0.85)

    # Panel 5 + 6: all verified matches at end of pipeline
    ax5 = fig.add_subplot(2, 3, 5)
    img_panel(ax5, f"Step 5: Pass-1 verified (n={r.get('n_matched_id', 0)})  →  rough R")
    if mp:
        ax5.scatter(det_xy[:, 0], det_xy[:, 1], facecolors="none",
                    edgecolors="red", s=15, linewidths=0.4, alpha=0.6)
        wcs_pred = build_pose_wcs(*r["pose_pred"], intr_keys, cd_base, cd_roll)
        cat_ra  = np.array([m["ra_deg"]  for m in mp])
        cat_dec = np.array([m["dec_deg"] for m in mp])
        px_proj, py_proj = wcs_pred.all_world2pix(cat_ra, cat_dec, 0)
        ax5.scatter(px_proj, py_proj, marker="+", c="cyan", s=50, linewidths=1.0)

    ax6 = fig.add_subplot(2, 3, 6)
    if not r["failed"]:
        title6 = (f"Step 6: Pass-3 plate-solve  →  err={r['angular_error_arcsec']:.2f}\""
                  f"  ({r['angular_error_arcsec']/label['pose']['plate_scale_arcsec_per_pix']:.2f} px),"
                  f"  median res={r['median_px_residual']:.3f} px")
    else:
        title6 = "Step 6: FAILED (quality gate)"
    img_panel(ax6, title6)
    if mp and not r["failed"]:
        wcs_pred = build_pose_wcs(*r["pose_pred"], intr_keys, cd_base, cd_roll)
        cat_ra  = np.array([m["ra_deg"]  for m in mp])
        cat_dec = np.array([m["dec_deg"] for m in mp])
        px_proj, py_proj = wcs_pred.all_world2pix(cat_ra, cat_dec, 0)
        # Draw match lines
        for m, pxp, pyp in zip(mp, px_proj, py_proj):
            ax6.plot([m["px"], pxp], [m["py"], pyp], "-", c="lime", linewidth=0.7)
        ax6.scatter([m["px"] for m in mp], [m["py"] for m in mp],
                    facecolors="none", edgecolors="red", s=18, linewidths=0.6,
                    label=f"matched detection (n={len(mp)})")
        ax6.scatter(px_proj, py_proj, marker="+", c="cyan", s=50, linewidths=1.0,
                    label="projected catalog")
        ax6.legend(loc="upper right", fontsize=9, framealpha=0.85)

    fig.suptitle(f"Pipeline trace: {args.image[-30:]}", fontsize=13, y=1.00)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
