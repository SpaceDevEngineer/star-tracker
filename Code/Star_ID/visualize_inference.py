"""
visualize_inference.py — render diagnostic overlays for each lost-in-space solve.

Reads the per-image artefacts written by `inference_full.py --out-dir <DIR>` and
produces, for each image:

  1. <stem>_overlay.png  — image + UNet detections (red), projected catalog
                           after plate-solve (blue), matched pairs (green lines).
                           Title carries the angular error, # matches, and
                           median Euclidean pixel residual.
  2. <stem>_residuals.png — per-star pixel-residual heatmap + histogram.

Plus one summary figure:

  3. summary.png — bar chart of angular error per image, time-budget breakdown,
                   error-vs-matches scatter.

Usage:
    python visualize_inference.py \
        --data-dir Data/dataset_tess_test \
        --run-dir  Results/star_id_run \
        --out-dir  Results/star_id_run/viz
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

CODE_DIR = Path(__file__).parent
sys.path.insert(0, str(CODE_DIR))
from inference_full import (
    extract_camera_intrinsics, build_pose_wcs, radec_to_unit_vec,
)

warnings.filterwarnings("ignore")


def per_image_overlay(img_arr, art, label, out_path):
    """Plot detections + projected catalog + matches over the image."""
    fig, ax = plt.subplots(figsize=(11, 11), dpi=100)
    # Stretch image for visibility
    vmin, vmax = np.percentile(img_arr, [1, 99.7])
    ax.imshow(img_arr, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")

    det = np.array(art["det_xy"])
    ax.scatter(det[:, 0], det[:, 1], facecolors="none", edgecolors="red",
               s=22, linewidths=0.7, label=f"U-Net detections (n={len(det)})")

    if not art["failed"]:
        # Project catalog stars near boresight through the predicted WCS
        intr, cd_intrinsic, _ = extract_camera_intrinsics(label["pose"]["wcs_header"])
        wcs_pred = build_pose_wcs(*art["pose_pred"], intr, cd_intrinsic)
        # Use the matched catalog stars (sufficient for visualisation)
        mp = art["matched_pairs"]
        cat_ra  = np.array([m["ra_deg"]  for m in mp])
        cat_dec = np.array([m["dec_deg"] for m in mp])
        px_proj, py_proj = wcs_pred.all_world2pix(cat_ra, cat_dec, 0)
        ax.scatter(px_proj, py_proj, marker="+", c="cyan", s=70, linewidths=1.2,
                   label=f"Catalog (matched, n={len(mp)})")
        # Match lines
        for m, pxp, pyp in zip(mp, px_proj, py_proj):
            ax.plot([m["px"], pxp], [m["py"], pyp], "-", c="lime",
                    linewidth=0.6, alpha=0.7)
        title = (f"{art['image']}\n"
                 f"Angular error: {art['angular_error_arcsec']:.2f}\" "
                 f"({art['angular_error_arcsec']/label['pose']['plate_scale_arcsec_per_pix']:.2f} px)   "
                 f"matches: {art['n_matched']}   "
                 f"plate-solve median residual: {art['median_px_residual']:.3f} px")
        ax.set_title(title, fontsize=10)
    else:
        ax.set_title(f"{art['image']}  —  FAILED ({art['reason']})", fontsize=10)

    ax.set_xlim(0, img_arr.shape[1])
    ax.set_ylim(img_arr.shape[0], 0)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def residual_map(art, label, out_path):
    """Per-star residual scatter + histogram."""
    if art["failed"] or not art["matched_pairs"]:
        return
    intr, cd_intrinsic, _ = extract_camera_intrinsics(label["pose"]["wcs_header"])
    wcs_pred = build_pose_wcs(*art["pose_pred"], intr, cd_intrinsic)
    mp = art["matched_pairs"]
    px_obs = np.array([m["px"]      for m in mp])
    py_obs = np.array([m["py"]      for m in mp])
    cat_ra  = np.array([m["ra_deg"]  for m in mp])
    cat_dec = np.array([m["dec_deg"] for m in mp])
    px_pred, py_pred = wcs_pred.all_world2pix(cat_ra, cat_dec, 0)
    px_pred = np.asarray(px_pred); py_pred = np.asarray(py_pred)
    res_px = np.hypot(px_obs - px_pred, py_obs - py_pred)
    ps = label["pose"]["plate_scale_arcsec_per_pix"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))

    sc = axL.scatter(px_obs, py_obs, c=res_px * ps, s=35, cmap="viridis")
    axL.invert_yaxis()
    axL.set_aspect("equal")
    axL.set_xlim(0, label["image_shape"][1]); axL.set_ylim(label["image_shape"][0], 0)
    axL.set_title(f"Per-star pixel residual ({art['image'][-28:]})\n"
                  f"median {np.median(res_px)*ps:.2f}\", max {res_px.max()*ps:.2f}\"")
    axL.set_xlabel("x [px]"); axL.set_ylabel("y [px]")
    plt.colorbar(sc, ax=axL, label='residual [arcsec]')

    axR.hist(res_px * ps, bins=20, color="steelblue", edgecolor="k")
    axR.set_xlabel("residual [arcsec]"); axR.set_ylabel("count")
    axR.set_title("Residual histogram")
    axR.axvline(np.median(res_px) * ps, color="r", ls="--", label=f"median {np.median(res_px)*ps:.2f}\"")
    axR.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def summary_figure(arts, out_path):
    """Bar chart of error per image, time per stage, error-vs-matches scatter."""
    names = [a["image"][-30:].replace("_tess_v01_img", "") for a in arts]
    err   = np.array([a.get("angular_error_arcsec", np.nan) for a in arts])
    nmatch = np.array([a.get("n_matched", 0) for a in arts])
    t_det = np.array([a.get("time_detect_s",   0) for a in arts])
    t_id  = np.array([a.get("time_identify_s", 0) for a in arts])
    ok    = np.array([not a["failed"] for a in arts])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Bar: angular error
    colors = ["seagreen" if o else "crimson" for o in ok]
    axes[0].bar(range(len(arts)), np.where(np.isnan(err), 0, err), color=colors)
    axes[0].set_xticks(range(len(arts)))
    axes[0].set_xticklabels(names, rotation=70, fontsize=7)
    axes[0].set_ylabel("angular error [arcsec]")
    axes[0].set_title(f"End-to-end attitude error  ({ok.sum()}/{len(arts)} solved)")
    axes[0].axhline(20.57, color="gray", ls="--", lw=1, label="1 px (plate scale)")
    if ok.any():
        axes[0].axhline(np.median(err[ok]), color="blue", ls=":", lw=1,
                        label=f"median {np.median(err[ok]):.1f}\"")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].set_yscale("log")

    # Scatter: error vs n_matched
    axes[1].scatter(nmatch[ok], err[ok], c="seagreen", s=40)
    axes[1].set_xlabel("# verified matches")
    axes[1].set_ylabel("angular error [arcsec]")
    axes[1].set_title("Accuracy improves with more matches")
    axes[1].grid(alpha=0.4)

    # Stacked bar: time per stage
    width = 0.6
    axes[2].bar(range(len(arts)), t_det, width, color="steelblue", label="U-Net detect")
    axes[2].bar(range(len(arts)), t_id,  width, bottom=t_det, color="darkorange",
                label="Star ID + plate-solve")
    axes[2].set_xticks(range(len(arts)))
    axes[2].set_xticklabels(names, rotation=70, fontsize=7)
    axes[2].set_ylabel("time [s]")
    axes[2].set_title("Time budget per image")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Directory containing images/ and labels/")
    ap.add_argument("--run-dir",  required=True, help="Directory containing <stem>.json artefacts")
    ap.add_argument("--out-dir",  required=True, help="Where to write the visualisations")
    ap.add_argument("--summary-only", action="store_true",
                    help="Only regenerate summary.png (skip per-image plots).")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    run_dir  = Path(args.run_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    artefacts = []
    for art_path in sorted(run_dir.glob("*.json")):
        art = json.load(open(art_path))
        artefacts.append(art)

        if args.summary_only:
            continue

        stem = art_path.stem
        img_path = data_dir / "images" / f"{stem}.png"
        lbl_path = data_dir / "labels" / f"{stem}.json"
        if not img_path.exists() or not lbl_path.exists():
            print(f"  skip {stem}: image or label missing")
            continue
        label = json.load(open(lbl_path))
        img_arr = np.array(Image.open(img_path), dtype=np.float32)

        per_image_overlay(img_arr, art, label, out_dir / f"{stem}_overlay.png")
        residual_map(art, label, out_dir / f"{stem}_residuals.png")

        status = "FAIL" if art["failed"] else f"{art['angular_error_arcsec']:.2f}\""
        print(f"  {stem[-30:]:<30}  {status}")

    summary_figure(artefacts, out_dir / "summary.png")
    if args.summary_only:
        print(f"Wrote summary.png for {len(artefacts)} artefacts to {out_dir}")
    else:
        print(f"\nWrote {len(artefacts)} per-image plots + summary.png to {out_dir}")


if __name__ == "__main__":
    main()
