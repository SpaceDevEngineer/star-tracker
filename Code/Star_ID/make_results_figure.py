"""Headline results figure from the per-frame CSVs (physical body-roll convention).

Usage:
    python Code/Star_ID/make_results_figure.py --out docs/images/results.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PLATE_SCALE = 20.57
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6a64", "#e4e3de"
SETS = [("Test split", "full_test_per_frame.csv", "#2a78d6"),
        ("Unseen sectors", "unseen_sectors_per_frame.csv", "#eb6834")]


def load(name):
    with open(ROOT / "Results" / name, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["attitude_error_arcsec"]]
    col = lambda k: np.array([float(r[k]) for r in rows])
    return col("attitude_error_arcsec"), col("cross_boresight_arcsec"), col("roll_arcsec")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "images" / "results.png")
    args = ap.parse_args()

    plt.rcParams.update({"font.size": 11, "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.edgecolor": MUTED})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=SURFACE,
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # (a) cumulative distribution of total attitude error
    for label, name, color in SETS:
        err, _, _ = load(name)
        x = np.sort(err)
        y = np.arange(1, len(x) + 1) / len(x) * 100
        ax1.step(x, y, where="post", color=color, linewidth=2,
                 label=f"{label} ({len(x)}/{len(x)} solved, median {np.median(x):.1f}″)")
    ax1.axvline(PLATE_SCALE, color=MUTED, linestyle="--", linewidth=1)
    ax1.text(PLATE_SCALE * 1.04, 6, "1 pixel = 20.57″", color=MUTED, fontsize=10)
    ax1.set_xscale("log")
    ax1.set_xlim(1, 100)
    ax1.set_ylim(0, 101)
    ax1.set_xlabel("attitude error (arcsec, log scale)")
    ax1.set_ylabel("frames with error ≤ x (%)")
    ax1.set_title("(a) Attitude error, 248 real TESS frames", loc="left", fontsize=12)
    ax1.legend(frameon=False, loc="upper left", fontsize=10)

    # (b) per-axis medians: pointing is tight, roll dominates
    width = 0.36
    for i, (label, name, color) in enumerate(SETS):
        _, cross, roll = load(name)
        vals = [np.median(cross), np.median(roll)]
        xs = np.arange(2) + (i - 0.5) * width
        ax2.bar(xs, vals, width - 0.04, color=color, label=label)
        for xv, v in zip(xs, vals):
            ax2.text(xv, v + 0.2, f"{v:.1f}″", ha="center", va="bottom", fontsize=10, color=INK)
    ax2.set_xticks([0, 1], ["cross-boresight\n(pointing)", "roll\n(about boresight)"])
    ax2.set_ylabel("median error (arcsec)")
    ax2.set_title("(b) Error by axis", loc="left", fontsize=12)
    ax2.legend(frameon=False, loc="upper left", fontsize=10)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, facecolor=SURFACE)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
