"""Export a compact, auditable per-frame table and metrics from pipeline JSON artefacts.

The final pipeline is the first attempt plus the refusal-retry, so a frame's
outcome comes from ``--retry-dir`` when it was retried and from ``--run-dir``
otherwise. The error is the angle of R_gtᵀ R_pred. Thesis-era artefacts store
roll as a CD-matrix angle that differs from the physical body roll by a
constant per frame; that constant multiplies both rotations from the right,
so it cancels in R_gtᵀ R_pred and the total, cross-boresight and roll errors
are exact either way.

Usage:
    python Code/Star_ID/summarize_results.py \
        --run-dir   /path/to/first_attempt \
        --retry-dir /path/to/retry_on_refusal \
        --timing    /path/to/bench_timing_fast/timing_w*.jsonl \
        --set-name  test \
        --output    Results/full_test_per_frame.csv \
        --metrics   Results/full_test_metrics.json
"""

import argparse
import csv
import json
import re
from datetime import date
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from inference_full import _R_from_pose

ARCSEC_PER_RAD = 206264.80624709636
PLATE_SCALE_ARCSEC = 20.57
# A false lock is a wrong star identification. Those land degrees away on every
# axis, so they are flagged on the cross-boresight error, which (unlike roll)
# a single CCD constrains to a few arcseconds.
FALSE_LOCK_ARCSEC = 60.0


def attitude_error(pose_pred, pose_gt):
    """Total, cross-boresight and roll error (arcsec) of R_gtᵀ R_pred."""
    R_pred = _R_from_pose(*map(float, pose_pred))
    R_gt = _R_from_pose(*map(float, pose_gt))
    rv = Rotation.from_matrix(R_gt.T @ R_pred).as_rotvec() * ARCSEC_PER_RAD
    return float(np.linalg.norm(rv)), float(np.hypot(rv[1], rv[2])), float(abs(rv[0]))


def load_dir(path):
    if path is None:
        return {}
    return {p.name: json.loads(p.read_text()) for p in sorted(path.glob("hlsp*.json"))}


def load_timing(paths, set_name):
    timing = {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            if set_name is None or row.get("set") == set_name:
                timing[row["frame"] + ".json"] = row
    return timing


def stats(values):
    v = np.asarray(values, dtype=float)
    return {"median": float(np.median(v)), "mean": float(np.mean(v)),
            "p90": float(np.percentile(v, 90)), "p95": float(np.percentile(v, 95)),
            "min": float(np.min(v)), "max": float(np.max(v))}


def wilson_interval(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(max(0.0, centre - half)), float(min(1.0, centre + half))]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--retry-dir", type=Path)
    parser.add_argument("--timing", nargs="*", default=[])
    parser.add_argument("--set-name", default=None,
                        help="Value of the 'set' field to select from --timing files.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    args = parser.parse_args()

    first = load_dir(args.run_dir)
    retry = load_dir(args.retry_dir)
    timing = load_timing(args.timing, args.set_name)

    rows = []
    first_attempt_errors = []
    for name, result in first.items():
        final = retry.get(name, result)
        attempts = int(final.get("n_attempts", 1))
        reason = final.get("reason") or ""
        row = {
            "image": final.get("image", name.replace(".json", ".png")),
            "sector": int(re.search(r"_s(\d{4})-", name).group(1)),
            "camera": int(re.search(r"-cam(\d)-", name).group(1)),
            "ccd": int(re.search(r"-ccd(\d)_", name).group(1)),
        }
        if final.get("failed"):
            status = ("data_excluded" if reason.startswith("label_missing_wcs")
                      else "algorithm_refusal")
            row.update(status=status, attitude_error_arcsec="",
                       cross_boresight_arcsec="", roll_arcsec="", pixel_error="")
        else:
            err, cross, roll = attitude_error(final["pose_pred"], final["pose_gt"])
            status = "false_lock" if cross > FALSE_LOCK_ARCSEC else "solved"
            row.update(status=status,
                       attitude_error_arcsec=round(err, 4),
                       cross_boresight_arcsec=round(cross, 4),
                       roll_arcsec=round(roll, 4),
                       pixel_error=round(err / PLATE_SCALE_ARCSEC, 4))
        if not result.get("failed"):
            first_attempt_errors.append(
                attitude_error(result["pose_pred"], result["pose_gt"])[0])
        t = timing.get(name, {})
        row.update({
            "attempts": attempts,
            "detections": final.get("n_det", ""),
            "id_matches": final.get("n_matched_id", ""),
            "final_matches": final.get("n_matched", ""),
            "ransac_iterations": final.get("iterations", ""),
            "residual_px": (round(final["median_px_residual"], 4)
                            if final.get("median_px_residual") is not None else ""),
            "frame_time_gpu_s": round(t["t_total"], 3) if "t_total" in t else "",
            "reason": reason,
        })
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    returned = [r for r in rows if r["status"] in ("solved", "false_lock")]
    errors = np.array([r["attitude_error_arcsec"] for r in returned], dtype=float)
    cross = np.array([r["cross_boresight_arcsec"] for r in returned], dtype=float)
    roll = np.array([r["roll_arcsec"] for r in returned], dtype=float)
    counts = {
        "total_frames": len(rows),
        "data_excluded": sum(r["status"] == "data_excluded" for r in rows),
        "solved_frames": len(returned),
        "algorithm_refusals": sum(r["status"] == "algorithm_refusal" for r in rows),
        "false_locks": sum(r["status"] == "false_lock" for r in rows),
        "solved_first_attempt": len(first_attempt_errors),
        "recovered_by_retry": sum(r["attempts"] > 1 and r["status"] != "algorithm_refusal"
                                  for r in rows),
    }
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"Counts: {counts}")
    if errors.size:
        print(f"Error: median={np.median(errors):.2f}″ p90={np.percentile(errors, 90):.2f}″ "
              f"max={np.max(errors):.2f}″  cross={np.median(cross):.2f}″ "
              f"roll={np.median(roll):.2f}″")

    if args.metrics and errors.size:
        times = [r["frame_time_gpu_s"] for r in rows if r["frame_time_gpu_s"] != ""]
        metrics = {
            "evaluation": f"lost-in-space TESS attitude, {args.set_name or 'all'} set, "
                          "final pipeline (first attempt + refusal-retry)",
            "attitude_error": "angle of R_gt^T R_pred (body-to-ICRS); cross-boresight = "
                              "|(y, z)| and roll = |x| of its rotation vector",
            "false_lock_definition": (f"wrong star identification: cross-boresight error "
                                      f"> {FALSE_LOCK_ARCSEC:.0f} arcsec"),
            "counts": counts,
            "solve_rate_percent": 100.0 * len(returned) / len(rows),
            "solve_rate_wilson95_percent": [100 * x for x in wilson_interval(len(returned), len(rows))],
            "attitude_error_arcsec": stats(errors),
            "attitude_error_px_median": float(np.median(errors) / PLATE_SCALE_ARCSEC),
            "axis_error_arcsec": {
                "cross_boresight_median": float(np.median(cross)),
                "cross_boresight_p90": float(np.percentile(cross, 90)),
                "roll_median": float(np.median(roll)),
                "roll_p90": float(np.percentile(roll, 90)),
            },
            "threshold_counts": {
                "subpixel_below_20_57_arcsec": int(np.sum(errors < PLATE_SCALE_ARCSEC)),
                "below_60_arcsec": int(np.sum(errors < FALSE_LOCK_ARCSEC)),
            },
            "first_attempt_attitude_error_arcsec": stats(first_attempt_errors),
            "frame_time_gpu_s": stats(times) if times else None,
            "provenance": {
                "source_artifacts": f"{len(rows)} per-frame JSON files from the thesis evaluation run",
                "check": "errors equal the rigid rotation between predicted and "
                         "reference WCS sky grids (tests/test_core_geometry.py)",
                "generated": date.today().isoformat(),
            },
        }
        args.metrics.write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"Wrote {args.metrics}")


if __name__ == "__main__":
    main()
