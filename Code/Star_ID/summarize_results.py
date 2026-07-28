"""Export a compact, auditable per-frame table from pipeline JSON artefacts.

Usage:
    python Code/Star_ID/summarize_results.py \
        --run-dir /path/to/full/results \
        --output Results/full_test_per_frame.csv \
        --legacy-cd-roll
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from inference_full import (
    _R_from_pose,
    angular_error_arcsec,
    rot_to_quat,
)


def corrected_legacy_error(row):
    pose_pred = list(map(float, row["pose_pred"]))
    pose_gt = list(map(float, row["pose_gt"]))
    pose_pred[2] = (180.0 - pose_pred[2]) % 360.0
    pose_gt[2] = (180.0 - pose_gt[2]) % 360.0
    return float(angular_error_arcsec(
        rot_to_quat(_R_from_pose(*pose_pred)),
        rot_to_quat(_R_from_pose(*pose_gt)),
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--legacy-cd-roll",
        action="store_true",
        help="Convert stored CD-angle poses to physical body roll before scoring.",
    )
    args = parser.parse_args()

    rows = []
    for path in sorted(args.run_dir.glob("*.json")):
        result = json.loads(path.read_text())
        reason = result.get("reason", "")
        if not result.get("failed"):
            status = "solved"
            error = (
                corrected_legacy_error(result)
                if args.legacy_cd_roll
                else float(result["angular_error_arcsec"])
            )
        elif reason.startswith("label_missing_wcs"):
            status = "data_excluded"
            error = ""
        else:
            status = "algorithm_refusal"
            error = ""

        rows.append({
            "image": result.get("image", path.stem + ".png"),
            "status": status,
            "attitude_error_arcsec": error,
            "detections": result.get("n_det", ""),
            "id_matches": result.get("n_matched_id", ""),
            "final_matches": result.get("n_matched", ""),
            "ransac_iterations": result.get("iterations", ""),
            "star_id_time_s": result.get("time_identify_s", ""),
            "reason": reason,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=rows[0].keys(),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    errors = np.array([
        row["attitude_error_arcsec"]
        for row in rows
        if row["status"] == "solved"
    ], dtype=float)
    status_counts = {
        status: sum(row["status"] == status for row in rows)
        for status in ("solved", "algorithm_refusal", "data_excluded")
    }
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"Counts: {status_counts}")
    if errors.size:
        print(
            f"Error: median={np.median(errors):.2f}″ "
            f"p90={np.percentile(errors, 90):.2f}″ max={np.max(errors):.2f}″"
        )


if __name__ == "__main__":
    main()
