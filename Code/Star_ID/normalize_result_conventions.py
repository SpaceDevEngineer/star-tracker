"""Normalize saved result artefacts to the physical body-roll convention.

Early experiments stored the FITS CD-matrix angle in the field named ``roll``.
For this camera convention, physical body roll is ``180° - CD angle``. This
utility performs that exact coordinate conversion, recomputes quaternion error,
and upgrades the quality metric from coordinate-wise absolute residuals to
per-star Euclidean pixel residuals.

The conversion does not rerun detection, matching, or optimization; it only
changes representation of already-solved poses.

Usage:
    python Code/Star_ID/normalize_result_conventions.py \
        --run-dir Results/star_id_run \
        --labels-dir Data/dataset_tess_test/labels \
        --in-place
"""

import argparse
import json
from pathlib import Path

import numpy as np

from inference_full import (
    _R_from_pose,
    angular_error_arcsec,
    build_pose_wcs,
    extract_camera_intrinsics,
    rot_to_quat,
)


CONVENTION = "physical_body_roll_deg"
CALIBRATION = "sip_crpix_derotated_cd_scale_shear"


def physical_roll(cd_angle_deg):
    return (180.0 - float(cd_angle_deg)) % 360.0


def normalize_pose(pose):
    ra, dec, legacy_cd_angle = map(float, pose)
    return [ra, dec, physical_roll(legacy_cd_angle)]


def radial_residuals_px(artefact, pose_pred, intrinsics, cd_intrinsic):
    pairs = artefact.get("matched_pairs", [])
    if not pairs:
        return np.zeros(0, dtype=np.float64)
    radec = np.array([(pair["ra_deg"], pair["dec_deg"]) for pair in pairs])
    observed = np.array([(pair["px"], pair["py"]) for pair in pairs])
    wcs = build_pose_wcs(*pose_pred, intrinsics, cd_intrinsic)
    px, py = wcs.all_world2pix(radec[:, 0], radec[:, 1], 0)
    return np.hypot(np.asarray(px) - observed[:, 0],
                    np.asarray(py) - observed[:, 1])


def normalize_artefact(artefact, label):
    if artefact.get("attitude_convention") == CONVENTION:
        return artefact

    header = label["pose"]["wcs_header"]
    intrinsics, cd_intrinsic, roll_gt = extract_camera_intrinsics(header)
    pose_gt = [float(header["CRVAL1"]), float(header["CRVAL2"]), roll_gt]

    failed = bool(artefact.get("failed"))
    pose_key = "pose_pred_if_kept" if failed else "pose_pred"
    error_key = "angular_error_arcsec_if_kept" if failed else "angular_error_arcsec"
    if pose_key not in artefact:
        artefact["attitude_convention"] = CONVENTION
        artefact["calibration"] = CALIBRATION
        return artefact

    pose_pred = normalize_pose(artefact[pose_key])
    artefact[pose_key] = pose_pred
    if not failed:
        artefact["pose_gt"] = pose_gt

    q_pred = rot_to_quat(_R_from_pose(*pose_pred))
    q_gt = rot_to_quat(_R_from_pose(*pose_gt))
    error = float(angular_error_arcsec(q_pred, q_gt))
    artefact[error_key] = error
    if not failed:
        plate_scale = float(label["pose"]["plate_scale_arcsec_per_pix"])
        artefact["pixel_error"] = error / plate_scale

    old_coordinate_residual = artefact.get("median_px_residual")
    radial = radial_residuals_px(
        artefact, pose_pred, intrinsics, cd_intrinsic)
    if radial.size:
        radial_median = float(np.median(radial))
        artefact["median_px_residual"] = radial_median
        artefact["median_coordinate_residual_px"] = old_coordinate_residual
        if failed and str(artefact.get("reason", "")).startswith("quality_gate:residual_"):
            artefact["reason"] = (
                f"quality_gate:residual_{radial_median:.1f}px_"
                f"n={artefact.get('n_matched', len(radial))}"
            )

    artefact["residual_metric"] = "median_euclidean_px"
    artefact["attitude_convention"] = CONVENTION
    artefact["attitude_seed"] = "ransac_wahba_ra_dec_roll"
    artefact["calibration"] = CALIBRATION
    return artefact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--in-place", action="store_true")
    args = parser.parse_args()

    if args.in_place == bool(args.out_dir):
        parser.error("choose exactly one of --in-place or --out-dir")
    destination = args.run_dir if args.in_place else args.out_dir
    destination.mkdir(parents=True, exist_ok=True)

    converted = 0
    for result_path in sorted(args.run_dir.glob("*.json")):
        artefact = json.loads(result_path.read_text())
        image_name = artefact.get("image", result_path.stem + ".png")
        label_path = args.labels_dir / (Path(image_name).stem + ".json")
        if not label_path.exists():
            print(f"skip {result_path.name}: missing {label_path.name}")
            continue
        label = json.loads(label_path.read_text())
        was_legacy = artefact.get("attitude_convention") != CONVENTION
        normalized = normalize_artefact(artefact, label)
        output_path = destination / result_path.name
        output_path.write_text(json.dumps(normalized, separators=(",", ":")))
        converted += int(was_legacy)

    print(f"Normalized {converted} legacy artefacts into {destination}")


if __name__ == "__main__":
    main()
