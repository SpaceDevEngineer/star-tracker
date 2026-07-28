import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code" / "Star_ID"))

from inference_full import (  # noqa: E402
    _R_from_pose,
    build_pose_wcs,
    extract_camera_intrinsics,
    pixel_residual_norms,
    pose_from_R,
)


def _angle_delta_deg(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def test_pose_rotation_round_trip():
    for pose in ((1.0, 2.0, 3.0), (142.3, 22.1, 74.4), (300.0, -20.0, 250.0)):
        recovered = pose_from_R(_R_from_pose(*pose))
        assert abs(_angle_delta_deg(recovered[0], pose[0])) < 1e-10
        assert abs(recovered[1] - pose[1]) < 1e-10
        assert abs(_angle_delta_deg(recovered[2], pose[2])) < 1e-10


def test_cd_factorization_removes_attitude_and_reconstructs_header():
    label_path = sorted((ROOT / "Data" / "dataset_tess_test" / "labels").glob("*.json"))[0]
    header = json.loads(label_path.read_text())["pose"]["wcs_header"]

    pc = np.array([
        [header.get("PC1_1", 1.0), header.get("PC1_2", 0.0)],
        [header.get("PC2_1", 0.0), header.get("PC2_2", 1.0)],
    ])
    raw_cd = pc @ np.diag([
        header.get("CDELT1", 1.0),
        header.get("CDELT2", 1.0),
    ])

    intrinsics, cd_intrinsic, physical_roll = extract_camera_intrinsics(header)
    reconstructed = build_pose_wcs(
        header["CRVAL1"],
        header["CRVAL2"],
        physical_roll,
        intrinsics,
        cd_intrinsic,
    )
    np.testing.assert_allclose(reconstructed.wcs.cd, raw_cd, atol=1e-15)


def test_pixel_residuals_are_radial_per_star():
    # least_squares layout is [dx0, dx1, ..., dy0, dy1, ...]
    residuals = np.array([3.0, 5.0, 4.0, 12.0])
    np.testing.assert_allclose(pixel_residual_norms(residuals), [5.0, 13.0])
