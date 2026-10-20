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
    radec_to_unit_vec,
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


def _sky_grid(wcs):
    gx, gy = np.meshgrid(np.linspace(50, 2080, 15), np.linspace(50, 2020, 15))
    ra, dec = wcs.all_pix2world(gx.ravel(), gy.ravel(), 0)
    return radec_to_unit_vec(ra, dec)


def _rigid_rotation(a, b):
    """Rotation R minimising |b - R a| (Kabsch / Wahba)."""
    u, _, vt = np.linalg.svd(b.T @ a)
    return u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt


def test_pose_parameters_match_the_sky_rotation_of_the_wcs():
    """A change of (RA, Dec, roll) must move the sky exactly as _R_from_pose says.

    This ties the scored attitude error to geometry: both the size and the sign
    of a roll change are checked against the rigid rotation between the sky
    grids of two synthesized WCS.
    """
    label_path = sorted((ROOT / "Data" / "dataset_tess_test" / "labels").glob("*.json"))[0]
    header = json.loads(label_path.read_text())["pose"]["wcs_header"]
    intrinsics, cd_intrinsic, roll = extract_camera_intrinsics(header)
    ra, dec = header["CRVAL1"], header["CRVAL2"]
    base = _sky_grid(build_pose_wcs(ra, dec, roll, intrinsics, cd_intrinsic))

    for d_ra, d_dec, d_roll in ((0.0, 0.0, 0.03), (0.01, 0.0, 0.0), (0.0, 0.01, -0.02)):
        moved = _sky_grid(build_pose_wcs(ra + d_ra, dec + d_dec, roll + d_roll,
                                         intrinsics, cd_intrinsic))
        r_sky = _rigid_rotation(base, moved)
        r_model = (_R_from_pose(ra + d_ra, dec + d_dec, roll + d_roll)
                   @ _R_from_pose(ra, dec, roll).T)
        np.testing.assert_allclose(r_sky, r_model, atol=1e-9)
