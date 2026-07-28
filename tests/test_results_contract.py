import csv
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_bundled_replay_set_is_complete_and_reproducible():
    files = sorted((ROOT / "Results" / "star_id_run").glob("*.json"))
    assert len(files) == 16

    rows = [json.loads(path.read_text()) for path in files]
    solved = [row for row in rows if not row.get("failed")]
    failed = [row for row in rows if row.get("failed")]

    assert len(solved) == 15
    assert len(failed) == 1
    assert failed[0]["reason"].startswith("quality_gate:")
    assert failed[0]["angular_error_arcsec_if_kept"] > 100_000

    assert all(row["attitude_convention"] == "physical_body_roll_deg" for row in rows)
    assert all(row["residual_metric"] == "median_euclidean_px" for row in rows)

    errors = np.array([row["angular_error_arcsec"] for row in solved])
    assert np.median(errors) == pytest.approx(6.6297, abs=5e-4)


def test_full_evaluation_summary_has_auditable_counts():
    summary = json.loads(
        (ROOT / "Results" / "full_test_metrics.json").read_text()
    )
    counts = summary["counts"]
    assert counts["total_frames"] == 120
    assert counts["valid_frames"] == 107
    assert counts["solved_frames"] == 104
    assert counts["algorithm_refusals"] == 3
    assert counts["solved_frames"] + counts["algorithm_refusals"] == counts["valid_frames"]

    with open(ROOT / "Results" / "full_test_per_frame.csv", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == counts["total_frames"]
    assert sum(row["status"] == "solved" for row in rows) == counts["solved_frames"]
    assert sum(row["status"] == "algorithm_refusal" for row in rows) == counts["algorithm_refusals"]
    assert sum(row["status"] == "data_excluded" for row in rows) == counts["missing_wcs_frames"]
