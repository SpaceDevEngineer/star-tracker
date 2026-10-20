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
    assert not any(row.get("failed") for row in rows)
    assert all(row["attitude_convention"] == "physical_body_roll_deg" for row in rows)
    assert all(row["residual_metric"] == "median_euclidean_px" for row in rows)
    assert all(row["median_px_residual"] < 2.0 for row in rows)

    errors = np.array([row["angular_error_arcsec"] for row in rows])
    assert np.median(errors) == pytest.approx(6.9253, abs=5e-4)
    assert errors.max() < 60.0


@pytest.mark.parametrize("name, total, first_attempt, median", [
    ("full_test", 120, 117, 7.788),
    ("unseen_sectors", 128, 124, 7.334),
])
def test_published_tables_match_their_metrics(name, total, first_attempt, median):
    metrics = json.loads((ROOT / "Results" / f"{name}_metrics.json").read_text())
    counts = metrics["counts"]
    assert counts["total_frames"] == total
    assert counts["solved_frames"] == total
    assert counts["algorithm_refusals"] == 0
    assert counts["false_locks"] == 0
    assert counts["solved_first_attempt"] == first_attempt
    assert counts["recovered_by_retry"] == total - first_attempt
    assert metrics["attitude_error_arcsec"]["median"] == pytest.approx(median, abs=1e-3)

    with open(ROOT / "Results" / f"{name}_per_frame.csv", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == total
    assert sum(row["status"] == "solved" for row in rows) == counts["solved_frames"]
    assert sum(int(row["attempts"]) > 1 for row in rows) == counts["recovered_by_retry"]
    errors = np.array([float(row["attitude_error_arcsec"]) for row in rows])
    assert np.median(errors) == pytest.approx(median, abs=1e-3)
    assert int(np.sum(errors < 20.57)) == metrics["threshold_counts"]["subpixel_below_20_57_arcsec"]
