"""Checks for post-hoc test operating-point diagnostics."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).parents[1] / "src" / "08_test_operating_points.py"
SPEC = importlib.util.spec_from_file_location("test_operating_points_module", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def synthetic_scores():
    rows = []
    for participant in ("sub_01", "sub_02"):
        for condition in ("dt_control", "dt_fatigue"):
            for distance in (0.1, 0.2, 0.3):
                rows.append({
                    "claimed_participant_id": participant,
                    "probe_participant_id": participant,
                    "condition": condition,
                    "is_genuine": True,
                    "distance": distance,
                })
            other = "sub_02" if participant == "sub_01" else "sub_01"
            for distance in (0.7, 0.8, 0.9):
                rows.append({
                    "claimed_participant_id": participant,
                    "probe_participant_id": other,
                    "condition": condition,
                    "is_genuine": False,
                    "distance": distance,
                })
    return pd.DataFrame(rows)


def test_separated_scores_have_zero_eer_and_zero_frr_at_one_percent_far():
    metrics = MODULE.calculate_test_operating_points(synthetic_scores(), 0.01)
    np.testing.assert_allclose(metrics["test_eer"], 0.0)
    np.testing.assert_allclose(metrics["test_frr_at_target_far"], 0.0)
    np.testing.assert_allclose(metrics["test_achieved_far"], 0.0)


def test_summary_macro_averages_participants_by_condition():
    metrics = MODULE.calculate_test_operating_points(synthetic_scores(), 0.01)
    summary = MODULE.summarize(metrics)
    assert len(summary) == 2
    assert set(summary["participant_count"]) == {2}
