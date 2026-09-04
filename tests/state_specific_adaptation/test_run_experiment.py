import numpy as np
import pandas as pd
import pytest

from src.state_specific_adaptation.run_experiment import (
    causal_median_scores,
    mine_hard_negatives,
    select_candidate,
    trusted_prefix,
)
from src.utils import FEATURE_COLS, ORIGINAL_RATE


class FirstFeatureModel:
    def decision_function(self, features):
        return features[FEATURE_COLS[0]].to_numpy()


def make_frame(values):
    data = {column: np.zeros(len(values)) for column in FEATURE_COLS}
    data[FEATURE_COLS[0]] = values
    data.update({
        "start_sample": np.arange(len(values)) * 128,
        "window_index": np.arange(len(values)),
        "participant_id": ["sub_01"] * len(values),
        "session_type": ["st_fatigue"] * len(values),
    })
    return pd.DataFrame(data)


def test_causal_median_uses_current_and_previous_scores_only():
    scores, aligned = causal_median_scores(
        FirstFeatureModel(), make_frame([1, 100, 2, 3, 4]), fusion_window=3
    )
    assert scores == pytest.approx([2, 3, 3])
    assert aligned.window_index.tolist() == [2, 3, 4]


def test_causal_median_rejects_invalid_window():
    with pytest.raises(ValueError, match="at least 1"):
        causal_median_scores(FirstFeatureModel(), make_frame([1, 2]), 0)
    with pytest.raises(ValueError, match="shorter"):
        causal_median_scores(FirstFeatureModel(), make_frame([1, 2]), 3)


def test_hard_negative_mining_keeps_highest_scores_per_participant():
    first = make_frame([1, 9, 2]).assign(participant_id="sub_01")
    second = make_frame([8, 3, 7]).assign(participant_id="sub_02")
    selected = mine_hard_negatives(
        FirstFeatureModel(), [first, second], maximum_per_participant=2
    )
    by_participant = {
        participant: sorted(group[FEATURE_COLS[0]].tolist())
        for participant, group in selected.groupby("participant_id")
    }
    assert by_participant == {"sub_01": [2, 9], "sub_02": [7, 8]}


def test_hard_negative_mining_rejects_invalid_limit():
    with pytest.raises(ValueError, match="at least 1"):
        mine_hard_negatives(FirstFeatureModel(), [make_frame([1])], 0)


def test_trusted_prefix_uses_only_requested_early_seconds():
    prefix = trusted_prefix(make_frame(range(80)), seconds=20)
    assert len(prefix) == 20
    assert prefix.start_sample.max() < 20 * ORIGINAL_RATE


def test_candidate_selection_prioritizes_frr_under_far_constraint():
    rows = [
        {"calibration_frr": 0.1, "calibration_far": 0.02,
         "update_seconds": 20, "hard_negatives_per_participant": 0},
        {"calibration_frr": 0.3, "calibration_far": 0.005,
         "update_seconds": 20, "hard_negatives_per_participant": 20},
        {"calibration_frr": 0.2, "calibration_far": 0.005,
         "update_seconds": 60, "hard_negatives_per_participant": 80},
    ]
    selected = select_candidate(rows, target_far=0.005)
    assert selected["calibration_frr"] == 0.2
