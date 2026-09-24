"""Checks for complete-experiment protocol helpers."""

from src.condition_invariant.run_experiment import _score_rows
from src.condition_invariant.scoring import ComparisonScore


def make_score(distance: float) -> ComparisonScore:
    return ComparisonScore(
        claimed_participant_id="sub_01",
        probe_participant_id="sub_01",
        condition="st_fatigue",
        window_index=3,
        start_sample=384,
        distance=distance,
    )


def test_saved_score_rows_apply_lower_distance_acceptance_rule() -> None:
    rows = _score_rows(
        (make_score(0.4), make_score(0.6)),
        fold_index=2,
        thresholds=0.5,
    )

    assert rows[0]["accepted"] is True
    assert rows[1]["accepted"] is False
    assert rows[0]["fold_index"] == 2
    assert rows[0]["is_genuine"] is True
