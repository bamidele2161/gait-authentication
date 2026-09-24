"""Checks for threshold selection, FRR, FAR, and macro-averaging."""

import math

import pytest

from src.condition_invariant.metrics import (
    AuthenticationRates,
    calculate_authentication_rates,
    macro_average_rates,
    select_global_threshold,
)
from src.condition_invariant.scoring import ComparisonScore


def make_score(
    claimed: str,
    probe: str,
    distance: float,
    condition: str = "st_control",
    window_index: int = 0,
) -> ComparisonScore:
    return ComparisonScore(
        claimed_participant_id=claimed,
        probe_participant_id=probe,
        condition=condition,
        window_index=window_index,
        start_sample=window_index * 128,
        distance=distance,
    )


def test_threshold_minimizes_frr_while_respecting_target_far() -> None:
    scores = (
        make_score("sub_01", "sub_01", 0.2),
        make_score("sub_01", "sub_01", 0.4, window_index=1),
        make_score("sub_01", "sub_02", 0.5),
        make_score("sub_01", "sub_03", 0.8),
    )

    selection = select_global_threshold(
        scores,
        development_participants=("sub_01", "sub_02", "sub_03"),
        target_far=0.0,
    )

    assert selection.threshold == pytest.approx(0.4)
    assert selection.validation_frr == 0.0
    assert selection.validation_far == 0.0


def test_threshold_rejects_non_development_scores() -> None:
    scores = (
        make_score("sub_01", "sub_01", 0.2),
        make_score("sub_01", "sub_99", 0.8),
    )

    with pytest.raises(ValueError, match="non-development participants"):
        select_global_threshold(scores, development_participants=("sub_01",))


def test_authentication_rates_use_lower_distance_as_a_match() -> None:
    scores = (
        make_score("sub_01", "sub_01", 0.2, "st_fatigue", 0),
        make_score("sub_01", "sub_01", 0.7, "st_fatigue", 1),
        make_score("sub_01", "sub_02", 0.3, "st_fatigue", 0),
        make_score("sub_01", "sub_02", 0.8, "st_fatigue", 1),
    )

    rates = calculate_authentication_rates(scores, threshold=0.5)

    assert rates.frr == 0.5
    assert rates.far == 0.5
    assert rates.false_rejection_count == 1
    assert rates.false_acceptance_count == 1
    assert rates.genuine_count == 2
    assert rates.impostor_count == 2


def test_macro_average_gives_participants_equal_weight() -> None:
    first = AuthenticationRates("sub_01", "dt_fatigue", 0.0, 0.2, 10, 30, 0, 6)
    second = AuthenticationRates("sub_02", "dt_fatigue", 1.0, 0.4, 100, 300, 100, 120)

    summary = macro_average_rates((first, second))

    assert summary.participant_count == 2
    assert summary.mean_frr == 0.5
    assert summary.mean_far == pytest.approx(0.3)
    assert summary.standard_deviation_frr == pytest.approx(math.sqrt(0.5))
    assert summary.standard_deviation_far == pytest.approx(math.sqrt(0.02))
