"""Checks for triplet records and sampling."""

import numpy as np

from src.condition_invariant.records import GaitWindow
from src.condition_invariant.config import CONDITIONS
from src.condition_invariant.triplets import (
    GaitTriplet,
    build_triplet_index,
    sample_triplet,
    window_identity,
)
from tests.condition_invariant.test_records import assert_raises


def make_window(
    participant_id: str,
    condition: str,
    window_index: int,
) -> GaitWindow:
    return GaitWindow(
        participant_id=participant_id,
        condition=condition,
        window_index=window_index,
        start_sample=window_index * 128,
        block_id=0,
        signal=np.zeros((256, 6), dtype=np.float32),
    )


def make_complete_learning_windows(
    participants: tuple[str, ...] = ("sub_02", "sub_01"),
) -> tuple[GaitWindow, ...]:
    windows = []
    for participant_id in participants:
        for condition in CONDITIONS:
            windows.extend([
                make_window(participant_id, condition, 1),
                make_window(participant_id, condition, 0),
            ])
    return tuple(windows)


def test_window_identity_uses_location_metadata() -> None:
    window = make_window("sub_01", "st_control", 3)

    assert window_identity(window) == ("sub_01", "st_control", 3, 384)


def test_valid_cross_condition_triplet_is_accepted() -> None:
    triplet = GaitTriplet(
        anchor=make_window("sub_01", "st_control", 0),
        positive=make_window("sub_01", "st_fatigue", 1),
        negative=make_window("sub_02", "st_control", 0),
    )

    assert triplet.anchor.participant_id == triplet.positive.participant_id
    assert triplet.anchor.participant_id != triplet.negative.participant_id
    assert triplet.anchor.condition != triplet.positive.condition


def test_triplet_rejects_positive_from_different_participant() -> None:
    assert_raises(
        ValueError,
        "same participant",
        lambda: GaitTriplet(
            anchor=make_window("sub_01", "st_control", 0),
            positive=make_window("sub_02", "st_fatigue", 1),
            negative=make_window("sub_03", "st_control", 0),
        ),
    )


def test_triplet_rejects_same_window_as_anchor_and_positive() -> None:
    anchor = make_window("sub_01", "st_control", 0)

    assert_raises(
        ValueError,
        "different windows",
        lambda: GaitTriplet(
            anchor=anchor,
            positive=anchor,
            negative=make_window("sub_02", "st_control", 0),
        ),
    )


def test_triplet_rejects_negative_from_anchor_participant() -> None:
    assert_raises(
        ValueError,
        "different participant",
        lambda: GaitTriplet(
            anchor=make_window("sub_01", "st_control", 0),
            positive=make_window("sub_01", "st_fatigue", 1),
            negative=make_window("sub_01", "dt_control", 2),
        ),
    )


def test_triplet_index_organizes_participants_and_conditions() -> None:
    index = build_triplet_index(make_complete_learning_windows())

    assert tuple(index) == ("sub_01", "sub_02")
    assert tuple(index["sub_01"]) == CONDITIONS
    assert len(index["sub_01"]["dt_fatigue"]) == 2


def test_triplet_index_sorts_each_recording_chronologically() -> None:
    index = build_triplet_index(make_complete_learning_windows())

    starts = [window.start_sample for window in index["sub_01"]["st_control"]]
    assert starts == [0, 128]


def test_triplet_index_rejects_missing_condition() -> None:
    windows = tuple(
        window
        for window in make_complete_learning_windows()
        if not (
            window.participant_id == "sub_01"
            and window.condition == "dt_fatigue"
        )
    )

    assert_raises(
        ValueError,
        "sub_01 is missing triplet conditions",
        lambda: build_triplet_index(windows),
    )


def test_triplet_index_rejects_duplicate_window() -> None:
    windows = make_complete_learning_windows()

    assert_raises(
        ValueError,
        "Duplicate gait window found",
        lambda: build_triplet_index((*windows, windows[0])),
    )


def test_sample_triplet_obeys_requested_identities_and_conditions() -> None:
    index = build_triplet_index(make_complete_learning_windows())
    random_generator = np.random.default_rng(42)

    triplet = sample_triplet(
        index=index,
        random_generator=random_generator,
        anchor_participant="sub_01",
        anchor_condition="st_control",
        positive_condition="st_fatigue",
        negative_condition="dt_control",
    )

    assert triplet.anchor.participant_id == "sub_01"
    assert triplet.anchor.condition == "st_control"
    assert triplet.positive.participant_id == "sub_01"
    assert triplet.positive.condition == "st_fatigue"
    assert triplet.negative.participant_id == "sub_02"
    assert triplet.negative.condition == "dt_control"


def test_sample_triplet_excludes_anchor_from_same_condition_positive() -> None:
    index = build_triplet_index(make_complete_learning_windows())
    random_generator = np.random.default_rng(7)

    triplet = sample_triplet(
        index=index,
        random_generator=random_generator,
        anchor_participant="sub_01",
        anchor_condition="st_control",
        positive_condition="st_control",
        negative_condition="st_control",
    )

    assert window_identity(triplet.anchor) != window_identity(triplet.positive)


def test_sample_triplet_is_reproducible_with_the_same_seed() -> None:
    index = build_triplet_index(make_complete_learning_windows())

    first = sample_triplet(
        index, np.random.default_rng(123),
        "sub_01", "st_control", "dt_fatigue", "st_fatigue",
    )
    second = sample_triplet(
        index, np.random.default_rng(123),
        "sub_01", "st_control", "dt_fatigue", "st_fatigue",
    )

    assert window_identity(first.anchor) == window_identity(second.anchor)
    assert window_identity(first.positive) == window_identity(second.positive)
    assert window_identity(first.negative) == window_identity(second.negative)
