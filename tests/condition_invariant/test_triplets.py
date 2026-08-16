"""Checks for triplet records and sampling."""

import numpy as np

from src.condition_invariant.records import GaitWindow
from src.condition_invariant.triplets import GaitTriplet, window_identity
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
