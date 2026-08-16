"""Checks for participant and chronological experiment splits."""

import numpy as np

from src.condition_invariant.dataset import discover_participants
from src.condition_invariant.folds import (
    OuterFold,
    create_outer_folds,
    prepare_development_fold_data,
    split_development_recording,
)
from src.condition_invariant.records import GaitWindow
from tests.condition_invariant.test_records import assert_raises


def make_block_windows(number_of_blocks: int = 10) -> list[GaitWindow]:
    """Create 20 windows per block, including an overlapping boundary window."""

    windows = []
    window_index = 0
    for block_id in range(number_of_blocks):
        block_start = block_id * 2560
        for offset in range(0, 2560, 128):
            windows.append(
                GaitWindow(
                    participant_id="sub_01",
                    condition="st_control",
                    window_index=window_index,
                    start_sample=block_start + offset,
                    block_id=block_id,
                    signal=np.zeros((256, 6), dtype=np.float32),
                )
            )
            window_index += 1
    return windows


def make_tiny_fold_dataset():
    """Create four conditions for three small synthetic participants."""

    dataset = {}
    for participant_id in ("sub_01", "sub_02", "sub_03"):
        dataset[participant_id] = {}
        for condition in ("st_control", "st_fatigue", "dt_control", "dt_fatigue"):
            participant_windows = make_block_windows(5)
            dataset[participant_id][condition] = [
                GaitWindow(
                    participant_id=participant_id,
                    condition=condition,
                    window_index=window.window_index,
                    start_sample=window.start_sample,
                    block_id=window.block_id,
                    signal=window.signal,
                )
                for window in participant_windows
            ]
    return dataset


def test_four_folds_have_twelve_development_and_four_evaluation_users() -> None:
    participants = tuple(f"sub_{index:02d}" for index in range(1, 17))

    folds = create_outer_folds(participants)

    assert len(folds) == 4
    for fold in folds:
        assert len(fold.development_participants) == 12
        assert len(fold.evaluation_participants) == 4


def test_development_and_evaluation_users_never_overlap() -> None:
    participants = tuple(f"sub_{index:02d}" for index in range(1, 17))

    folds = create_outer_folds(participants)

    for fold in folds:
        assert not (
            set(fold.development_participants)
            & set(fold.evaluation_participants)
        )


def test_every_participant_is_evaluated_exactly_once() -> None:
    participants = tuple(f"sub_{index:02d}" for index in range(1, 17))

    folds = create_outer_folds(participants)
    evaluated = [
        participant
        for fold in folds
        for participant in fold.evaluation_participants
    ]

    assert sorted(evaluated) == sorted(participants)
    assert len(evaluated) == len(set(evaluated))


def test_fold_assignment_is_independent_of_input_order() -> None:
    participants = tuple(f"sub_{index:02d}" for index in range(1, 17))

    forward = create_outer_folds(participants)
    reversed_order = create_outer_folds(tuple(reversed(participants)))

    assert forward == reversed_order


def test_unequal_fold_size_is_rejected() -> None:
    participants = tuple(f"sub_{index:02d}" for index in range(1, 16))

    assert_raises(
        ValueError,
        "Cannot divide 15 participants equally",
        lambda: create_outer_folds(participants),
    )


def test_real_participants_form_valid_outer_folds() -> None:
    participants = discover_participants()

    folds = create_outer_folds(participants)

    assert len(folds) == 4
    assert all(len(fold.evaluation_participants) == 4 for fold in folds)


def test_development_recording_uses_early_blocks_for_learning() -> None:
    split = split_development_recording(make_block_windows(10))

    assert {window.block_id for window in split.learning_windows} == set(range(8))
    assert {window.block_id for window in split.validation_windows} == {8, 9}


def test_development_recording_boundary_has_no_shared_samples() -> None:
    split = split_development_recording(make_block_windows(10))

    last_learning = split.learning_windows[-1]
    first_validation = split.validation_windows[0]
    assert last_learning.start_sample + 256 <= first_validation.start_sample


def test_development_recording_rejects_mixed_participants() -> None:
    windows = make_block_windows(2)
    mixed_window = GaitWindow(
        participant_id="sub_02",
        condition="st_control",
        window_index=windows[-1].window_index,
        start_sample=windows[-1].start_sample,
        block_id=windows[-1].block_id,
        signal=windows[-1].signal,
    )
    windows[-1] = mixed_window

    assert_raises(
        ValueError,
        "one participant and condition",
        lambda: split_development_recording(windows),
    )


def test_prepare_development_data_excludes_evaluation_participant() -> None:
    dataset = make_tiny_fold_dataset()
    fold = OuterFold(
        fold_index=0,
        development_participants=("sub_01", "sub_02"),
        evaluation_participants=("sub_03",),
    )

    prepared = prepare_development_fold_data(dataset, fold)
    used = {
        window.participant_id
        for window in (*prepared.learning_windows, *prepared.validation_windows)
    }

    assert used == {"sub_01", "sub_02"}
    assert "sub_03" not in used


def test_prepare_development_data_includes_every_condition() -> None:
    dataset = make_tiny_fold_dataset()
    fold = OuterFold(
        fold_index=0,
        development_participants=("sub_01", "sub_02"),
        evaluation_participants=("sub_03",),
    )

    prepared = prepare_development_fold_data(dataset, fold)

    assert {window.condition for window in prepared.learning_windows} == {
        "st_control", "st_fatigue", "dt_control", "dt_fatigue"
    }
    assert {window.condition for window in prepared.validation_windows} == {
        "st_control", "st_fatigue", "dt_control", "dt_fatigue"
    }
