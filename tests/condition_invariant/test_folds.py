"""Checks for participant-disjoint outer folds."""

from src.condition_invariant.dataset import discover_participants
from src.condition_invariant.folds import create_outer_folds
from tests.condition_invariant.test_records import assert_raises


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
