"""Participant-disjoint splits for condition-invariant evaluation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class OuterFold:
    """One run's development population and completely unseen users."""

    fold_index: int
    development_participants: tuple[str, ...]
    evaluation_participants: tuple[str, ...]


def create_outer_folds(
    participants: tuple[str, ...],
    number_of_folds: int = 4,
) -> tuple[OuterFold, ...]:
    """Create deterministic folds where every participant is unseen once."""

    unique_participants = tuple(sorted(set(participants)))
    if len(unique_participants) != len(participants):
        raise ValueError("Participant IDs must be unique")
    if number_of_folds < 2:
        raise ValueError("number_of_folds must be at least 2")
    if len(unique_participants) % number_of_folds != 0:
        raise ValueError(
            f"Cannot divide {len(unique_participants)} participants equally "
            f"across {number_of_folds} folds"
        )

    folds = []
    for fold_index in range(number_of_folds):
        evaluation_participants = unique_participants[
            fold_index::number_of_folds
        ]
        evaluation_set = set(evaluation_participants)
        development_participants = tuple(
            participant
            for participant in unique_participants
            if participant not in evaluation_set
        )

        if set(development_participants) & evaluation_set:
            raise AssertionError("Development and evaluation participants overlap")
        if set(development_participants) | evaluation_set != set(unique_participants):
            raise AssertionError("Fold does not contain every participant")

        folds.append(
            OuterFold(
                fold_index=fold_index,
                development_participants=development_participants,
                evaluation_participants=evaluation_participants,
            )
        )

    evaluated_once = [
        participant
        for fold in folds
        for participant in fold.evaluation_participants
    ]
    if sorted(evaluated_once) != list(unique_participants):
        raise AssertionError("Every participant must be evaluated exactly once")

    return tuple(folds)
