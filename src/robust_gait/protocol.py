"""Evaluation-protocol helpers shared by the experiment and 08 diagnostics.

Copied unchanged from the retired ``src/condition_invariant`` package so this
system stands alone: deterministic participant-disjoint outer folds, the
target-FAR threshold selector, and the threshold-free equal error rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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


def select_threshold_from_distances(
    genuine_distances: np.ndarray,
    impostor_distances: np.ndarray,
    target_far: float,
) -> tuple[float, float, float]:
    """Select a low-distance threshold without constructing score records."""

    genuine = np.asarray(genuine_distances, dtype=np.float64)
    impostor = np.asarray(impostor_distances, dtype=np.float64)
    if genuine.ndim != 1 or genuine.size == 0:
        raise ValueError("Genuine distances must be a non-empty vector")
    if impostor.ndim != 1 or impostor.size == 0:
        raise ValueError("Impostor distances must be a non-empty vector")
    if not np.isfinite(genuine).all() or not np.isfinite(impostor).all():
        raise ValueError("Threshold distances must be finite")
    if np.any(genuine < 0) or np.any(impostor < 0):
        raise ValueError("Threshold distances must be non-negative")
    if not 0.0 <= target_far <= 1.0:
        raise ValueError("target_far must be between 0 and 1")

    genuine = np.sort(genuine)
    impostor = np.sort(impostor)
    observed = np.concatenate((genuine, impostor))
    candidates = np.concatenate(
        ([np.nextafter(observed.min(), -np.inf)], np.unique(observed))
    )
    accepted_genuine = np.searchsorted(genuine, candidates, side="right")
    accepted_impostors = np.searchsorted(impostor, candidates, side="right")
    frr_values = 1.0 - accepted_genuine / genuine.size
    far_values = accepted_impostors / impostor.size
    valid = far_values <= target_far
    if not np.any(valid):
        raise RuntimeError("No threshold satisfies the requested target FAR")
    valid_indices = np.flatnonzero(valid)
    order = np.lexsort(
        (
            candidates[valid_indices],
            far_values[valid_indices],
            frr_values[valid_indices],
        )
    )
    best_index = valid_indices[int(order[0])]
    return (
        float(candidates[best_index]),
        float(frr_values[best_index]),
        float(far_values[best_index]),
    )


def equal_error_rate(
    genuine_distances: np.ndarray,
    impostor_distances: np.ndarray,
) -> tuple[float, float]:
    """Return the threshold-free EER and the distance that achieves it."""

    genuine = np.sort(np.asarray(genuine_distances, dtype=np.float64))
    impostor = np.sort(np.asarray(impostor_distances, dtype=np.float64))
    if genuine.ndim != 1 or genuine.size == 0:
        raise ValueError("Genuine distances must be a non-empty vector")
    if impostor.ndim != 1 or impostor.size == 0:
        raise ValueError("Impostor distances must be a non-empty vector")
    if not np.isfinite(genuine).all() or not np.isfinite(impostor).all():
        raise ValueError("EER distances must be finite")

    observed = np.concatenate((genuine, impostor))
    candidates = np.concatenate(
        ([np.nextafter(observed.min(), -np.inf)], np.unique(observed))
    )
    frr_values = 1.0 - np.searchsorted(genuine, candidates, side="right") / genuine.size
    far_values = np.searchsorted(impostor, candidates, side="right") / impostor.size
    index = int(np.argmin(np.abs(frr_values - far_values)))
    return (
        float((frr_values[index] + far_values[index]) / 2.0),
        float(candidates[index]),
    )
