"""Threshold selection and biometric error metrics for distance scores."""

from dataclasses import dataclass

import numpy as np

from src.condition_invariant.scoring import ComparisonScore


@dataclass(frozen=True)
class ThresholdSelection:
    """One global operating threshold selected on development data."""

    threshold: float
    target_far: float
    validation_frr: float
    validation_far: float
    genuine_count: int
    impostor_count: int


@dataclass(frozen=True)
class AuthenticationRates:
    """FRR and FAR for one claimed participant under one condition."""

    claimed_participant_id: str
    condition: str
    frr: float
    far: float
    genuine_count: int
    impostor_count: int
    false_rejection_count: int
    false_acceptance_count: int


@dataclass(frozen=True)
class MacroAverage:
    """Equal-participant summary for one evaluation condition."""

    condition: str
    participant_count: int
    mean_frr: float
    standard_deviation_frr: float
    mean_far: float
    standard_deviation_far: float


def _error_rates(
    scores: tuple[ComparisonScore, ...],
    threshold: float,
) -> tuple[float, float, int, int, int, int]:
    """Calculate errors using lower-distance-means-better decisions."""

    genuine_distances = np.asarray(
        [score.distance for score in scores if score.is_genuine], dtype=np.float64
    )
    impostor_distances = np.asarray(
        [score.distance for score in scores if not score.is_genuine], dtype=np.float64
    )
    if genuine_distances.size == 0:
        raise ValueError("At least one genuine comparison is required")
    if impostor_distances.size == 0:
        raise ValueError("At least one impostor comparison is required")

    false_rejections = int(np.count_nonzero(genuine_distances > threshold))
    false_acceptances = int(np.count_nonzero(impostor_distances <= threshold))
    frr = false_rejections / genuine_distances.size
    far = false_acceptances / impostor_distances.size
    return (
        float(frr),
        float(far),
        int(genuine_distances.size),
        int(impostor_distances.size),
        false_rejections,
        false_acceptances,
    )


def select_global_threshold(
    development_scores: tuple[ComparisonScore, ...],
    development_participants: tuple[str, ...],
    target_far: float = 0.01,
) -> ThresholdSelection:
    """Select one threshold using development-validation comparisons only."""

    if not development_scores:
        raise ValueError("Development scores must not be empty")
    if not 0.0 <= target_far <= 1.0:
        raise ValueError("target_far must be between 0 and 1")

    allowed_participants = set(development_participants)
    if not allowed_participants:
        raise ValueError("Development participants must not be empty")
    score_participants = {
        participant_id
        for score in development_scores
        for participant_id in (
            score.claimed_participant_id,
            score.probe_participant_id,
        )
    }
    unexpected = sorted(score_participants - allowed_participants)
    if unexpected:
        raise ValueError(
            "Threshold scores contain non-development participants: "
            f"{unexpected}"
        )

    genuine_distances = np.asarray(
        [score.distance for score in development_scores if score.is_genuine]
    )
    impostor_distances = np.asarray(
        [score.distance for score in development_scores if not score.is_genuine]
    )
    threshold, frr, far = select_threshold_from_distances(
        genuine_distances, impostor_distances, target_far
    )
    return ThresholdSelection(
        threshold=threshold,
        target_far=target_far,
        validation_frr=frr,
        validation_far=far,
        genuine_count=len(genuine_distances),
        impostor_count=len(impostor_distances),
    )


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


def calculate_authentication_rates(
    scores: tuple[ComparisonScore, ...],
    threshold: float,
) -> AuthenticationRates:
    """Apply a frozen threshold to one claimed user and condition."""

    if not scores:
        raise ValueError("Scores must not be empty")
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")

    claimed_participants = {score.claimed_participant_id for score in scores}
    conditions = {score.condition for score in scores}
    if len(claimed_participants) != 1:
        raise ValueError("Rates require scores for exactly one claimed participant")
    if len(conditions) != 1:
        raise ValueError("Rates require scores from exactly one condition")

    frr, far, genuine_count, impostor_count, false_rejections, false_acceptances = (
        _error_rates(scores, threshold)
    )
    return AuthenticationRates(
        claimed_participant_id=next(iter(claimed_participants)),
        condition=next(iter(conditions)),
        frr=frr,
        far=far,
        genuine_count=genuine_count,
        impostor_count=impostor_count,
        false_rejection_count=false_rejections,
        false_acceptance_count=false_acceptances,
    )


def macro_average_rates(
    participant_rates: tuple[AuthenticationRates, ...],
) -> MacroAverage:
    """Give every participant equal weight in one condition summary."""

    if not participant_rates:
        raise ValueError("Participant rates must not be empty")
    conditions = {rates.condition for rates in participant_rates}
    participants = {
        rates.claimed_participant_id for rates in participant_rates
    }
    if len(conditions) != 1:
        raise ValueError("Macro-averaging requires exactly one condition")
    if len(participants) != len(participant_rates):
        raise ValueError("Each participant must appear exactly once")

    frr_values = np.asarray([rates.frr for rates in participant_rates])
    far_values = np.asarray([rates.far for rates in participant_rates])
    ddof = 1 if len(participant_rates) > 1 else 0
    return MacroAverage(
        condition=next(iter(conditions)),
        participant_count=len(participant_rates),
        mean_frr=float(frr_values.mean()),
        standard_deviation_frr=float(frr_values.std(ddof=ddof)),
        mean_far=float(far_values.mean()),
        standard_deviation_far=float(far_values.std(ddof=ddof)),
    )
