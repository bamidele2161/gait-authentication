"""Enrolment-conditioned verification in the learned embedding space."""

from dataclasses import dataclass

import numpy as np
from sklearn.svm import SVC

from src.condition_invariant.config import WINDOW_SAMPLES
from src.condition_invariant.metrics import select_threshold_from_distances
from src.condition_invariant.model import GaitEncoder
from src.condition_invariant.normalization import ChannelNormalizer
from src.condition_invariant.records import GaitWindow
from src.condition_invariant.scoring import (
    ComparisonScore,
    causal_mean_fusion,
    encode_probe_windows,
)


@dataclass
class EnrollmentVerifier:
    """A user-specific boundary learned from normal gait and cohort impostors."""

    participant_id: str
    classifier: SVC
    threshold: float
    training_genuine_count: int
    training_impostor_count: int
    calibration_genuine_count: int
    calibration_impostor_count: int
    fusion_window: int


def _split_enrollment_windows(
    windows: tuple[GaitWindow, ...],
    learning_fraction: float = 0.80,
) -> tuple[tuple[GaitWindow, ...], tuple[GaitWindow, ...]]:
    """Separate normal enrolment into verifier-learning and calibration parts."""

    if not 0 < learning_fraction < 1:
        raise ValueError("learning_fraction must be between 0 and 1")
    ordered = tuple(sorted(windows, key=lambda window: window.start_sample))
    if len(ordered) < 5:
        raise ValueError("At least five normal enrolment windows are required")
    participant_ids = {window.participant_id for window in ordered}
    conditions = {window.condition for window in ordered}
    if len(participant_ids) != 1 or conditions != {"st_control"}:
        raise ValueError("Verifier enrolment requires one user's ST-control windows only")

    split_index = int(len(ordered) * learning_fraction)
    learning = ordered[:split_index]
    calibration = ordered[split_index + 1 :]
    if not learning or not calibration:
        raise ValueError("Enrolment split produced an empty partition")
    if learning[-1].start_sample + WINDOW_SAMPLES > calibration[0].start_sample:
        raise AssertionError("Verifier learning and calibration share raw samples")
    return learning, calibration


def _sample_cohort(
    windows: tuple[GaitWindow, ...],
    excluded_participant: str,
    maximum_per_participant_condition: int,
    seed: int,
) -> tuple[GaitWindow, ...]:
    """Sample condition-balanced generic impostors from development users."""

    if maximum_per_participant_condition < 1:
        raise ValueError("Cohort sample limit must be positive")
    grouped: dict[tuple[str, str], list[GaitWindow]] = {}
    for window in windows:
        if window.participant_id == excluded_participant:
            raise ValueError("Enrolled participant leaked into the development cohort")
        grouped.setdefault((window.participant_id, window.condition), []).append(window)
    if not grouped:
        raise ValueError("Development cohort must not be empty")

    random_generator = np.random.default_rng(seed)
    selected = []
    for key in sorted(grouped):
        candidates = grouped[key]
        take = min(maximum_per_participant_condition, len(candidates))
        positions = np.sort(
            random_generator.choice(len(candidates), size=take, replace=False)
        )
        selected.extend(candidates[int(position)] for position in positions)
    return tuple(selected)


def _dissimilarity(classifier: SVC, embeddings: np.ndarray) -> np.ndarray:
    """Convert the SVM score to a bounded value where smaller means better."""

    decision = np.asarray(classifier.decision_function(embeddings), dtype=np.float64)
    clipped = np.clip(decision, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(clipped))


def _score_records(
    claimed_participant_id: str,
    windows: tuple[GaitWindow, ...],
    distances: np.ndarray,
) -> tuple[ComparisonScore, ...]:
    """Attach gait-window metadata to verifier dissimilarities."""

    return tuple(
        ComparisonScore(
            claimed_participant_id=claimed_participant_id,
            probe_participant_id=window.participant_id,
            condition=window.condition,
            window_index=window.window_index,
            start_sample=window.start_sample,
            distance=float(distance),
        )
        for window, distance in zip(windows, distances)
    )


def fit_enrollment_verifier(
    participant_id: str,
    enrollment_windows: tuple[GaitWindow, ...],
    cohort_learning_windows: tuple[GaitWindow, ...],
    cohort_validation_windows: tuple[GaitWindow, ...],
    model: GaitEncoder,
    normalizer: ChannelNormalizer,
    target_far: float = 0.01,
    c_value: float = 10.0,
    cohort_windows_per_group: int = 40,
    seed: int = 42,
    batch_size: int = 128,
    fusion_window: int = 5,
) -> EnrollmentVerifier:
    """Fit a verifier without using the enrolled user's changed-condition data."""

    learning_genuine, calibration_genuine = _split_enrollment_windows(
        enrollment_windows
    )
    if learning_genuine[0].participant_id != participant_id:
        raise ValueError("participant_id does not match the enrolment windows")
    learning_impostors = _sample_cohort(
        cohort_learning_windows,
        participant_id,
        cohort_windows_per_group,
        seed,
    )
    calibration_impostors = _sample_cohort(
        cohort_validation_windows,
        participant_id,
        cohort_windows_per_group,
        seed + 1,
    )

    positive_embeddings = encode_probe_windows(
        learning_genuine, model, normalizer, batch_size
    )
    negative_embeddings = encode_probe_windows(
        learning_impostors, model, normalizer, batch_size
    )
    training_embeddings = np.concatenate((positive_embeddings, negative_embeddings))
    training_labels = np.concatenate(
        (np.ones(len(positive_embeddings)), np.zeros(len(negative_embeddings)))
    )
    classifier = SVC(
        kernel="rbf",
        C=c_value,
        gamma="scale",
        class_weight="balanced",
    )
    classifier.fit(training_embeddings, training_labels)

    genuine_calibration_embeddings = encode_probe_windows(
        calibration_genuine, model, normalizer, batch_size
    )
    impostor_calibration_embeddings = encode_probe_windows(
        calibration_impostors, model, normalizer, batch_size
    )
    genuine_scores = causal_mean_fusion(
        _score_records(
            participant_id,
            calibration_genuine,
            _dissimilarity(classifier, genuine_calibration_embeddings),
        ),
        fusion_window,
    )
    impostor_scores = causal_mean_fusion(
        _score_records(
            participant_id,
            calibration_impostors,
            _dissimilarity(classifier, impostor_calibration_embeddings),
        ),
        fusion_window,
    )
    threshold, _, _ = select_threshold_from_distances(
        np.asarray([score.distance for score in genuine_scores]),
        np.asarray([score.distance for score in impostor_scores]),
        target_far,
    )
    return EnrollmentVerifier(
        participant_id=participant_id,
        classifier=classifier,
        threshold=threshold,
        training_genuine_count=len(positive_embeddings),
        training_impostor_count=len(negative_embeddings),
        calibration_genuine_count=len(genuine_calibration_embeddings),
        calibration_impostor_count=len(impostor_calibration_embeddings),
        fusion_window=fusion_window,
    )


def score_with_enrollment_verifier(
    verifier: EnrollmentVerifier,
    windows: tuple[GaitWindow, ...],
    model: GaitEncoder,
    normalizer: ChannelNormalizer,
    batch_size: int = 128,
) -> tuple[ComparisonScore, ...]:
    """Return one user-specific dissimilarity score per probe window."""

    embeddings = encode_probe_windows(windows, model, normalizer, batch_size)
    raw_scores = _score_records(
        verifier.participant_id,
        windows,
        _dissimilarity(verifier.classifier, embeddings),
    )
    return causal_mean_fusion(raw_scores, verifier.fusion_window)
