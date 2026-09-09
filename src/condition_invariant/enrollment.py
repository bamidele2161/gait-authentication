"""Create an unseen user's template from ST-control enrolment windows only."""

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from src.condition_invariant.config import WINDOW_SAMPLES
from src.condition_invariant.model import GaitEncoder
from src.condition_invariant.normalization import ChannelNormalizer
from src.condition_invariant.records import GaitWindow
from src.condition_invariant.triplets import window_identity


@dataclass(frozen=True)
class EvaluationEnrollmentSplit:
    """Normal-walking enrolment and later baseline-test windows."""

    enrollment_windows: tuple[GaitWindow, ...]
    test_windows: tuple[GaitWindow, ...]


@dataclass(frozen=True)
class UserTemplate:
    """One enrolled user's unit-length reference embedding."""

    participant_id: str
    embedding: np.ndarray
    enrollment_window_count: int

    def __post_init__(self) -> None:
        embedding = np.asarray(self.embedding, dtype=np.float32).copy()

        if not self.participant_id.startswith("sub_"):
            raise ValueError(f"Invalid participant ID: {self.participant_id!r}")
        if embedding.ndim != 1 or embedding.size == 0:
            raise ValueError("Template embedding must be a non-empty vector")
        if not np.isfinite(embedding).all():
            raise ValueError("Template embedding contains NaN or infinite values")
        if self.enrollment_window_count < 1:
            raise ValueError("enrollment_window_count must be at least 1")
        if not np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5):
            raise ValueError("Template embedding must have unit L2 length")

        embedding.setflags(write=False)
        object.__setattr__(self, "embedding", embedding)


def split_evaluation_st_control(
    windows: tuple[GaitWindow, ...],
    enrollment_fraction: float = 0.60,
    test_start_fraction: float = 0.80,
) -> EvaluationEnrollmentSplit:
    """Keep early ST-control for enrolment and late ST-control for testing."""

    if not 0 < enrollment_fraction < test_start_fraction < 1:
        raise ValueError(
            "Fractions must satisfy 0 < enrollment_fraction "
            "< test_start_fraction < 1"
        )
    if not windows:
        raise ValueError("Cannot split an empty ST-control recording")

    ordered = tuple(sorted(windows, key=lambda w: w.start_sample))
    participants = {w.participant_id for w in ordered}
    conditions   = {w.condition for w in ordered}
    if len(participants) != 1 or conditions != {"st_control"}:
        raise ValueError("Evaluation split requires one participant's ST-control data")

    enrollment_end = int(len(ordered) * enrollment_fraction)
    test_start     = int(len(ordered) * test_start_fraction) + 1
    enrollment_windows = ordered[:enrollment_end]
    test_windows       = ordered[test_start:]
    if not enrollment_windows or not test_windows:
        raise ValueError("Split produced empty enrolment or test data")
    if (
        enrollment_windows[-1].start_sample + WINDOW_SAMPLES
        > test_windows[0].start_sample
    ):
        raise AssertionError("Enrolment and baseline test windows share raw samples")

    return EvaluationEnrollmentSplit(
        enrollment_windows=enrollment_windows,
        test_windows=test_windows,
    )


def _validate_enrollment_windows(
    participant_id: str,
    windows: tuple[GaitWindow, ...],
) -> None:
    """Enforce normal-walking-only enrolment for exactly one unseen user."""

    if not windows:
        raise ValueError("Cannot create a template without enrolment windows")
    if any(w.participant_id != participant_id for w in windows):
        raise ValueError("Every enrolment window must belong to the claimed participant")
    if any(w.condition != "st_control" for w in windows):
        raise ValueError("Unseen-user enrolment may use ST-control windows only")

    identities = [window_identity(w) for w in windows]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate enrolment window found")
    start_samples = [w.start_sample for w in windows]
    if start_samples != sorted(start_samples):
        raise ValueError("Enrolment windows must be in chronological order")


def create_user_template(
    participant_id: str,
    enrollment_windows: tuple[GaitWindow, ...],
    model: GaitEncoder,
    normalizer: ChannelNormalizer,
    batch_size: int = 64,
) -> UserTemplate:
    """Encode enrolment windows and select the medoid as the user template.

    Why medoid instead of mean?
    ---------------------------
    On the unit hypersphere the arithmetic mean is pulled off-centre by
    outlier embeddings (e.g. windows with sensor artefacts or noisy strides).
    The medoid — the actual embedding *closest* to the mean — is always a
    real, valid point on the sphere and is robust to those outliers.  This
    keeps the template well inside the genuine cluster rather than at a
    potentially empty average location.
    """

    _validate_enrollment_windows(participant_id, enrollment_windows)
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    model.eval()
    embedding_batches = []
    with torch.no_grad():
        for start in range(0, len(enrollment_windows), batch_size):
            batch = enrollment_windows[start : start + batch_size]
            normalized_signals = np.stack(
                [normalizer.transform(w.signal) for w in batch]
            )
            tensor = torch.from_numpy(normalized_signals).to(device)
            embedding_batches.append(model(tensor))

        embeddings = torch.cat(embedding_batches, dim=0)   # (N, E), unit-norm

        # Compute mean direction and find the closest actual embedding (medoid).
        mean_direction = embeddings.mean(dim=0)            # (E,)
        if float(torch.linalg.vector_norm(mean_direction)) <= 1e-12:
            raise ValueError("Enrolment embeddings cancel to a zero-length template")

        # Squared L2 distance from mean; minimise to find medoid index.
        dists_sq   = ((embeddings - mean_direction.unsqueeze(0)) ** 2).sum(dim=1)
        medoid_idx = int(dists_sq.argmin())
        template_embedding = embeddings[medoid_idx]        # already unit-norm

    return UserTemplate(
        participant_id=participant_id,
        embedding=template_embedding.cpu().numpy(),
        enrollment_window_count=len(enrollment_windows),
    )
