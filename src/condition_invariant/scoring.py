"""Score probe gait windows against one claimed user's frozen template."""

from dataclasses import dataclass

import numpy as np
import torch

from src.condition_invariant.enrollment import UserTemplate
from src.condition_invariant.model import GaitEncoder
from src.condition_invariant.normalization import ChannelNormalizer
from src.condition_invariant.records import GaitWindow


@dataclass(frozen=True)
class ComparisonScore:
    """One window-level comparison with identity and condition metadata."""

    claimed_participant_id: str
    probe_participant_id: str
    condition: str
    window_index: int
    start_sample: int
    distance: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.distance) or self.distance < 0:
            raise ValueError("Comparison distance must be finite and non-negative")

    @property
    def is_genuine(self) -> bool:
        """Return whether the probe really belongs to the claimed user."""

        return self.claimed_participant_id == self.probe_participant_id

    @property
    def similarity_score(self) -> float:
        """Express distance as a score where a larger value is a better match."""

        return -self.distance


def encode_probe_windows(
    windows: tuple[GaitWindow, ...],
    model: GaitEncoder,
    normalizer: ChannelNormalizer,
    batch_size: int = 64,
) -> np.ndarray:
    """Encode probe windows independently using frozen development components."""

    if not windows:
        raise ValueError("Cannot score an empty probe-window collection")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    model.eval()
    encoded_batches = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = windows[start : start + batch_size]
            normalized_signals = np.stack(
                [normalizer.transform(window.signal) for window in batch]
            )
            tensor = torch.from_numpy(normalized_signals).to(device)
            encoded_batches.append(model(tensor))

        embeddings = torch.cat(encoded_batches, dim=0)

    if embeddings.ndim != 2 or embeddings.shape[0] != len(windows):
        raise ValueError("Encoder returned an invalid probe-embedding shape")
    if not torch.isfinite(embeddings).all():
        raise ValueError("Probe embeddings contain NaN or infinite values")
    norms = torch.linalg.vector_norm(embeddings, dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5):
        raise ValueError("Probe embeddings must have unit L2 length")

    return embeddings.cpu().numpy().astype(np.float32, copy=False)


def score_windows(
    template: UserTemplate,
    windows: tuple[GaitWindow, ...],
    model: GaitEncoder,
    normalizer: ChannelNormalizer,
    batch_size: int = 64,
) -> tuple[ComparisonScore, ...]:
    """Calculate one Euclidean template distance for every probe window."""

    embeddings = encode_probe_windows(
        windows=windows,
        model=model,
        normalizer=normalizer,
        batch_size=batch_size,
    )
    if embeddings.shape[1] != template.embedding.size:
        raise ValueError(
            "Probe and template embedding sizes differ: "
            f"{embeddings.shape[1]} and {template.embedding.size}"
        )

    distances = np.linalg.norm(embeddings - template.embedding[None, :], axis=1)
    return tuple(
        ComparisonScore(
            claimed_participant_id=template.participant_id,
            probe_participant_id=window.participant_id,
            condition=window.condition,
            window_index=window.window_index,
            start_sample=window.start_sample,
            distance=float(distance),
        )
        for window, distance in zip(windows, distances)
    )


def causal_mean_fusion(
    scores: tuple[ComparisonScore, ...],
    fusion_window: int,
) -> tuple[ComparisonScore, ...]:
    """Average the current and previous scores within each probe stream."""

    if fusion_window < 1:
        raise ValueError("fusion_window must be at least 1")
    if not scores:
        raise ValueError("Cannot fuse an empty score collection")

    grouped: dict[tuple[str, str, str], list[ComparisonScore]] = {}
    for score in scores:
        key = (
            score.claimed_participant_id,
            score.probe_participant_id,
            score.condition,
        )
        grouped.setdefault(key, []).append(score)

    fused = []
    for key in sorted(grouped):
        stream = sorted(grouped[key], key=lambda score: score.start_sample)
        if len(stream) < fusion_window:
            raise ValueError(
                f"Probe stream {key} is shorter than fusion_window={fusion_window}"
            )
        for index in range(fusion_window - 1, len(stream)):
            current = stream[index]
            recent = stream[index - fusion_window + 1 : index + 1]
            fused.append(
                ComparisonScore(
                    claimed_participant_id=current.claimed_participant_id,
                    probe_participant_id=current.probe_participant_id,
                    condition=current.condition,
                    window_index=current.window_index,
                    start_sample=current.start_sample,
                    distance=float(np.mean([score.distance for score in recent])),
                )
            )
    return tuple(fused)
