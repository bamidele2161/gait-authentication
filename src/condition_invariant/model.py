"""Shared LSTM encoder and loss for condition-invariant gait learning."""

import torch
from torch import nn
from torch.nn import functional as F

from src.condition_invariant.config import NUMBER_OF_CHANNELS, WINDOW_SAMPLES


class GaitEncoder(nn.Module):
    """Convert one normalized raw gait window into a unit-length embedding."""

    def __init__(
        self,
        hidden_size: int = 64,
        embedding_size: int = 64,
        dropout_probability: float = 0.2,
    ) -> None:
        super().__init__()

        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if embedding_size <= 0:
            raise ValueError("embedding_size must be positive")
        if not 0.0 <= dropout_probability < 1.0:
            raise ValueError("dropout_probability must be in [0, 1)")

        self.lstm = nn.LSTM(
            input_size=NUMBER_OF_CHANNELS,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout_probability)
        self.projection = nn.Linear(hidden_size, embedding_size)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """Encode a batch shaped ``(batch, 256 samples, 6 channels)``."""

        expected_tail = (WINDOW_SAMPLES, NUMBER_OF_CHANNELS)
        if windows.ndim != 3 or tuple(windows.shape[1:]) != expected_tail:
            raise ValueError(
                "Expected windows with shape "
                f"(batch, {WINDOW_SAMPLES}, {NUMBER_OF_CHANNELS}), "
                f"received {tuple(windows.shape)}"
            )
        if not windows.is_floating_point():
            raise TypeError("windows must be a floating-point tensor")
        if not torch.isfinite(windows).all():
            raise ValueError("windows contain NaN or infinite values")

        _, (final_hidden_state, _) = self.lstm(windows)
        summary = final_hidden_state[-1]
        projected = self.projection(self.dropout(summary))
        return F.normalize(projected, p=2, dim=1)


class DevelopmentIdentityClassifier(nn.Module):
    """Temporary identity head used only while developing the shared encoder."""

    def __init__(self, embedding_size: int, number_of_identities: int) -> None:
        super().__init__()
        if embedding_size < 1:
            raise ValueError("embedding_size must be positive")
        if number_of_identities < 2:
            raise ValueError("number_of_identities must be at least 2")
        self.output = nn.Linear(embedding_size, number_of_identities)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return development-identity logits for auxiliary supervision."""

        return self.output(embeddings)


def triplet_loss(
    anchor_embedding: torch.Tensor,
    positive_embedding: torch.Tensor,
    negative_embedding: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """Penalize triplets whose negative is not sufficiently farther away."""

    if margin <= 0:
        raise ValueError("margin must be positive")
    if anchor_embedding.shape != positive_embedding.shape:
        raise ValueError("Anchor and positive embeddings must have the same shape")
    if anchor_embedding.shape != negative_embedding.shape:
        raise ValueError("Anchor and negative embeddings must have the same shape")
    if anchor_embedding.ndim != 2:
        raise ValueError("Embeddings must have shape (batch, embedding_size)")
    if not all(
        torch.isfinite(embedding).all()
        for embedding in (
            anchor_embedding,
            positive_embedding,
            negative_embedding,
        )
    ):
        raise ValueError("Embeddings contain NaN or infinite values")

    return F.triplet_margin_loss(
        anchor_embedding,
        positive_embedding,
        negative_embedding,
        margin=margin,
        p=2,
        reduction="mean",
    )


def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    identity_labels: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """Use each anchor's hardest positive and closest valid negative."""

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must have shape (batch, embedding_size)")
    if identity_labels.ndim != 1 or identity_labels.shape[0] != embeddings.shape[0]:
        raise ValueError("Identity labels must have shape (batch,)")
    if margin <= 0:
        raise ValueError("margin must be positive")
    unique_labels, counts = torch.unique(identity_labels, return_counts=True)
    if unique_labels.numel() < 2:
        raise ValueError("Batch-hard loss requires at least two identities")
    if torch.any(counts < 2):
        raise ValueError("Every identity needs at least two windows in the batch")

    distances = torch.cdist(embeddings, embeddings, p=2)
    same_identity = identity_labels[:, None] == identity_labels[None, :]
    diagonal = torch.eye(
        embeddings.shape[0], dtype=torch.bool, device=embeddings.device
    )
    positive_mask = same_identity & ~diagonal
    negative_mask = ~same_identity

    hardest_positive = distances.masked_fill(~positive_mask, -torch.inf).max(dim=1).values

    # Prefer the closest negative that is farther than the hardest positive.
    # If none exists, fall back to the closest different-person embedding.
    semi_hard_mask = (
        negative_mask
        & (distances > hardest_positive[:, None])
        & (distances < hardest_positive[:, None] + margin)
    )
    semi_hard_negative = distances.masked_fill(~semi_hard_mask, torch.inf).min(dim=1).values
    closest_negative = distances.masked_fill(~negative_mask, torch.inf).min(dim=1).values
    selected_negative = torch.where(
        torch.isfinite(semi_hard_negative), semi_hard_negative, closest_negative
    )

    return F.relu(hardest_positive - selected_negative + margin).mean()


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    identity_labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Pull all same-identity windows together and repel all other identities."""

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must have shape (batch, embedding_size)")
    if identity_labels.ndim != 1 or identity_labels.shape[0] != embeddings.shape[0]:
        raise ValueError("Identity labels must have shape (batch,)")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    _, counts = torch.unique(identity_labels, return_counts=True)
    if torch.any(counts < 2):
        raise ValueError("Every identity needs at least two windows in the batch")

    normalized = F.normalize(embeddings, p=2, dim=1)
    logits = normalized @ normalized.T / temperature
    diagonal = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    positive_mask = (
        identity_labels[:, None] == identity_labels[None, :]
    ) & ~diagonal

    # Subtracting each row maximum leaves the probabilities unchanged and
    # prevents overflow when exponentiating low-temperature similarities.
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    denominator_mask = ~diagonal
    exp_logits = torch.exp(logits) * denominator_mask
    log_probability = logits - torch.log(
        exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12)
    )
    mean_positive_log_probability = (
        (positive_mask * log_probability).sum(dim=1)
        / positive_mask.sum(dim=1)
    )
    return -mean_positive_log_probability.mean()
