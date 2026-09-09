"""Shared-weight Siamese encoder and triplet objective."""

import torch
from torch import nn
from torch.nn import functional as F

from src.condition_invariant.config import NUMBER_OF_CHANNELS, WINDOW_SAMPLES


class GaitEncoder(nn.Module):
    """Map one two-second IMU window to a unit-length identity embedding."""

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
        _, (hidden, _) = self.lstm(windows)
        return F.normalize(self.projection(self.dropout(hidden[-1])), p=2, dim=1)


def triplet_loss(
    anchor_embedding: torch.Tensor,
    positive_embedding: torch.Tensor,
    negative_embedding: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """Pull the positive closer than the negative by at least the margin."""
    if margin <= 0:
        raise ValueError("margin must be positive")
    if anchor_embedding.shape != positive_embedding.shape:
        raise ValueError("Anchor and positive embeddings must have the same shape")
    if anchor_embedding.shape != negative_embedding.shape:
        raise ValueError("Anchor and negative embeddings must have the same shape")
    if anchor_embedding.ndim != 2:
        raise ValueError("Embeddings must have shape (batch, embedding_size)")
    return F.triplet_margin_loss(
        anchor_embedding,
        positive_embedding,
        negative_embedding,
        margin=margin,
        p=2,
        reduction="mean",
    )
