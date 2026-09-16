"""CNN-BiLSTM-attention Siamese encoder and metric-learning losses."""

import torch
from torch import nn
from torch.nn import functional as F

from src.condition_invariant.config import NUMBER_OF_CHANNELS, WINDOW_SAMPLES


class GaitEncoder(nn.Module):
    """Learn local motion, temporal structure, and informative gait moments."""

    def __init__(
        self,
        hidden_size: int = 64,
        embedding_size: int = 64,
        dropout_probability: float = 0.25,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or embedding_size <= 0:
            raise ValueError("hidden_size and embedding_size must be positive")
        if not 0 <= dropout_probability < 1:
            raise ValueError("dropout_probability must be in [0, 1)")
        self.cnn = nn.Sequential(
            nn.Conv1d(NUMBER_OF_CHANNELS, 32, 7, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 96, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(96), nn.GELU(),
        )
        self.temporal = nn.LSTM(
            input_size=96,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.projection = nn.Sequential(
            nn.Linear(hidden_size * 4, 128),
            nn.GELU(),
            nn.Dropout(dropout_probability),
            nn.Linear(128, embedding_size),
        )

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        expected = (WINDOW_SAMPLES, NUMBER_OF_CHANNELS)
        if windows.ndim != 3 or tuple(windows.shape[1:]) != expected:
            raise ValueError(
                f"Expected windows with shape (batch, {WINDOW_SAMPLES}, "
                f"{NUMBER_OF_CHANNELS}), received {tuple(windows.shape)}"
            )
        features = self.cnn(windows.transpose(1, 2)).transpose(1, 2)
        sequence, _ = self.temporal(features)
        weights = torch.softmax(self.attention(sequence), dim=1)
        attended = (weights * sequence).sum(dim=1)
        maximum = sequence.amax(dim=1)
        embedding = self.projection(torch.cat((attended, maximum), dim=1))
        return F.normalize(embedding, p=2, dim=1)


class DevelopmentIdentityClassifier(nn.Module):
    """Temporary training head; it is discarded before unseen-user enrollment."""

    def __init__(self, embedding_size: int, number_of_identities: int):
        super().__init__()
        self.output = nn.Linear(embedding_size, number_of_identities)

    def forward(self, embeddings):
        return self.output(embeddings)


def session1_batch_hard_loss(
    embeddings: torch.Tensor,
    identities: torch.Tensor,
    conditions: torch.Tensor,
    margin: float = 0.2,
    soft_margin: bool = False,
) -> torch.Tensor:
    """Use ST-control anchors, farthest same-user positives, and closest impostors."""
    if margin <= 0:
        raise ValueError("margin must be positive")
    # The hinge below reaches exactly zero once a positive sits `margin` closer
    # than the nearest impostor, and the gradient dies with it. On unit-length
    # embeddings that bar is cleared within ~15 epochs, after which training
    # stops separating identities. The soft-margin form never saturates, so the
    # encoder keeps pulling same-person windows together throughout training.
    distances = torch.cdist(embeddings, embeddings)
    same_person = identities[:, None].eq(identities[None, :])
    different_person = ~same_person
    diagonal = torch.eye(len(embeddings), dtype=torch.bool, device=embeddings.device)
    positive_mask = same_person & ~diagonal
    anchor_mask = conditions.eq(0)  # 0 is ST-control; ST-fatigue is 1.

    hardest_positive = distances.masked_fill(~positive_mask, -torch.inf).max(1).values
    hardest_negative = distances.masked_fill(~different_person, torch.inf).min(1).values
    valid = anchor_mask & torch.isfinite(hardest_positive) & torch.isfinite(hardest_negative)
    if not torch.any(valid):
        raise ValueError("Batch contains no valid ST-control anchors")
    violation = hardest_positive[valid] - hardest_negative[valid]
    if soft_margin:
        return F.softplus(violation).mean()
    return F.relu(violation + margin).mean()
