"""Layered LSTM sister network and Euclidean triplet loss."""

import torch
from torch import nn
from torch.nn import functional as F

from src.paper_triplet.config import EMBEDDING_SIZE, MARGIN


class PaperTripletEncoder(nn.Module):
    def __init__(self, hidden_size=64, layers=2, embedding_size=EMBEDDING_SIZE):
        super().__init__()
        self.register_buffer("input_mean", torch.zeros(1, 1, 6))
        self.register_buffer("input_std", torch.ones(1, 1, 6))
        self.local_patterns = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(32, hidden_size, num_layers=layers, batch_first=True)
        self.projection = nn.Linear(hidden_size, embedding_size)

    def set_input_normalization(self, mean, standard_deviation):
        mean = torch.as_tensor(mean, dtype=torch.float32).reshape(1, 1, 6)
        standard_deviation = torch.as_tensor(
            standard_deviation, dtype=torch.float32
        ).reshape(1, 1, 6)
        if not torch.isfinite(mean).all() or not torch.isfinite(standard_deviation).all():
            raise ValueError("Input-normalization values must be finite")
        if torch.any(standard_deviation <= 0):
            raise ValueError("Input standard deviations must be positive")
        self.input_mean.copy_(mean)
        self.input_std.copy_(standard_deviation)

    def forward(self, windows):
        if windows.ndim != 3 or tuple(windows.shape[1:]) != (256, 6):
            raise ValueError(f"Expected (batch, 256, 6), received {tuple(windows.shape)}")
        windows = (windows - self.input_mean) / self.input_std
        local = self.local_patterns(windows.transpose(1, 2)).transpose(1, 2)
        _, (hidden, _) = self.lstm(local)
        return F.normalize(torch.tanh(self.projection(hidden[-1])), p=2, dim=1)


def paper_triplet_loss(anchor, positive, negative, margin=MARGIN):
    positive_distance = torch.linalg.vector_norm(anchor - positive, dim=1)
    negative_distance = torch.linalg.vector_norm(anchor - negative, dim=1)
    return F.relu(positive_distance - negative_distance + margin).mean()


def semi_hard_triplet_loss(anchor, positive, negative_candidates, margin=MARGIN):
    """Choose a confusing candidate negative separately for every anchor."""
    positive_distance = torch.linalg.vector_norm(anchor - positive, dim=1)
    negative_distances = torch.linalg.vector_norm(
        anchor[:, None, :] - negative_candidates, dim=2
    )
    semi_hard = (
        (negative_distances > positive_distance[:, None])
        & (negative_distances < positive_distance[:, None] + margin)
    )
    semi_hard_distance = negative_distances.masked_fill(~semi_hard, torch.inf).min(1).values
    closest_distance = negative_distances.min(1).values
    selected = torch.where(torch.isfinite(semi_hard_distance), semi_hard_distance, closest_distance)
    return F.relu(positive_distance - selected + margin).mean()
