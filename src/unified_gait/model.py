"""One condition-robust encoder and its batch-hard metric objective."""

import torch
from torch import nn
from torch.nn import functional as F


class UnifiedEncoder(nn.Module):
    def __init__(self, embedding_size=64):
        super().__init__()
        self.register_buffer("mean", torch.zeros(1, 1, 8))
        self.register_buffer("std", torch.ones(1, 1, 8))
        self.cnn = nn.Sequential(
            nn.Conv1d(8, 48, 9, padding=4, bias=False),
            nn.BatchNorm1d(48), nn.GELU(),
            nn.Conv1d(48, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 96, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(96), nn.GELU(),
        )
        self.temporal = nn.GRU(96, 64, batch_first=True, bidirectional=True)
        self.projection = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, embedding_size),
        )

    @staticmethod
    def channels(windows):
        gyro, acceleration = windows[..., :3], windows[..., 3:]
        return torch.cat((
            gyro, acceleration,
            torch.linalg.vector_norm(gyro, dim=-1, keepdim=True),
            torch.linalg.vector_norm(acceleration, dim=-1, keepdim=True),
        ), dim=-1)

    def set_normalizer(self, mean, std):
        self.mean.copy_(torch.as_tensor(mean).reshape(1, 1, 8))
        self.std.copy_(torch.as_tensor(std).reshape(1, 1, 8))

    def forward(self, windows):
        values = (self.channels(windows) - self.mean) / self.std
        values = self.cnn(values.transpose(1, 2)).transpose(1, 2)
        sequence, _ = self.temporal(values)
        pooled = torch.cat((sequence.mean(1), sequence.amax(1)), dim=1)
        return F.normalize(self.projection(pooled), dim=1)


def batch_hard_triplet_loss(embeddings, identities, margin=0.2):
    """Pull every same-person condition together and push other people away."""
    distances = torch.cdist(embeddings, embeddings)
    same = identities[:, None].eq(identities[None, :])
    eye = torch.eye(len(identities), dtype=torch.bool, device=identities.device)
    positives = same & ~eye
    negatives = ~same
    hardest_positive = distances.masked_fill(~positives, -torch.inf).max(1).values
    hardest_negative = distances.masked_fill(~negatives, torch.inf).min(1).values
    valid = torch.isfinite(hardest_positive) & torch.isfinite(hardest_negative)
    return F.relu(hardest_positive[valid] - hardest_negative[valid] + margin).mean()
