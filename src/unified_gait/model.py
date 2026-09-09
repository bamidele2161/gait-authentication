"""One condition-robust encoder and its batch-hard metric objective."""

import torch
from torch import nn
from torch.nn import functional as F


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, strength):
        ctx.strength = strength
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.strength * gradient, None


def reverse_gradient(values, strength=1.0):
    """Keep the forward values, but reverse their encoder-training gradient."""
    return _GradientReversal.apply(values, strength)


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


class ConditionClassifier(nn.Module):
    """Condition head used to discourage condition information in embeddings."""

    def __init__(self, embedding_size=64, condition_count=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_size, 32), nn.GELU(),
            nn.Linear(32, condition_count),
        )

    def forward(self, embeddings, reversal_strength=1.0):
        return self.network(reverse_gradient(embeddings, reversal_strength))


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


def supervised_contrastive_loss(embeddings, identities, temperature=0.07):
    """Use every other same-identity sample in the batch as a positive."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    count = len(embeddings)
    identity_matches = identities[:, None].eq(identities[None, :])
    self_mask = torch.eye(count, dtype=torch.bool, device=embeddings.device)
    positive_mask = identity_matches & ~self_mask
    if not torch.all(positive_mask.any(dim=1)):
        raise ValueError("every anchor requires another same-identity sample")

    logits = embeddings @ embeddings.T / temperature
    # Subtracting each row maximum improves numerical stability without
    # changing the softmax probability.
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
    log_probability = logits - torch.log(
        exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12)
    )
    mean_positive_log_probability = (
        log_probability.masked_fill(~positive_mask, 0.0).sum(dim=1)
        / positive_mask.sum(dim=1)
    )
    return -mean_positive_log_probability.mean()
