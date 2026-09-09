"""Shared LSTM encoder and losses for condition-invariant gait learning."""

import torch
from torch import nn
from torch.nn import functional as F

from src.condition_invariant.config import NUMBER_OF_CHANNELS, WINDOW_SAMPLES


# ---------------------------------------------------------------------------
# Attention pooling
# ---------------------------------------------------------------------------

class _ScaledDotAttention(nn.Module):
    """Learn which timesteps carry identity-stable information.

    Cross-session domain shift (fatigue, dual-task) distorts *some* parts of
    the stride cycle while leaving others stable.  Attending over all LSTM
    outputs lets the encoder focus on the stable parts rather than averaging
    everything or reading only the final state.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = hidden_dim ** -0.5

    def forward(self, outputs: torch.Tensor) -> torch.Tensor:
        """Pool a sequence of LSTM outputs into one context vector.

        Args:
            outputs: ``(batch, time, hidden_dim)``

        Returns:
            context: ``(batch, hidden_dim)``
        """
        # Global query: mean of all outputs.
        q = self.query(outputs.mean(dim=1, keepdim=True))   # (B, 1, H)
        k = self.key(outputs)                                # (B, T, H)
        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (B, 1, T)
        weights = F.softmax(scores, dim=-1)                  # (B, 1, T)
        context = torch.bmm(weights, outputs).squeeze(1)    # (B, H)
        return context


# ---------------------------------------------------------------------------
# Main encoder
# ---------------------------------------------------------------------------

class GaitEncoder(nn.Module):
    """Convert one normalised raw gait window into a unit-length embedding.

    Architecture
    ------------
    256×6 window
        → 2-layer bidirectional LSTM  (hidden = hidden_size per direction)
        → scaled-dot attention over all T timestep outputs
        → context vector (2 * hidden_size)
        → dropout
        → Linear → embedding_size
        → L2 normalisation  (unit sphere)

    The bidirectional design is appropriate because at both training and
    inference the full 2-second window is already buffered before encoding.
    Reading both directions gives the model access to the complete stride
    cycle, not just its tail.
    """

    def __init__(
        self,
        hidden_size: int = 128,
        embedding_size: int = 128,
        dropout_probability: float = 0.3,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if embedding_size <= 0:
            raise ValueError("embedding_size must be positive")
        if not 0.0 <= dropout_probability < 1.0:
            raise ValueError("dropout_probability must be in [0, 1)")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.lstm = nn.LSTM(
            input_size=NUMBER_OF_CHANNELS,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_probability if num_layers > 1 else 0.0,
        )
        # Attention operates on the concatenated fwd+bwd output (2*hidden_size).
        self.attention  = _ScaledDotAttention(hidden_dim=hidden_size * 2)
        self.dropout    = nn.Dropout(dropout_probability)
        self.projection = nn.Linear(hidden_size * 2, embedding_size)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """Encode a batch shaped ``(batch, 256, 6)`` → ``(batch, embedding_size)``."""

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

        outputs, _ = self.lstm(windows)          # (B, T, 2*H)
        context    = self.attention(outputs)     # (B, 2*H)
        projected  = self.projection(self.dropout(context))  # (B, E)
        return F.normalize(projected, p=2, dim=1)


# ---------------------------------------------------------------------------
# Development identity head (unchanged)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def triplet_loss(
    anchor_embedding: torch.Tensor,
    positive_embedding: torch.Tensor,
    negative_embedding: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """Penalise triplets whose negative is not sufficiently farther away."""

    if margin <= 0:
        raise ValueError("margin must be positive")
    if anchor_embedding.shape != positive_embedding.shape:
        raise ValueError("Anchor and positive embeddings must have the same shape")
    if anchor_embedding.shape != negative_embedding.shape:
        raise ValueError("Anchor and negative embeddings must have the same shape")
    if anchor_embedding.ndim != 2:
        raise ValueError("Embeddings must have shape (batch, embedding_size)")
    if not all(
        torch.isfinite(e).all()
        for e in (anchor_embedding, positive_embedding, negative_embedding)
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

    semi_hard_mask = (
        negative_mask
        & (distances > hardest_positive[:, None])
        & (distances < hardest_positive[:, None] + margin)
    )
    semi_hard_negative = distances.masked_fill(~semi_hard_mask, torch.inf).min(dim=1).values
    closest_negative   = distances.masked_fill(~negative_mask, torch.inf).min(dim=1).values
    selected_negative  = torch.where(
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
    logits     = normalized @ normalized.T / temperature
    diagonal   = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    positive_mask = (
        identity_labels[:, None] == identity_labels[None, :]
    ) & ~diagonal

    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * (~diagonal)
    log_probability = logits - torch.log(
        exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12)
    )
    mean_positive_log_probability = (
        (positive_mask * log_probability).sum(dim=1)
        / positive_mask.sum(dim=1).clamp_min(1)
    )
    return -mean_positive_log_probability.mean()


def cross_condition_supcon_loss(
    embeddings: torch.Tensor,
    identity_labels: torch.Tensor,
    condition_labels: torch.Tensor,
    temperature: float = 0.1,
    cross_condition_weight: float = 2.0,
) -> torch.Tensor:
    """SupCon loss that up-weights cross-session positive pairs.

    Same-identity pairs from *different* conditions are up-weighted by
    ``cross_condition_weight`` relative to same-condition pairs.  This
    directly trains the encoder to treat fatigue, dual-task, and normal walks
    of the same person as more similar than walks of two different people,
    which is the core requirement for cross-session authentication.

    Args:
        embeddings:             ``(batch, embedding_size)`` unit-norm embeddings.
        identity_labels:        ``(batch,)`` integer identity IDs.
        condition_labels:       ``(batch,)`` integer condition IDs (0–3).
        temperature:            Softmax temperature (annealed during training).
        cross_condition_weight: Extra pull strength for cross-session positives.

    Returns:
        Scalar loss.
    """

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must have shape (batch, embedding_size)")
    if identity_labels.shape != condition_labels.shape:
        raise ValueError("identity_labels and condition_labels must have the same shape")
    if identity_labels.ndim != 1 or identity_labels.shape[0] != embeddings.shape[0]:
        raise ValueError("Labels must have shape (batch,)")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if cross_condition_weight < 1.0:
        raise ValueError("cross_condition_weight must be >= 1")

    _, counts = torch.unique(identity_labels, return_counts=True)
    if torch.any(counts < 2):
        raise ValueError("Every identity needs at least two windows in the batch")

    normalized    = F.normalize(embeddings, p=2, dim=1)
    logits        = normalized @ normalized.T / temperature           # (B, B)
    diagonal      = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    same_identity = (identity_labels[:, None] == identity_labels[None, :]) & ~diagonal
    same_condition = condition_labels[:, None] == condition_labels[None, :]

    # Positive pair weights: cross-condition same-identity gets extra pull.
    cross_cond_pos = same_identity & ~same_condition
    same_cond_pos  = same_identity &  same_condition
    pair_weight = torch.ones_like(logits)
    pair_weight[cross_cond_pos] = cross_condition_weight

    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * (~diagonal)
    log_prob   = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

    # Weighted mean over all positive pairs.
    weighted_log_prob = (same_identity * pair_weight * log_prob).sum(dim=1)
    normaliser        = (same_identity * pair_weight).sum(dim=1).clamp_min(1e-12)
    return -(weighted_log_prob / normaliser).mean()
