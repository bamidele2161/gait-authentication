"""Cross-session training: annealed SupCon, grad clipping, and
cross-condition threshold calibration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.condition_invariant.batches import IdentityConditionBatch, sample_identity_condition_batches
from src.condition_invariant.config import CONDITIONS
from src.condition_invariant.enrollment import create_user_template
from src.condition_invariant.metrics import select_threshold_from_distances
from src.condition_invariant.model import (
    DevelopmentIdentityClassifier,
    GaitEncoder,
    cross_condition_supcon_loss,
)
from src.condition_invariant.normalization import ChannelNormalizer, fit_channel_normalizer
from src.condition_invariant.records import GaitWindow
from src.condition_invariant.scoring import encode_probe_windows
from src.condition_invariant.triplets import build_triplet_index, window_identity


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingConfig:
    """Cross-session training settings.

    Key differences from the original config:
    - Larger hidden/embedding sizes for the bidirectional encoder.
    - More participants and windows per batch → richer cross-session negatives.
    - Temperature anneals from a warm start to avoid early embedding collapse.
    - Gradient clipping and a cosine LR schedule for stable convergence.
    - Longer patience so cross-session invariance has time to emerge.
    """

    # Encoder capacity
    hidden_size: int = 128
    embedding_size: int = 128
    dropout_probability: float = 0.3
    num_layers: int = 2

    # Optimiser
    epochs: int = 80
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0

    # Loss
    temperature_start: float = 0.5      # warm start — avoids early collapse
    temperature_end: float = 0.07       # target temperature after annealing
    cross_condition_weight: float = 2.0 # extra pull for cross-session positives
    classification_weight: float = 0.3  # auxiliary identity-head weight

    # Batching
    participants_per_batch: int = 12
    windows_per_condition: int = 4
    batches_per_epoch: int = 100
    validation_batches: int = 20
    scoring_batch_size: int = 128

    # Early stopping
    operating_target_far: float = 0.01
    patience: int = 15

    # Reproducibility
    seed: int = 42

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.temperature_start <= self.temperature_end:
            raise ValueError("temperature_start must be greater than temperature_end")
        if self.classification_weight < 0:
            raise ValueError("classification_weight must be non-negative")
        if self.participants_per_batch < 2:
            raise ValueError("participants_per_batch must be at least 2")
        if self.windows_per_condition < 1:
            raise ValueError("windows_per_condition must be at least 1")
        if self.batches_per_epoch < 1 or self.validation_batches < 1:
            raise ValueError("Training and validation batch counts must be positive")
        if self.scoring_batch_size < 1:
            raise ValueError("scoring_batch_size must be positive")
        if not 0 <= self.operating_target_far <= 1:
            raise ValueError("operating_target_far must be between 0 and 1")
        if self.patience < 1:
            raise ValueError("patience must be at least 1")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")

    def temperature_at(self, epoch: int) -> float:
        """Cosine anneal temperature from start → end over the full run."""
        progress = min(epoch - 1, self.epochs - 1) / max(self.epochs - 1, 1)
        cosine   = 0.5 * (1.0 + np.cos(np.pi * progress))
        return self.temperature_end + (self.temperature_start - self.temperature_end) * cosine


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EpochResult:
    """Learning losses and biometric validation point after one epoch."""

    epoch: int
    training_loss: float
    training_contrastive_loss: float
    training_classification_loss: float
    validation_loss: float
    validation_frr: float
    validation_far: float
    validation_threshold: float
    temperature: float


@dataclass(frozen=True)
class TrainingResult:
    """The best restored encoder and its reproducibility information."""

    model: GaitEncoder
    normalizer: ChannelNormalizer
    history: tuple[EpochResult, ...]
    best_epoch: int
    best_validation_frr: float
    best_validation_far: float
    best_validation_threshold: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_development_partitions(learning_windows, validation_windows) -> None:
    if not learning_windows or not validation_windows:
        raise ValueError("Learning and validation windows must not be empty")
    learning_participants   = {w.participant_id for w in learning_windows}
    validation_participants = {w.participant_id for w in validation_windows}
    if learning_participants != validation_participants:
        raise ValueError("Learning and validation must contain the same development participants")
    learning_ids   = {window_identity(w) for w in learning_windows}
    validation_ids = {window_identity(w) for w in validation_windows}
    if learning_ids & validation_ids:
        raise ValueError("A gait window appears in both learning and validation")


def _condition_to_int(condition: str) -> int:
    """Map condition name to a stable integer label."""
    return CONDITIONS.index(condition)


def _batch_tensors(batch: IdentityConditionBatch, normalizer: ChannelNormalizer, device: torch.device):
    """Return (windows_tensor, identity_labels, condition_labels) on device."""
    signals = np.stack([normalizer.transform(w.signal) for w in batch.windows])
    identity_labels   = torch.tensor(batch.identity_labels,  dtype=torch.long, device=device)
    condition_labels  = torch.tensor(
        [_condition_to_int(w.condition) for w in batch.windows],
        dtype=torch.long,
        device=device,
    )
    return torch.from_numpy(signals).to(device), identity_labels, condition_labels


def _run_batches(
    encoder: GaitEncoder,
    classifier: DevelopmentIdentityClassifier,
    batches: tuple[IdentityConditionBatch, ...],
    normalizer: ChannelNormalizer,
    config: TrainingConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    temperature: float,
) -> tuple[float, float, float]:
    """Return mean (total, contrastive, classification) loss."""

    is_training = optimizer is not None
    encoder.train(is_training)
    classifier.train(is_training)
    total_sum = contrastive_sum = classification_sum = 0.0
    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for batch in batches:
            windows, id_labels, cond_labels = _batch_tensors(batch, normalizer, device)
            if optimizer is not None:
                optimizer.zero_grad()

            embeddings = encoder(windows)
            metric_loss = cross_condition_supcon_loss(
                embeddings,
                id_labels,
                cond_labels,
                temperature=temperature,
                cross_condition_weight=config.cross_condition_weight,
            )
            identity_loss = F.cross_entropy(classifier(embeddings), id_labels)
            loss = metric_loss + config.classification_weight * identity_loss

            if optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(classifier.parameters()),
                    config.max_grad_norm,
                )
                optimizer.step()

            total_sum         += float(loss.detach())
            contrastive_sum   += float(metric_loss.detach())
            classification_sum += float(identity_loss.detach())

    count = len(batches)
    return total_sum / count, contrastive_sum / count, classification_sum / count


def _validation_operating_point(
    encoder: GaitEncoder,
    normalizer: ChannelNormalizer,
    learning_windows: tuple[GaitWindow, ...],
    validation_windows: tuple[GaitWindow, ...],
    participants: tuple[str, ...],
    config: TrainingConfig,
) -> tuple[float, float, float]:
    """Measure development-validation FRR at the requested FAR.

    Critical fix vs the original: this function now scores probes from **all
    four conditions**, not just ``st_control``.  This means the calibration
    threshold is selected on the same session-shift distribution that the
    unseen evaluation participants will face.
    """

    # Build one template per development participant from their ST-control
    # learning windows only (mirrors the evaluation enrollment protocol).
    templates = []
    for participant_id in participants:
        enrollment = tuple(
            w for w in learning_windows
            if w.participant_id == participant_id and w.condition == "st_control"
        )
        if not enrollment:
            raise ValueError(
                f"No ST-control learning windows found for development participant "
                f"{participant_id}"
            )
        templates.append(
            create_user_template(
                participant_id, enrollment, encoder, normalizer, config.scoring_batch_size
            ).embedding
        )

    template_matrix = np.stack(templates)         # (P, E)
    participant_array = np.asarray(participants)  # (P,)

    genuine_distances  = []
    impostor_distances = []

    # Score validation probes from every condition — not just ST-control.
    for condition in CONDITIONS:
        condition_windows = tuple(
            w for w in validation_windows if w.condition == condition
        )
        if not condition_windows:
            continue

        probe_embeddings = encode_probe_windows(
            condition_windows, encoder, normalizer, config.scoring_batch_size
        )                                          # (N, E)
        probe_ids = np.asarray([w.participant_id for w in condition_windows])

        # Distance from every template to every probe in this condition.
        dists = np.linalg.norm(
            template_matrix[:, None, :] - probe_embeddings[None, :, :], axis=2
        )  # (P, N)

        genuine_mask = participant_array[:, None] == probe_ids[None, :]  # (P, N)
        genuine_distances.append(dists[genuine_mask])
        impostor_distances.append(dists[~genuine_mask])

    genuine_all  = np.concatenate(genuine_distances)
    impostor_all = np.concatenate(impostor_distances)
    threshold, frr, far = select_threshold_from_distances(
        genuine_all, impostor_all, config.operating_target_far
    )
    return frr, far, threshold


# ---------------------------------------------------------------------------
# Public training entry point
# ---------------------------------------------------------------------------

def train_encoder(
    learning_windows: tuple[GaitWindow, ...],
    validation_windows: tuple[GaitWindow, ...],
    config: TrainingConfig = TrainingConfig(),
    device: str = "cpu",
) -> TrainingResult:
    """Train with cross-session SupCon and retain the best biometric epoch."""

    _check_development_partitions(learning_windows, validation_windows)
    torch.manual_seed(config.seed)
    selected_device = torch.device(device)
    normalizer      = fit_channel_normalizer(learning_windows)
    learning_index  = build_triplet_index(learning_windows)
    validation_index = build_triplet_index(validation_windows)
    participants    = tuple(sorted(learning_index))
    if config.participants_per_batch > len(participants):
        raise ValueError("participants_per_batch exceeds development population")
    identity_to_label = {p: i for i, p in enumerate(participants)}

    encoder = GaitEncoder(
        hidden_size=config.hidden_size,
        embedding_size=config.embedding_size,
        dropout_probability=config.dropout_probability,
        num_layers=config.num_layers,
    ).to(selected_device)
    classifier = DevelopmentIdentityClassifier(
        config.embedding_size, len(participants)
    ).to(selected_device)

    optimizer = torch.optim.Adam(
        (*encoder.parameters(), *classifier.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.learning_rate * 0.05
    )

    # Pre-sample fixed validation batches (same batches every epoch for fair comparison).
    validation_batches = sample_identity_condition_batches(
        validation_index,
        identity_to_label,
        config.seed + 1_000_000,
        config.validation_batches,
        config.participants_per_batch,
        config.windows_per_condition,
    )

    history: list[EpochResult] = []
    best_key   = (float("inf"), float("inf"), float("inf"))
    best_epoch = 0
    best_state = None
    best_point = (float("inf"), float("inf"), float("nan"))
    stale_epochs = 0

    for epoch in range(1, config.epochs + 1):
        temperature = config.temperature_at(epoch)

        learning_batches = sample_identity_condition_batches(
            learning_index,
            identity_to_label,
            config.seed + epoch,
            config.batches_per_epoch,
            config.participants_per_batch,
            config.windows_per_condition,
        )
        training_loss, metric_loss, identity_loss = _run_batches(
            encoder, classifier, learning_batches,
            normalizer, config, selected_device, optimizer, temperature,
        )
        validation_loss, _, _ = _run_batches(
            encoder, classifier, validation_batches,
            normalizer, config, selected_device, None, temperature,
        )
        validation_frr, validation_far, validation_threshold = _validation_operating_point(
            encoder, normalizer, learning_windows, validation_windows, participants, config
        )

        history.append(EpochResult(
            epoch=epoch,
            training_loss=training_loss,
            training_contrastive_loss=metric_loss,
            training_classification_loss=identity_loss,
            validation_loss=validation_loss,
            validation_frr=validation_frr,
            validation_far=validation_far,
            validation_threshold=validation_threshold,
            temperature=temperature,
        ))

        # Best-model key: minimise (FRR + FAR) sum, break ties on loss.
        combined_error = validation_frr + validation_far
        current_key   = (combined_error, validation_loss)
        best_combined = best_key[0]
        if current_key < (best_combined, best_key[1]):
            best_key   = current_key
            best_epoch = epoch
            best_state = deepcopy(encoder.state_dict())
            best_point = (validation_frr, validation_far, validation_threshold)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

        scheduler.step()

    if best_state is None:
        raise RuntimeError("Training did not produce a valid encoder state")
    encoder.load_state_dict(best_state)
    encoder.eval()
    return TrainingResult(
        model=encoder,
        normalizer=normalizer,
        history=tuple(history),
        best_epoch=best_epoch,
        best_validation_frr=best_point[0],
        best_validation_far=best_point[1],
        best_validation_threshold=best_point[2],
    )
