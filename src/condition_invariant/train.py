"""Batch-hard development training and operating-point early stopping."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from src.condition_invariant.batches import IdentityConditionBatch, sample_identity_condition_batches
from src.condition_invariant.enrollment import create_user_template
from src.condition_invariant.metrics import select_threshold_from_distances
from src.condition_invariant.model import (
    DevelopmentIdentityClassifier,
    GaitEncoder,
    supervised_contrastive_loss,
)
from src.condition_invariant.normalization import ChannelNormalizer, fit_channel_normalizer
from src.condition_invariant.records import GaitWindow
from src.condition_invariant.scoring import encode_probe_windows
from src.condition_invariant.triplets import build_triplet_index, window_identity


@dataclass(frozen=True)
class TrainingConfig:
    """Initial batch-hard settings; final values require development validation."""

    epochs: int = 40
    learning_rate: float = 1e-3
    contrastive_temperature: float = 0.1
    classification_weight: float = 0.5
    participants_per_batch: int = 8
    windows_per_condition: int = 2
    batches_per_epoch: int = 100
    validation_batches: int = 20
    scoring_batch_size: int = 128
    operating_target_far: float = 0.01
    patience: int = 7
    seed: int = 42
    hidden_size: int = 64
    embedding_size: int = 64
    dropout_probability: float = 0.2

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.contrastive_temperature <= 0:
            raise ValueError("contrastive_temperature must be positive")
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


def _check_development_partitions(learning_windows, validation_windows) -> None:
    if not learning_windows or not validation_windows:
        raise ValueError("Learning and validation windows must not be empty")
    learning_participants = {window.participant_id for window in learning_windows}
    validation_participants = {window.participant_id for window in validation_windows}
    if learning_participants != validation_participants:
        raise ValueError("Learning and validation must contain the same development participants")
    learning_ids = {window_identity(window) for window in learning_windows}
    validation_ids = {window_identity(window) for window in validation_windows}
    if learning_ids & validation_ids:
        raise ValueError("A gait window appears in both learning and validation")


def _batch_tensors(batch, normalizer, device):
    signals = np.stack([normalizer.transform(window.signal) for window in batch.windows])
    labels = torch.tensor(batch.identity_labels, dtype=torch.long, device=device)
    return torch.from_numpy(signals).to(device), labels


def _run_batches(encoder, classifier, batches, normalizer, config, device, optimizer):
    """Return mean total, triplet, and classification loss."""

    is_training = optimizer is not None
    encoder.train(is_training)
    classifier.train(is_training)
    total_sum = triplet_sum = classification_sum = 0.0
    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for batch in batches:
            windows, labels = _batch_tensors(batch, normalizer, device)
            if optimizer is not None:
                optimizer.zero_grad()
            embeddings = encoder(windows)
            metric_loss = supervised_contrastive_loss(
                embeddings, labels, config.contrastive_temperature
            )
            identity_loss = F.cross_entropy(classifier(embeddings), labels)
            loss = metric_loss + config.classification_weight * identity_loss
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_sum += float(loss.detach())
            triplet_sum += float(metric_loss.detach())
            classification_sum += float(identity_loss.detach())
    count = len(batches)
    return total_sum / count, triplet_sum / count, classification_sum / count


def _validation_operating_point(encoder, normalizer, learning_windows, validation_windows, participants, config):
    """Measure development-validation FRR at the requested FAR."""

    templates = []
    for participant_id in participants:
        enrollment = tuple(
            window for window in learning_windows
            if window.participant_id == participant_id and window.condition == "st_control"
        )
        templates.append(create_user_template(
            participant_id, enrollment, encoder, normalizer, config.scoring_batch_size
        ).embedding)

    probe_embeddings = encode_probe_windows(
        validation_windows, encoder, normalizer, config.scoring_batch_size
    )
    distances = np.linalg.norm(
        np.stack(templates)[:, None, :] - probe_embeddings[None, :, :], axis=2
    )
    probe_ids = np.asarray([window.participant_id for window in validation_windows])
    genuine_mask = np.asarray(participants)[:, None] == probe_ids[None, :]
    threshold, frr, far = select_threshold_from_distances(
        distances[genuine_mask], distances[~genuine_mask], config.operating_target_far
    )
    return frr, far, threshold


def train_encoder(learning_windows, validation_windows, config=TrainingConfig(), device="cpu"):
    """Train with batch-hard mining and retain the best biometric epoch."""

    _check_development_partitions(learning_windows, validation_windows)
    torch.manual_seed(config.seed)
    selected_device = torch.device(device)
    normalizer = fit_channel_normalizer(learning_windows)
    learning_index = build_triplet_index(learning_windows)
    validation_index = build_triplet_index(validation_windows)
    participants = tuple(sorted(learning_index))
    if config.participants_per_batch > len(participants):
        raise ValueError("participants_per_batch exceeds development population")
    identity_to_label = {participant: label for label, participant in enumerate(participants)}

    encoder = GaitEncoder(config.hidden_size, config.embedding_size, config.dropout_probability).to(selected_device)
    classifier = DevelopmentIdentityClassifier(config.embedding_size, len(participants)).to(selected_device)
    optimizer = torch.optim.Adam(
        (*encoder.parameters(), *classifier.parameters()), lr=config.learning_rate
    )
    validation_batches = sample_identity_condition_batches(
        validation_index, identity_to_label, config.seed + 1_000_000,
        config.validation_batches, config.participants_per_batch, config.windows_per_condition,
    )

    history = []
    best_key = (float("inf"), float("inf"), float("inf"))
    best_epoch = 0
    best_state = None
    best_point = (float("inf"), float("inf"), float("nan"))
    stale_epochs = 0
    for epoch in range(1, config.epochs + 1):
        learning_batches = sample_identity_condition_batches(
            learning_index, identity_to_label, config.seed + epoch,
            config.batches_per_epoch, config.participants_per_batch, config.windows_per_condition,
        )
        training_loss, metric_loss, identity_loss = _run_batches(
            encoder, classifier, learning_batches, normalizer, config, selected_device, optimizer
        )
        validation_loss, _, _ = _run_batches(
            encoder, classifier, validation_batches, normalizer, config, selected_device, None
        )
        validation_frr, validation_far, validation_threshold = _validation_operating_point(
            encoder, normalizer, learning_windows, validation_windows, participants, config
        )
        history.append(EpochResult(
            epoch, training_loss, metric_loss, identity_loss, validation_loss,
            validation_frr, validation_far, validation_threshold,
        ))
        current_key = (validation_frr, validation_far, validation_loss)
        if current_key < best_key:
            best_key = current_key
            best_epoch = epoch
            best_state = deepcopy(encoder.state_dict())
            best_point = (validation_frr, validation_far, validation_threshold)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid encoder state")
    encoder.load_state_dict(best_state)
    encoder.eval()
    return TrainingResult(
        encoder, normalizer, tuple(history), best_epoch,
        best_point[0], best_point[1], best_point[2],
    )
