"""Train the deeper session-1 encoder with online hard mining."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from src.condition_invariant.batches import sample_session1_batch
from src.condition_invariant.enrollment import create_user_template
from src.condition_invariant.metrics import select_threshold_from_distances
from src.condition_invariant.model import (
    DevelopmentIdentityClassifier, GaitEncoder, session1_batch_hard_loss,
)
from src.condition_invariant.normalization import ChannelNormalizer, fit_channel_normalizer
from src.condition_invariant.records import GaitWindow
from src.condition_invariant.scoring import encode_probe_windows
from src.condition_invariant.triplets import build_triplet_index, window_identity


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 40
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    margin: float = 0.2
    soft_margin: bool = False
    identity_loss_weight: float = 0.3
    participants_per_batch: int = 8
    windows_per_condition: int = 3
    batches_per_epoch: int = 100
    validation_batches: int = 20
    scoring_batch_size: int = 128
    operating_target_far: float = 0.01
    patience: int = 7
    seed: int = 42
    hidden_size: int = 64
    embedding_size: int = 64
    dropout_probability: float = 0.25

    def __post_init__(self):
        if self.epochs < 1 or self.batches_per_epoch < 1 or self.validation_batches < 1:
            raise ValueError("Epoch and batch counts must be positive")
        if self.learning_rate <= 0 or self.margin <= 0:
            raise ValueError("learning_rate and margin must be positive")
        if self.participants_per_batch < 2 or self.windows_per_condition < 2:
            raise ValueError("A batch needs multiple identities and windows")
        if self.identity_loss_weight < 0:
            raise ValueError("identity_loss_weight must be non-negative")


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    training_loss: float
    training_metric_loss: float
    training_identity_loss: float
    validation_loss: float
    validation_frr: float
    validation_far: float
    validation_threshold: float


@dataclass(frozen=True)
class TrainingResult:
    model: GaitEncoder
    normalizer: ChannelNormalizer
    history: tuple[EpochResult, ...]
    best_epoch: int
    best_validation_frr: float
    best_validation_far: float
    best_validation_threshold: float


def _check_partitions(learning, validation):
    if not learning or not validation:
        raise ValueError("Learning and validation windows must not be empty")
    if {w.participant_id for w in learning} != {w.participant_id for w in validation}:
        raise ValueError("Learning and validation participants differ")
    if {window_identity(w) for w in learning} & {window_identity(w) for w in validation}:
        raise ValueError("A gait window appears in both learning and validation")


def _run_batches(
    encoder, classifier, index, identity_to_label, normalizer, config,
    device, rng, count, optimizer,
):
    training = optimizer is not None
    encoder.train(training)
    classifier.train(training)
    totals, metrics, identities = [], [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for _ in range(count):
            batch = sample_session1_batch(
                index, identity_to_label, rng, config.participants_per_batch,
                config.windows_per_condition,
            )
            signals = np.stack([normalizer.transform(w.signal) for w in batch.windows])
            tensor = torch.from_numpy(signals).to(device)
            labels = torch.tensor(batch.identity_labels, dtype=torch.long, device=device)
            conditions = torch.tensor(batch.condition_labels, dtype=torch.long, device=device)
            if training:
                optimizer.zero_grad()
            embeddings = encoder(tensor)
            metric_loss = session1_batch_hard_loss(
                embeddings, labels, conditions, config.margin, config.soft_margin
            )
            identity_loss = F.cross_entropy( (embeddings), labels)
            total = metric_loss + config.identity_loss_weight * identity_loss
            if training:
                total.backward()
                torch.nn.utils.clip_grad_norm_(
                    (*encoder.parameters(), *classifier.parameters()), 5.0
                )
                optimizer.step()
            totals.append(float(total.detach()))
            metrics.append(float(metric_loss.detach()))
            identities.append(float(identity_loss.detach()))
    return float(np.mean(totals)), float(np.mean(metrics)), float(np.mean(identities))


def _operating_point(encoder, normalizer, learning, validation, participants, config):
    templates = np.vstack([
        create_user_template(
            person,
            tuple(w for w in learning if
                  w.participant_id == person and w.condition == "st_control"),
            encoder, normalizer, config.scoring_batch_size,
        ).embedding
        for person in participants
    ])
    probes = encode_probe_windows(validation, encoder, normalizer, config.scoring_batch_size)
    distances = np.linalg.norm(templates[:, None] - probes[None, :], axis=2)
    owners = np.asarray([w.participant_id for w in validation])
    genuine = np.asarray(participants)[:, None] == owners[None, :]
    return select_threshold_from_distances(
        distances[genuine], distances[~genuine], config.operating_target_far
    )

def train_encoder(
    learning_windows: tuple[GaitWindow, ...],
    validation_windows: tuple[GaitWindow, ...],
    config: TrainingConfig = TrainingConfig(),
    device: str = "cpu",
) -> TrainingResult:
    _check_partitions(learning_windows, validation_windows)
    torch.manual_seed(config.seed)
    selected_device = torch.device(device)
    normalizer = fit_channel_normalizer(learning_windows)
    learning_index = build_triplet_index(learning_windows)
    validation_index = build_triplet_index(validation_windows)
    participants = tuple(learning_index)
    if config.participants_per_batch > len(participants):
        raise ValueError("participants_per_batch exceeds development identities")
    identity_to_label = {person: i for i, person in enumerate(participants)}

    encoder = GaitEncoder(
        config.hidden_size, config.embedding_size, config.dropout_probability
    ).to(selected_device)
    classifier = DevelopmentIdentityClassifier(
        config.embedding_size, len(participants)
    ).to(selected_device)
    optimizer = torch.optim.AdamW(
        (*encoder.parameters(), *classifier.parameters()),
        lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    train_rng = np.random.default_rng(config.seed)
    validation_rng = np.random.default_rng(config.seed + 1_000_000)
    history, best_state = [], None
    best_key = (float("inf"), float("inf"), float("inf"))
    best_epoch, stale = 0, 0
    best_point = (float("inf"), float("inf"), float("nan"))
    for epoch in range(1, config.epochs + 1):
        train_total, train_metric, train_identity = _run_batches(
            encoder, classifier, learning_index, identity_to_label, normalizer,
            config, selected_device, train_rng, config.batches_per_epoch, optimizer,
        )
        validation_total, _, _ = _run_batches(
            encoder, classifier, validation_index, identity_to_label, normalizer,
            config, selected_device, validation_rng, config.validation_batches, None,
        )
        threshold, frr, far = _operating_point(
            encoder, normalizer, learning_windows, validation_windows,
            participants, config,
        )
        history.append(EpochResult(
            epoch, train_total, train_metric, train_identity, validation_total,
            frr, far, threshold,
        ))
        print(
            f"  epoch={epoch:02d} train={train_total:.5f} "
            f"metric={train_metric:.5f} identity={train_identity:.5f} "
            f"validation={validation_total:.5f} FRR/FAR={frr:.2%}/{far:.2%}"
        )
        key = (frr, far, validation_total)
        if key < best_key:
            best_key = key
            best_state = deepcopy(encoder.state_dict())
            best_epoch = epoch
            best_point = (frr, far, threshold)
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("No valid encoder state was produced")
    encoder.load_state_dict(best_state)
    encoder.eval()
    return TrainingResult(
        encoder, normalizer, tuple(history), best_epoch,
        best_point[0], best_point[1], best_point[2],
    )
