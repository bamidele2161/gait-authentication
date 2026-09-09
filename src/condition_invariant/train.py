"""Train the session-1 Siamese encoder without exposing session-2 data."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch

from src.condition_invariant.enrollment import create_user_template
from src.condition_invariant.metrics import select_threshold_from_distances
from src.condition_invariant.model import GaitEncoder, triplet_loss
from src.condition_invariant.normalization import ChannelNormalizer, fit_channel_normalizer
from src.condition_invariant.records import GaitWindow
from src.condition_invariant.scoring import encode_probe_windows
from src.condition_invariant.triplets import (
    build_triplet_index, sample_session1_triplets, window_identity,
)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 40
    learning_rate: float = 1e-3
    margin: float = 0.2
    triplets_per_batch: int = 64
    batches_per_epoch: int = 100
    validation_batches: int = 20
    scoring_batch_size: int = 128
    operating_target_far: float = 0.01
    patience: int = 7
    seed: int = 42
    hidden_size: int = 64
    embedding_size: int = 64
    dropout_probability: float = 0.2

    def __post_init__(self):
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.margin <= 0:
            raise ValueError("margin must be positive")
        if self.triplets_per_batch < 1:
            raise ValueError("triplets_per_batch must be positive")
        if self.batches_per_epoch < 1 or self.validation_batches < 1:
            raise ValueError("Training and validation batch counts must be positive")
        if self.patience < 1:
            raise ValueError("patience must be at least 1")


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    training_loss: float
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
        raise ValueError("Learning and validation must contain the same development participants")
    if {window_identity(w) for w in learning} & {window_identity(w) for w in validation}:
        raise ValueError("A gait window appears in both learning and validation")


def _triplet_tensors(triplets, normalizer, device):
    def stack(role):
        return torch.from_numpy(np.stack([
            normalizer.transform(getattr(item, role).signal) for item in triplets
        ])).to(device)
    return stack("anchor"), stack("positive"), stack("negative")


def _run_triplet_batches(model, index, normalizer, config, device, rng, count, optimizer):
    training = optimizer is not None
    model.train(training)
    losses = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for _ in range(count):
            triplets = sample_session1_triplets(
                index, rng, config.triplets_per_batch
            )
            anchor, positive, negative = _triplet_tensors(
                triplets, normalizer, device
            )
            if training:
                optimizer.zero_grad()
            # One shared encoder processes all three roles.
            combined = model(torch.cat((anchor, positive, negative), dim=0))
            size = len(triplets)
            loss = triplet_loss(
                combined[:size], combined[size:2 * size], combined[2 * size:],
                config.margin,
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            losses.append(float(loss.detach()))
    return float(np.mean(losses))


def _validation_operating_point(
    model, normalizer, learning, validation, participants, config
):
    templates = np.vstack([
        create_user_template(
            person,
            tuple(w for w in learning if
                  w.participant_id == person and w.condition == "st_control"),
            model, normalizer, config.scoring_batch_size,
        ).embedding
        for person in participants
    ])
    probes = encode_probe_windows(
        validation, model, normalizer, config.scoring_batch_size
    )
    distances = np.linalg.norm(templates[:, None, :] - probes[None, :, :], axis=2)
    probe_people = np.asarray([w.participant_id for w in validation])
    genuine_mask = np.asarray(participants)[:, None] == probe_people[None, :]
    return select_threshold_from_distances(
        distances[genuine_mask], distances[~genuine_mask],
        config.operating_target_far,
    )

def train_encoder(
    learning_windows: tuple[GaitWindow, ...],
    validation_windows: tuple[GaitWindow, ...],
    config: TrainingConfig = TrainingConfig(),
    device: str = "cpu",
) -> TrainingResult:
    """Train only on ST-control/ST-fatigue and restore the best validation epoch."""
    _check_partitions(learning_windows, validation_windows)
    torch.manual_seed(config.seed)
    selected_device = torch.device(device)
    normalizer = fit_channel_normalizer(learning_windows)
    learning_index = build_triplet_index(learning_windows)
    validation_index = build_triplet_index(validation_windows)
    participants = tuple(learning_index)

    model = GaitEncoder(
        config.hidden_size, config.embedding_size, config.dropout_probability
    ).to(selected_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_rng = np.random.default_rng(config.seed)
    validation_rng = np.random.default_rng(config.seed + 1_000_000)

    history = []
    best_key = (float("inf"), float("inf"), float("inf"))
    best_state = None
    best_epoch = 0
    best_point = (float("inf"), float("inf"), float("nan"))
    stale = 0
    for epoch in range(1, config.epochs + 1):
        training_loss = _run_triplet_batches(
            model, learning_index, normalizer, config, selected_device,
            train_rng, config.batches_per_epoch, optimizer,
        )
        validation_loss = _run_triplet_batches(
            model, validation_index, normalizer, config, selected_device,
            validation_rng, config.validation_batches, None,
        )
        threshold, frr, far = _validation_operating_point(
            model, normalizer, learning_windows, validation_windows,
            participants, config,
        )
        history.append(EpochResult(
            epoch, training_loss, validation_loss, frr, far, threshold
        ))
        print(
            f"  epoch={epoch:02d} train_loss={training_loss:.5f} "
            f"validation_loss={validation_loss:.5f} "
            f"validation FRR/FAR={frr:.2%}/{far:.2%}"
        )
        key = (frr, far, validation_loss)
        if key < best_key:
            best_key, best_state, best_epoch = key, deepcopy(model.state_dict()), epoch
            best_point = (frr, far, threshold)
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a valid encoder")
    model.load_state_dict(best_state)
    model.eval()
    return TrainingResult(
        model, normalizer, tuple(history), best_epoch,
        best_point[0], best_point[1], best_point[2],
    )
