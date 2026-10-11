"""Balanced per-user CNN training with training-only augmentation."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from src.augmented_cnn_verifier.augmentation import augment_signal
from src.augmented_cnn_verifier.model import PaperCNNVerifier
from src.condition_invariant.normalization import ChannelNormalizer, fit_channel_normalizer


@dataclass(frozen=True)
class VerifierTrainingConfig:
    epochs: int = 40
    batches_per_epoch: int = 50
    validation_batches: int = 15
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 7
    augmentation_probability: float = 0.80
    maximum_distortion: float = 0.15
    seed: int = 42


@dataclass(frozen=True)
class TrainedVerifier:
    model: PaperCNNVerifier
    normalizer: ChannelNormalizer
    best_epoch: int
    history: tuple[dict, ...]


def _sample_class(windows, count, rng):
    positions = rng.choice(len(windows), count, replace=len(windows) < count)
    return [windows[int(position)] for position in positions]


def _make_batch(positive, negative, normalizer, config, rng, augment):
    positive_count = config.batch_size // 2
    negative_count = config.batch_size - positive_count
    selected = _sample_class(positive, positive_count, rng) + _sample_class(
        negative, negative_count, rng
    )
    labels = np.concatenate((np.ones(positive_count), np.zeros(negative_count)))
    signals = []
    for window in selected:
        signal = window.signal
        if augment and rng.random() < config.augmentation_probability:
            signal = augment_signal(signal, rng, config.maximum_distortion)
        signals.append(normalizer.transform(signal))
    order = rng.permutation(len(selected))
    return (
        torch.from_numpy(np.stack(signals)[order]),
        torch.from_numpy(labels[order].astype(np.float32)),
    )


def _epoch(model, positive, negative, normalizer, config, rng, batches, device, optimizer):
    training = optimizer is not None
    model.train(training)
    losses = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for _ in range(batches):
            inputs, labels = _make_batch(
                positive, negative, normalizer, config, rng, augment=training
            )
            inputs, labels = inputs.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(inputs), labels)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            losses.append(float(loss.detach()))
    return float(np.mean(losses))


def train_verifier(
    positive_learning,
    negative_learning,
    config,
    device="cpu",
    positive_validation=None,
    negative_validation=None,
    fixed_epochs=None,
):
    if not positive_learning or not negative_learning:
        raise ValueError("Both positive and negative training windows are required")
    selected_device = torch.device(device)
    torch.manual_seed(config.seed)
    # Both classes are legitimate training inputs for a binary verifier, so its
    # scaler is fitted from both without using validation or DT probes.
    normalizer = fit_channel_normalizer(tuple((*positive_learning, *negative_learning)))
    model = PaperCNNVerifier().to(selected_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    training_rng = np.random.default_rng(config.seed)
    validation_rng = np.random.default_rng(config.seed + 1_000_000)
    epoch_limit = fixed_epochs or config.epochs
    use_validation = positive_validation is not None and negative_validation is not None
    history, best_state, best_epoch, best_loss, stale = [], None, 0, float("inf"), 0
    for epoch in range(1, epoch_limit + 1):
        training_loss = _epoch(
            model, positive_learning, negative_learning, normalizer, config,
            training_rng, config.batches_per_epoch, selected_device, optimizer,
        )
        validation_loss = (
            _epoch(
                model, positive_validation, negative_validation, normalizer, config,
                validation_rng, config.validation_batches, selected_device, None,
            )
            if use_validation else training_loss
        )
        history.append({
            "epoch": epoch,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
        })
        print(
            f"    epoch={epoch:02d} train={training_loss:.5f} "
            f"validation={validation_loss:.5f}"
        )
        if not use_validation:
            best_state, best_epoch = deepcopy(model.state_dict()), epoch
            continue
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state, stale = deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("Training produced no model checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return TrainedVerifier(model, normalizer, best_epoch, tuple(history))


def predict_legitimate_probability(model, normalizer, windows, batch_size, device="cpu"):
    if not windows:
        return np.empty(0, dtype=np.float64)
    selected_device = torch.device(device)
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = windows[start:start + batch_size]
            signals = np.stack([normalizer.transform(window.signal) for window in batch])
            logits = model(torch.from_numpy(signals).to(selected_device))
            outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)

