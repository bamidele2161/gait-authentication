"""Checks for development-only encoder training."""

import numpy as np
import pytest
import torch

from src.condition_invariant.config import CONDITIONS
from src.condition_invariant.records import GaitWindow
from src.condition_invariant.train import TrainingConfig, train_encoder


def make_partition(
    start_window_index: int,
    windows_per_condition: int,
    offset: float = 0.0,
) -> tuple[GaitWindow, ...]:
    windows = []
    for participant_number in (1, 2):
        participant_id = f"sub_{participant_number:02d}"
        for condition_number, condition in enumerate(CONDITIONS):
            for local_index in range(windows_per_condition):
                window_index = start_window_index + local_index
                time = np.arange(256, dtype=np.float32)[:, None]
                channels = np.arange(6, dtype=np.float32)[None, :]
                signal = (
                    np.sin(time / (8 + participant_number))
                    + channels
                    + condition_number * 0.1
                    + local_index * 0.01
                    + offset
                ).astype(np.float32)
                windows.append(
                    GaitWindow(
                        participant_id=participant_id,
                        condition=condition,
                        window_index=window_index,
                        start_sample=window_index * 128,
                        block_id=window_index,
                        signal=signal,
                    )
                )
    return tuple(windows)


def test_train_encoder_returns_losses_and_best_model() -> None:
    learning = make_partition(0, windows_per_condition=2)
    validation = make_partition(10, windows_per_condition=2)
    config = TrainingConfig(
        epochs=2,
        patience=2,
        participants_per_batch=2,
        windows_per_condition=1,
        batches_per_epoch=2,
        validation_batches=1,
        hidden_size=8,
        embedding_size=4,
        dropout_probability=0.0,
    )

    result = train_encoder(learning, validation, config)

    assert len(result.history) == 2
    assert result.best_epoch in (1, 2)
    assert all(np.isfinite(epoch.training_loss) for epoch in result.history)
    assert all(np.isfinite(epoch.validation_loss) for epoch in result.history)
    assert result.model.training is False
    with torch.no_grad():
        embedding = result.model(torch.randn(1, 256, 6))
    assert embedding.shape == (1, 4)


def test_normalizer_uses_learning_windows_only() -> None:
    learning = make_partition(0, windows_per_condition=2)
    validation = make_partition(10, windows_per_condition=2, offset=10_000.0)
    config = TrainingConfig(
        epochs=1,
        participants_per_batch=2,
        windows_per_condition=1,
        batches_per_epoch=1,
        validation_batches=1,
        hidden_size=4,
        embedding_size=2,
        dropout_probability=0.0,
    )

    result = train_encoder(learning, validation, config)
    learning_samples = np.concatenate([window.signal for window in learning], axis=0)

    np.testing.assert_allclose(
        result.normalizer.mean,
        learning_samples.mean(axis=0),
        rtol=1e-5,
    )
    assert np.all(result.normalizer.mean < 100.0)


def test_training_rejects_a_window_shared_with_validation() -> None:
    learning = make_partition(0, windows_per_condition=2)
    validation = (learning[0], *make_partition(10, windows_per_condition=2)[1:])

    with pytest.raises(ValueError, match="both learning and validation"):
        train_encoder(learning, validation, TrainingConfig(epochs=1))


def test_training_settings_are_validated() -> None:
    with pytest.raises(ValueError, match="epochs must be at least 1"):
        TrainingConfig(epochs=0)
    with pytest.raises(ValueError, match="learning_rate must be positive"):
        TrainingConfig(learning_rate=0.0)
    with pytest.raises(ValueError, match="contrastive_temperature must be positive"):
        TrainingConfig(contrastive_temperature=0.0)
