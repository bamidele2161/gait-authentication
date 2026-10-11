import numpy as np

from src.augmented_cnn_verifier.train import VerifierTrainingConfig, train_verifier
from src.condition_invariant.records import GaitWindow


def _windows(participant, count, value):
    return tuple(
        GaitWindow(
            participant, "st_control", index, index * 128, index // 20,
            np.full((256, 2), value + index * 0.001, dtype=np.float32),
        )
        for index in range(count)
    )


def test_one_epoch_binary_training_runs_without_validation():
    result = train_verifier(
        _windows("sub_01", 8, 1.0),
        _windows("sub_02", 8, 3.0),
        VerifierTrainingConfig(epochs=1, batches_per_epoch=1, batch_size=4),
        fixed_epochs=1,
    )
    assert result.best_epoch == 1
    assert len(result.history) == 1

