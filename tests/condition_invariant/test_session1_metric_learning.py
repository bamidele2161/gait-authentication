"""Protocol tests for session-1 training and session-2 evaluation."""

import numpy as np
import pytest
import torch

from src.condition_invariant.config import TRAINING_CONDITIONS
from src.condition_invariant.model import GaitEncoder, session1_batch_hard_loss
from src.condition_invariant.records import GaitWindow
from src.condition_invariant.triplets import (
    GaitTriplet, build_triplet_index, sample_session1_triplets,
)
from src.condition_invariant.train import TrainingConfig, train_encoder


def window(person, condition, index):
    return GaitWindow(
        participant_id=person, condition=condition, window_index=index,
        start_sample=index * 128, block_id=index,
        signal=np.full((256, 6), index + 1, dtype=np.float32),
    )


def learning_windows():
    return tuple(
        window(person, condition, index)
        for person in ("sub_01", "sub_02", "sub_03")
        for condition in TRAINING_CONDITIONS
        for index in range(3)
    )


def test_encoder_returns_unit_64d_embeddings():
    encoder = GaitEncoder(dropout_probability=0.)
    output = encoder(torch.randn(3, 256, 6))
    assert output.shape == (3, 64)
    torch.testing.assert_close(torch.linalg.vector_norm(output, dim=1), torch.ones(3))


def test_batch_hard_loss_rewards_correct_separation():
    embeddings = torch.tensor([[1., 0.], [.99, .01], [-1., 0.], [-.99, .01]])
    identities = torch.tensor([0, 0, 1, 1])
    conditions = torch.tensor([0, 1, 0, 1])
    assert session1_batch_hard_loss(
        embeddings, identities, conditions, .2
    ).item() == 0.


def test_every_sampled_anchor_is_st_control_and_dt_is_absent():
    index = build_triplet_index(learning_windows())
    triplets = sample_session1_triplets(index, np.random.default_rng(42), 48)
    assert {item.anchor.condition for item in triplets} == {"st_control"}
    assert {item.positive.condition for item in triplets} == set(TRAINING_CONDITIONS)
    assert {item.negative.condition for item in triplets} == set(TRAINING_CONDITIONS)
    assert all(item.anchor.participant_id == item.positive.participant_id for item in triplets)
    assert all(item.anchor.participant_id != item.negative.participant_id for item in triplets)


def test_index_rejects_session2_leakage():
    contaminated = learning_windows() + (window("sub_01", "dt_control", 8),)
    with pytest.raises(ValueError, match="test-only conditions"):
        build_triplet_index(contaminated)


def test_triplet_record_rejects_non_st_control_anchor():
    with pytest.raises(ValueError, match="anchor must come from ST-control"):
        GaitTriplet(
            window("sub_01", "st_fatigue", 0),
            window("sub_01", "st_control", 1),
            window("sub_02", "st_control", 0),
        )


def test_trainer_runs_using_session1_triplets_only():
    learning = learning_windows()
    validation = tuple(
        GaitWindow(
            participant_id=w.participant_id,
            condition=w.condition,
            window_index=w.window_index + 10,
            start_sample=w.start_sample + 10_000,
            block_id=w.block_id + 10,
            signal=w.signal + .01,
        )
        for w in learning
    )
    result = train_encoder(
        learning, validation,
        TrainingConfig(
            epochs=1, batches_per_epoch=1, validation_batches=1,
            participants_per_batch=3, windows_per_condition=2,
            hidden_size=4, embedding_size=4,
            dropout_probability=0.,
        ),
    )
    assert result.best_epoch == 1
    assert len(result.history) == 1
