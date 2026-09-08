"""Checks for the shared gait encoder and triplet loss."""

import pytest
import torch

from src.condition_invariant.model import (
    DevelopmentConditionClassifier,
    DevelopmentIdentityClassifier,
    GaitEncoder,
    batch_hard_triplet_loss,
    supervised_contrastive_loss,
    triplet_loss,
)


def test_encoder_returns_one_unit_embedding_per_window() -> None:
    encoder = GaitEncoder(dropout_probability=0.0)
    windows = torch.randn(3, 256, 6)

    embeddings = encoder(windows)

    assert embeddings.shape == (3, 64)
    torch.testing.assert_close(
        torch.linalg.vector_norm(embeddings, dim=1),
        torch.ones(3),
    )


def test_encoder_rejects_incorrect_window_shape() -> None:
    encoder = GaitEncoder()

    with pytest.raises(ValueError, match="Expected windows with shape"):
        encoder(torch.randn(2, 128, 6))


def test_all_triplet_roles_share_one_encoder() -> None:
    encoder = GaitEncoder(dropout_probability=0.0)
    windows = torch.randn(2, 256, 6)

    first = encoder(windows)
    second = encoder(windows)

    torch.testing.assert_close(first, second)


def test_triplet_loss_is_zero_when_margin_is_already_satisfied() -> None:
    anchor = torch.tensor([[1.0, 0.0]])
    positive = torch.tensor([[1.0, 0.0]])
    negative = torch.tensor([[-1.0, 0.0]])

    loss = triplet_loss(anchor, positive, negative, margin=0.2)

    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_triplet_loss_penalizes_a_close_negative() -> None:
    anchor = torch.tensor([[1.0, 0.0]], requires_grad=True)
    positive = torch.tensor([[0.0, 1.0]], requires_grad=True)
    negative = torch.tensor([[1.0, 0.0]], requires_grad=True)

    loss = triplet_loss(anchor, positive, negative, margin=0.2)
    loss.backward()

    assert loss.item() > 0
    assert anchor.grad is not None
    assert positive.grad is not None
    assert negative.grad is not None


def test_model_settings_are_validated() -> None:
    with pytest.raises(ValueError, match="hidden_size must be positive"):
        GaitEncoder(hidden_size=0)
    with pytest.raises(ValueError, match="dropout_probability"):
        GaitEncoder(dropout_probability=1.0)
    with pytest.raises(ValueError, match="margin must be positive"):
        triplet_loss(
            torch.zeros(1, 2),
            torch.zeros(1, 2),
            torch.ones(1, 2),
            margin=0.0,
        )


def test_batch_hard_loss_backpropagates_through_difficult_examples() -> None:
    embeddings = torch.tensor(
        [[1.0, 0.0], [0.8, 0.2], [0.7, 0.3], [-1.0, 0.0]],
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1])

    loss = batch_hard_triplet_loss(embeddings, labels, margin=0.2)
    loss.backward()

    assert loss.item() > 0
    assert embeddings.grad is not None


def test_development_classifier_returns_one_logit_per_identity() -> None:
    classifier = DevelopmentIdentityClassifier(embedding_size=4, number_of_identities=3)

    logits = classifier(torch.randn(5, 4))

    assert logits.shape == (5, 3)


def test_condition_adversary_reverses_encoder_gradient() -> None:
    classifier = DevelopmentConditionClassifier(embedding_size=4, number_of_conditions=4)
    embeddings = torch.randn(5, 4, requires_grad=True)

    classifier(embeddings, reversal_strength=0.5).sum().backward()

    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()


def test_supervised_contrastive_loss_prefers_separated_identities() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    separated = torch.tensor([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, 0.1]])
    mixed = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.9, 0.1], [-0.9, 0.1]])

    separated_loss = supervised_contrastive_loss(separated, labels)
    mixed_loss = supervised_contrastive_loss(mixed, labels)

    assert separated_loss < mixed_loss
