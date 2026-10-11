import pytest
import torch

from src.augmented_cnn_verifier.model import PaperCNNVerifier


def test_paper_cnn_returns_one_logit_per_window():
    model = PaperCNNVerifier()
    assert model(torch.zeros(5, 256, 2)).shape == (5,)


def test_paper_cnn_rejects_wrong_input_shape():
    with pytest.raises(ValueError, match="Expected"):
        PaperCNNVerifier()(torch.zeros(5, 128, 2))

