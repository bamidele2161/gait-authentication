"""Checks for unseen-user ST-control template creation."""

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.condition_invariant.enrollment import (
    create_user_template,
    split_evaluation_st_control,
)
from src.condition_invariant.normalization import ChannelNormalizer
from src.condition_invariant.records import GaitWindow


class PredictableEncoder(nn.Module):
    """Small test encoder whose output can be calculated by hand."""

    def __init__(self) -> None:
        super().__init__()
        self.marker = nn.Parameter(torch.tensor(0.0))

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        values = windows[:, 0, 0]
        return F.normalize(torch.stack((values, torch.ones_like(values)), dim=1), dim=1)


def make_window(
    window_index: int,
    value: float,
    participant_id: str = "sub_01",
    condition: str = "st_control",
) -> GaitWindow:
    signal = np.full((256, 6), value, dtype=np.float32)
    return GaitWindow(
        participant_id=participant_id,
        condition=condition,
        window_index=window_index,
        start_sample=window_index * 128,
        block_id=window_index,
        signal=signal,
    )


def test_template_is_normalized_average_of_window_embeddings() -> None:
    model = PredictableEncoder()
    model.train()
    normalizer = ChannelNormalizer(mean=np.zeros(6), standard_deviation=np.ones(6))
    windows = (make_window(0, 1.0), make_window(1, 3.0))

    template = create_user_template("sub_01", windows, model, normalizer)

    individual = F.normalize(torch.tensor([[1.0, 1.0], [3.0, 1.0]]), dim=1)
    expected = F.normalize(individual.mean(dim=0), dim=0).numpy()
    np.testing.assert_allclose(template.embedding, expected, atol=1e-6)
    assert template.participant_id == "sub_01"
    assert template.enrollment_window_count == 2
    assert np.isclose(np.linalg.norm(template.embedding), 1.0)
    assert model.training is False
    assert template.embedding.flags.writeable is False


def test_enrollment_rejects_changed_condition_data() -> None:
    model = PredictableEncoder()
    normalizer = ChannelNormalizer(mean=np.zeros(6), standard_deviation=np.ones(6))
    windows = (make_window(0, 1.0), make_window(1, 2.0, condition="st_fatigue"))

    with pytest.raises(ValueError, match="ST-control windows only"):
        create_user_template("sub_01", windows, model, normalizer)


def test_enrollment_rejects_another_participants_window() -> None:
    model = PredictableEncoder()
    normalizer = ChannelNormalizer(mean=np.zeros(6), standard_deviation=np.ones(6))
    windows = (make_window(0, 1.0), make_window(1, 2.0, participant_id="sub_02"))

    with pytest.raises(ValueError, match="claimed participant"):
        create_user_template("sub_01", windows, model, normalizer)


def test_enrollment_requires_chronological_unique_windows() -> None:
    model = PredictableEncoder()
    normalizer = ChannelNormalizer(mean=np.zeros(6), standard_deviation=np.ones(6))

    with pytest.raises(ValueError, match="chronological order"):
        create_user_template(
            "sub_01", (make_window(2, 1.0), make_window(1, 2.0)), model, normalizer
        )
    with pytest.raises(ValueError, match="Duplicate enrolment window"):
        create_user_template(
            "sub_01", (make_window(1, 1.0), make_window(1, 1.0)), model, normalizer
        )


def test_evaluation_split_uses_early_enrollment_and_late_test_windows() -> None:
    windows = tuple(make_window(index, float(index)) for index in range(20))

    split = split_evaluation_st_control(windows)

    assert [window.window_index for window in split.enrollment_windows] == list(range(12))
    assert [window.window_index for window in split.test_windows] == [17, 18, 19]
    assert (
        split.enrollment_windows[-1].start_sample + 256
        <= split.test_windows[0].start_sample
    )


def test_evaluation_split_rejects_changed_condition_data() -> None:
    windows = tuple(make_window(index, 1.0) for index in range(9)) + (
        make_window(9, 1.0, condition="st_fatigue"),
    )

    with pytest.raises(ValueError, match="ST-control data"):
        split_evaluation_st_control(windows)
