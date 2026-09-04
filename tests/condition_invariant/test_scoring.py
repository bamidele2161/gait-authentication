"""Checks for window-level template scoring."""

import math

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.condition_invariant.enrollment import UserTemplate
from src.condition_invariant.normalization import ChannelNormalizer
from src.condition_invariant.records import GaitWindow
from src.condition_invariant.scoring import (
    ComparisonScore,
    causal_mean_fusion,
    encode_probe_windows,
    score_windows,
)


class PredictableEncoder(nn.Module):
    """Map the first signal value to a calculable two-value embedding."""

    def __init__(self) -> None:
        super().__init__()
        self.marker = nn.Parameter(torch.tensor(0.0))

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        values = windows[:, 0, 0]
        vectors = torch.stack((values, torch.ones_like(values)), dim=1)
        return F.normalize(vectors, p=2, dim=1)


def make_window(
    window_index: int,
    value: float,
    participant_id: str,
    condition: str,
) -> GaitWindow:
    return GaitWindow(
        participant_id=participant_id,
        condition=condition,
        window_index=window_index,
        start_sample=window_index * 128,
        block_id=window_index,
        signal=np.full((256, 6), value, dtype=np.float32),
    )


def identity_normalizer() -> ChannelNormalizer:
    return ChannelNormalizer(mean=np.zeros(6), standard_deviation=np.ones(6))


def test_scores_are_euclidean_distances_from_claimed_template() -> None:
    template = UserTemplate("sub_01", np.array([1.0, 0.0]), 5)
    windows = (
        make_window(0, 0.0, "sub_01", "st_fatigue"),
        make_window(1, 1.0, "sub_02", "dt_control"),
    )

    scores = score_windows(
        template, windows, PredictableEncoder(), identity_normalizer()
    )

    assert len(scores) == 2
    assert scores[0].distance == pytest.approx(math.sqrt(2.0))
    expected_second = np.linalg.norm(
        np.array([1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)])
        - np.array([1.0, 0.0])
    )
    assert scores[1].distance == pytest.approx(expected_second)
    assert scores[0].is_genuine is True
    assert scores[1].is_genuine is False
    assert scores[0].similarity_score == -scores[0].distance
    assert scores[0].condition == "st_fatigue"


def test_every_probe_window_remains_a_separate_comparison() -> None:
    template = UserTemplate("sub_01", np.array([1.0, 0.0]), 5)
    windows = tuple(
        make_window(index, float(index), "sub_01", "dt_fatigue")
        for index in range(3)
    )

    scores = score_windows(
        template, windows, PredictableEncoder(), identity_normalizer(), batch_size=2
    )

    assert [score.window_index for score in scores] == [0, 1, 2]
    assert len(scores) == len(windows)


def test_scoring_rejects_empty_input_and_embedding_size_mismatch() -> None:
    model = PredictableEncoder()
    normalizer = identity_normalizer()

    with pytest.raises(ValueError, match="empty probe-window"):
        encode_probe_windows((), model, normalizer)

    wrong_size_template = UserTemplate(
        "sub_01", np.array([1.0, 0.0, 0.0]), enrollment_window_count=2
    )
    windows = (make_window(0, 1.0, "sub_01", "st_control"),)
    with pytest.raises(ValueError, match="embedding sizes differ"):
        score_windows(wrong_size_template, windows, model, normalizer)


def test_causal_fusion_uses_only_current_and_previous_scores() -> None:
    scores = tuple(
        ComparisonScore("sub_01", "sub_01", "st_fatigue", index, index * 128, value)
        for index, value in enumerate((1.0, 2.0, 6.0, 4.0))
    )

    fused = causal_mean_fusion(scores, fusion_window=3)

    assert [score.distance for score in fused] == pytest.approx((3.0, 4.0))
    assert [score.window_index for score in fused] == [2, 3]
