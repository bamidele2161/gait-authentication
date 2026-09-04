"""Checks for normal-only enrolment-conditioned verification."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.condition_invariant.config import CONDITIONS
from src.condition_invariant.normalization import ChannelNormalizer
from src.condition_invariant.records import GaitWindow
from src.condition_invariant.verifier import fit_enrollment_verifier, score_with_enrollment_verifier


class DirectionEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.marker = nn.Parameter(torch.tensor(0.0))

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        value = windows[:, 0, 0]
        return F.normalize(torch.stack((value, torch.ones_like(value)), dim=1), dim=1)


def make_window(participant, condition, index, value):
    return GaitWindow(
        participant_id=participant,
        condition=condition,
        window_index=index,
        start_sample=index * 128,
        block_id=index,
        signal=np.full((256, 6), value, dtype=np.float32),
    )


def make_cohort(start_index):
    return tuple(
        make_window(participant, condition, start_index + index, -3.0 + index * 0.01)
        for participant in ("sub_02", "sub_03")
        for condition in CONDITIONS
        for index in range(4)
    )


def test_verifier_uses_normal_user_data_and_development_impostors() -> None:
    enrollment = tuple(
        make_window("sub_01", "st_control", index, 3.0 + index * 0.01)
        for index in range(12)
    )
    normalizer = ChannelNormalizer(np.zeros(6), np.ones(6))
    model = DirectionEncoder()

    verifier = fit_enrollment_verifier(
        "sub_01",
        enrollment,
        make_cohort(100),
        make_cohort(200),
        model,
        normalizer,
        cohort_windows_per_group=2,
        fusion_window=1,
    )
    probes = (
        make_window("sub_01", "st_fatigue", 300, 3.1),
        make_window("sub_02", "st_fatigue", 301, -3.1),
    )
    scores = score_with_enrollment_verifier(verifier, probes, model, normalizer)

    assert scores[0].distance < scores[1].distance
    assert 0 <= verifier.threshold <= 1
    assert verifier.training_genuine_count > 0
    assert verifier.calibration_genuine_count > 0
