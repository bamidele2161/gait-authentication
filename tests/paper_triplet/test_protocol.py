"""Protocol checks for the paper-faithful experiment."""

import numpy as np
import torch
import json

from src.paper_triplet.config import CONDITIONS
from src.paper_triplet.data import WindowSet, paper_enrollment_split
from src.paper_triplet.model import PaperTripletEncoder, paper_triplet_loss
from src.paper_triplet.triplets import PaperTripletDataset
from src.paper_triplet.run_experiment import (
    causal_mean_fusion, cohort_relative_distances, evaluation_eer_summary,
    fit_enrollment_normalizer, fit_input_normalization, normalize_scores,
    split_template_calibration,
    threshold_at_eer, threshold_at_far,
)


def fake_recordings():
    return {
        participant: {
            condition: np.full(
                (5, 256, 6), participant_index + condition_index / 10,
                dtype=np.float32,
            )
            for condition_index, condition in enumerate(CONDITIONS)
        }
        for participant_index, participant in enumerate(("sub_01", "sub_02"))
    }


def test_paper_enrollment_uses_first_half_up_to_400_windows():
    values = np.zeros((1_000, 256, 6), dtype=np.float32)
    recording = WindowSet("sub_01", "st_control", values, np.arange(1_000) * 128)
    template, calibration, test = paper_enrollment_split(recording)
    assert len(template) == 320
    assert len(calibration) == 79
    assert len(test) == 599


def test_four_conditions_produce_ten_protocol_types_per_anchor():
    dataset = PaperTripletDataset(fake_recordings(), anchors_per_group=1)
    assert dataset.types_per_anchor == 10
    assert len(dataset) == 2 * 4 * 10
    anchor, positive, negative = dataset[0]
    assert anchor.shape == positive.shape == (256, 6)
    assert negative.shape == (4, 256, 6)


def test_encoder_and_triplet_loss_follow_paper_dimensions():
    model = PaperTripletEncoder(hidden_size=8, layers=2)
    anchor = model(torch.randn(4, 256, 6))
    positive = model(torch.randn(4, 256, 6))
    negative = model(torch.randn(4, 256, 6))
    loss = paper_triplet_loss(anchor, positive, negative)
    assert anchor.shape == (4, 48)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_encoder_applies_frozen_channel_standardization():
    model = PaperTripletEncoder(hidden_size=8, layers=2)
    model.set_input_normalization(np.arange(6), np.arange(1, 7))
    np.testing.assert_allclose(model.input_mean.numpy().reshape(6), np.arange(6))
    np.testing.assert_allclose(model.input_std.numpy().reshape(6), np.arange(1, 7))


def test_input_normalization_balances_different_channel_scales():
    values = np.zeros((2, 4, 6), dtype=np.float32)
    values[..., 0] = np.arange(8).reshape(2, 4)
    values[..., 1] = np.arange(8).reshape(2, 4) * 100
    recordings = {"sub_01": {"st_control": values}}
    mean, standard_deviation = fit_input_normalization(recordings)
    standardized = (values - mean) / standard_deviation
    np.testing.assert_allclose(standardized.mean((0, 1))[:2], 0.0, atol=1e-6)
    np.testing.assert_allclose(standardized.std((0, 1))[:2], 1.0, atol=1e-6)


def test_thresholds_are_json_serializable():
    genuine = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    impostor = np.asarray([0.4, 0.5, 0.6], dtype=np.float32)
    values = {
        "eer": threshold_at_eer(genuine, impostor),
        "strict": threshold_at_far(genuine, impostor, 0.01),
    }
    json.dumps(values)


def test_evaluation_eer_is_explicitly_post_hoc():
    import pandas as pd
    scores = pd.DataFrame({
        "condition": ["st_control"] * 4,
        "is_genuine": [True, True, False, False],
        "enrollment_normalized_score": [0.1, 0.2, 0.8, 0.9],
    })
    summary = evaluation_eer_summary(scores)
    assert summary.evaluation_eer.iloc[0] == 0.0
    assert "post-hoc" in summary.note.iloc[0]


def test_cohort_relative_distance_rewards_closest_claimed_template():
    probes = np.asarray([[0.1, 0.0], [9.9, 0.0]])
    target = np.asarray([0.0, 0.0])
    cohort = np.asarray([[5.0, 0.0], [10.0, 0.0], [15.0, 0.0]])
    normalized, raw, means, deviations = cohort_relative_distances(
        probes, target, cohort
    )
    assert normalized[0] < normalized[1]
    assert np.all(raw >= 0)
    assert np.all(means > 0)
    assert np.all(deviations > 0)


def test_causal_fusion_never_uses_future_scores():
    values = causal_mean_fusion([1.0, 3.0, 5.0, 100.0], window_size=3)
    np.testing.assert_allclose(values, [1.0, 2.0, 3.0, 36.0])


def test_enrollment_normalization_uses_one_robust_scale():
    center, scale = fit_enrollment_normalizer([1.0, 2.0, 2.0, 3.0])
    assert center == 2.0
    assert scale > 0
    np.testing.assert_allclose(normalize_scores([center], center, scale), [0.0])


def test_template_calibration_split_has_overlap_guard():
    template, calibration = split_template_calibration(np.arange(10))
    np.testing.assert_array_equal(template, np.arange(8))
    np.testing.assert_array_equal(calibration, np.asarray([9]))
