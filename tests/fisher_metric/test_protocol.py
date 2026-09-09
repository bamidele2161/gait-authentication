import numpy as np
import pandas as pd

from src.fisher_metric.data import (
    chronological_split, evaluation_enrollment_split, participant_folds,
)
from src.fisher_metric.metric import FisherMetric
from src.fisher_metric.run_experiment import (
    causal_fusion, rank_condition_stable_features, threshold_at_eer,
)


def test_participants_are_evaluated_once():
    folds = participant_folds([f"sub_{index:02d}" for index in range(16)], 4)
    evaluated = [person for _, evaluation in folds for person in evaluation]
    assert len(evaluated) == len(set(evaluated)) == 16


def test_chronological_split_respects_blocks_and_overlap():
    frame = pd.DataFrame({
        "block_id": np.repeat(np.arange(5), 2),
        "start_sample": np.arange(10) * 128,
    })
    learning, validation = chronological_split(frame)
    assert set(learning.block_id).isdisjoint(validation.block_id)
    assert learning.start_sample.max() + 256 <= validation.start_sample.min()


def test_normal_enrollment_has_separate_template_calibration_and_test():
    frame = pd.DataFrame({"start_sample": np.arange(100) * 128})
    template, calibration, test = evaluation_enrollment_split(frame)
    assert len(template) == 40
    assert len(calibration) == 9
    assert len(test) == 49
    assert template.index.max() + 1 < calibration.index.min()
    assert calibration.index.max() + 1 < test.index.min()


def test_fisher_metric_projects_unseen_rows_to_unit_vectors():
    rng = np.random.default_rng(2)
    features = np.vstack([rng.normal(index, 0.2, (12, 10)) for index in range(4)])
    labels = np.repeat(np.arange(4), 12)
    metric = FisherMetric(8, 3).fit(features, labels)
    projected = metric.transform(features[:5])
    assert projected.shape == (5, 3)
    np.testing.assert_allclose(np.linalg.norm(projected, axis=1), 1.0)


def test_threshold_and_fusion():
    threshold, frr, far = threshold_at_eer(
        np.asarray([0.1, 0.2]), np.asarray([0.8, 0.9])
    )
    assert 0.2 <= threshold < 0.8
    assert frr == far == 0
    np.testing.assert_allclose(causal_fusion([1, 3, 5], 2), [1, 2, 4])


def test_stability_ranking_prefers_identity_over_condition_feature():
    rows = []
    for identity in ("a", "b", "c"):
        for condition_index, condition in enumerate(("st_control", "st_fatigue")):
            for repeat in range(4):
                row = {name: 0.0 for name in __import__(
                    "src.fisher_metric.config", fromlist=["FEATURES"]
                ).FEATURES}
                row.update({
                    "metric_identity": identity, "session_type": condition,
                    "GyrX_mean": float(ord(identity) - ord("a")),
                    "GyrX_std": float(condition_index * 10),
                })
                rows.append(row)
    ranking, _ = rank_condition_stable_features(pd.DataFrame(rows))
    assert ranking.index("GyrX_mean") < ranking.index("GyrX_std")
