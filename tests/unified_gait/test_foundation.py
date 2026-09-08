import numpy as np
import torch

from src.unified_gait.data import chronological_split, enrollment_split, outer_folds
from src.unified_gait.model import UnifiedEncoder, batch_hard_triplet_loss
from src.unified_gait.run_experiment import (
    causal_fusion, cosine_distances, eer_threshold, fit_global_normalizer,
    rotate_windows,
)


def test_chronological_split_discards_overlap_boundary():
    learning, validation = chronological_split(np.arange(10))
    np.testing.assert_array_equal(learning, np.arange(8))
    np.testing.assert_array_equal(validation, np.asarray([9]))


def test_enrollment_and_test_do_not_touch():
    enrollment, test = enrollment_split(np.arange(10))
    np.testing.assert_array_equal(enrollment, np.arange(5))
    np.testing.assert_array_equal(test, np.arange(6, 10))


def test_outer_folds_evaluate_each_person_once():
    folds = outer_folds([f"sub_{index:02d}" for index in range(16)])
    evaluated = [person for _, evaluation in folds for person in evaluation]
    assert len(folds) == 4
    assert len(evaluated) == len(set(evaluated)) == 16


def test_encoder_produces_unit_embeddings():
    model = UnifiedEncoder(32)
    output = model(torch.randn(4, 256, 6))
    assert output.shape == (4, 32)
    torch.testing.assert_close(torch.linalg.vector_norm(output, dim=1), torch.ones(4))


def test_batch_hard_loss_rewards_identity_separation():
    identities = torch.tensor([0, 0, 1, 1])
    good = torch.tensor([[1.0, 0.0], [.99, .01], [0.0, 1.0], [.01, .99]])
    bad = torch.tensor([[1.0, 0.0], [0.0, 1.0], [.99, .01], [.01, .99]])
    assert batch_hard_triplet_loss(good, identities) < batch_hard_triplet_loss(bad, identities)


def test_rotation_preserves_vector_magnitude():
    windows = np.random.default_rng(1).normal(size=(2, 256, 6)).astype(np.float32)
    rotated = rotate_windows(windows, 12, np.random.default_rng(2))
    np.testing.assert_allclose(
        np.linalg.norm(rotated[..., :3], axis=-1),
        np.linalg.norm(windows[..., :3], axis=-1), atol=1e-5,
    )


def test_cosine_distance_and_causal_fusion():
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(cosine_distances(embeddings, [1.0, 0.0]), [0.0, 1.0])
    np.testing.assert_allclose(causal_fusion([1.0, 3.0, 5.0], 2), [1.0, 2.0, 4.0])


def test_eer_threshold_separates_simple_scores():
    threshold, frr, far = eer_threshold(np.asarray([0.1, 0.2]), np.asarray([0.8, 0.9]))
    assert 0.2 <= threshold < 0.8
    assert frr == far == 0.0
