import pandas as pd

from src.secure_adaptation.run_experiment import (
    _class_balanced_group_cv,
    _fit_verifier,
    _refit_adapted_verifier,
    should_activate_adaptation,
)
from src.utils import FEATURE_COLS


def make_features(participant, label_offset, rows=60):
    values = {
        column: [label_offset + index / rows for index in range(rows)]
        for column in FEATURE_COLS
    }
    values.update({
        "participant_id": [participant] * rows,
        "session_type": ["st_control"] * rows,
        "block_id": [index // 5 for index in range(rows)],
    })
    return pd.DataFrame(values)


def test_adapted_refit_preserves_enrollment_selected_hyperparameters():
    positive = make_features("sub_01", 1.0)
    negative = make_features("sub_02", -1.0)
    static = _fit_verifier(positive, negative, seed=42)
    update = make_features("sub_01", 1.5, rows=10)
    adapted = _refit_adapted_verifier(
        pd.concat((positive, update), ignore_index=True), negative, static
    )
    assert adapted.named_steps["svm"].C == static.named_steps["svm"].C
    assert adapted.named_steps["svm"].gamma == static.named_steps["svm"].gamma


def test_group_cv_keeps_groups_together_and_both_classes_in_each_fold():
    labels = [0] * 10 + [1] * 10
    groups = [f"negative-{index // 2}" for index in range(10)] + [
        f"positive-{index // 2}" for index in range(10)
    ]
    for training, validation in _class_balanced_group_cv(labels, groups, folds=5):
        assert set(pd.Series(labels).iloc[training]) == {0, 1}
        assert set(pd.Series(labels).iloc[validation]) == {0, 1}
        assert not set(pd.Series(groups).iloc[training]) & set(pd.Series(groups).iloc[validation])


def test_gate_requires_lower_frr_and_security_compliance():
    assert should_activate_adaptation(0.50, 0.20, 0.01)
    assert not should_activate_adaptation(0.20, 0.30, 0.00)
    assert not should_activate_adaptation(0.50, 0.20, 0.02)
