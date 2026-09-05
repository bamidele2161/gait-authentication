import numpy as np
import pandas as pd

from src.rolling_adaptation.run_experiment import select_group_robust_threshold
from src.utils import FEATURE_COLS


class FirstFeatureModel:
    def decision_function(self, frame):
        return frame[FEATURE_COLS[0]].to_numpy()


def frame(scores, participants):
    data = {column: np.zeros(len(scores)) for column in FEATURE_COLS}
    data[FEATURE_COLS[0]] = scores
    data["participant_id"] = participants
    return pd.DataFrame(data)


def test_group_threshold_controls_each_identity():
    positive = frame([0.8, 0.9], ["target", "target"])
    negative = frame([0.85, 0.1, 0.2], ["hard", "easy", "easy"])
    threshold, frr, pooled, worst = select_group_robust_threshold(
        FirstFeatureModel(), positive, negative, target_far=0.0
    )
    assert threshold == 0.9
    assert frr == 0.5
    assert pooled == 0.0
    assert worst == 0.0
