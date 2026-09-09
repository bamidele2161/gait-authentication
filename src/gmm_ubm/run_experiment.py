"""Evaluate a population GMM with normal-only MAP user adaptation."""

import argparse
from dataclasses import dataclass, asdict
import json

import joblib
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.fisher_metric.config import CONDITIONS, FEATURES
from src.fisher_metric.data import (
    balanced_learning_frame, load_features, participant_folds, prepare_development,
)


@dataclass(frozen=True)
class Config:
    components: int = 8
    relevance_factor: float = 8.0
    pca_components: int = 30
    maximum_windows_per_group: int = 120
    target_far: float = 0.01
    fusion_window: int = 30
    seed: int = 42


def three_way_control_split(frame):
    frame = frame.sort_values("start_sample").reset_index(drop=True)
    first, second = int(len(frame) * .60), int(len(frame) * .80)
    enrollment = frame.iloc[:first]
    calibration = frame.iloc[first + 1:second]
    test = frame.iloc[second + 1:]
    if enrollment.empty or calibration.empty or test.empty:
        raise ValueError("ST-control recording is too short")
    return enrollment, calibration, test


class GMMUBM:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.pca = PCA(config.pca_components, whiten=True, random_state=config.seed)
        self.ubm = GaussianMixture(
            n_components=config.components, covariance_type="diag", max_iter=300,
            reg_covar=1e-5, n_init=3, random_state=config.seed,
        )

    def fit(self, features):
        values = self.scaler.fit_transform(np.asarray(features))
        values = self.pca.fit_transform(values)
        self.ubm.fit(values)
        return self

    def transform(self, features):
        return self.pca.transform(self.scaler.transform(np.asarray(features)))

    def adapt(self, enrollment_features):
        values = self.transform(enrollment_features)
        responsibilities = self.ubm.predict_proba(values)
        counts = responsibilities.sum(axis=0)
        first_order = responsibilities.T @ values
        observed_means = first_order / np.maximum(counts[:, None], 1e-9)
        weight = counts / (counts + self.config.relevance_factor)
        means = weight[:, None] * observed_means + (1 - weight[:, None]) * self.ubm.means_
        return means

    def _log_likelihood(self, values, means):
        precision = 1.0 / self.ubm.covariances_
        dimension = values.shape[1]
        constants = (
            np.log(self.ubm.weights_) - .5 * dimension * np.log(2 * np.pi)
            - .5 * np.log(self.ubm.covariances_).sum(axis=1)
        )
        difference = values[:, None, :] - means[None, :, :]
        component = constants[None, :] - .5 * np.sum(
            np.square(difference) * precision[None, :, :], axis=2
        )
        return logsumexp(component, axis=1)

    def score(self, features, adapted_means):
        values = self.transform(features)
        return self._log_likelihood(values, adapted_means) - self._log_likelihood(
            values, self.ubm.means_
        )


def causal_fusion(values, size):
    values = np.asarray(values, dtype=np.float64)
    cumulative = np.r_[0.0, np.cumsum(values)]
    positions = np.arange(len(values))
    starts = np.maximum(0, positions - size + 1)
    return (cumulative[positions + 1] - cumulative[starts]) / (positions - starts + 1)


def target_far_threshold(genuine, impostor, target_far):
    candidates = np.r_[-np.inf, np.unique(np.concatenate((genuine, impostor))), np.inf]
    valid = []
    for threshold in candidates:
        far = float((impostor >= threshold).mean())
        if far <= target_far + 1e-12:
            valid.append((float((genuine < threshold).mean()), far, threshold))
    frr, far, threshold = min(valid)
    return float(threshold), frr, far


def sample_impostor_frames(dataset, participants, maximum, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for person in participants:
        for condition in CONDITIONS:
            frame = dataset[person][condition]
            take = min(maximum, len(frame))
            rows.append(frame.iloc[np.sort(rng.choice(len(frame), take, replace=False))])
    return pd.concat(rows, ignore_index=True)


def run_fold(fold_number=1, config=Config()):
    dataset = load_features()
    development, evaluation = participant_folds(dataset, 4)[fold_number - 1]
    learning, _ = prepare_development(dataset, development, .8)
    population = balanced_learning_frame(
        learning, config.maximum_windows_per_group, config.seed + fold_number
    )
    model = GMMUBM(config).fit(population[list(FEATURES)])
    impostor_calibration = sample_impostor_frames(
        dataset, development, 40, config.seed + 100 + fold_number
    )
    rows, score_rows, user_models = [], [], {}
    for claim in evaluation:
        enrollment, calibration, baseline_test = three_way_control_split(
            dataset[claim]["st_control"]
        )
        adapted = model.adapt(enrollment[list(FEATURES)])
        genuine_calibration = causal_fusion(
            model.score(calibration[list(FEATURES)], adapted), config.fusion_window
        )
        impostor_scores = causal_fusion(
            model.score(impostor_calibration[list(FEATURES)], adapted),
            config.fusion_window,
        )
        threshold, calibration_frr, calibration_far = target_far_threshold(
            genuine_calibration, impostor_scores, config.target_far
        )
        probes = {"st_control": baseline_test}
        probes.update({condition: dataset[claim][condition] for condition in CONDITIONS[1:]})
        for condition in CONDITIONS:
            genuine = causal_fusion(
                model.score(probes[condition][list(FEATURES)], adapted),
                config.fusion_window,
            )
            impostor = []
            for probe in evaluation:
                if probe == claim:
                    continue
                impostor.extend(causal_fusion(
                    model.score(dataset[probe][condition][list(FEATURES)], adapted),
                    config.fusion_window,
                ))
            impostor = np.asarray(impostor)
            rows.append({
                "fold": fold_number, "participant": claim, "condition": condition,
                "threshold": threshold, "frr": float((genuine < threshold).mean()),
                "far": float((impostor >= threshold).mean()),
                "calibration_frr": calibration_frr, "calibration_far": calibration_far,
            })
            score_rows.extend({
                "fold": fold_number, "participant": claim, "condition": condition,
                "is_genuine": True, "score_index": index, "llr": float(value),
            } for index, value in enumerate(genuine))
        user_models[claim] = {"means": adapted, "threshold": threshold}
    metrics = pd.DataFrame(rows)
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    output = root / f"src/results/gmm_ubm/fold_{fold_number}"
    models = root / f"src/models/gmm_ubm/fold_{fold_number}"
    output.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "participant_metrics.csv", index=False)
    pd.DataFrame(score_rows).to_csv(output / "genuine_scores.csv", index=False)
    summary = metrics.groupby("condition")[["frr", "far"]].agg(["mean", "std"])
    summary.to_csv(output / "summary.csv")
    joblib.dump({"ubm": model, "users": user_models}, models / "model.joblib")
    (output / "protocol.json").write_text(json.dumps({
        "development": development, "evaluation": evaluation, "config": asdict(config)
    }, indent=2))
    print(f"Fold {fold_number}/4: evaluation={','.join(evaluation)}")
    print(summary.to_string())
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--components", type=int, default=8)
    parser.add_argument("--relevance-factor", type=float, default=8.0)
    parser.add_argument("--fusion-window", type=int, default=30)
    args = parser.parse_args()
    run_fold(args.fold, Config(
        components=args.components, relevance_factor=args.relevance_factor,
        fusion_window=args.fusion_window,
    ))


if __name__ == "__main__":
    main()
