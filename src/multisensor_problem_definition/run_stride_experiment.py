"""Problem-definition baseline on strides, under the proposed method's evaluation.

Purpose: measure how fatigue, cognitive load (dual task) and session change
affect a *conventional* gait verifier, with the data pipeline and evaluation
protocol identical to the proposed method and none of its fixes.

Same as the proposed method (src/robust_gait):
* data: nine IMUs, raw 6 axes, recording trims, left-shank stride
  segmentation (6 Hz), 128-sample phase normalisation, mean pooling to 32
  points -> 1728 features per stride (``load_strides`` + ``stride_features``);
* 4 participant-disjoint folds (12 development, 4 test users);
* test users enrol on their full ST-control recording and are tested on
  ST-fatigue, DT-control and DT-fatigue; impostors are the other 3 test users;
* negatives: 240 strides per development user;
* decisions: mean of 10 consecutive strides, advanced by one stride;
* one threshold per fold from the calibration mirror (6 pairs of development
  users as pseudo-users); operating point minimax (as the proposed method) or
  a target FAR (the most lenient threshold whose mirror FAR <= target).

Deliberately conventional (the proposed fixes are NOT used):
* negatives from development users' ST-control only (no fatigue);
* the mirror calibrates on normal walking only: pseudo-users enrol on the first
  80 % of ST-control and are tested on the held-out last 20 %;
* one classifier per user on all 1728 features (early fusion): L2 logistic
  regression (default) or RBF-SVM, hyper-parameters by grouped
  cross-validation as in the original baseline;
* no augmentation, no learned drift, no score normalisation, no per-sensor
  models, no faulty-sensor exclusion.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.robust_gait.data import CONDITIONS, LOCATIONS, SRC_DIR, load_strides
from src.robust_gait.protocol import create_outer_folds, equal_error_rate
from src.robust_gait.run_experiment import (
    ENROLMENT_FRACTION, OFFSET, _background, _minimax_threshold, fuse, participants,
)
from src.robust_gait.verifier import stride_features
from src.utils import PARAM_GRID

RESULTS_DIR = SRC_DIR / "results" / "multisensor_problem_definition_stride"
TEST_CONDITIONS = ("st_fatigue", "dt_control", "dt_fatigue")
NEGATIVE_CONDITIONS = ("st_control",)
GROUP_STRIDES = 18  # ~20 s blocks of enrolment strides for grouped CV
SVM_GRID = {key.replace("svm__", "clf__"): value for key, value in PARAM_GRID.items()}
LR_GRID = {"clf__C": [0.01, 0.1, 1, 10, 100]}


@dataclass(frozen=True)
class Config:
    classifier: str = "lr"
    target_far: float | None = None  # None -> minimax operating point
    fusion_strides: int = 10
    background_strides_per_user: int = 240
    mirror_group_size: int = 2
    cv_folds: int = 5
    downsample: int = 4
    seed: int = 42
    jobs: int = 4


def features(strides: np.ndarray, config: Config) -> np.ndarray:
    return stride_features(strides, tuple(LOCATIONS), config.downsample, "mean", False, "axes")


def fit_user(enrolment, negative_people, data, rng, config):
    """One classifier: the user's strides against other people's ST-control strides."""

    negatives = [
        _background([p], data, rng, config.background_strides_per_user, NEGATIVE_CONDITIONS)
        for p in negative_people
    ]
    x = features(np.vstack([enrolment, *negatives]), config)
    y = np.r_[np.ones(len(enrolment)), np.zeros(sum(map(len, negatives)))]
    groups = np.r_[
        np.arange(len(enrolment)) // GROUP_STRIDES,
        np.concatenate([np.full(len(n), 10_000 + i) for i, n in enumerate(negatives)]),
    ]
    if config.classifier == "svm":
        classifier, grid = SVC(kernel="rbf", class_weight="balanced"), SVM_GRID
    else:
        classifier, grid = LogisticRegression(class_weight="balanced", max_iter=5000), LR_GRID
    search = GridSearchCV(
        Pipeline((("scaler", StandardScaler()), ("clf", classifier))),
        grid, scoring="roc_auc",
        cv=StratifiedGroupKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed),
        n_jobs=config.jobs, error_score="raise",
    ).fit(x, y, groups=groups)
    model = search.best_estimator_
    return (lambda strides: model.decision_function(features(strides, config))), search


def _target_far_threshold(per_claim, target_far):
    """Most lenient distance threshold whose pooled mirror FAR <= target_far."""

    genuine = np.concatenate([np.concatenate(g) for g, _ in per_claim.values()])
    impostor = np.sort(np.concatenate([np.concatenate(i) for _, i in per_claim.values()]))
    allowed = int(np.floor(target_far * len(impostor)))  # impostor decisions that may pass
    threshold = float(np.nextafter(impostor[allowed], -np.inf))
    return threshold, float(np.mean(genuine > threshold)), float(np.mean(impostor <= threshold))


def calibrate(development, data, config, rng, decisions):
    """Calibration mirror on development users' normal walking only."""

    groups = len(development) // config.mirror_group_size
    per_claim = {name: {} for name, _ in decisions}
    for g in range(groups):
        group = development[g::groups]
        others = [p for p in development if p not in group]
        held_out = {}
        scorers = {}
        for p in group:
            strides = data[p, "st_control"].strides
            cut = int(len(strides) * ENROLMENT_FRACTION)
            held_out[p] = strides[cut + 1:]  # skip the boundary stride
            scorers[p], _ = fit_user(strides[:cut], others, data, rng, config)
        for claimed in group:
            for owner in group:
                raw = scorers[claimed](held_out[owner])
                for name, count in decisions:
                    per_claim[name].setdefault((g, claimed, "st_control"), ([], []))[
                        0 if claimed == owner else 1
                    ].append(OFFSET - fuse(raw, count))
    if config.target_far is None:
        return {name: _minimax_threshold(claims) for name, claims in per_claim.items()}
    return {name: _target_far_threshold(claims, config.target_far) for name, claims in per_claim.items()}


def run(config: Config) -> pd.DataFrame:
    people = participants()
    data = {(p, c): load_strides(p, c) for p in people for c in CONDITIONS}
    decisions = (("1stride", 1), (f"{config.fusion_strides}strides", config.fusion_strides))
    point = "" if config.target_far is None else f"_far{config.target_far:g}"
    output = RESULTS_DIR / f"{config.classifier}_proposed_protocol{point}_seed{config.seed}"
    output.mkdir(parents=True, exist_ok=True)
    rows, mirror_rows, scores = [], [], {}
    for fold in create_outer_folds(people, 4):
        rng = np.random.default_rng(config.seed + fold.fold_index)
        development = list(fold.development_participants)
        thresholds = calibrate(development, data, config, rng, decisions)
        users = list(fold.evaluation_participants)
        raw = {}
        for claimed in users:
            scorer, search = fit_user(data[claimed, "st_control"].strides, development, data, rng, config)
            print(f"  {claimed}: {search.best_params_} CV AUC={search.best_score_:.3f}", flush=True)
            for owner in users:
                for condition in TEST_CONDITIONS:
                    raw[claimed, owner, condition] = scorer(data[owner, condition].strides)
        for name, count in decisions:
            threshold, mirror_frr, mirror_far = thresholds[name]
            mirror_rows.append({"fold_index": fold.fold_index, "decision": name,
                                "mirror_frr": mirror_frr, "mirror_far": mirror_far,
                                "threshold": OFFSET - threshold})
            scores[f"threshold|{name}|{fold.fold_index}"] = np.array([OFFSET - threshold])
            for claimed in users:
                for condition in TEST_CONDITIONS:
                    for owner in users:
                        scores[f"{name}|{fold.fold_index}|{claimed}|{owner}|{condition}"] = fuse(
                            raw[claimed, owner, condition], count)
                    genuine = OFFSET - fuse(raw[claimed, claimed, condition], count)
                    impostor = np.concatenate([
                        OFFSET - fuse(raw[claimed, other, condition], count)
                        for other in users if other != claimed
                    ])
                    rows.append({
                        "fold_index": fold.fold_index, "decision": name,
                        "claimed_participant_id": claimed, "condition": condition,
                        "frr": float(np.mean(genuine > threshold)),
                        "far": float(np.mean(impostor <= threshold)),
                        "eer": equal_error_rate(genuine, impostor)[0],
                        "threshold": OFFSET - threshold,
                        "genuine_count": len(genuine), "impostor_count": len(impostor),
                    })
        print(
            f"fold {fold.fold_index + 1}: users={','.join(users)} mirror "
            + ", ".join(f"{name} FRR/FAR={thresholds[name][1]:.2%}/{thresholds[name][2]:.2%}"
                        for name, _ in decisions), flush=True,
        )

    metrics = pd.DataFrame(rows)
    summary = metrics.groupby(["decision", "condition"])[["frr", "far", "eer"]].agg(["mean", "std"])
    metrics.to_csv(output / "participant_metrics.csv", index=False)
    pd.DataFrame(mirror_rows).to_csv(output / "mirror_normal_walking.csv", index=False)
    summary.to_csv(output / "macro_summary.csv")
    np.savez_compressed(output / "scores.npz", **scores)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2))
    print((metrics.groupby(["decision", "condition"])[["frr", "far", "eer"]].mean() * 100).round(2))
    print(f"Results saved under {output}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--classifier", choices=("lr", "svm"), default="lr")
    parser.add_argument("--target-far", type=float, default=None,
                        help="mirror FAR target (e.g. 0.01); default: minimax")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    run(Config(classifier=args.classifier, target_far=args.target_far, seed=args.seed, jobs=args.jobs))


if __name__ == "__main__":
    main()
