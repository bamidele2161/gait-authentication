"""Problem-definition experiment with logistic regression instead of the RBF-SVM.

Reuses src/multisensor_problem_definition/run_experiment.py unchanged (features, 60/20/20
split, negatives, 1 % FAR threshold on validation, 2-s and 11-s decisions); only the
classifier differs: L2 logistic regression, C chosen from {0.01, 0.1, 1, 10, 100} by the same
grouped 5-fold cross-validation (ROC AUC). Saves metrics and all decision scores.

Run from the repository root: PYTHONPATH=. venv/bin/python docs/build/paper/problem_definition_lr.py
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.multisensor_problem_definition import run_experiment as B
from src.robust_gait.data import CONDITIONS

OUT = B.SRC_DIR / "results" / "multisensor_problem_definition_lr" / "raw_axes_9imu_targetfar0.01"
OUT.mkdir(parents=True, exist_ok=True)
GRID = {"lr__C": [0.01, 0.1, 1, 10, 100]}
config = B.Config()

people = B.participants()
data = {(p, c): B.load_features(p, c) for p in people for c in CONDITIONS}
splits = {p: B.chronological_split(data[p, "st_control"]) for p in people}
rows, scores = [], {}
for i, target in enumerate(people):
    x, y, groups = B._training_partition(splits, target, np.random.default_rng(config.seed + i))
    pipe = Pipeline((("scaler", StandardScaler()),
                     ("lr", LogisticRegression(class_weight="balanced", max_iter=5000))))
    search = GridSearchCV(pipe, GRID, scoring="roc_auc",
                          cv=StratifiedGroupKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed),
                          n_jobs=1, error_score="raise").fit(x, y, groups=groups)
    model, scaler = search.best_estimator_.named_steps["lr"], search.best_estimator_.named_steps["scaler"]
    validation = B._streams(model, scaler, {p: splits[p]["validation"] for p in people}, target)
    for decision, windows in (("2s", 1), ("11s", config.fusion_windows)):
        vfrr, threshold, vfar = B._threshold(validation, target, config.target_far, windows)
        scores[f"threshold|{decision}|{target}"] = np.array([threshold])
        for condition in CONDITIONS:
            frames = ({p: splits[p]["test"] for p in people} if condition == "st_control"
                      else {p: data[p, condition] for p in people})
            streams = B._streams(model, scaler, frames, target)
            for owner, s in streams.items():
                scores[f"{decision}|{target}|{owner}|{condition}"] = B.fuse(s, windows)
            frr, far, eer, ng, ni = B._rates(streams, target, threshold, windows)
            rows.append(dict(participant_id=target, decision=decision, condition=condition, frr=frr, far=far,
                             eer=eer, threshold=threshold, validation_frr=vfrr, validation_far=vfar,
                             genuine_count=ng, impostor_count=ni, C=search.best_params_["lr__C"]))
    print(f"{target}: C={search.best_params_['lr__C']} CV AUC={search.best_score_:.3f}", flush=True)

m = pd.DataFrame(rows)
m.to_csv(OUT / "participant_metrics.csv", index=False)
m.groupby(["decision", "condition"])[["frr", "far", "eer"]].agg(["mean", "std"]).to_csv(OUT / "macro_summary.csv")
np.savez_compressed(OUT / "scores.npz", **scores)
(OUT / "config.json").write_text(json.dumps({"classifier": "logistic regression", "grid": GRID, **vars(config)}, indent=2))
print((m.groupby(["decision", "condition"])[["frr", "far", "eer"]].mean() * 100).round(2))
