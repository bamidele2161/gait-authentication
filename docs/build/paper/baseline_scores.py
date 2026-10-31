"""Re-score the saved problem-definition SVMs to export every decision score.

Uses the cached per-person models and thresholds from the 9-IMU raw-axes re-run
(no retraining). Output: docs/build/paper/baseline_scores.npz with keys
  "<decision>|<claimed>|<owner>|<condition>" -> fused decision scores (higher = genuine)
  "threshold|<decision>|<claimed>"          -> that person's frozen threshold
Run from the repository root: PYTHONPATH=. venv/bin/python docs/build/paper/baseline_scores.py
"""
import json
from pathlib import Path

import joblib
import numpy as np

from src.multisensor_problem_definition import run_experiment as B
from src.robust_gait.data import CONDITIONS

MODELS = B.SRC_DIR / "models" / "multisensor_problem_definition_rerun" / "raw_axes_9imu_targetfar0.01"
OUT = Path(__file__).resolve().parent / "baseline_scores.npz"

people = B.participants()
data = {(p, c): B.load_features(p, c) for p in people for c in CONDITIONS}
splits = {p: B.chronological_split(data[p, "st_control"]) for p in people}
out = {}
for target in people:
    pipe = joblib.load(MODELS / f"{target}.joblib")
    model, scaler = pipe.named_steps["svm"], pipe.named_steps["scaler"]
    thresholds = json.loads((MODELS / f"{target}_thresholds.json").read_text())
    for decision, windows in (("2s", 1), ("11s", 10)):
        out[f"threshold|{decision}|{target}"] = np.array([thresholds[decision]])
    for condition in CONDITIONS:
        frames = ({p: splits[p]["test"] for p in people} if condition == "st_control"
                  else {p: data[p, condition] for p in people})
        streams = B._streams(model, scaler, frames, target)
        for owner, s in streams.items():
            out[f"2s|{target}|{owner}|{condition}"] = s
            out[f"11s|{target}|{owner}|{condition}"] = B.fuse(s, 10)
    print(target, flush=True)
np.savez_compressed(OUT, **out)
print("saved", OUT, len(out))
