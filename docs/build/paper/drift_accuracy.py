"""How well do the hand-made and learned augmentations predict real later walking?

Leakage-free: per outer fold, drift vectors and feature scaling come from the
12 development users only; the 4 test users' real later strides are only used
as the comparison target.
"""
import numpy as np
import pandas as pd
from dataclasses import replace

from src.robust_gait import run_experiment as R
from src.robust_gait.data import CONDITIONS, LOCATIONS, load_strides
from src.robust_gait.protocol import create_outer_folds
from src.robust_gait.verifier import VerifierConfig, augment_strides, drift_strides

LATER = R.DRIFT_CONDITIONS
NAMES = list(LOCATIONS)
cfg = VerifierConfig(locations=tuple(NAMES), enrolment_augmentation=4, amplitude_range=0.2,
                     warp_range=0.1, drift_augmentation=4)
exp = R.ExperimentConfig(verifier=cfg)
people = R.participants()
data = {(p, c): load_strides(p, c) for p in people for c in CONDITIONS}


def feats(strides, keep):
    """Raw axes of working sensors, mean-pooled to 32 points, flattened."""
    cols = [18 + 6 * l + j for l in keep for j in range(6)]
    x = strides[:, :, cols]
    return x.reshape(len(x), 32, 4, len(cols)).mean(2).reshape(len(x), -1)


def nn_dist(queries, reference):
    """Mean distance from each query to its nearest reference row."""
    r2 = (reference ** 2).sum(1)
    out = []
    for chunk in np.array_split(queries, max(1, len(queries) // 500)):
        d2 = (chunk ** 2).sum(1)[:, None] + r2[None] - 2 * chunk @ reference.T
        out.append(np.sqrt(np.maximum(d2.min(1), 0)))
    return np.concatenate(out).mean()


rows = []
for fold in create_outer_folds(people, 4):
    dev = list(fold.development_participants)
    test = list(fold.evaluation_participants)
    bank = R.drift_bank(dev, data, exp)                     # (36, 128, 72), dev users only
    bank_by_cond = {c: bank[i::3] for i, c in enumerate(LATER)}
    for u in test:
        enrol = data[u, "st_control"].strides
        keep = [NAMES.index(l) for l in R.working_locations(enrol, dev, data, exp)]
        # scaling from development users only, as in the models
        bg = np.vstack([feats(data[p, c].strides, keep) for p in dev for c in R.TRAINING_CONDITIONS])
        mu, sd = bg.mean(0), bg.std(0) + 1e-9
        f = lambda s: (feats(s, keep) - mu) / sd
        rng = np.random.default_rng(42)
        hand = augment_strides(enrol, 4, cfg, rng)          # original + 4 hand copies
        drift = drift_strides(enrol, bank, 4, cfg, np.random.default_rng(43))  # 4 drift copies
        sets = {
            "none": f(enrol),
            "hand": f(hand),
            "learned": np.vstack((f(enrol), f(drift))),
            "hand+learned": np.vstack((f(hand), f(drift))),
        }
        # within-session floor: last 20 % of ST-control vs first 80 %
        cut = int(0.8 * len(enrol))
        floor = nn_dist(f(enrol[cut + 1:]), f(enrol[:cut]))
        st_mean = f(enrol).mean(0)
        for c in LATER:
            real = f(data[u, c].strides)
            actual = real.mean(0) - st_mean
            preds = {
                "hand": sets["hand"].mean(0) - st_mean,
                "learned (as used: all 36 vectors x 0.75)": f(enrol + 0.75 * bank.mean(0)).mean(0) - st_mean,
                "learned (condition-specific mean)": f(enrol + bank_by_cond[c].mean(0)).mean(0) - st_mean,
            }
            imp = np.vstack([f(data[o, c].strides) for o in test if o != u])
            row = {"fold": fold.fold_index + 1, "user": u, "condition": c,
                   "dropped_sensors": len(NAMES) - len(keep), "floor": floor,
                   "actual_change": np.linalg.norm(actual)}
            for name, p in preds.items():
                row[f"cos|{name}"] = actual @ p / (np.linalg.norm(actual) * np.linalg.norm(p) + 1e-12)
                row[f"R2|{name}"] = 1 - np.sum((actual - p) ** 2) / np.sum(actual ** 2)
            for name, s in sets.items():
                row[f"gen|{name}"] = nn_dist(real, s)
                row[f"imp|{name}"] = nn_dist(imp, s)
            rows.append(row)
    print("fold", fold.fold_index + 1, "done", flush=True)

df = pd.DataFrame(rows)
df.to_csv("drift_accuracy.csv", index=False)
pd.set_option("display.width", 220)
print(df.groupby("condition")[[c for c in df if c.startswith(("cos|", "R2|"))]].mean().round(2).T)
print(df.groupby("condition")[[c for c in df if c.startswith(("gen|", "imp|"))] + ["floor"]].mean().round(2).T)
print((df.filter(like="cos|learned (as").iloc[:, 0] > 0).mean(), "share of user-conditions with positive cosine")
print(df[["user", "condition", "cos|learned (as used: all 36 vectors x 0.75)", "R2|learned (as used: all 36 vectors x 0.75)"]].round(2).to_string())
