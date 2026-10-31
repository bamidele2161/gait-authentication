"""Evaluate frozen user thresholds across all DUO-GAIT conditions."""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (  # noqa: E402
    DT_CONTROL_FEATURES, DT_FATIGUE_FEATURES, EXCLUDED_PARTICIPANTS,
    FEATURE_COLS, MODELS_DIR, RESULTS_DIR, ST_FATIGUE_FEATURES,
)


def compute_rates(y_true, scores, threshold):
    predictions = scores >= threshold
    positives = y_true == 1
    negatives = ~positives
    frr = np.mean(~predictions[positives]) if positives.any() else np.nan
    far = np.mean(predictions[negatives]) if negatives.any() else np.nan
    return float(frr), float(far)


def load_condition(directory):
    frames = [pd.read_csv(path) for path in sorted(directory.glob("*_features.csv"))]
    if not frames:
        raise FileNotFoundError(f"No feature files in {directory}")
    data = pd.concat(frames, ignore_index=True)
    return data[~data["participant_id"].isin(EXCLUDED_PARTICIPANTS)].copy()


def evaluate_external(participant, svm, scaler, threshold, data):
    X = scaler.transform(data[FEATURE_COLS].to_numpy())
    y = (data["participant_id"] == participant).astype(int).to_numpy()
    return (*compute_rates(y, svm.decision_function(X), threshold), int(y.sum()), int((y == 0).sum()))


def evaluate_all_participants():
    conditions = {
        "cross_session": load_condition(ST_FATIGUE_FEATURES),
        "dt_control": load_condition(DT_CONTROL_FEATURES),
        "dt_fatigue": load_condition(DT_FATIGUE_FEATURES),
    }
    rows = []
    for model_path in sorted(MODELS_DIR.glob("*_svm.pkl")):
        participant = model_path.stem.replace("_svm", "")
        svm = joblib.load(model_path)
        scaler = joblib.load(MODELS_DIR / f"{participant}_scaler.pkl")
        with open(MODELS_DIR / f"{participant}_threshold.json") as handle:
            threshold_info = json.load(handle)
        threshold = threshold_info["threshold"]

        holdout = np.load(MODELS_DIR / f"{participant}_holdout.npz")
        baseline_frr, baseline_far = compute_rates(
            holdout["y"], svm.decision_function(holdout["X"]), threshold
        )
        external = {
            name: evaluate_external(participant, svm, scaler, threshold, data)
            for name, data in conditions.items()
        }
        row = {
            "participant_id": participant,
            "threshold": threshold,
            "validation_eer": threshold_info["validation_eer"],
            "frr_baseline": baseline_frr,
            "far_baseline": baseline_far,
            "n_baseline_legitimate": int((holdout["y"] == 1).sum()),
            "n_baseline_impostor": int((holdout["y"] == 0).sum()),
        }
        for name, (frr, far, n_legitimate, n_impostor) in external.items():
            row[f"frr_{name}"] = frr
            row[f"far_{name}"] = far
            row[f"n_{name}_legitimate"] = n_legitimate
            row[f"n_{name}_impostor"] = n_impostor
        row["frr_delta"] = row["frr_cross_session"] - baseline_frr
        row["far_delta"] = row["far_cross_session"] - baseline_far
        row["frr_dt_delta"] = row["frr_dt_fatigue"] - row["frr_dt_control"]
        row["far_dt_delta"] = row["far_dt_fatigue"] - row["far_dt_control"]
        rows.append(row)
        print(
            f"{participant}: baseline {baseline_frr:.1%}/{baseline_far:.1%}, "
            f"ST-fatigue {row['frr_cross_session']:.1%}/{row['far_cross_session']:.1%}, "
            f"DT-control {row['frr_dt_control']:.1%}/{row['far_dt_control']:.1%}, "
            f"DT-fatigue {row['frr_dt_fatigue']:.1%}/{row['far_dt_fatigue']:.1%} (FRR/FAR)"
        )

    results = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_DIR / "evaluation_results.csv", index=False)
    return results


def print_summary(results):
    mapping = [
        ("baseline", "Baseline ST-control"),
        ("cross_session", "ST-fatigue"),
        ("dt_control", "DT-control"),
        ("dt_fatigue", "DT-fatigue"),
    ]
    rows = []
    print("\nMacro-average across participants (each participant has equal weight)")
    for key, label in mapping:
        frr = results[f"frr_{key}"]
        far = results[f"far_{key}"]
        print(f"{label:22s} FRR={frr.mean():.2%} (SD {frr.std(ddof=1):.2%})  FAR={far.mean():.2%} (SD {far.std(ddof=1):.2%})")
        rows.extend([
            {"metric": f"mean_{key}_frr", "value": frr.mean()},
            {"metric": f"std_{key}_frr", "value": frr.std(ddof=1)},
            {"metric": f"mean_{key}_far", "value": far.mean()},
            {"metric": f"std_{key}_far", "value": far.std(ddof=1)},
        ])
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "evaluation_summary.csv", index=False)


def main():
    results = evaluate_all_participants()
    print_summary(results)
    print(f"Results saved under {RESULTS_DIR}")


if __name__ == "__main__":
    main()
