"""Post-hoc test operating points for an already completed four-fold run.

These metrics deliberately select thresholds with evaluation labels. They are
diagnostic upper bounds and must not be presented as deployable frozen-threshold
performance.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.robust_gait.protocol import (
    equal_error_rate,
    select_threshold_from_distances,
)

# Four-fold score files of the retired CNN experiment, kept for this diagnostic.
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "condition_invariant"


def load_four_fold_scores(method_name, results_root=RESULTS_DIR):
    frames = []
    for fold in range(1, 5):
        path = results_root / f"{method_name}_fold_{fold}" / "evaluation_scores.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing fold score file: {path}")
        frame = pd.read_csv(path)
        frame["source_fold"] = fold
        frames.append(frame)
    scores = pd.concat(frames, ignore_index=True)
    required = {
        "claimed_participant_id", "probe_participant_id", "condition",
        "is_genuine", "distance",
    }
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"Evaluation scores are missing columns: {missing}")
    scores["is_genuine"] = scores["is_genuine"].astype(str).str.lower().eq("true")
    if not np.isfinite(scores["distance"].to_numpy(dtype=float)).all():
        raise ValueError("Evaluation distances contain NaN or infinity")
    return scores


def calculate_test_operating_points(scores, target_far=0.01):
    rows = []
    grouped = scores.groupby(["claimed_participant_id", "condition"], sort=True)
    for (participant, condition), group in grouped:
        genuine = group.loc[group["is_genuine"], "distance"].to_numpy(dtype=float)
        impostor = group.loc[~group["is_genuine"], "distance"].to_numpy(dtype=float)
        if genuine.size == 0 or impostor.size == 0:
            raise ValueError(f"Missing genuine/impostor scores for {participant} {condition}")

        eer, eer_threshold = equal_error_rate(genuine, impostor)
        far_threshold, frr_at_far, achieved_far = select_threshold_from_distances(
            genuine, impostor, target_far
        )
        rows.append({
            "participant_id": participant,
            "condition": condition,
            "test_eer": eer,
            "test_eer_threshold": eer_threshold,
            "target_far": target_far,
            "test_frr_at_target_far": frr_at_far,
            "test_achieved_far": achieved_far,
            "test_far_threshold": far_threshold,
            "genuine_count": genuine.size,
            "impostor_count": impostor.size,
        })
    return pd.DataFrame(rows)


def summarize(participant_metrics):
    rows = []
    for condition, group in participant_metrics.groupby("condition", sort=True):
        rows.append({
            "condition": condition,
            "participant_count": len(group),
            "test_eer_mean": group["test_eer"].mean(),
            "test_eer_sd": group["test_eer"].std(ddof=1),
            "test_frr_at_target_far_mean": group["test_frr_at_target_far"].mean(),
            "test_frr_at_target_far_sd": group["test_frr_at_target_far"].std(ddof=1),
            "test_achieved_far_mean": group["test_achieved_far"].mean(),
            "test_achieved_far_sd": group["test_achieved_far"].std(ddof=1),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate post-hoc EER and FRR at a target FAR on test scores"
    )
    parser.add_argument(
        "--method",
        default="session1_cnn_bilstm_hard",
        help="four-fold result directory prefix",
    )
    parser.add_argument("--target-far", type=float, default=0.01)
    args = parser.parse_args()
    if not 0 <= args.target_far <= 1:
        raise ValueError("target FAR must be between zero and one")

    scores = load_four_fold_scores(args.method)
    participant_metrics = calculate_test_operating_points(scores, args.target_far)
    summary = summarize(participant_metrics)

    output_dir = RESULTS_DIR / "test_operating_points" / args.method
    output_dir.mkdir(parents=True, exist_ok=True)
    participant_metrics.to_csv(output_dir / "participant_metrics.csv", index=False)
    summary.to_csv(output_dir / "macro_summary.csv", index=False)

    print("POST-HOC TEST DIAGNOSTIC — evaluation labels select these thresholds")
    print("Do not report these as frozen development-threshold performance.\n")
    for row in summary.itertuples(index=False):
        print(
            f"{row.condition:12s} "
            f"EER={row.test_eer_mean:.2%} (SD {row.test_eer_sd:.2%})  "
            f"FRR@FAR<={args.target_far:.1%}="
            f"{row.test_frr_at_target_far_mean:.2%} "
            f"(SD {row.test_frr_at_target_far_sd:.2%})  "
            f"achieved FAR={row.test_achieved_far_mean:.2%} "
            f"(SD {row.test_achieved_far_sd:.2%})"
        )
    print(f"\nSaved under {output_dir}")


if __name__ == "__main__":
    main()
