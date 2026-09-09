"""Run the unified non-deep Fisher gait-verification experiment."""

import argparse
from dataclasses import asdict
import json

import joblib
import numpy as np
import pandas as pd

from src.fisher_metric.config import CONDITIONS, FEATURES, MODEL_DIR, RESULT_DIR, Config
from src.fisher_metric.data import (
    balanced_learning_frame, evaluation_enrollment_split, load_features,
    participant_folds, prepare_development,
)
from src.fisher_metric.metric import FisherMetric


def template(metric, frame, feature_names=FEATURES):
    value = metric.transform(frame[list(feature_names)]).mean(axis=0)
    return value / max(np.linalg.norm(value), 1e-9)


def distances(metric, frame, claimed_template, feature_names=FEATURES):
    return 1.0 - metric.transform(frame[list(feature_names)]) @ claimed_template


def causal_fusion(values, size):
    values = np.asarray(values, dtype=np.float64)
    cumulative = np.r_[0.0, np.cumsum(values)]
    positions = np.arange(len(values))
    starts = np.maximum(0, positions - size + 1)
    return (cumulative[positions + 1] - cumulative[starts]) / (positions - starts + 1)


def threshold_at_eer(genuine, impostor):
    candidates = np.unique(np.concatenate((genuine, impostor)))
    frr = np.asarray([(genuine > threshold).mean() for threshold in candidates])
    far = np.asarray([(impostor <= threshold).mean() for threshold in candidates])
    index = int(np.argmin(np.abs(frr - far)))
    return float(candidates[index]), float(frr[index]), float(far[index])


def calibration_scores(
    metric, learning, validation, participants, fusion_window, feature_names=FEATURES,
):
    templates = {
        person: template(metric, learning[person]["st_control"], feature_names)
        for person in participants
    }
    genuine, impostor = [], []
    for claim in participants:
        for probe in participants:
            for condition in CONDITIONS:
                scores = causal_fusion(
                    distances(
                        metric, validation[probe][condition], templates[claim], feature_names
                    ),
                    fusion_window,
                )
                (genuine if claim == probe else impostor).extend(scores)
    return np.asarray(genuine), np.asarray(impostor)


def nested_shared_threshold(learning, validation, config):
    """Pool scores from identities unseen by each inner Fisher model."""
    genuine_parts, impostor_parts, rows = [], [], []
    for inner_index, (train_people, calibration_people) in enumerate(
        participant_folds(learning, config.inner_folds), start=1
    ):
        frame = balanced_learning_frame(
            {person: learning[person] for person in train_people},
            config.maximum_windows_per_group, config.seed + inner_index,
        )
        metric = FisherMetric(
            config.pca_components, config.fisher_dimensions, config.seed + inner_index
        ).fit(
            frame[list(FEATURES)], frame.metric_identity
        )
        genuine, impostor = calibration_scores(
            metric,
            {person: learning[person] for person in calibration_people},
            {person: validation[person] for person in calibration_people},
            calibration_people, config.fusion_window,
        )
        genuine_parts.append(genuine)
        impostor_parts.append(impostor)
        rows.append({
            "inner_fold": inner_index, "training_participants": ",".join(train_people),
            "calibration_participants": ",".join(calibration_people),
            "genuine_scores": len(genuine), "impostor_scores": len(impostor),
        })
    threshold, frr, far = threshold_at_eer(
        np.concatenate(genuine_parts), np.concatenate(impostor_parts)
    )
    return threshold, frr, far, pd.DataFrame(rows)


def evaluate(metric, dataset, participants, config, fold_number, feature_names=FEATURES):
    enrollment, probes = {}, {condition: {} for condition in CONDITIONS}
    calibration = {}
    for person in participants:
        enrollment[person], calibration[person], probes["st_control"][person] = (
            evaluation_enrollment_split(dataset[person]["st_control"])
        )
        for condition in CONDITIONS[1:]:
            probes[condition][person] = dataset[person][condition]
    templates = {
        person: template(metric, enrollment[person], feature_names)
        for person in participants
    }
    calibration_genuine, calibration_impostor = [], []
    for claim in participants:
        for probe in participants:
            scores = causal_fusion(
                distances(metric, calibration[probe], templates[claim], feature_names),
                config.fusion_window,
            )
            (calibration_genuine if claim == probe else calibration_impostor).extend(scores)
    threshold, calibration_frr, calibration_far = threshold_at_eer(
        np.asarray(calibration_genuine), np.asarray(calibration_impostor)
    )
    metric_rows, score_rows = [], []
    for claim in participants:
        for condition in CONDITIONS:
            genuine, impostor = [], []
            for probe in participants:
                scores = causal_fusion(
                    distances(
                        metric, probes[condition][probe], templates[claim], feature_names
                    ),
                    config.fusion_window,
                )
                is_genuine = claim == probe
                (genuine if is_genuine else impostor).extend(scores)
                score_rows.extend({
                    "fold": fold_number, "claim": claim, "probe": probe,
                    "condition": condition, "is_genuine": is_genuine,
                    "score_index": index, "distance": float(score),
                } for index, score in enumerate(scores))
            genuine, impostor = np.asarray(genuine), np.asarray(impostor)
            metric_rows.append({
                "fold": fold_number, "participant": claim, "condition": condition,
                "threshold": threshold,
                "frr": float((genuine > threshold).mean()),
                "far": float((impostor <= threshold).mean()),
            })
    return (
        pd.DataFrame(metric_rows), pd.DataFrame(score_rows), threshold,
        calibration_frr, calibration_far,
    )


def rank_condition_stable_features(frame):
    """Rank features that vary by identity more than by recording condition."""
    feature_names = list(FEATURES)
    group_means = frame.groupby(
        ["metric_identity", "session_type"], sort=True
    )[feature_names].mean()
    participant_means = group_means.groupby(level=0).mean()
    between_identity = participant_means.var(axis=0, ddof=1)
    aligned_participant_means = group_means.index.get_level_values(0).map(
        participant_means.to_dict("index")
    )
    aligned = np.asarray([
        [row[name] for name in feature_names] for row in aligned_participant_means
    ])
    condition_shift = np.square(group_means.to_numpy() - aligned).mean(axis=0)
    within_condition = frame.groupby(
        ["metric_identity", "session_type"], sort=True
    )[feature_names].var(ddof=0).mean(axis=0).to_numpy()
    denominator = condition_shift + 0.1 * within_condition + 1e-9
    scores = between_identity.to_numpy() / denominator
    order = np.argsort(-scores)
    return tuple(feature_names[index] for index in order), scores[order]


def select_features_and_fit(learning, validation, config, fold_number):
    """Tune only the feature count on development data, then freeze the metric."""
    frame = balanced_learning_frame(
        learning, config.maximum_windows_per_group, config.seed + fold_number
    )
    ranking, ranking_scores = rank_condition_stable_features(frame)
    rows, best = [], None
    for count in config.candidate_feature_counts:
        selected = ranking[:min(count, len(ranking))]
        metric = FisherMetric(
            min(config.pca_components, len(selected)), config.fisher_dimensions,
            config.seed + fold_number,
        ).fit(frame[list(selected)], frame.metric_identity)
        genuine, impostor = calibration_scores(
            metric, learning, validation, tuple(sorted(learning)),
            config.fusion_window, selected,
        )
        threshold, frr, far = threshold_at_eer(genuine, impostor)
        objective = (frr + far) / 2
        rows.append({
            "feature_count": len(selected), "development_eer": objective,
            "development_threshold": threshold,
            "selected_features": ",".join(selected),
        })
        candidate = (objective, len(selected), metric, selected)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    _, _, metric, selected = best
    ranking_frame = pd.DataFrame({
        "feature": ranking, "condition_stability_score": ranking_scores,
    })
    return metric, selected, pd.DataFrame(rows), ranking_frame


def run_fold(fold_number=1, config=Config()):
    dataset = load_features()
    development, evaluation = participant_folds(dataset, config.outer_folds)[fold_number - 1]
    learning, validation = prepare_development(
        dataset, development, config.learning_fraction
    )
    print(
        f"Fold {fold_number}/{config.outer_folds}: development={','.join(development)} "
        f"evaluation={','.join(evaluation)}"
    )
    metric, selected_features, tuning, ranking = select_features_and_fit(
        learning, validation, config, fold_number
    )
    winning = tuning.loc[tuning.development_eer.idxmin()]
    print(
        f"  selected {len(selected_features)} condition-stable features, "
        f"development EER={winning.development_eer:.2%}"
    )
    metrics, scores, threshold, calibration_frr, calibration_far = evaluate(
        metric, dataset, evaluation, config, fold_number, selected_features
    )
    print(
        f"  one normal-enrollment shared threshold={threshold:.6f}, enrollment "
        f"FRR/FAR={calibration_frr:.2%}/{calibration_far:.2%}"
    )
    output = RESULT_DIR / f"fold_{fold_number}"
    models = MODEL_DIR / f"fold_{fold_number}"
    output.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "participant_metrics.csv", index=False)
    scores.to_csv(output / "scores.csv", index=False)
    tuning.to_csv(output / "feature_tuning.csv", index=False)
    ranking.to_csv(output / "feature_ranking.csv", index=False)
    summary = metrics.groupby("condition")[["frr", "far"]].agg(["mean", "std"])
    summary.to_csv(output / "summary.csv")
    joblib.dump(metric, models / "fisher_metric.joblib")
    (output / "protocol.json").write_text(json.dumps({
        "development": development, "evaluation": evaluation,
        "threshold_source": "reserved normal enrollment only",
        "selected_features": selected_features,
        "shared_threshold": threshold, "calibration_frr": calibration_frr,
        "calibration_far": calibration_far, "config": asdict(config),
    }, indent=2))
    print(summary.to_string())
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--fusion-window", type=int, default=30)
    parser.add_argument("--pca-components", type=int, default=30)
    args = parser.parse_args()
    run_fold(args.fold, Config(
        fusion_window=args.fusion_window, pca_components=args.pca_components
    ))


if __name__ == "__main__":
    main()
