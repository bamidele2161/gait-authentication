"""Run participant-disjoint domain-adversarial SVM verification."""

import argparse
from dataclasses import asdict, replace
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC

from src.domain_adversarial_svm.config import (
    CONDITIONS, FEATURES, MODEL_DIR, RESULT_DIR, Config,
)
from src.domain_adversarial_svm.projector import DomainAdversarialProjector
from src.fisher_metric.data import (
    balanced_learning_frame, evaluation_enrollment_split, load_features,
    participant_folds, prepare_development,
)


def causal_fusion(values, size):
    values = np.asarray(values, dtype=np.float64)
    cumulative = np.r_[0.0, np.cumsum(values)]
    positions = np.arange(len(values))
    starts = np.maximum(0, positions - size + 1)
    return (cumulative[positions + 1] - cumulative[starts]) / (positions - starts + 1)


def similarity_threshold_at_eer(genuine, impostor):
    candidates = np.unique(np.concatenate((genuine, impostor)))
    frr = np.asarray([(genuine < value).mean() for value in candidates])
    far = np.asarray([(impostor >= value).mean() for value in candidates])
    index = int(np.argmin(np.abs(frr - far)))
    return float(candidates[index]), float(frr[index]), float(far[index])


def sample_negative_cohort(learning, participants, maximum, seed):
    frame = balanced_learning_frame(
        {person: learning[person] for person in participants}, maximum, seed
    )
    return frame


def fit_verifier(projector, positive_frame, negative_frame, config):
    positive = projector.transform(positive_frame[list(FEATURES)])
    negative = projector.transform(negative_frame[list(FEATURES)])
    features = np.vstack((positive, negative))
    labels = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
    return SVC(
        C=config.svm_c, gamma="scale", kernel="rbf", class_weight="balanced"
    ).fit(features, labels)


def score(verifier, projector, frame, fusion_window):
    values = verifier.decision_function(projector.transform(frame[list(FEATURES)]))
    return causal_fusion(values, fusion_window)


def inner_unseen_scores(learning, validation, config, strength):
    genuine_parts, impostor_parts = [], []
    for inner_index, (cohort_people, claim_people) in enumerate(
        participant_folds(learning, config.inner_folds), start=1
    ):
        population = balanced_learning_frame(
            {person: learning[person] for person in cohort_people}, 120,
            config.seed + inner_index,
        )
        projector = DomainAdversarialProjector(
            config.pca_components, strength, config.seed + inner_index
        ).fit(population[list(FEATURES)], population.session_type)
        negative = sample_negative_cohort(
            learning, cohort_people, config.maximum_negative_windows_per_group,
            config.seed + 100 + inner_index,
        )
        verifiers = {
            claim: fit_verifier(
                projector, learning[claim]["st_control"], negative, config
            )
            for claim in claim_people
        }
        for claim in claim_people:
            for probe in claim_people:
                for condition in CONDITIONS:
                    values = score(
                        verifiers[claim], projector, validation[probe][condition],
                        config.fusion_window,
                    )
                    (genuine_parts if claim == probe else impostor_parts).extend(values)
    return np.asarray(genuine_parts), np.asarray(impostor_parts)


def tune_strength(learning, validation, config):
    rows, best = [], None
    for strength in config.removal_strengths:
        genuine, impostor = inner_unseen_scores(learning, validation, config, strength)
        threshold, frr, far = similarity_threshold_at_eer(genuine, impostor)
        objective = (frr + far) / 2
        rows.append({
            "removal_strength": strength, "unseen_development_eer": objective,
            "shared_threshold": threshold, "frr": frr, "far": far,
        })
        candidate = (objective, -strength, strength, threshold)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    return best[2], best[3], pd.DataFrame(rows)


def evaluate(projector, negative, dataset, participants, threshold, config, fold_number):
    templates, calibration, probes = {}, {}, {condition: {} for condition in CONDITIONS}
    for person in participants:
        templates[person], calibration[person], probes["st_control"][person] = (
            evaluation_enrollment_split(dataset[person]["st_control"])
        )
        for condition in CONDITIONS[1:]:
            probes[condition][person] = dataset[person][condition]
    verifiers = {
        person: fit_verifier(projector, templates[person], negative, config)
        for person in participants
    }
    metric_rows, score_rows = [], []
    for claim in participants:
        for condition in CONDITIONS:
            genuine, impostor = [], []
            for probe in participants:
                values = score(
                    verifiers[claim], projector, probes[condition][probe],
                    config.fusion_window,
                )
                is_genuine = claim == probe
                (genuine if is_genuine else impostor).extend(values)
                score_rows.extend({
                    "fold": fold_number, "claim": claim, "probe": probe,
                    "condition": condition, "is_genuine": is_genuine,
                    "score_index": index, "similarity": float(value),
                } for index, value in enumerate(values))
            genuine, impostor = np.asarray(genuine), np.asarray(impostor)
            metric_rows.append({
                "fold": fold_number, "participant": claim, "condition": condition,
                "threshold": threshold,
                "frr": float((genuine < threshold).mean()),
                "far": float((impostor >= threshold).mean()),
            })
    return pd.DataFrame(metric_rows), pd.DataFrame(score_rows), verifiers


def run_fold(fold_number=1, config=Config()):
    dataset = load_features()
    development, evaluation = participant_folds(dataset, 4)[fold_number - 1]
    learning, validation = prepare_development(dataset, development, 0.8)
    print(f"Fold {fold_number}/4: development={','.join(development)} evaluation={','.join(evaluation)}")
    strength, threshold, tuning = tune_strength(learning, validation, config)
    selected = tuning.loc[tuning.unseen_development_eer.idxmin()]
    print(
        f"  selected domain-removal strength={strength:.2f}, one shared "
        f"threshold={threshold:.6f}, unseen-development "
        f"FRR/FAR={selected.frr:.2%}/{selected.far:.2%}"
    )
    population = balanced_learning_frame(
        learning, 120, config.seed + fold_number
    )
    projector = DomainAdversarialProjector(
        config.pca_components, strength, config.seed + fold_number
    ).fit(population[list(FEATURES)], population.session_type)
    negative = sample_negative_cohort(
        learning, development, config.maximum_negative_windows_per_group,
        config.seed + 500 + fold_number,
    )
    metrics, scores, verifiers = evaluate(
        projector, negative, dataset, evaluation, threshold, config, fold_number
    )
    output = RESULT_DIR / f"fold_{fold_number}"
    models = MODEL_DIR / f"fold_{fold_number}"
    output.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "participant_metrics.csv", index=False)
    scores.to_csv(output / "scores.csv", index=False)
    tuning.to_csv(output / "domain_removal_tuning.csv", index=False)
    summary = metrics.groupby("condition")[["frr", "far"]].agg(["mean", "std"])
    summary.to_csv(output / "summary.csv")
    joblib.dump({"projector": projector, "verifiers": verifiers}, models / "models.joblib")
    (output / "protocol.json").write_text(json.dumps({
        "development": development, "evaluation": evaluation,
        "domain_removal_strength": strength, "shared_threshold": threshold,
        "config": asdict(config),
    }, indent=2))
    print(summary.to_string())
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--fusion-window", type=int, default=30)
    args = parser.parse_args()
    run_fold(args.fold, replace(Config(), fusion_window=args.fusion_window))


if __name__ == "__main__":
    main()
