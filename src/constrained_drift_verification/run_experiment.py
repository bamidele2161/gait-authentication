"""Tune and evaluate the complete constrained-drift verification architecture."""

from __future__ import annotations

import argparse
from itertools import product
import json

import numpy as np
import pandas as pd

from src.constrained_drift_verification.scoring import (
    fit_drift_model, normalized_constrained_distance,
)
from src.posthoc_supcon.export_embeddings import output_directory
from src.posthoc_supcon.run_experiment import (
    condition_metrics, load_cache, prepared_development, split_evaluation,
)
from src.posthoc_supcon.scoring import centroid
from src.unified_gait.config import CONDITIONS, RESULT_DIR
from src.unified_gait.run_experiment import causal_fusion, eer_threshold


def candidates():
    for rank, quantile, scale, top_k in product(
        (1, 3, 5), (.50, .90), (.50, 1.0), (None, 5, 8)
    ):
        yield {
            "rank": rank, "quantile": quantile,
            "scale": scale, "top_k": top_k,
        }


def context(learning, population, parameters):
    basis, limits = fit_drift_model(
        learning, population, CONDITIONS, parameters["rank"],
        parameters["quantile"], parameters["scale"],
    )
    cohort_templates = np.vstack([
        centroid(learning[(person, "st_control")]) for person in population
    ])
    return basis, limits, cohort_templates


def loso_scores(learning, probes, development, parameters, fusion_window):
    rows = []
    for held_out in development:
        population = tuple(person for person in development if person != held_out)
        basis, limits, cohort_templates = context(learning, population, parameters)
        claimed_template = centroid(learning[(held_out, "st_control")])
        for probe_person in development:
            for condition in CONDITIONS:
                values = normalized_constrained_distance(
                    probes[(probe_person, condition)], claimed_template,
                    cohort_templates, basis, limits, parameters["top_k"],
                )
                values = causal_fusion(values, fusion_window)
                rows.extend({
                    "held_out": held_out, "probe": probe_person,
                    "condition": condition, "genuine": held_out == probe_person,
                    "distance": float(value),
                } for value in values)
    return pd.DataFrame(rows)


def tune(learning, probes, development, fusion_window):
    trials, selected = [], None
    for parameters in candidates():
        frame = loso_scores(
            learning, probes, development, parameters, fusion_window
        )
        genuine = frame.loc[frame.genuine, "distance"].to_numpy()
        impostor = frame.loc[~frame.genuine, "distance"].to_numpy()
        threshold, frr, far = eer_threshold(genuine, impostor)
        trial = {
            "parameters": parameters, "threshold": threshold,
            "frr": frr, "far": far, "objective": (frr + far) / 2,
        }
        trials.append(trial)
        if selected is None or trial["objective"] < selected["objective"]:
            selected = trial
    table = pd.DataFrame([{
        "parameters": json.dumps(item["parameters"], sort_keys=True),
        "threshold": item["threshold"], "frr": item["frr"],
        "far": item["far"], "objective": item["objective"],
    } for item in trials])
    return selected, table


def evaluation_scores(embeddings, development, evaluation, parameters, fusion_window):
    learning, _ = prepared_development(embeddings, development)
    basis, limits, cohort_templates = context(learning, development, parameters)
    enrollment, probes = {}, {}
    for person in evaluation:
        for condition in CONDITIONS:
            enrolled, tested = split_evaluation(
                embeddings[(person, condition)], condition
            )
            probes[(person, condition)] = tested
            if condition == "st_control":
                enrollment[person] = centroid(enrolled)
    rows = []
    for claim in evaluation:
        for probe_person in evaluation:
            for condition in CONDITIONS:
                values = normalized_constrained_distance(
                    probes[(probe_person, condition)], enrollment[claim],
                    cohort_templates, basis, limits, parameters["top_k"],
                )
                values = causal_fusion(values, fusion_window)
                rows.extend({
                    "claim": claim, "probe": probe_person,
                    "condition": condition, "genuine": claim == probe_person,
                    "distance": float(value),
                } for value in values)
    return pd.DataFrame(rows), basis, limits


def run(fold=1, fusion_window=30, device="cpu"):
    embeddings, manifest = load_cache(fold, device)
    development = tuple(manifest["development"])
    evaluation = tuple(manifest["evaluation"])
    learning, probes = prepared_development(embeddings, development)
    print("Tuning constrained drift verifier with development LOSO only")
    selected, trials = tune(learning, probes, development, fusion_window)
    print(
        f"  selected={selected['parameters']} threshold={selected['threshold']:.6f} "
        f"development LOSO FRR/FAR={selected['frr']:.2%}/{selected['far']:.2%}"
    )
    scores, basis, limits = evaluation_scores(
        embeddings, development, evaluation, selected["parameters"], fusion_window
    )
    metrics = condition_metrics(scores, selected["threshold"])
    metrics.insert(0, "method", "constrained_drift_cohort")

    output = RESULT_DIR / "constrained_drift_verification" / f"fold_{fold}"
    output.mkdir(parents=True, exist_ok=True)
    trials.to_csv(output / "development_loso_tuning.csv", index=False)
    scores.to_csv(output / "evaluation_scores.csv", index=False)
    metrics.to_csv(output / "condition_metrics.csv", index=False)
    np.save(output / "drift_basis.npy", basis)
    np.save(output / "drift_limits.npy", limits)
    (output / "selected.json").write_text(json.dumps(selected, indent=2))
    print(metrics.to_string(index=False))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--fusion-window", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(args.fold, args.fusion_window, args.device)


if __name__ == "__main__":
    main()
