"""Tune and evaluate isolated post-hoc methods on frozen SupCon embeddings."""

from __future__ import annotations

import argparse
from itertools import product
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from src.posthoc_supcon.export_embeddings import export, output_directory
from src.posthoc_supcon.scoring import (
    augmented_templates, centroid, cohort_normalized_distance, cosine_distance,
    drift_aware_distance, drift_basis, population_shifts,
)
from src.unified_gait.config import CONDITIONS
from src.unified_gait.run_experiment import causal_fusion, eer_threshold


METHODS = ("raw", "cohort", "augmentation", "drift")


def load_cache(fold, device):
    cache_path = export(fold, device)
    manifest = json.loads((cache_path.parent / "embedding_manifest.json").read_text())
    archive = np.load(cache_path)
    embeddings = {
        tuple(key.split("__", 1)): archive[key].astype(np.float64)
        for key in archive.files
    }
    return embeddings, manifest


def split_development(values, fraction=.8):
    boundary = min(max(int(len(values) * fraction), 1), len(values) - 2)
    return values[:boundary], values[boundary + 1:]


def split_evaluation(values, condition):
    if condition != "st_control":
        return None, values
    boundary = len(values) // 2
    return values[:boundary], values[boundary + 1:]


def prepared_development(embeddings, participants):
    learning, probes = {}, {}
    for person in participants:
        for condition in CONDITIONS:
            learning[(person, condition)], probes[(person, condition)] = (
                split_development(embeddings[(person, condition)])
            )
    return learning, probes


def method_parameters(method):
    if method == "raw":
        return ({},)
    if method == "cohort":
        return tuple({"top_k": value} for value in (None, 3, 5, 8))
    if method == "augmentation":
        return tuple({"shift_scale": value} for value in (.25, .5, .75, 1.0, 1.25))
    if method == "drift":
        return tuple(
            {"rank": rank, "drift_weight": weight}
            for rank, weight in product((1, 2, 3, 5, 8), (0., .1, .25, .5, .75))
        )
    raise ValueError(method)


def build_context(method, learning, population, parameters):
    templates = {
        person: centroid(learning[(person, "st_control")]) for person in population
    }
    if method == "cohort":
        return {"cohort_templates": np.vstack(list(templates.values()))}
    if method == "augmentation":
        return {"shifts": population_shifts(
            learning, population, CONDITIONS, parameters["shift_scale"]
        )}
    if method == "drift":
        return {"basis": drift_basis(
            learning, population, CONDITIONS, parameters["rank"]
        )}
    return {}


def score(method, probes, claimed_template, context, parameters):
    if method == "raw":
        return cosine_distance(probes, claimed_template)
    if method == "cohort":
        return cohort_normalized_distance(
            probes, claimed_template, context["cohort_templates"], parameters["top_k"]
        )
    if method == "augmentation":
        templates = augmented_templates(claimed_template, context["shifts"])
        return cosine_distance(probes, templates)
    if method == "drift":
        return drift_aware_distance(
            probes, claimed_template, context["basis"], parameters["drift_weight"]
        )
    raise ValueError(method)


def loso_scores(method, learning, probes, development, parameters, fusion_window):
    rows = []
    for held_out in development:
        population = tuple(person for person in development if person != held_out)
        context = build_context(method, learning, population, parameters)
        claimed_template = centroid(learning[(held_out, "st_control")])
        for probe_person in development:
            for condition in CONDITIONS:
                values = score(
                    method, probes[(probe_person, condition)], claimed_template,
                    context, parameters,
                )
                values = causal_fusion(values, fusion_window)
                rows.extend({
                    "held_out": held_out, "probe": probe_person,
                    "condition": condition, "genuine": held_out == probe_person,
                    "distance": float(value),
                } for value in values)
    return pd.DataFrame(rows)


def pooled_eer(frame):
    genuine = frame.loc[frame.genuine, "distance"].to_numpy()
    impostor = frame.loc[~frame.genuine, "distance"].to_numpy()
    return eer_threshold(genuine, impostor)


def tune(method, learning, probes, development, fusion_window):
    trials = []
    best = None
    for parameters in method_parameters(method):
        frame = loso_scores(
            method, learning, probes, development, parameters, fusion_window
        )
        threshold, frr, far = pooled_eer(frame)
        trial = {"parameters": parameters, "threshold": threshold,
                 "frr": frr, "far": far, "objective": (frr + far) / 2}
        trials.append(trial)
        if best is None or trial["objective"] < best["objective"]:
            best = trial
    return best, pd.DataFrame([{
        "parameters": json.dumps(item["parameters"], sort_keys=True),
        "threshold": item["threshold"], "frr": item["frr"],
        "far": item["far"], "objective": item["objective"],
    } for item in trials])


def evaluation_scores(method, embeddings, development, evaluation, parameters,
                      fusion_window):
    development_learning, _ = prepared_development(embeddings, development)
    context = build_context(method, development_learning, development, parameters)
    enrollment, probes = {}, {}
    for person in evaluation:
        for condition in CONDITIONS:
            enrolled, tested = split_evaluation(embeddings[(person, condition)], condition)
            probes[(person, condition)] = tested
            if condition == "st_control":
                enrollment[person] = centroid(enrolled)
    rows = []
    for claim in evaluation:
        for probe_person in evaluation:
            for condition in CONDITIONS:
                values = score(
                    method, probes[(probe_person, condition)], enrollment[claim],
                    context, parameters,
                )
                values = causal_fusion(values, fusion_window)
                rows.extend({
                    "claim": claim, "probe": probe_person, "condition": condition,
                    "genuine": claim == probe_person, "distance": float(value),
                } for value in values)
    return pd.DataFrame(rows)


def condition_metrics(frame, threshold):
    rows = []
    for condition, group in frame.groupby("condition", sort=False):
        genuine = group.loc[group.genuine, "distance"].to_numpy()
        impostor = group.loc[~group.genuine, "distance"].to_numpy()
        labels = np.r_[np.ones(len(genuine)), np.zeros(len(impostor))]
        similarities = -np.r_[genuine, impostor]
        fpr, tpr, thresholds = roc_curve(labels, similarities)
        fnr = 1 - tpr
        index = int(np.argmin(np.abs(fnr - fpr)))
        rows.append({
            "condition": condition,
            "frr": float((genuine > threshold).mean()),
            "far": float((impostor <= threshold).mean()),
            "eer": float((fnr[index] + fpr[index]) / 2),
            "auc": float(roc_auc_score(labels, similarities)),
            "genuine_mean_distance": float(genuine.mean()),
            "impostor_mean_distance": float(impostor.mean()),
            "threshold": threshold,
        })
    return pd.DataFrame(rows)


def run(fold=1, method="all", fusion_window=30, device="cpu"):
    embeddings, manifest = load_cache(fold, device)
    development = tuple(manifest["development"])
    evaluation = tuple(manifest["evaluation"])
    learning, probes = prepared_development(embeddings, development)
    selected_methods = METHODS if method == "all" else (method,)
    output_root = output_directory(fold)
    summaries = []
    for name in selected_methods:
        print(f"Tuning isolated method={name} with development LOSO only")
        selected, trials = tune(
            name, learning, probes, development, fusion_window
        )
        print(
            f"  selected={selected['parameters']} threshold={selected['threshold']:.6f} "
            f"development LOSO FRR/FAR={selected['frr']:.2%}/{selected['far']:.2%}"
        )
        evaluation_frame = evaluation_scores(
            name, embeddings, development, evaluation, selected["parameters"],
            fusion_window,
        )
        metrics = condition_metrics(evaluation_frame, selected["threshold"])
        metrics.insert(0, "method", name)
        method_output = output_root / name
        method_output.mkdir(parents=True, exist_ok=True)
        trials.to_csv(method_output / "development_loso_tuning.csv", index=False)
        evaluation_frame.to_csv(method_output / "evaluation_scores.csv", index=False)
        metrics.to_csv(method_output / "condition_metrics.csv", index=False)
        (method_output / "selected.json").write_text(json.dumps(selected, indent=2))
        summaries.append(metrics)
        print(metrics.to_string(index=False))
    # Preserve summaries from separately executed methods so isolation does not
    # require rerunning completed experiments merely to build the comparison.
    available = []
    for name in METHODS:
        path = output_root / name / "condition_metrics.csv"
        if path.exists():
            available.append(pd.read_csv(path))
    combined = pd.concat(available or summaries, ignore_index=True)
    combined.to_csv(output_root / "all_condition_metrics.csv", index=False)
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--method", choices=("all",) + METHODS, default="all")
    parser.add_argument("--fusion-window", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(args.fold, args.method, args.fusion_window, args.device)


if __name__ == "__main__":
    main()
