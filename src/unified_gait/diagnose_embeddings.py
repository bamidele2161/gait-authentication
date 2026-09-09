"""Diagnose whether a frozen encoder represents identity or recording condition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.unified_gait.config import CONDITIONS, MODEL_DIR, RESULT_DIR
from src.unified_gait.data import load_dataset
from src.unified_gait.model import UnifiedEncoder
from src.unified_gait.run_experiment import cosine_distances, encode, template


def evenly_spaced(values, maximum):
    if len(values) <= maximum:
        return values
    return values[np.linspace(0, len(values) - 1, maximum, dtype=int)]


def load_frozen_encoder(fold, device):
    checkpoint_path = MODEL_DIR / f"fold_{fold}" / "encoder.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = UnifiedEncoder(config["embedding_size"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def collect_embeddings(model, dataset, participants, maximum, device):
    rows, vectors = [], []
    for participant in participants:
        for condition in CONDITIONS:
            windows = evenly_spaced(dataset[participant][condition].windows, maximum)
            encoded = encode(model, windows, device)
            vectors.append(encoded)
            rows.extend({
                "participant": participant, "condition": condition,
                "window_index": index,
            } for index in range(len(encoded)))
    return np.vstack(vectors), pd.DataFrame(rows)


def distance_summary(model, dataset, participants, device):
    templates = {}
    encoded = {}
    for participant in participants:
        control = dataset[participant]["st_control"].windows
        templates[participant] = template(model, control[:len(control) // 2], device)
        for condition in CONDITIONS:
            windows = dataset[participant][condition].windows
            if condition == "st_control":
                windows = windows[len(windows) // 2 + 1:]
            encoded[(participant, condition)] = encode(model, windows, device)
    rows = []
    for claim in participants:
        for probe in participants:
            for condition in CONDITIONS:
                values = cosine_distances(encoded[(probe, condition)], templates[claim])
                rows.extend({
                    "claim": claim, "probe": probe, "condition": condition,
                    "is_genuine": claim == probe, "distance": float(value),
                } for value in values)
    scores = pd.DataFrame(rows)
    summary = scores.groupby(["condition", "is_genuine"]).distance.agg(
        mean="mean", median="median", std="std",
        q05=lambda values: values.quantile(.05),
        q95=lambda values: values.quantile(.95), count="count",
    ).reset_index()
    return scores, summary


def run(fold=1, maximum=30, device="cpu"):
    dataset = load_dataset()
    protocol_path = RESULT_DIR / f"fold_{fold}" / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    model = load_frozen_encoder(fold, device)
    output = RESULT_DIR / f"fold_{fold}" / "diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    for population_name in ("development", "evaluation"):
        participants = tuple(protocol[population_name])
        vectors, labels = collect_embeddings(
            model, dataset, participants, maximum, device
        )
        np.save(output / f"embeddings_{population_name}.npy", vectors)
        labels.to_csv(output / f"embedding_labels_{population_name}.csv", index=False)
        scores, summary = distance_summary(model, dataset, participants, device)
        scores.to_csv(output / f"distances_{population_name}.csv", index=False)
        summary.to_csv(output / f"distance_summary_{population_name}.csv", index=False)
    print(f"Embeddings and distance diagnostics saved to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--maximum-per-group", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(args.fold, args.maximum_per_group, args.device)


if __name__ == "__main__":
    main()
