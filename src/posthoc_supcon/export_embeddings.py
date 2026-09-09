"""Export every window embedding once from the frozen SupCon encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.unified_gait.config import CONDITIONS, MODEL_DIR, RESULT_DIR
from src.unified_gait.data import load_dataset
from src.unified_gait.model import UnifiedEncoder
from src.unified_gait.run_experiment import encode


def output_directory(fold: int) -> Path:
    return RESULT_DIR / "posthoc_supcon" / f"fold_{fold}"


def export(fold: int = 1, device: str = "cpu", force: bool = False) -> Path:
    output = output_directory(fold)
    cache_path = output / "embeddings.npz"
    manifest_path = output / "embedding_manifest.json"
    if cache_path.exists() and manifest_path.exists() and not force:
        print(f"Reusing frozen embeddings: {cache_path}")
        return cache_path

    checkpoint_path = MODEL_DIR / "supcon" / f"fold_{fold}" / "encoder.pt"
    protocol_path = RESULT_DIR / "supcon" / f"fold_{fold}" / "protocol.json"
    if not checkpoint_path.exists() or not protocol_path.exists():
        raise FileNotFoundError("Run the SupCon fold before exporting embeddings")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = UnifiedEncoder(checkpoint["config"]["embedding_size"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    dataset = load_dataset()
    arrays, counts = {}, {}
    for participant in sorted(dataset):
        for condition in CONDITIONS:
            key = f"{participant}__{condition}"
            arrays[key] = encode(model, dataset[participant][condition].windows, device)
            counts[key] = len(arrays[key])

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **arrays)
    protocol = json.loads(protocol_path.read_text())
    manifest_path.write_text(json.dumps({
        "checkpoint": str(checkpoint_path),
        "checkpoint_config": checkpoint["config"],
        "development": protocol["development"],
        "evaluation": protocol["evaluation"],
        "conditions": CONDITIONS,
        "counts": counts,
    }, indent=2))
    print(f"Exported frozen embeddings once to {cache_path}")
    return cache_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    export(args.fold, args.device, args.force)


if __name__ == "__main__":
    main()
