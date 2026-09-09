"""Train and evaluate the single unified gait-verification method."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
import random

import numpy as np
import pandas as pd
import torch

from src.unified_gait.config import CONDITIONS, MODEL_DIR, RESULT_DIR, Config
from src.unified_gait.data import (
    chronological_split, enrollment_split, load_dataset, outer_folds,
)
from src.unified_gait.model import (
    ConditionClassifier, UnifiedEncoder, batch_hard_triplet_loss,
    supervised_contrastive_loss,
)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def derived_channels(windows):
    gyro, acceleration = windows[..., :3], windows[..., 3:]
    return np.concatenate((
        gyro, acceleration,
        np.linalg.norm(gyro, axis=-1, keepdims=True),
        np.linalg.norm(acceleration, axis=-1, keepdims=True),
    ), axis=-1)


def fit_global_normalizer(learning):
    total = np.zeros(8, dtype=np.float64)
    squared = np.zeros(8, dtype=np.float64)
    count = 0
    for conditions in learning.values():
        for windows in conditions.values():
            values = derived_channels(windows).reshape(-1, 8).astype(np.float64)
            total += values.sum(0)
            squared += np.square(values).sum(0)
            count += len(values)
    mean = total / count
    std = np.sqrt(np.maximum(squared / count - np.square(mean), 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)


def rotate_windows(windows, maximum_degrees, rng):
    """Apply one small random 3-D rotation to both sensors in each window."""
    output = windows.copy()
    for index in range(len(output)):
        angles = np.deg2rad(rng.uniform(-maximum_degrees, maximum_degrees, 3))
        x, y, z = angles
        rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
        ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
        rotation = (rz @ ry @ rx).astype(np.float32)
        output[index, :, :3] = output[index, :, :3] @ rotation.T
        output[index, :, 3:] = output[index, :, 3:] @ rotation.T
    return output


def balanced_batch(learning, config, rng, augment):
    participants = sorted(learning)
    chosen = rng.choice(
        participants, min(config.identities_per_batch, len(participants)), replace=False
    )
    windows, identities, condition_labels = [], [], []
    for identity, participant in enumerate(chosen):
        for condition_index, condition in enumerate(CONDITIONS):
            values = learning[participant][condition]
            indices = rng.choice(
                len(values), config.samples_per_condition,
                replace=len(values) < config.samples_per_condition,
            )
            windows.extend(values[indices])
            identities.extend([identity] * len(indices))
            condition_labels.extend([condition_index] * len(indices))
    windows = np.asarray(windows, dtype=np.float32)
    if augment:
        windows = rotate_windows(windows, config.rotation_degrees, rng)
    return (
        windows,
        np.asarray(identities, dtype=np.int64),
        np.asarray(condition_labels, dtype=np.int64),
    )


def identity_loss(embeddings, identities, config):
    if config.objective == "triplet":
        return batch_hard_triplet_loss(embeddings, identities, config.margin)
    if config.objective in ("supcon", "supcon_adv"):
        return supervised_contrastive_loss(
            embeddings, identities, config.temperature
        )
    raise ValueError(f"Unknown training objective: {config.objective}")


def train_encoder(learning, validation, config, device):
    seed_everything(config.seed)
    selected_device = torch.device(device)
    model = UnifiedEncoder(config.embedding_size).to(selected_device)
    condition_classifier = ConditionClassifier(config.embedding_size).to(selected_device)
    mean, std = fit_global_normalizer(learning)
    model.set_normalizer(mean, std)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(condition_classifier.parameters()),
        lr=config.learning_rate, weight_decay=config.weight_decay
    )
    training_rng = np.random.default_rng(config.seed)
    validation_rng = np.random.default_rng(config.seed + 10_000)
    best_loss, best_state, best_condition_state, stale = float("inf"), None, None, 0
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        condition_classifier.train()
        training_losses, training_correct, training_total = [], 0, 0
        for _ in range(config.batches_per_epoch):
            windows, identities, conditions = balanced_batch(
                learning, config, training_rng, True
            )
            tensor = torch.as_tensor(windows, device=selected_device)
            labels = torch.as_tensor(identities, device=selected_device)
            condition_labels = torch.as_tensor(conditions, device=selected_device)
            optimizer.zero_grad()
            embeddings = model(tensor)
            loss = identity_loss(embeddings, labels, config)
            if config.objective == "supcon_adv":
                logits = condition_classifier(embeddings)
                loss = loss + config.adversarial_weight * torch.nn.functional.cross_entropy(
                    logits, condition_labels
                )
                training_correct += int((logits.argmax(1) == condition_labels).sum())
                training_total += len(condition_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            training_losses.append(float(loss.detach()))
        model.eval()
        condition_classifier.eval()
        validation_losses, condition_correct, condition_total = [], 0, 0
        with torch.no_grad():
            for _ in range(max(10, config.batches_per_epoch // 5)):
                windows, identities, conditions = balanced_batch(
                    validation, config, validation_rng, False
                )
                embeddings = model(torch.as_tensor(windows, device=selected_device))
                validation_losses.append(float(identity_loss(
                    embeddings, torch.as_tensor(identities, device=selected_device), config,
                )))
                if config.objective == "supcon_adv":
                    condition_labels = torch.as_tensor(conditions, device=selected_device)
                    logits = condition_classifier.network(embeddings)
                    condition_correct += int((logits.argmax(1) == condition_labels).sum())
                    condition_total += len(condition_labels)
        training_loss = float(np.mean(training_losses))
        validation_loss = float(np.mean(validation_losses))
        train_condition_accuracy = (
            training_correct / training_total if training_total else float("nan")
        )
        validation_condition_accuracy = (
            condition_correct / condition_total if condition_total else float("nan")
        )
        history.append((epoch, training_loss, validation_loss,
                        train_condition_accuracy, validation_condition_accuracy))
        print(
            f"  epoch={epoch:02d} train_loss={training_loss:.5f} "
            f"validation_loss={validation_loss:.5f}"
            + (f" condition_accuracy={validation_condition_accuracy:.2%}"
               if condition_total else "")
        )
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            best_condition_state = deepcopy(condition_classifier.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    model.load_state_dict(best_state)
    condition_classifier.load_state_dict(best_condition_state)
    model.eval()
    return model, condition_classifier, pd.DataFrame(
        history,
        columns=("epoch", "train_loss", "validation_loss",
                 "train_condition_accuracy", "validation_condition_accuracy"),
    )


def encode(model, windows, device, batch_size=512):
    results = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.as_tensor(windows[start:start + batch_size], device=device)
            results.append(model(batch).cpu().numpy())
    return np.concatenate(results)


def template(model, windows, device):
    value = encode(model, windows, device).mean(0)
    return value / max(np.linalg.norm(value), 1e-9)


def cosine_distances(embeddings, claimed_template):
    return 1.0 - np.asarray(embeddings) @ np.asarray(claimed_template)


def causal_fusion(scores, size):
    scores = np.asarray(scores, dtype=np.float64)
    cumulative = np.r_[0.0, np.cumsum(scores)]
    positions = np.arange(len(scores))
    starts = np.maximum(0, positions - size + 1)
    return (cumulative[positions + 1] - cumulative[starts]) / (positions - starts + 1)


def eer_threshold(genuine, impostor):
    candidates = np.unique(np.concatenate((genuine, impostor)))
    frr = np.asarray([(genuine > value).mean() for value in candidates])
    far = np.asarray([(impostor <= value).mean() for value in candidates])
    index = int(np.argmin(np.abs(frr - far)))
    return float(candidates[index]), float(frr[index]), float(far[index])


def select_shared_threshold(model, learning, validation, config, device):
    participants = tuple(sorted(learning))
    templates = {
        person: template(model, learning[person]["st_control"], device)
        for person in participants
    }
    encoded = {
        (person, condition): encode(model, validation[person][condition], device)
        for person in participants for condition in CONDITIONS
    }
    genuine, impostor = [], []
    for claim in participants:
        for probe in participants:
            for condition in CONDITIONS:
                scores = causal_fusion(
                    cosine_distances(encoded[(probe, condition)], templates[claim]),
                    config.fusion_window,
                )
                (genuine if claim == probe else impostor).extend(scores)
    return eer_threshold(np.asarray(genuine), np.asarray(impostor))


def evaluate(model, dataset, participants, threshold, config, fold, device):
    enrollment, probes = {}, {condition: {} for condition in CONDITIONS}
    for person in participants:
        enrollment[person], probes["st_control"][person] = enrollment_split(
            dataset[person]["st_control"].windows
        )
        for condition in CONDITIONS[1:]:
            probes[condition][person] = dataset[person][condition].windows
    templates = {
        person: template(model, enrollment[person], device) for person in participants
    }
    encoded = {
        (person, condition): encode(model, probes[condition][person], device)
        for person in participants for condition in CONDITIONS
    }
    rows = []
    for claim in participants:
        for condition in CONDITIONS:
            genuine, impostor = [], []
            for probe in participants:
                scores = causal_fusion(
                    cosine_distances(encoded[(probe, condition)], templates[claim]),
                    config.fusion_window,
                )
                (genuine if claim == probe else impostor).extend(scores)
            genuine, impostor = np.asarray(genuine), np.asarray(impostor)
            rows.append({
                "fold": fold, "participant": claim, "condition": condition,
                "threshold": threshold,
                "frr": float((genuine > threshold).mean()),
                "far": float((impostor <= threshold).mean()),
            })
    return pd.DataFrame(rows)


def run_fold(fold_number, config, device="cpu"):
    dataset = load_dataset()
    development, evaluation = outer_folds(dataset)[fold_number - 1]
    fold_config = replace(config, seed=config.seed + fold_number - 1)
    learning, validation = {}, {}
    for person in development:
        learning[person], validation[person] = {}, {}
        for condition in CONDITIONS:
            learning[person][condition], validation[person][condition] = chronological_split(
                dataset[person][condition].windows
            )
    print(f"Fold {fold_number}/4: development={','.join(development)} evaluation={','.join(evaluation)}")
    model, condition_classifier, history = train_encoder(
        learning, validation, fold_config, device
    )
    threshold, development_frr, development_far = select_shared_threshold(
        model, learning, validation, fold_config, device
    )
    print(
        f"  one shared threshold={threshold:.6f}, development "
        f"FRR/FAR={development_frr:.2%}/{development_far:.2%}"
    )
    metrics = evaluate(
        model, dataset, evaluation, threshold, fold_config, fold_number, device
    )
    if fold_config.objective == "triplet":
        output = RESULT_DIR / f"fold_{fold_number}"
        model_dir = MODEL_DIR / f"fold_{fold_number}"
    else:
        output = RESULT_DIR / fold_config.objective / f"fold_{fold_number}"
        model_dir = MODEL_DIR / fold_config.objective / f"fold_{fold_number}"
    output.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(output / "training_history.csv", index=False)
    metrics.to_csv(output / "participant_metrics.csv", index=False)
    summary = metrics.groupby("condition")[["frr", "far"]].agg(["mean", "std"])
    summary.to_csv(output / "summary.csv")
    torch.save({"state_dict": model.state_dict(), "config": asdict(fold_config)}, model_dir / "encoder.pt")
    if fold_config.objective == "supcon_adv":
        torch.save(condition_classifier.state_dict(), model_dir / "condition_classifier.pt")
    (output / "protocol.json").write_text(json.dumps({
        "development": development, "evaluation": evaluation,
        "shared_threshold": threshold, "development_frr": development_frr,
        "development_far": development_far, "config": asdict(fold_config),
    }, indent=2))
    print(summary.to_string())
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batches-per-epoch", type=int, default=100)
    parser.add_argument("--fusion-window", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--objective", choices=("triplet", "supcon", "supcon_adv"), default="triplet"
    )
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--adversarial-weight", type=float, default=0.1)
    args = parser.parse_args()
    config = Config(
        epochs=args.epochs, patience=args.patience,
        batches_per_epoch=args.batches_per_epoch, fusion_window=args.fusion_window,
        objective=args.objective, temperature=args.temperature,
        adversarial_weight=args.adversarial_weight,
    )
    run_fold(args.fold, config, args.device)


if __name__ == "__main__":
    main()
