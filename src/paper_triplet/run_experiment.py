"""Train and evaluate the paper-faithful triplet method on DUO-GAIT."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, roc_auc_score

from src.condition_invariant.folds import create_outer_folds
from src.paper_triplet.config import CONDITIONS, MODEL_DIR, RESULT_DIR, TrainingConfig
from src.paper_triplet.data import (
    chronological_development_split, load_dataset, paper_enrollment_split,
)
from src.paper_triplet.model import PaperTripletEncoder, semi_hard_triplet_loss
from src.paper_triplet.triplets import PaperTripletDataset


def _tensor(values, device):
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def fit_input_normalization(recordings):
    """Fit one six-channel scaler using representation-training windows only."""
    channel_sum = np.zeros(6, dtype=np.float64)
    channel_squared_sum = np.zeros(6, dtype=np.float64)
    count = 0
    for conditions in recordings.values():
        for windows in conditions.values():
            flattened = np.asarray(windows, dtype=np.float64).reshape(-1, 6)
            channel_sum += flattened.sum(axis=0)
            channel_squared_sum += np.square(flattened).sum(axis=0)
            count += len(flattened)
    if count < 2:
        raise ValueError("At least two IMU samples are required for normalization")
    mean = channel_sum / count
    variance = channel_squared_sum / count - np.square(mean)
    standard_deviation = np.sqrt(np.maximum(variance, 1e-12))
    return mean.astype(np.float32), standard_deviation.astype(np.float32)


def train_encoder(learning, validation, config, device="cpu"):
    torch.manual_seed(config.seed)
    selected_device = torch.device(device)
    model = PaperTripletEncoder(
        config.hidden_size, config.lstm_layers
    ).to(selected_device)
    input_mean, input_std = fit_input_normalization(learning)
    model.set_input_normalization(input_mean, input_std)
    training_data = PaperTripletDataset(
        learning, config.anchors_per_participant_condition, config.seed,
        config.negative_candidates,
    )
    validation_data = PaperTripletDataset(
        validation,
        max(20, config.anchors_per_participant_condition // 5),
        config.seed + 1,
        config.negative_candidates,
    )
    training_loader = DataLoader(
        training_data, batch_size=config.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    validation_loader = DataLoader(
        validation_data, batch_size=config.batch_size, shuffle=False
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best_loss = float("inf")
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        training_sum = 0.0
        for anchor, positive, negative in training_loader:
            optimizer.zero_grad()
            anchor_embedding = model(_tensor(anchor, selected_device))
            positive_embedding = model(_tensor(positive, selected_device))
            negative_tensor = _tensor(negative, selected_device)
            batch, candidates = negative_tensor.shape[:2]
            negative_embedding = model(
                negative_tensor.reshape(batch * candidates, *negative_tensor.shape[2:])
            ).reshape(batch, candidates, -1)
            loss = semi_hard_triplet_loss(
                anchor_embedding, positive_embedding, negative_embedding
            )
            loss.backward()
            optimizer.step()
            training_sum += float(loss.detach()) * len(anchor)

        model.eval()
        validation_sum = 0.0
        with torch.no_grad():
            for anchor, positive, negative in validation_loader:
                anchor_embedding = model(_tensor(anchor, selected_device))
                positive_embedding = model(_tensor(positive, selected_device))
                negative_tensor = _tensor(negative, selected_device)
                batch, candidates = negative_tensor.shape[:2]
                negative_embedding = model(
                    negative_tensor.reshape(batch * candidates, *negative_tensor.shape[2:])
                ).reshape(batch, candidates, -1)
                loss = semi_hard_triplet_loss(
                    anchor_embedding, positive_embedding, negative_embedding
                )
                validation_sum += float(loss) * len(anchor)
        training_loss = training_sum / len(training_data)
        validation_loss = validation_sum / len(validation_data)
        history.append({
            "epoch": epoch,
            "training_triplet_loss": training_loss,
            "validation_triplet_loss": validation_loss,
        })
        print(
            f"  epoch={epoch:02d} train_loss={training_loss:.5f} "
            f"validation_loss={validation_loss:.5f}"
        )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("No valid encoder checkpoint was produced")
    model.load_state_dict(best_state)
    model.eval()
    return model, pd.DataFrame(history)


def encode(model, windows, device="cpu", batch_size=512):
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            output.append(
                model(_tensor(windows[start:start + batch_size], device)).cpu().numpy()
            )
    return np.concatenate(output)


def create_template(model, enrollment, device):
    """Average enrolment features exactly as described by the source paper."""
    return encode(model, enrollment, device).mean(axis=0)


def _unit(vector):
    return vector / max(float(np.linalg.norm(vector)), 1e-9)


def distances(model, template, probes, device):
    return np.linalg.norm(encode(model, probes, device) - template, axis=1)


def threshold_at_eer(genuine, impostor):
    candidates = np.unique(np.concatenate((genuine, impostor)))
    frr = np.asarray([(genuine > threshold).mean() for threshold in candidates])
    far = np.asarray([(impostor <= threshold).mean() for threshold in candidates])
    index = int(np.argmin(np.abs(frr - far)))
    return float(candidates[index]), float(frr[index]), float(far[index])


def threshold_at_far(genuine, impostor, target_far):
    candidates = np.r_[-np.inf, np.unique(np.concatenate((genuine, impostor))), np.inf]
    valid = []
    for threshold in candidates:
        far = float((impostor <= threshold).mean())
        if far <= target_far + 1e-12:
            valid.append((float((genuine > threshold).mean()), -threshold, far))
    frr, negative_threshold, far = min(valid)
    return float(-negative_threshold), float(frr), float(far)


def prepare_development(dataset, participants):
    learning, validation = {}, {}
    for participant in participants:
        learning[participant], validation[participant] = {}, {}
        for condition in CONDITIONS:
            first, second = chronological_development_split(dataset[participant][condition])
            learning[participant][condition] = first
            validation[participant][condition] = second
    return learning, validation


def cohort_relative_distances(probe_embeddings, target_template, cohort_templates):
    """Normalize target distance by how each probe relates to cohort templates.

    Smaller remains better. A negative value means the claimed template is closer
    than the average development-cohort template for that particular probe.
    """
    probes = np.asarray(probe_embeddings)
    cohort = np.asarray(cohort_templates)
    raw_target = np.linalg.norm(probes - np.asarray(target_template), axis=1)
    cohort_distances = np.linalg.norm(probes[:, None, :] - cohort[None, :, :], axis=2)
    cohort_mean = cohort_distances.mean(axis=1)
    cohort_std = np.maximum(cohort_distances.std(axis=1, ddof=1), 1e-6)
    normalized = (raw_target - cohort_mean) / cohort_std
    return normalized, raw_target, cohort_mean, cohort_std


def split_template_calibration(windows, fraction=0.8):
    """Chronologically split normal enrollment with one overlap guard window."""
    boundary = min(max(int(len(windows) * fraction), 1), len(windows) - 2)
    template = windows[:boundary]
    calibration = windows[boundary + 1:]
    if not len(template) or not len(calibration):
        raise ValueError("Normal enrollment is too short for score normalization")
    return template, calibration


def fit_enrollment_normalizer(scores):
    """Estimate a robust personal score scale from normal enrollment only."""
    values = np.asarray(scores, dtype=np.float64)
    center = float(np.median(values))
    mad_scale = float(1.4826 * np.median(np.abs(values - center)))
    scale = max(mad_scale, 1e-3)
    return center, scale


def normalize_scores(scores, center, scale):
    """Put every claimant's scores on the same dimensionless scale."""
    return (np.asarray(scores, dtype=np.float64) - center) / scale


def global_development_threshold(
    model, learning, validation, participants, cohort_templates, fusion_window, device,
):
    """Select one threshold on identities unseen during encoder training."""
    templates = {
        participant: _unit(create_template(model, split_template_calibration(
            learning[participant]["st_control"]
        )[0], device))
        for participant in participants
    }
    normalizers = {}
    for participant in participants:
        calibration_windows = split_template_calibration(
            learning[participant]["st_control"]
        )[1]
        calibration_scores = cohort_relative_distances(
            encode(model, calibration_windows, device), templates[participant],
            cohort_templates,
        )[0]
        calibration_scores = causal_mean_fusion(calibration_scores, fusion_window)
        normalizers[participant] = fit_enrollment_normalizer(calibration_scores)
    encoded = {
        (participant, condition): encode(model, validation[participant][condition], device)
        for participant in participants
        for condition in CONDITIONS
    }
    genuine, impostor = [], []
    for claimed_id in participants:
        for probe_id in participants:
            for condition in CONDITIONS:
                scores = cohort_relative_distances(
                    encoded[(probe_id, condition)], templates[claimed_id],
                    cohort_templates,
                )[0]
                scores = causal_mean_fusion(scores, fusion_window)
                scores = normalize_scores(scores, *normalizers[claimed_id])
                (genuine if claimed_id == probe_id else impostor).extend(scores)
    threshold, frr, far = threshold_at_eer(np.asarray(genuine), np.asarray(impostor))
    return {
        "threshold": threshold,
        "development_frr": frr,
        "development_far": far,
        "genuine_comparisons": len(genuine),
        "impostor_comparisons": len(impostor),
    }


def causal_mean_fusion(values, window_size):
    """Average only the current and preceding scores in one continuous stream."""
    values = np.asarray(values, dtype=np.float64)
    if window_size < 1:
        raise ValueError("fusion window must be positive")
    if not len(values):
        return values
    cumulative = np.r_[0.0, np.cumsum(values)]
    positions = np.arange(len(values))
    starts = np.maximum(0, positions - window_size + 1)
    totals = cumulative[positions + 1] - cumulative[starts]
    return totals / (positions - starts + 1)


def evaluate_fold(
    model, dataset, participants, cohort_learning, threshold_learning,
    threshold_validation, fold_number, config, device,
):
    enrollment, probes = {}, {condition: {} for condition in CONDITIONS}
    for participant in participants:
        template_windows, calibration_windows, test_windows = paper_enrollment_split(
            dataset[participant]["st_control"]
        )
        enrollment[participant] = (template_windows, calibration_windows)
        probes["st_control"][participant] = test_windows
        for condition in CONDITIONS[1:]:
            probes[condition][participant] = dataset[participant][condition].values

    encoded_probes = {
        (condition, participant): encode(model, probes[condition][participant], device)
        for condition in CONDITIONS
        for participant in participants
    }
    templates = {
        participant: _unit(create_template(
            model, enrollment[participant][0], device,
        ))
        for participant in participants
    }
    cohort_templates = np.stack([
        _unit(create_template(
            model, cohort_learning[participant]["st_control"], device
        ))
        for participant in sorted(cohort_learning)
    ])
    calibration = global_development_threshold(
        model, threshold_learning, threshold_validation,
        tuple(sorted(threshold_learning)), cohort_templates,
        config.fusion_window, device,
    )
    shared_threshold = calibration["threshold"]
    print(
        f"  shared development threshold={shared_threshold:.6f}, "
        f"development FRR/FAR={calibration['development_frr']:.2%}/"
        f"{calibration['development_far']:.2%}"
    )
    metric_rows, score_rows = [], []
    for claimed_id in participants:
        template = templates[claimed_id]
        enrollment_scores = cohort_relative_distances(
            encode(model, enrollment[claimed_id][1], device), template,
            cohort_templates,
        )[0]
        enrollment_scores = causal_mean_fusion(
            enrollment_scores, config.fusion_window
        )
        score_center, score_scale = fit_enrollment_normalizer(enrollment_scores)
        for condition in CONDITIONS:
            genuine_distances, impostor_distances = [], []
            for probe_id in participants:
                values, raw_values, cohort_means, cohort_stds = cohort_relative_distances(
                    encoded_probes[(condition, probe_id)], template, cohort_templates
                )
                cohort_values = causal_mean_fusion(values, config.fusion_window)
                values = normalize_scores(cohort_values, score_center, score_scale)
                is_genuine = probe_id == claimed_id
                (genuine_distances if is_genuine else impostor_distances).extend(values)
                for index, (raw_value, cohort_value, value, cohort_mean, cohort_std) in enumerate(
                    zip(raw_values, cohort_values, values, cohort_means, cohort_stds)
                ):
                    score_rows.append({
                        "fold": fold_number,
                        "claimed_participant_id": claimed_id,
                        "probe_participant_id": probe_id,
                        "condition": condition,
                        "probe_window_index": index,
                        "is_genuine": is_genuine,
                        "raw_distance": float(raw_value),
                        "cohort_normalized_distance": float(cohort_value),
                        "enrollment_normalized_score": float(value),
                        "probe_cohort_mean": float(cohort_mean),
                        "probe_cohort_std": float(cohort_std),
                        "enrollment_score_center": score_center,
                        "enrollment_score_scale": score_scale,
                    })
            genuine_distances = np.asarray(genuine_distances)
            impostor_distances = np.asarray(impostor_distances)
            metric_rows.append({
                "fold": fold_number,
                "claimed_participant_id": claimed_id,
                "condition": condition,
                "operating_point": "shared_development_eer_threshold",
                "threshold": shared_threshold,
                "frr": float((genuine_distances > shared_threshold).mean()),
                "far": float((impostor_distances <= shared_threshold).mean()),
                "development_frr": calibration["development_frr"],
                "development_far": calibration["development_far"],
                "calibration_genuine_count": calibration["genuine_comparisons"],
                "calibration_impostor_count": calibration["impostor_comparisons"],
                "fusion_window": config.fusion_window,
                "templates_per_user": 1,
                "score_normalization": "enrollment_median_mad",
                "genuine_comparisons": len(genuine_distances),
                "impostor_comparisons": len(impostor_distances),
            })
    return pd.DataFrame(metric_rows), pd.DataFrame(score_rows)


def evaluation_eer_summary(scores):
    """Report post-hoc evaluation EER for comparison with the source paper.

    These thresholds use evaluation labels and are descriptive, not deployable.
    """
    rows = []
    for condition, frame in scores.groupby("condition", sort=False):
        labels = frame.is_genuine.astype(int).to_numpy()
        similarities = -frame.enrollment_normalized_score.to_numpy()
        far, true_positive_rate, _ = roc_curve(labels, similarities)
        frr = 1 - true_positive_rate
        index = int(np.argmin(np.abs(far - frr)))
        eer = float((far[index] + frr[index]) / 2)
        rows.append({
            "condition": condition,
            "evaluation_eer": eer,
            "one_minus_eer": 1 - eer,
            "evaluation_auc": float(roc_auc_score(labels, similarities)),
            "note": "post-hoc descriptive metric; threshold uses evaluation labels",
        })
    return pd.DataFrame(rows)


def run_fold(
    fold_number=1, config=TrainingConfig(), device="cpu", reuse_checkpoint=False
):
    dataset = load_dataset()
    folds = create_outer_folds(tuple(sorted(dataset)), 4)
    fold = folds[fold_number - 1]
    fold_config = replace(config, seed=config.seed + fold_number - 1)
    print(
        f"Fold {fold_number}/4: development={','.join(fold.development_participants)} "
        f"evaluation={','.join(fold.evaluation_participants)}"
    )
    learning, validation = prepare_development(dataset, fold.development_participants)
    representation_participants = tuple(fold.development_participants[:8])
    threshold_participants = tuple(fold.development_participants[8:])
    representation_learning = {
        participant: learning[participant] for participant in representation_participants
    }
    representation_validation = {
        participant: validation[participant] for participant in representation_participants
    }
    checkpoint_path = MODEL_DIR / f"fold_{fold_number}" / "encoder.pt"
    if reuse_checkpoint:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if checkpoint.get("protocol_version") != "standardized_shared_threshold_v2":
            raise ValueError(
                "Checkpoint predates the shared-threshold protocol; rerun without "
                "--reuse-checkpoint"
            )
        saved = checkpoint["training_config"]
        model = PaperTripletEncoder(
            hidden_size=saved["hidden_size"], layers=saved["lstm_layers"]
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        history = None
        print(f"  reused encoder checkpoint: {checkpoint_path}")
    else:
        model, history = train_encoder(
            representation_learning, representation_validation, fold_config, device
        )
    metrics, scores = evaluate_fold(
        model, dataset, fold.evaluation_participants, representation_learning,
        {participant: learning[participant] for participant in threshold_participants},
        {participant: validation[participant] for participant in threshold_participants},
        fold_number, fold_config, device,
    )
    output = RESULT_DIR / f"shared_threshold_fold_{fold_number}_fusion_{fold_config.fusion_window}"
    models = MODEL_DIR / f"fold_{fold_number}"
    output.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    if history is not None:
        history.to_csv(output / "training_history.csv", index=False)
    metrics.to_csv(output / "participant_metrics.csv", index=False)
    scores.to_csv(output / "evaluation_scores.csv", index=False)
    summary = metrics.groupby(["condition", "operating_point"])[["frr", "far"]].agg(
        ["mean", "std"]
    )
    summary.to_csv(output / "macro_summary.csv")
    evaluation_eer = evaluation_eer_summary(scores)
    evaluation_eer.to_csv(output / "evaluation_eer_summary.csv", index=False)
    torch.save({
        "state_dict": model.state_dict(),
        "training_config": asdict(fold_config),
        "development_participants": fold.development_participants,
        "representation_participants": representation_participants,
        "threshold_participants": threshold_participants,
        "evaluation_participants": fold.evaluation_participants,
        "protocol_version": "standardized_shared_threshold_v2",
    }, models / "encoder.pt")
    print(summary.to_string())
    print("\nPaper-comparable post-hoc evaluation metrics")
    print(evaluation_eer.to_string(index=False))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--anchors-per-group", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fusion-window", type=int, default=5)
    parser.add_argument("--reuse-checkpoint", action="store_true")
    args = parser.parse_args()
    config = TrainingConfig(
        epochs=args.epochs,
        patience=args.patience,
        anchors_per_participant_condition=args.anchors_per_group,
        batch_size=args.batch_size,
        fusion_window=args.fusion_window,
    )
    run_fold(args.fold, config, args.device, args.reuse_checkpoint)


if __name__ == "__main__":
    main()
