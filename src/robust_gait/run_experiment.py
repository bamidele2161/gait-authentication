"""Multi-sensor stride-aligned per-user gait verification on DUO-GAIT.

Final nine-sensor protocol:
- 4 outer folds, 12 development and 4 completely unseen evaluation users;
- evaluation users enrol with their complete ST-control recording only;
- probes are session-2 DT-control and DT-fatigue only;
- one threshold per outer fold, calibrated on development users only; and
- development users' session-2 data may calibrate the operating point, but
  evaluation identities never contribute to fitting, normalization, or
  threshold selection.

Per fold:
1. Background: strides of the 12 development users (ST-control + ST-fatigue).
2. Each evaluation user gets a per-user model (``verifier.UserModel``) trained
   on their ST-control strides against that background.
3. Each probe stride is scored by the claimed user's model; decisions average
   ``fusion_strides`` consecutive stride scores (10 strides ~ 12 s).
4. Calibration mirror: development users are divided into pseudo-unseen
   groups.  In the final configuration, each group contains two users, uses
   full ST-control enrolment and session-2 DT probes, and selects a minimax
   threshold from fused, S-normalized scores.  Other development users provide
   the background population and score-normalization cohorts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.robust_gait.protocol import (
    create_outer_folds, equal_error_rate, select_threshold_from_distances,
)
from src.robust_gait.data import CONDITIONS, LOCATIONS, SRC_DIR, load_strides
from src.robust_gait.verifier import UserModel, VerifierConfig

RESULTS_DIR = SRC_DIR / "results" / "robust_gait"
TRAINING_CONDITIONS = ("st_control", "st_fatigue")
EVALUATION_CONDITIONS = ("dt_control", "dt_fatigue")
ENROLMENT_FRACTION = 0.8
NORMALIZATION_SPLITS = 3
SENSOR_FAULT_RATIO = 0.25
_REPORTED_FAULTS: set[str] = set()
OFFSET = 1000.0  # turns "higher score = genuine" into a non-negative distance


@dataclass(frozen=True)
class ExperimentConfig:
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    fusion_strides: int = 10
    target_far: float = 0.01
    score_normalization: str = "none"   # none | znorm | tnorm | snorm | fnorm
    # session1: mirror uses development users' late ST-control + ST-fatigue.
    # session2: mirror uses development users' DT-control + DT-fatigue (never
    #           the evaluation users'), matching the one-week deployment gap.
    calibration_probes: str = "session1"
    # target-far: strictest threshold meeting target_far on the mirror.
    # eer: threshold where pooled mirror FAR equals pooled mirror FRR.
    # macro-eer: threshold where per-user-averaged mirror FAR equals FRR.
    # minimax: threshold minimizing the worst per-condition macro FRR/FAR.
    operating_point: str = "target-far"
    # Pseudo-unseen users per mirror group.  Smaller groups leave a larger
    # background (12 - size), closer to the 12 used for real evaluation users.
    mirror_group_size: int = 4
    seed: int = 42
    exclude: tuple[str, ...] = ()


def participants() -> tuple[str, ...]:
    folder = SRC_DIR / "data" / "processed" / "windows" / "sacrum_time" / "st_control"
    return tuple(sorted(p.name.removesuffix("_windows.npy") for p in folder.glob("*_windows.npy")))


def fuse(scores: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return scores
    cumulative = np.cumsum(np.insert(scores, 0, 0.0))
    return (cumulative[n:] - cumulative[:-n]) / n


def _background(people, data, rng, per_user):
    return np.vstack([
        data[p, c].strides[np.sort(rng.choice(
            len(data[p, c].strides), min(per_user, len(data[p, c].strides)), replace=False,
        ))]
        for p in people for c in TRAINING_CONDITIONS
    ])


def _user_scorer(enrolment, background_people, data, rng, config, seed, normalize):
    """Return a function mapping strides to raw or z-normalized scores.

    With ``normalize`` (Z-norm) the background people are split into
    ``NORMALIZATION_SPLITS`` groups.  For each group a model is trained without
    that group, and the group's strides (people the model never saw) give the
    impostor mean/SD used to standardize that model's scores.  The normalized
    scores of the rotated models are averaged, which puts every user's scores
    on a common "distance from a stranger" scale.
    """

    per_user = config.verifier.background_strides_per_user
    if not normalize:
        model = UserModel(config.verifier).fit(
            enrolment, _background(background_people, data, rng, per_user), seed
        )
        return model.score
    scorers = []
    for k in range(NORMALIZATION_SPLITS):
        cohort = background_people[k::NORMALIZATION_SPLITS]
        training = [p for p in background_people if p not in cohort]
        background = _background(training, data, rng, per_user)
        model = UserModel(config.verifier).fit(enrolment, background, seed + k)
        cohort_scores = model.score(_background(cohort, data, rng, per_user))
        if config.score_normalization == "fnorm":
            # Genuine level from 2-fold chronological cross-validation on the
            # user's own enrolment: each half is scored by a model trained on
            # the other half.  F-norm maps impostor mean -> 0, genuine mean -> 1.
            half = len(enrolment) // 2
            held_out = np.concatenate([
                UserModel(config.verifier).fit(train, background, seed + k).score(test)
                for train, test in (
                    (enrolment[:half], enrolment[half:]), (enrolment[half:], enrolment[:half])
                )
            ])
            impostor_mean = cohort_scores.mean()
            spread = max(held_out.mean() - impostor_mean, 1e-6)
            scorers.append((model, impostor_mean, spread))
            continue
        scorers.append((model, cohort_scores.mean(), cohort_scores.std() + 1e-9))
    return lambda strides: np.mean(
        [(model.score(strides) - mean) / sd for model, mean, sd in scorers], axis=0
    )


def _cohort_scores(probes, background_people, data, rng, config, seed):
    """T-norm cohort: one model per background person, trained against the
    other background people.  Returns {probe key: (decisions, cohort) fused
    scores}; the probe owners are never part of the cohort."""

    per_user = config.verifier.background_strides_per_user
    columns = {key: [] for key in probes}
    for i, person in enumerate(background_people):
        others = [p for p in background_people if p != person]
        model = UserModel(config.verifier).fit(
            data[person, "st_control"].strides,
            _background(others, data, rng, per_user), seed + 1000 + i,
        )
        for key, strides in probes.items():
            columns[key].append(fuse(model.score(strides), config.fusion_strides))
    return {key: np.column_stack(value) for key, value in columns.items()}


def working_locations(enrolment, background_people, data, config):
    """Drop sensors that were not working during a user's enrolment.

    A location is treated as faulty when its median gyroscope magnitude over
    the enrolment strides is below ``SENSOR_FAULT_RATIO`` of the development
    population's median for that location (e.g. a detached sensor recording
    almost no rotation).  Only development users define the reference.
    """

    names = list(LOCATIONS)
    reference = np.median(np.concatenate([
        data[p, c].strides[:, :, 1:2 * len(names):2].reshape(-1, len(names))
        for p in background_people for c in TRAINING_CONDITIONS
    ]), axis=0)
    level = np.median(enrolment[:, :, 1:2 * len(names):2].reshape(-1, len(names)), axis=0)
    kept = tuple(
        location for location in config.verifier.locations
        if level[names.index(location)] >= SENSOR_FAULT_RATIO * reference[names.index(location)]
    )
    return kept or config.verifier.locations


def score_streams(users, enrolment, probes, background_people, data, rng, config, seed):
    """{(claimed, probe_owner, condition): fused chronological scores}.

    ``score_normalization``: ``none`` raw scores; ``znorm`` claimed-model
    normalization (above); ``tnorm`` each decision is standardized by the same
    probe's scores against the cohort models, which cancels probe/session-wide
    shifts; ``snorm`` averages the Z- and T-normalized scores.
    """

    mode = config.score_normalization
    cohort = (
        _cohort_scores(probes, background_people, data, rng, config, seed)
        if mode in ("tnorm", "snorm") else None
    )
    streams = {}
    base_config = config
    for claimed in users:
        locations = working_locations(enrolment[claimed], background_people, data, base_config)
        if locations != base_config.verifier.locations and claimed not in _REPORTED_FAULTS:
            _REPORTED_FAULTS.add(claimed)
            dropped = sorted(set(base_config.verifier.locations) - set(locations))
            print(f"  {claimed}: faulty enrolment sensor(s) dropped: {', '.join(dropped)}")
        config = replace(base_config, verifier=replace(base_config.verifier, locations=locations))
        raw = (
            _user_scorer(enrolment[claimed], background_people, data, rng, config, seed, False)
            if mode in ("none", "tnorm", "snorm") else None
        )
        z = (
            _user_scorer(enrolment[claimed], background_people, data, rng, config, seed, True)
            if mode in ("znorm", "snorm", "fnorm") else None
        )
        for key, strides in probes.items():
            if mode in ("znorm", "fnorm"):
                score = fuse(z(strides), config.fusion_strides)
            elif mode == "none":
                score = fuse(raw(strides), config.fusion_strides)
            else:
                fused = fuse(raw(strides), config.fusion_strides)
                c = cohort[key]
                score = (fused - c.mean(axis=1)) / (c.std(axis=1) + 1e-9)
                if mode == "snorm":
                    score = 0.5 * (score + fuse(z(strides), config.fusion_strides))
            streams[(claimed, *key)] = score
    return streams


def _macro_balanced_threshold(per_claim):
    """Threshold where the mirror's per-user-averaged FAR equals its FRR.

    The reported metrics are macro averages over claimed users and conditions,
    so the threshold is balanced on that same quantity rather than on pooled
    scores, where users with broad score distributions dominate.
    """

    claims = [
        (np.sort(np.concatenate(g)), np.sort(np.concatenate(i)))
        for g, i in per_claim.values()
    ]
    candidates = np.unique(np.concatenate([np.concatenate(c) for c in claims]))
    candidates = np.quantile(candidates, np.linspace(0, 1, 4001))
    frr = np.mean([1 - np.searchsorted(g, candidates, "right") / len(g) for g, _ in claims], axis=0)
    far = np.mean([np.searchsorted(i, candidates, "right") / len(i) for _, i in claims], axis=0)
    best = int(np.argmin(np.abs(frr - far)))
    return float(candidates[best]), float(frr[best]), float(far[best])


def _minimax_threshold(per_claim):
    """Threshold minimizing the worst per-condition macro FRR or FAR.

    The requirement is that FRR and FAR stay low in *every* condition, so the
    mirror's per-condition macro averages are computed and the threshold that
    minimizes their maximum is chosen.  Returns (threshold, worst, worst) so the
    selection criterion (FRR + FAR) / 2 equals the worst-case error.
    """

    by_condition = {}
    for (_, _, condition), (g, i) in per_claim.items():
        by_condition.setdefault(condition, []).append(
            (np.sort(np.concatenate(g)), np.sort(np.concatenate(i)))
        )
    pooled = np.unique(np.concatenate([
        np.concatenate(pair) for claims in by_condition.values() for pair in claims
    ]))
    candidates = np.quantile(pooled, np.linspace(0, 1, 4001))
    worst = np.zeros_like(candidates)
    for claims in by_condition.values():
        frr = np.mean([1 - np.searchsorted(g, candidates, "right") / len(g) for g, _ in claims], axis=0)
        far = np.mean([np.searchsorted(i, candidates, "right") / len(i) for _, i in claims], axis=0)
        worst = np.maximum(worst, np.maximum(frr, far))
    best = int(np.argmin(worst))
    return float(candidates[best]), float(worst[best]), float(worst[best])


def calibrate(development, data, config, rng):
    """Mirror the unseen-user procedure on development users only."""

    genuine, impostor, per_claim = [], [], {}
    groups = len(development) // config.mirror_group_size
    for g in range(groups):
        group = development[g::groups]
        others = [p for p in development if p not in group]
        enrolment, probes = {}, {}
        for p in group:
            strides = data[p, "st_control"].strides
            if config.calibration_probes == "session2":
                # Same situation as the evaluation users: full ST-control
                # enrolment, session-2 DT probes, one week later.
                enrolment[p] = strides
                for condition in EVALUATION_CONDITIONS:
                    probes[p, condition] = data[p, condition].strides
                continue
            cut = int(len(strides) * ENROLMENT_FRACTION)
            enrolment[p] = strides[:cut]
            probes[p, "st_control"] = strides[cut + 1:]  # skip the boundary stride
            probes[p, "st_fatigue"] = data[p, "st_fatigue"].strides
        streams = score_streams(
            group, enrolment, probes, others, data, rng, config, config.seed + g
        )
        for (claimed, owner, condition), scores in streams.items():
            (genuine if claimed == owner else impostor).append(OFFSET - scores)
            per_claim.setdefault((g, claimed, condition), ([], []))[
                0 if claimed == owner else 1
            ].append(OFFSET - scores)
    genuine, impostor = np.concatenate(genuine), np.concatenate(impostor)
    if config.operating_point == "macro-eer":
        return _macro_balanced_threshold(per_claim)
    if config.operating_point == "minimax":
        return _minimax_threshold(per_claim)
    if config.operating_point == "eer":
        # Balanced operating point: development FAR == development FRR.
        _, threshold = equal_error_rate(genuine, impostor)
        return (
            threshold,
            float(np.mean(genuine > threshold)),
            float(np.mean(impostor <= threshold)),
        )
    return select_threshold_from_distances(genuine, impostor, config.target_far)


def run(config: ExperimentConfig, results_dir: Path) -> pd.DataFrame:
    people = participants()
    data = {(p, c): load_strides(p, c) for p in people for c in CONDITIONS}
    rows = []
    for fold in create_outer_folds(people, 4):
        rng = np.random.default_rng(config.seed + fold.fold_index)
        development = list(fold.development_participants)
        threshold, cal_frr, cal_far = calibrate(development, data, config, rng)
        users = [p for p in fold.evaluation_participants if p not in config.exclude]
        enrolment = {p: data[p, "st_control"].strides for p in users}
        probes = {(p, c): data[p, c].strides for p in users for c in EVALUATION_CONDITIONS}
        streams = score_streams(
            users, enrolment, probes, development, data, rng, config,
            config.seed + 50 + fold.fold_index,
        )
        for claimed in users:
            for condition in EVALUATION_CONDITIONS:
                genuine = OFFSET - streams[claimed, claimed, condition]
                impostor = np.concatenate([
                    OFFSET - streams[claimed, other, condition]
                    for other in users if other != claimed
                ])
                rows.append({
                    "fold_index": fold.fold_index,
                    "claimed_participant_id": claimed,
                    "condition": condition,
                    "frr": float(np.mean(genuine > threshold)),
                    "far": float(np.mean(impostor <= threshold)),
                    "eer": equal_error_rate(genuine, impostor)[0],
                    "threshold": threshold,
                    "genuine_count": len(genuine),
                    "impostor_count": len(impostor),
                })
        print(
            f"fold {fold.fold_index + 1}: users={','.join(users)} "
            f"calibration FRR/FAR={cal_frr:.2%}/{cal_far:.2%}", flush=True,
        )
    metrics = pd.DataFrame(rows)
    summary = metrics.groupby("condition")[["frr", "far", "eer"]].agg(["mean", "std"])
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(results_dir / "participant_metrics.csv", index=False)
    summary.to_csv(results_dir / "macro_summary.csv")
    (results_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))
    print(
        f"\nsensors={'+'.join(config.verifier.locations)} fusion={config.verifier.fusion} "
        f"({config.fusion_strides} strides per decision)"
    )
    for condition, row in summary.iterrows():
        print(
            f"{condition:12s} FRR={row[('frr', 'mean')]:.2%}  "
            f"FAR={row[('far', 'mean')]:.2%}  EER={row[('eer', 'mean')]:.2%}"
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--locations", nargs="+", choices=tuple(LOCATIONS),
        default=["sacrum", "left_leg", "right_leg"],
    )
    parser.add_argument("--fusion", choices=("early", "late"), default="early")
    parser.add_argument("--sensor-combination", choices=("mean", "median"), default="mean",
                        help="how late fusion combines per-sensor scores")
    parser.add_argument("--regularization", type=float, default=0.1)
    parser.add_argument("--amplitude-range", type=float, default=0.10)
    parser.add_argument("--warp-range", type=float, default=0.05)
    parser.add_argument("--background-strides", type=int, default=120,
                        help="background strides per development user and condition")
    parser.add_argument("--channels", choices=("magnitude", "axes", "both"), default="magnitude",
                        help="per-sensor input: 2 magnitudes, 6 raw axes, or all 8")
    parser.add_argument("--pooling", choices=("slice", "mean"), default="slice")
    parser.add_argument("--derivative", action="store_true")
    parser.add_argument("--enrolment-augmentation", type=int, default=0,
                        help="augmented copies per enrolment stride")
    parser.add_argument("--score-normalization", choices=("none", "znorm", "tnorm", "snorm", "fnorm"), default="none")
    parser.add_argument("--calibration-probes", choices=("session1", "session2"),
                        default="session1")
    parser.add_argument("--operating-point", choices=("target-far", "eer", "macro-eer", "minimax"),
                        default="target-far")
    parser.add_argument("--mirror-group-size", type=int, default=4)
    parser.add_argument("--fusion-strides", type=int, default=10)
    parser.add_argument("--target-far", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude", nargs="*", default=[])
    args = parser.parse_args()
    config = ExperimentConfig(
        verifier=VerifierConfig(
            locations=tuple(args.locations), fusion=args.fusion,
            combination=args.sensor_combination,
            regularization=args.regularization,
            enrolment_augmentation=args.enrolment_augmentation,
            pooling=args.pooling, derivative=args.derivative, channels=args.channels,
            amplitude_range=args.amplitude_range, warp_range=args.warp_range,
            background_strides_per_user=args.background_strides,
        ),
        fusion_strides=args.fusion_strides, target_far=args.target_far,
        score_normalization=args.score_normalization,
        calibration_probes=args.calibration_probes,
        operating_point=args.operating_point,
        mirror_group_size=args.mirror_group_size,
        seed=args.seed, exclude=tuple(args.exclude),
    )
    name = (
        f"{'+'.join(args.locations)}_{args.fusion}"
        f"{'-median' if args.sensor_combination == 'median' else ''}_C{args.regularization:g}"
        f"{'' if args.channels == 'magnitude' else '_' + args.channels}"
        f"_{args.pooling}{'_deriv' if args.derivative else ''}"
        f"_aug{args.enrolment_augmentation}a{args.amplitude_range:g}w{args.warp_range:g}"
        f"_bg{args.background_strides}_{args.score_normalization}"
        f"_cal{args.calibration_probes}_{args.operating_point}_g{args.mirror_group_size}"
        f"_fuse{args.fusion_strides}"
    )
    if args.exclude:
        name += "_without_" + "_".join(args.exclude)
    if args.seed != 42:
        name += f"_seed{args.seed}"
    run(config, RESULTS_DIR / name)


if __name__ == "__main__":
    main()
