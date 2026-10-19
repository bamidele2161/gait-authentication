"""Development-only model selection for the robust gait verifier.

Each candidate is scored with the calibration mirror (session-1 or, with
``--calibration-probes session2``, development users' own DT recordings) of ``run_experiment``:
development users in groups of four act as unseen users (enrolment on early
ST-control, probes from late ST-control and ST-fatigue, background from the
other eight development users).  The criterion is the mirror FRR at the target
FAR, averaged over the four outer folds.  DT-control/DT-fatigue data and the
evaluation users are never touched.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import replace

import numpy as np

from src.robust_gait.protocol import create_outer_folds
from src.robust_gait.data import CONDITIONS, LOCATIONS, load_strides
from src.robust_gait.run_experiment import ExperimentConfig, calibrate, participants
from src.robust_gait.verifier import VerifierConfig


def mirror_frr(config: ExperimentConfig, data, people) -> float:
    """Mirror error at the configured operating point, averaged over folds.

    For ``target-far`` this is the FRR at the target FAR; for ``eer`` it is the
    half total error rate (FAR + FRR) / 2 at the balanced threshold.
    """

    values = []
    for fold in create_outer_folds(people, 4):
        rng = np.random.default_rng(config.seed + fold.fold_index)
        _, frr, far = calibrate(list(fold.development_participants), data, config, rng)
        values.append(frr if config.operating_point == "target-far" else (frr + far) / 2)
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--locations", nargs="+", choices=tuple(LOCATIONS),
                        default=["sacrum", "left_leg", "right_leg"])
    parser.add_argument("--fusion", choices=("early", "late"), default="early")
    parser.add_argument("--regularization", type=float, nargs="+", default=[0.01, 0.03, 0.1, 0.3])
    parser.add_argument("--augmentation", type=int, nargs="+", default=[0, 2])
    parser.add_argument("--normalization", nargs="+", default=["none", "znorm", "tnorm", "snorm"])
    parser.add_argument("--fusion-strides", type=int, default=10)
    parser.add_argument("--calibration-probes", choices=("session1", "session2"),
                        default="session1")
    parser.add_argument("--operating-point", choices=("target-far", "eer", "macro-eer", "minimax"),
                        default="target-far")
    args = parser.parse_args()
    people = participants()
    data = {(p, c): load_strides(p, c) for p in people for c in CONDITIONS}
    base = ExperimentConfig(
        VerifierConfig(locations=tuple(args.locations), fusion=args.fusion),
        fusion_strides=args.fusion_strides,
        calibration_probes=args.calibration_probes,
        operating_point=args.operating_point,
    )
    results = []
    for c, aug, norm in itertools.product(
        args.regularization, args.augmentation, args.normalization
    ):
        config = replace(base, score_normalization=norm, verifier=replace(
            base.verifier, regularization=c, enrolment_augmentation=aug))
        frr = mirror_frr(config, data, people)
        results.append((frr, c, aug, norm))
        print(f"C={c:<6g} augmentation={aug} normalization={norm:5s} "
              f"mirror error={frr:.2%}", flush=True)
    frr, c, aug, norm = min(results)
    print(f"\nselected: C={c:g} augmentation={aug} normalization={norm} "
          f"(mirror error {frr:.2%})")


if __name__ == "__main__":
    main()
