# Nine-sensor problem-definition baseline

This package repeats the original statistical-feature RBF-SVM experiment with
all nine synchronized DUO-GAIT IMUs. It exists to answer whether the robust
stride verifier improves over a conventional baseline when both systems can
observe the same sensor locations.

The baseline uses the six raw axes from sacrum, both shanks, both wrists, both
feet, head, and sternum. Every two-second, 50%-overlapping window therefore
produces 594 time-domain features (9 sensors x 6 axes x 11 statistics). One participant-
specific RBF-SVM is trained from ST-control only. Its thresholds are selected
from held-out ST-control at a 1% FAR target and frozen before ST-fatigue,
DT-control, or DT-fatigue is evaluated.

It reports both 2-second decisions and fusion of ten windows (11 seconds). The
11-second threshold is selected from equally fused validation streams; an
unfused threshold is never applied to fused test scores.

This baseline intentionally does **not** use stride alignment, augmentation,
sensor-fault removal, S-norm, session-2 calibration, or minimax thresholding.

```bash
source venv/bin/activate
python3 -u -m src.multisensor_problem_definition.run_experiment
```

Features are cached under
`src/data/processed/features/multisensor_9imu_raw_axes/`. Results and models
are isolated under `src/results/multisensor_problem_definition/` and
`src/models/multisensor_problem_definition/`.
