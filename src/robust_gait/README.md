# Robust multi-sensor gait verification (few-participant design)

Cross-session (7 days), cross-condition (dual-task, fatigue) gait
authentication on DUO-GAIT with only 16 participants.

## Why this design

Diagnostics on the earlier sacrum CNN-BiLSTM (since retired; its four-fold
scores remain in `src/results/condition_invariant/` for `08_test_operating_points.py`) showed:

- With 12 training identities per fold the encoder memorizes those people:
  median genuine distance 0.19 for development users vs 0.95 for unseen users,
  whose genuine scores then overlap impostors (EER 14-21%).
- Thresholds and early stopping chosen on memorized identities do not transfer
  (target 1% FAR, observed 12-20%).
- A linear per-user model on phase-aligned strides already beats the deep model
  on the sacrum alone (EER 6% / 12%), because every user brings ~320 enrolment
  strides even though there are only 12 training identities.

So instead of learning a global identity embedding from 12 people, the system
learns **one small discriminative model per enrolled user**, against a
background of other people, on rich, phase-aligned multi-sensor stride
profiles.

## Pipeline

1. **Data** (`data.py`): all nine synchronized IMUs (sacrum, shanks, wrists,
   feet, head, sternum) read from `raw_data/interim` with the project's known
   recording trims; acceleration and gyroscope magnitudes per location
   (orientation-free) and, with `--channels axes`, the raw six axes.
2. **Stride segmentation**: low-pass (6 Hz) left-shank gyroscope magnitude;
   its mid-swing peak occurs once per stride, so every sensor is cut at the same
   gait phase; strides of 0.7-1.8 s are resampled to 128 samples (removes
   cadence changes caused by dual-task walking and fatigue).
3. **Noise filtering**: mean pooling (box low-pass) to 32 points per channel.
4. **Sensor-fault gate**: a sensor whose enrolment gyroscope level is below 25%
   of the development population's is dropped for that user (sub_07's
   ST-control sacrum and left-foot sensors were not recording movement).
5. **Per-user models** (`verifier.py`): L2 logistic regression per sensor,
   enrolled user vs development background, trained on ST-control strides plus
   augmented copies (gravity-preserving amplitude scaling, within-stride timing
   warp) that mimic fatigue/dual-task variation.
6. **Late sensor fusion**: per-sensor scores averaged (or median, robust to a
   shifted sensor).
7. **Z-norm**: each user's score is standardized by its scores on development
   people held out from that user's training, so scores are comparable across
   users.
8. **Temporal fusion**: mean of 10 consecutive strides (~12 s) per decision.
9. **Calibration mirror**: development users, two at a time, play the unseen
   users (enrolment ST-control, probes DT-control/DT-fatigue, background the
   other 10 development users). One global threshold is fixed on this mirror.

Protocol helpers (outer folds, EER, target-FAR threshold) live in
`protocol.py`, which `src/08_test_operating_points.py` also uses.

Evaluation users' DT data is never used for training, normalization,
calibration or model selection. Every configuration choice was made with
`select_config.py`-style development-mirror scores only.

## Protocol options

| option | default | meaning |
| --- | --- | --- |
| `--calibration-probes` | session1 | `session2` uses *development* users' DT recordings in the mirror (never evaluation users') |
| `--operating-point` | target-far | `eer`, `macro-eer` or `minimax` (minimize the worst per-condition FRR/FAR) |
| `--score-normalization` | none | `znorm`, `tnorm`, `snorm`, `fnorm` |
| `--fusion` / `--sensor-combination` | early / mean | late fusion, mean or median across sensors |
| `--channels` | magnitude | `axes`: raw GyrX/Y/Z + AccX/Y/Z per sensor; `both`: magnitudes + axes |

## Results (4 folds, 16 users, mean of seeds 42/142/242; FRR / FAR / EER %)

| channels | DT-control | DT-fatigue |
| --- | --- | --- |
| magnitude | 6.2 / 14.0 / 2.4 | 14.0 / 10.4 / 4.8 |
| axes | 0.7 / 8.3 / 0.3 | 9.4 / 6.9 / 0.3 |

Raw axes keep the direction of each segment's motion, which magnitudes discard.
Gravity direction alone (sensor mounting, no motion) gives only ~20-23% EER,
so the gain comes from motion direction rather than mounting orientation.

## Commands

```bash
venv/bin/python -u -m src.robust_gait.run_experiment \
  --locations sacrum left_leg right_leg left_foot right_foot sternum \
  --fusion late --regularization 0.3 --enrolment-augmentation 4 \
  --amplitude-range 0.2 --warp-range 0.1 --background-strides 240 \
  --pooling mean --score-normalization znorm --calibration-probes session2 \
  --operating-point minimax --mirror-group-size 2 --channels axes

venv/bin/python -m pytest tests/robust_gait -q
```

Results: `src/results/robust_gait/<configuration>/` (`macro_summary.csv`,
`participant_metrics.csv`, `config.json`). Stride cache:
`src/data/processed/strides/robust_gait_9imu/`.
