# Paper-inspired augmented CNN verifier

This experiment adapts Choi, Choi, and Kang (2025) to the stricter DUO-GAIT
protocol. It retains the paper's four-block CNN and four training-only signal
augmentations, but it does **not** copy the paper's shuffled 60/40 split.

## Leakage-aware protocol

1. Four participant-disjoint outer folds: 12 development users and four unseen
   evaluation users.
2. Development recordings are split chronologically 80/20 using complete
   blocks and an overlap guard.
3. Each development verifier is enrolled from that user's ST-control learning
   windows only. ST-fatigue is validation-only.
4. The four augmentations are applied only while optimizing the CNN, with at
   most 15% distortion: noise, temporal shift, amplitude stretch, time stretch.
5. Development verifiers determine the median enrollment epoch and one global
   FAR<=1% threshold.
6. Each unseen user receives a fresh binary CNN trained from all of that user's
   sacrum ST-control windows and generic development-user ST-control impostors.
7. DT-control and DT-fatigue remain untouched final-test probes.

## Run fold 1

```bash
python3 -u -m src.augmented_cnn_verifier.run_experiment \
  --fold 1 \
  --epochs 40 \
  --batches-per-epoch 50 \
  --validation-batches 15 \
  --batch-size 64 \
  --fusion-window 1 \
  --windows-dir /Users/user/Documents/gait-authentication/src/data/processed/windows/sacrum_time
```

Use `--fusion-window 10` for an 11-second decision, after evaluating the
two-second system first.
