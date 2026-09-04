# Secure online gait adaptation

This experiment preserves normal-walking-only initial enrollment. When the
static gait verifier encounters a changed walking state, a trusted secondary
factor (for example, fingerprint or PIN) verifies a short segment. That segment
is then added as genuine data to a temporary participant-specific verifier.

For each changed DUO-GAIT recording, the protocol uses the first 20 seconds for
the trusted update, the next 10 seconds for threshold calibration, and the
remaining recording for evaluation. Guard windows prevent shared raw samples
between these portions. Static and adapted results use exactly the same final
test windows.

The experiment also reports a `gated` method. It activates the adapted model
only when the trusted calibration segment has lower FRR than the static model
and its development-cohort FAR remains at or below 1%. If either check fails,
the original normal-walking verifier remains active.

Run a pilot fold:

```bash
source venv/bin/activate
python3 -m src.secure_adaptation.run_experiment --fold 1
```

The outputs are stored under `src/results/secure_adaptation/`. Changed-session
data from an evaluation participant is never used unless it belongs to that
claimed participant and is inside the simulated trusted update/calibration
period.
