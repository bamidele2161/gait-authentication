# Temporary state-specific gait verification

This experiment keeps the normal-walking verifier unchanged. After a trusted
secondary authentication, it trains a separate temporary verifier using only
the user's current-state gait as genuine data. Development participants in the
matching condition provide the impostor cohort.

The default protocol uses 30 seconds for the trusted update, 20 seconds for
threshold calibration, a causal median of five window scores, and a development
FAR target of 0.5%. All partitions are chronological and separated by guard
windows.

After fitting the first temporary verifier, a hard-negative stage scores every
matching-condition window from the development participants. It selects the 80
most genuine-looking windows per participant and retrains against these cases.
Evaluation participants are never used for hard-negative selection.

Run a pilot fold:

```bash
source venv/bin/activate
python3 -m src.state_specific_adaptation.run_experiment --fold 1
```

The output reports both single-window `state_specific` results and
`state_specific_fused_5` results. It also reports `static` and
`static_fused_5` controls on exactly the same untouched evaluation windows so
that improvements are paired and directly comparable.
The `state_specific_hard_negative` row reports the security-refined verifier.

To select genuine-update duration and hard-negative strength from trusted
calibration data without inspecting test windows, run:

```bash
python3 -m src.state_specific_adaptation.run_experiment \
  --fold 1 --update-seconds 60 --adaptive-search
```

This evaluates 20/30/60-second trusted prefixes and 0/20/40/80 hard negatives
per development participant. The chosen `state_specific_selected` model is then
evaluated once on the untouched remainder of the recording.
