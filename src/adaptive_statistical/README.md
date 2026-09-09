# Condition-robust adaptive statistical verification

This separate ablation implements enrollment-only raw-channel centering, 88
time-domain statistical features, development-only feature scaling, cosine
similarity, per-user ST-control thresholds, and causal accepted-window EMA
template updates.

The threshold is selected at a 1% target FAR using held-out genuine ST-control
windows and a development-participant impostor cohort. No changed-condition
data from an evaluation user calibrates the threshold.

EMA is deliberately causal. A probe is scored before it can update the
template, rejected probes never update it, and no condition label is used.
Because accepting an impostor can poison an adaptive template, FAR must be
reported alongside FRR and the method should not be described as secure unless
that risk is acceptably small.

```bash
python3 -m src.adaptive_statistical.run_experiment --fold 1
```
