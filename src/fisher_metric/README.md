# Regularized condition-invariant Fisher metric

This non-deep method is designed for DUO-GAIT's small participant population.

It uses the existing 88 sacrum time-domain features. Standardization and PCA
remove scale and redundant noise. Shrinkage Fisher LDA learns a projection in
which windows from the same participant across all conditions are compact and
different participants are separated.

Evaluation users enroll with ST-control only and receive one mean template.
Every condition uses cosine distance and one shared global threshold. Each
unseen user's normal enrollment is divided chronologically into template and
reserved calibration portions. Genuine and cross-user impostor calibration
scores are pooled to select one threshold before any test condition is scored.
No fatigue, dual-task, or test rows participate in threshold selection.

Before final fitting, the development data ranks features by between-person
variation divided by condition/session and within-recording variation. Candidate
feature counts are compared on development validation rows only. The chosen
subset is frozen before evaluation-user enrollment and testing.

Run one outer fold:

```bash
python3 -m src.fisher_metric.run_experiment --fold 1
```
