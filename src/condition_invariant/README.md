# Condition-Invariant Gait Authentication

This package contains the proposed cross-condition metric-learning experiment.

It is intentionally separate from the existing statistical-feature RBF-SVM baseline in the numbered files under `src/`.

## Boundaries

- Reuse the raw sacrum windows created by `src/02_preprocess.py`.
- Do not modify or overwrite the baseline models or results.
- Write proposed-model artifacts under `src/models/condition_invariant/`.
- Write proposed-model results under `src/results/condition_invariant/`.
- Keep evaluated participants out of encoder training, normalization, validation, and threshold selection.
- Enrol evaluated participants with ST-control only.

## Planned implementation order

1. `config.py`
2. `records.py`
3. `dataset.py`
4. `folds.py`
5. `triplets.py`
6. `batches.py`
7. `normalization.py`
8. `model.py`
9. `train.py`
10. `enrollment.py`
11. `scoring.py`
12. `metrics.py`
13. `verifier.py`
14. `run_experiment.py`

Each file will be implemented and tested before work begins on the next file.
