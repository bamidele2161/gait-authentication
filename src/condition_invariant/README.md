# Condition-Invariant Gait Authentication

This package contains the proposed cross-condition triplet-learning experiment.

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
6. `model.py`
7. `train.py`
8. `enrollment.py`
9. `scoring.py`
10. `metrics.py`
11. `run_experiment.py`

Each file will be implemented and tested before work begins on the next file.
