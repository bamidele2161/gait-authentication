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

## Implemented architecture

1. Split participants into 12 development and 4 evaluation identities per fold.
2. Use only session-1 ST-control and ST-fatigue from development identities;
   split those recordings chronologically into learning and validation.
3. Fit six-channel normalization on development-learning windows only.
4. Encode each window with three temporal convolutions, a bidirectional LSTM,
   attention/max pooling, and a 64-D L2-normalized projection.
5. In each balanced session-1 batch, treat ST-control windows as anchors. Online
   hard mining selects the farthest same-person ST-control/ST-fatigue positive
   and closest other-person ST-control/ST-fatigue negative for every anchor.
6. Optimize batch-hard metric loss plus a temporary development-identity
   classification loss. The classifier is discarded after encoder training.
7. Enrol each unseen evaluation user using ST-control windows only; average and
   L2-normalize those embeddings to create one template.
8. Compare every probe with the claimed template using Euclidean distance.
   Because embeddings have unit length, this gives the same ordering as cosine
   distance.
9. Select one shared threshold using session-1 development-validation scores,
   freeze it, and apply it unchanged to DT-control and DT-fatigue from session 2.

DT-control and DT-fatigue are never used for encoder training, normalization,
validation, early stopping, or threshold calibration. There is no per-user SVM,
condition-specific model, condition-specific template, or evaluation-time retraining.

## Source-file order

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
13. `run_experiment.py`

The test suite enforces both the participant boundary and the session boundary.
