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
4. Train one shared 64-unit LSTM encoder with explicit triplets: every anchor is
   ST-control; positives are the same person's ST-control or ST-fatigue; negatives
   are another person's ST-control or ST-fatigue.
5. Apply triplet loss so the positive is closer to the anchor than the negative
   by the configured margin.
6. Convert each 2-second window into one L2-normalized 64-D embedding.
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
13. `verifier.py`
14. `run_experiment.py`

Each file will be implemented and tested before work begins on the next file.
