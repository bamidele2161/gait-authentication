# Paper-faithful action-invariant triplet experiment

This package extends the supplied action-invariant IMU-gait method for a
sacrum-mounted DUO-GAIT sensor without changing the unseen-user enrolment
assumption.

- Eight development participants learn the encoder from all four conditions.
- Four separate development participants select the single global threshold.
- Evaluation participants are unseen during encoder development.
- Evaluation users enrol using ST-control only.
- The same encoder, scoring rule and one fold-global threshold are used for
  every evaluation condition.
- Authentication never receives a condition label.
- Each input is an existing two-second, 50%-overlapping sacrum window with all
  three accelerometer and three gyroscope axes.
- A small temporal CNN detects local motion patterns before the layered LSTM.
- One frozen six-channel standardizer is fitted on representation-training
  participants, preventing gyroscope magnitude from overwhelming accelerometer
  information. It is unchanged for threshold and evaluation participants.
- Four candidate negatives are mined per triplet and a semi-hard candidate is
  selected during training.
- Euclidean distances are normalized against development-cohort embeddings.
- One global threshold is selected from pooled development-participant scores
  and then frozen. No evaluation user or evaluation condition calibrates it.
- Each unseen user has one template, built only from ST-control enrolment.
- A reserved part of ST-control enrollment estimates a robust median/MAD score
  scale. The same formula maps every user's scores to a common scale before the
  single global threshold is applied.

The source dataset has six actions, giving `3(6-1)+1 = 16` mining combinations
per anchor. DUO-GAIT has four conditions, giving `3(4-1)+1 = 10` combinations.

Run one fold:

```bash
python3 -m src.paper_triplet.run_experiment --fold 1
```

The main output reports the deployable shared threshold. A clearly marked
post-hoc EER is retained only for comparison with the source paper and must not
be presented as the deployed operating point.
