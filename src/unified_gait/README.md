# Unified condition-robust gait verifier

This is the only active proposed-method implementation on this branch.

## Fixed deployment contract

1. Population development may use all four DUO-GAIT conditions.
2. Outer-fold evaluation participants never train the encoder or select the threshold.
3. A new user enrolls with ST-control walking only.
4. The user receives one template: the mean of their enrollment embeddings.
5. Every probe uses the same cosine-distance scoring rule.
6. One global threshold is selected using development data and then frozen.
7. Authentication receives no condition label and never switches methods.

## Representation

The encoder receives the six sacrum axes plus acceleration and gyroscope
magnitudes. Small 3-D rotations are applied during training to simulate sensor
reattachment. Balanced batches contain every condition for each selected
identity. Batch-hard triplet loss uses the most separated same-person pair and
the closest different-person pair, directly optimizing the failure we observed.

The experiment remains an ablation until all four folds have been evaluated.
