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

## Objective ablations

`--objective triplet` uses the original hardest-positive/hardest-negative loss.
`--objective supcon` uses every other same-identity window in the balanced batch
as a positive, including positives across all four conditions. The architecture
and evaluation protocol are otherwise identical, and SupCon artifacts are saved
under separate `supcon/fold_*` directories.

`--objective supcon_adv` adds one condition classifier through a gradient
reversal layer. The classifier learns to recognise ST-control, ST-fatigue,
DT-control, and DT-fatigue, while the reversed gradient tells the encoder to
remove condition information. Four-class chance accuracy is 25%. The verifier
still uses one encoder, normal ST-control enrollment, cosine distance, and one
shared development threshold.

```bash
python3 -m src.unified_gait.run_experiment \
  --fold 1 \
  --objective supcon_adv \
  --temperature 0.07 \
  --adversarial-weight 0.1 \
  --epochs 40 \
  --patience 7 \
  --batches-per-epoch 100 \
  --fusion-window 30
```
