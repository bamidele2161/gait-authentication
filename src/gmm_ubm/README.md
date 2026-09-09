# GMM–UBM gait verifier

This classical generative verifier learns one Universal Background Model from
all conditions of the development population. An unseen user's GMM means are
MAP-adapted using only their normal ST-control enrollment. Each probe receives
a log-likelihood-ratio score: user model likelihood minus population model
likelihood. Common population/condition variation is therefore discounted.

Thresholds are calibrated per user from held-out ST-control and development
cohort scores at a 1% target FAR. No changed-condition evaluation data is used
for enrollment, adaptation or threshold selection.

```bash
python3 -m src.gmm_ubm.run_experiment --fold 1
```
