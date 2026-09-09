# Domain-adversarial SVM

A standard SVM cannot directly perform neural gradient-reversal training. This
package implements the non-deep equivalent: a development-only condition
classifier identifies linear PCA directions that predict ST/DT and
fatigue/control domains; those directions are suppressed before RBF-SVM
verification.

Each unseen user supplies ST-control enrollment only. Every user verifier uses
the same projection, RBF-SVM configuration and one global score threshold.
Authentication does not receive a condition label. Domain-removal strength and
the global threshold are selected using inner participant-disjoint scores, so
each calibration identity is unseen by its corresponding population projector
and negative cohort.

```bash
python3 -m src.domain_adversarial_svm.run_experiment --fold 1
```
