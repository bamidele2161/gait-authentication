# Statistical Condition Augmentation

This experiment is separate from both the frozen RBF-SVM baseline and the
condition-invariant neural experiments.

Development participants provide condition transformations. An unseen user
provides ST-control gait only. Their normal features are augmented with the
development-derived transformations before a participant-specific verifier is
trained. The unseen user's fatigue and dual-task data remain test-only.
