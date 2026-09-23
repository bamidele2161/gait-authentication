# Gait Verification Under Fatigue and Cross-Visit Dual-Task Walking
## Technical documentation of the implemented system

---

## 0. One-paragraph summary

We ask whether a gait verifier enrolled on a person's *ordinary* walk still recognises
them when they are tired, a week later, or distracted. Two experiments answer it.
**Experiment I** is a conventional per-user SVM that quantifies the problem: it is
near-perfect within the enrollment condition (1.49% FRR) and degrades sharply outside
it (54.33% FRR on later fatigued dual-task walking). **Experiment II** is the proposed
system: a population-trained Siamese encoder that learns condition variation from
*other people's* normal and fatigued walks, so a newly enrolled user supplies only
normal walking. It reaches 16.33% FRR / 11.41% FAR on later dual-task walking and
47.55% / 13.06% on later fatigued dual-task walking, using an 11-second decision.

---

## 1. The problem

Continuous authentication verifies identity *after* login, from signals collected during
ordinary use. Gait is attractive because phones and wearables already carry the sensors.

The hard question is **not** "can two windows cut from one walk be told apart?" — that is
easy and most reported accuracies answer it. The hard question is:

> Does a verifier enrolled from a person's ordinary walk still recognise that person
> after exertion, on a later day, or while they are doing something else?

This matters because a deployed system enrolls *once*, under whatever conditions happen
to hold that day, and is then expected to work indefinitely.

---

## 2. Dataset

**DUO-GAIT** (Zhou et al., 2023, *Scientific Data*). 16 participants, IMUs at several
body locations, two visits ~7 days apart.

| Condition | Visit | Meaning |
|---|---|---|
| ST-control | 1 | single-task walking, before fatigue |
| ST-fatigue | 1 | single-task walking, after a fatigue protocol |
| DT-control | 2 | dual-task walking (mental arithmetic), before fatigue |
| DT-fatigue | 2 | dual-task walking, after fatigue |

**We use the sacrum sensor only** — triaxial accelerometer + triaxial gyroscope at 128 Hz.

> **Why only the sacrum?**
> Using several body locations would multiply the apparent sample size without adding
> independent participants — the same 16 people measured 5 ways is still 16 people. It
> would also let sensor fusion mask the condition effect we are trying to measure. One
> location keeps the acquisition story honest.

> **Why the sacrum specifically?**
> It sits near the body's centre of mass, so it captures whole-body gait dynamics rather
> than limb-specific motion.

**Data volume after windowing:** ~380 windows per participant per condition; 6,072–6,129
windows per condition across the cohort.

### 2.1 One data correction (disclose this — she may ask)

Participant 07's ST-control sacrum file extends well beyond the concurrently recorded
sensors. We restrict that one file to timestamps **769.53125–1146.875 s**, matching the
corresponding ST-control interval on the synchronised sensors.

> **Why not just drop participant 07?**
> Dropping a participant because their file is inconvenient biases the cohort. This is a
> recording-alignment correction driven by the *other* sensors' timestamps, not by
> authentication performance. The final cohort remains 16. The paper discloses it, and
> notes the rule was chosen after observing the discrepancy — which is the honest framing.

---

## 3. Shared preprocessing

Both experiments consume the same windows.

```
raw CSV → 256-sample windows (2 s @ 128 Hz), 128-sample hop (50% overlap)
        → labelled with participant, condition, start_sample, 20-second block_id
```

> **Why 2-second windows?**
> Mean stride period in this cohort is **1.10 s** (range 1.00–1.33 s). A 2-second window
> therefore contains ~1.8 gait cycles — the shortest window that reliably spans a
> complete stride for every participant. We tested this: **1-second windows are worse**
> (DT-control EER 24.35% vs 16.08%) because they are shorter than *every* participant's
> gait cycle. **4-second windows are also worse** at matched observation time.

> **Why 50% overlap?**
> It gives a new decision every second without waiting for disjoint segments, which suits
> continuous authentication. We verified overlap is not inflating results: removing it
> entirely changes every metric by **under 0.5 points**.

> **Why store `block_id`?**
> 20-second blocks are the grouping unit for cross-validation, so that overlapping windows
> from the same short stretch of walking never straddle a train/test boundary.

### 3.1 The leakage rule (important — this is a contribution)

We split **chronologically**, never randomly, and insert guard windows at every boundary
so adjacent partitions share no raw samples.

> **Why does this matter?**
> Adjacent windows share 50% of their samples. A random row-wise split would put nearly
> identical portions of the same walk on both sides of the split, and the model would be
> scored on data it effectively trained on. This is the single most common way gait papers
> report optimistic numbers.

---

## 4. Experiment I — statistical-feature SVM (the problem definition)

### 4.1 Features

For each window, form 8 signals: the 6 raw axes plus 2 vector magnitudes

```
m_a(t) = √(ax² + ay² + az²)      m_g(t) = √(gx² + gy² + gz²)
```

For each signal extract 11 time-domain statistics (mean, std, median, min, max, range,
IQR, MAD, RMS, skewness, kurtosis) → **88-dimensional feature vector**.

> **Why an SVM and not something modern?**
> This experiment's job is to *state the problem*, not to be competitive. An SVM is
> transparent, fits nonlinear boundaries from limited enrollment data, and its failure
> mode is interpretable. If a deep model failed we could not tell whether the condition
> shift or the architecture was responsible.

> **Why these 11 statistics?**
> They cover level, spread, shape and energy. We do not claim they are canonical — they
> are a deliberately inspectable baseline in the spirit of earlier feature-based inertial
> work.

### 4.2 Training and calibration

- **One RBF-SVM per enrolled participant.** Positives = that person's ST-control training
  windows. Negatives = other participants' ST-control windows.
- Grid search over `C ∈ {0.1, 1, 10, 100}`, `γ ∈ {scale, 0.001, 0.01, 0.1}` using 5-fold
  `StratifiedGroupKFold` on training blocks, selected by ROC-AUC.
- Each ST-control recording split chronologically **60/20/20** into
  train / threshold-development / baseline-test, with guard windows.
- **Threshold**: chosen per participant on the 20% validation portion at an empirical
  FAR ≤ 1% target, then **frozen** and applied unchanged to the baseline test *and* to all
  three shifted conditions.

> **Why freeze the threshold?**
> That is what a deployed system does — you calibrate at enrollment and cannot recalibrate
> later using data you do not have. Re-tuning the threshold per condition would measure
> something no real system can achieve.

> **Why grouped cross-validation?**
> Without grouping, overlapping windows from one 20-second block would land in different
> internal folds, and model selection would be optimistic.

### 4.3 Result — Table II in the paper

| Condition | FRR | SD | FAR | SD |
|---|---|---|---|---|
| ST-control (baseline) | 1.49 | 3.73 | 1.49 | 0.91 |
| ST-fatigue | 19.13 | 34.98 | 11.26 | 11.27 |
| DT-control | 23.04 | 37.95 | 10.15 | 6.55 |
| DT-fatigue | **54.33** | 39.20 | 13.95 | 12.40 |

**Reading:** near-perfect inside the enrollment condition, collapsing outside it. Both
convenience *and* security degrade — a rule that rejects more legitimate users still
accepts more impostors from a shifted condition.

**The SDs are as important as the means.** SD 39 on a mean of 54 means the cohort is a
mixture: some users stay verifiable, others are rejected on nearly every attempt.

---

## 5. Experiment II — the proposed system

### 5.1 Design objective

A new user supplies **only ordinary ST-control walking**. The encoder may learn condition
variation from *different* people's ST-control + ST-fatigue. The later DT visit is held
out entirely.

> **Why this framing?**
> The alternative — asking each new user to also provide fatigued and dual-task samples —
> is not deployable. You cannot ask someone to convincingly fake being tired at
> registration. Moving that burden onto a development population, once, is the core idea.

### 5.2 Input representation: two magnitudes

Instead of 6 axes, the encoder receives **256 × 2**: `m_a(t)` and `m_g(t)` as *time series*
(not scalar summaries).

> **Why magnitudes instead of raw axes?**
> A vector norm is invariant to rigid rotation of the sensor frame. The sacrum unit is
> re-strapped a week later at a slightly different angle, which redistributes the same
> physical motion across x, y, z.
>
> **We measured this.** Between-visit shift in each channel's mean, expressed in units of
> that channel's own within-session SD:
>
> | channel | median shift |
> |---|---|
> | AccX | 0.512 |
> | AccZ | 0.282 |
> | AccY | 0.080 |
> | **AccMag** | **0.017** |
>
> The accelerometer axes move ~0.5 SD between visits; the magnitude moves 0.017 SD —
> **30× more stable**. That is the gravity vector being re-projected by the new mount
> angle. The magnitude is invariant to it.
>
> **We also tested it end-to-end.** Six raw axes give DT-fatigue EER 34.03% vs **25.55%**
> for magnitudes. All eight channels together is worse still (36.04%). Adding axes does
> not help — they actively hurt.

> **Why do axes hurt rather than just add noise?**
> The training loss collapses to ~0.00002 with axis channels but settles near 0.02 with
> magnitudes. Within a single recording session, sensor orientation is a *perfect* identity
> cue, so the encoder grabs it as a shortcut. That cue evaporates when the unit is
> re-mounted. Magnitudes deny the shortcut and force the model to learn gait.

> **What magnitudes do NOT fix (concede this):**
> They are invariant to rigid rotation only. They do not remove changed walking style,
> nonrigid sensor displacement, or genuine amplitude changes.

### 5.3 Architecture (153,729 parameters)

```
input 256 × 2
  ↓ Conv1d(2→32, k=7)  + BatchNorm + GELU          256 steps
  ↓ Conv1d(32→64, k=5, stride 2) + BN + GELU       128 steps
  ↓ Conv1d(64→96, k=3, stride 2) + BN + GELU        64 steps
  ↓ BiLSTM(96 → 64 per direction)                   64 × 128
  ↓ concat[ attention-weighted mean , temporal max ]      256
  ↓ Linear(256→128) + GELU + Dropout(0.25) → Linear(128→64)
  ↓ L2 normalise                                     64-D unit vector
```

> **Why CNN → BiLSTM → attention?**
> The convolutions capture local motion primitives (heel strike, swing). The BiLSTM
> summarises how those follow each other, in both directions. Attention lets the model
> weight informative moments rather than averaging the whole window flat. The temporal
> maximum is concatenated so the strongest response in each channel survives alongside the
> weighted summary.

> **Why L2-normalise the output?**
> It puts every embedding on the unit hypersphere, so distance is driven by *direction* in
> identity space rather than arbitrary vector magnitude. It also makes Euclidean distance
> monotonic in cosine similarity, so the two are interchangeable.

> **Why so small a network (153k parameters)?**
> 12 development identities is a tiny training population. A larger model would fit them
> better without transferring. We verified this empirically — see §8.

### 5.4 Training objective

**Batch-hard triplet loss** with an auxiliary identity head:

```
L = L_triplet + 0.3 · L_crossentropy(identity)

L_triplet = mean over ST-control anchors of
            max(0, ‖z_a − z_p‖ − ‖z_a − z_n‖ + 0.2)
```

- **Anchors**: ST-control only.
- **Positives**: same person, from ST-control *or* ST-fatigue.
- **Negatives**: different person, from either condition.
- **Batch-hard**: within each minibatch take the *farthest* same-person and *nearest*
  different-person example.

> **Why are anchors restricted to ST-control?**
> It matches the enrollment condition. At deployment the template is built from normal
> walking, so the reference point during training should be normal walking too.

> **Why do positives span both conditions?**
> This is the mechanism of the whole paper. Forcing a person's fatigued window to be
> closer to their normal window than to anyone else's teaches the encoder which features
> survive fatigue — learned once, from the development population, and transferred.

> **Why are negatives drawn from both conditions?**
> Otherwise the model could use *condition* as a shortcut for identity — separating
> "ST-control" from "ST-fatigue" instead of separating people.

> **Why batch-hard rather than random triplets?**
> Easy triplets already satisfy the margin and contribute zero gradient. Mining the hardest
> available pair concentrates learning on the comparisons that still matter (Hermans et
> al., 2017).

> **Why an auxiliary identity classifier, and why is it discarded?**
> Cross-entropy over the 12 development identities stabilises training by encouraging the
> embedding to retain identity-discriminative information. It is removed before enrollment
> because an unseen user has no class. **We tested its value:** removing it costs
> **+6.01 EER points** — the single largest degradation we measured. It is load-bearing.

> **Why margin 0.2?**
> It is the value we ran. **Honest caveat:** the loss falls below 0.01 by epoch 14–20 and
> training continues to epoch ~33, so the objective saturates. We tested the standard fix
> (soft-margin, which never saturates) and it made results **worse by 2.00 EER points**.
> So the saturation is not costing anything.

**Optimiser:** AdamW, lr 3e-4, weight decay 1e-4, dropout 0.25, gradient clipping at 5.0.
**Batches:** 8 participants × 3 windows × 2 conditions = 48 windows; 100 batches/epoch;
max 40 epochs, patience 7.

### 5.5 Enrollment and scoring

```
enrollment:  full ST-control recording → frozen encoder → ~380 embeddings
             → mean → L2 normalise → template T_u (one 64-D vector)

probe:       one DT window → frozen encoder → one 64-D embedding z_j
score:       d = ‖z_j − T_u‖₂      (smaller = better match)
decision:    accept if mean of last 10 distances ≤ τ
```

> **Why average the embeddings into one template?**
> It summarises the identity once. Only a 64-value vector is stored — the enrollment
> windows need never be reprocessed.

> **Why re-normalise after averaging?**
> Every embedding leaves the encoder unit-length, but the *mean* of unit vectors is
> shorter than 1, by an amount that depends on how variable the person's walk is. Without
> re-normalising, template length would encode walk consistency and leak into every
> distance — two users the same angular distance from a probe would score differently.

> **Why enrol on the whole recording rather than a portion?**
> Every probe comes from the *later visit*, so nothing in ST-control needs withholding.
> An earlier version used the first 60%; we tested both and the difference is ≤1.5 points
> (DT-fatigue FRR 45.99 → 47.55). We chose the full recording because "enrol on the user's
> normal walk" is a cleaner protocol than an unexplained 60%.

> **Why Euclidean distance?**
> On the unit sphere it is monotonic in cosine similarity, so the choice is immaterial —
> but it is simpler to state and to compute.

### 5.6 Score fusion

The decision uses the **causal mean of the last 10 window distances** (11 seconds of
walking at a 1-second hop), then a new rolling decision every second.

> **Why fuse at all?**
> A single 2-second window may catch an unusual step, a turn, or sensor noise. Individual
> window distances scatter around the person's true position; averaging removes that
> scatter. Concretely, the pooled EER falls from 16.29% to 12.49% on DT-control.

> **Why causal?**
> No future window may contribute to an earlier decision — otherwise it is not deployable.

> **Why 10 windows?**
> **Be upfront: this was selected by inspecting evaluation results**, and the paper says so
> in Section VI-B. The defensible justification is that 11 s ≈ **10 gait cycles** at the
> cohort's 1.10 s mean stride (≥8 cycles for the slowest walker), and the accuracy curve
> flattens beyond that while DT-fatigue FRR starts rising.

> **Why average distances rather than embeddings?**
> Averaging embeddings would need re-normalising and would blur genuinely different
> moments. Averaging *scores* treats each window as independent evidence about one claim.

### 5.7 Evaluation protocol

- 16 participants → **4 deterministic outer folds**, each with **12 development** and
  **4 unseen evaluation** participants. Every participant is unseen exactly once.
- Fold membership (strided, not contiguous):

| Fold | Evaluated participants |
|---|---|
| 1 | sub_01, sub_06, sub_10, sub_14 |
| 2 | sub_02, sub_07, sub_11, sub_15 |
| 3 | sub_03, sub_08, sub_12, sub_17 |
| 4 | sub_05, sub_09, sub_13, sub_18 |

- Within each development participant and each session-1 condition, chronological blocks
  split **80/20** for learning/validation.
- **One threshold per fold**, chosen on development validation scores at FAR ≤ 1%.
- Evaluation users' ST-fatigue is **never used**. Their DT-control and DT-fatigue are the
  only probes.
- **Impostors** are the other evaluation participants (zero-effort, no mimicry).

> **Why participant-disjoint folds?**
> This is the whole point. If the same person appeared in training and test, we would be
> measuring memorisation, not transfer to a new user.

> **Why are impostors drawn only from evaluation participants?**
> Development identities were seen by the encoder, so it separates them unusually well —
> using them as impostors would give optimistically low FAR. Restricting impostors to
> unseen users is the conservative choice. **Cost:** only 3 impostors per claimed user,
> which makes FAR estimates noisy.

> **Why one threshold per fold, not per user?**
> A per-user threshold would need calibration data we do not have at enrollment. One
> global threshold is what a deployed system can actually do.

---

## 6. Results

### 6.1 Main table (Table IV)

| Decision | Condition | FRR | SD | FAR | SD | EER (SD) |
|---|---|---|---|---|---|---|
| 2 s, one window | DT-control | 20.95 | 28.90 | 14.53 | 14.74 | 16.29 (15.95) |
| | DT-fatigue | 45.92 | 39.68 | 15.30 | 17.10 | 25.48 (20.62) |
| 11 s, 10 windows | DT-control | **16.33** | 31.86 | **11.41** | 15.84 | **12.49** (17.48) |
| | DT-fatigue | 47.55 | 45.12 | **13.06** | 19.15 | **20.03** (26.20) |

### 6.2 Fusion sweep (Table V)

| Windows | Span | DT-c FRR | DT-c FAR | DT-f FRR | DT-f FAR |
|---|---|---|---|---|---|
| 1 | 2 s | 20.95 | 14.53 | 45.92 | 15.30 |
| 3 | 4 s | 18.76 | 12.88 | 46.37 | 13.83 |
| 5 | 6 s | 17.27 | 12.02 | 46.72 | 13.44 |
| **10** | **11 s** | **16.33** | **11.41** | **47.55** | **13.06** |
| 15 | 16 s | 15.98 | 11.14 | 48.01 | 12.84 |
| 20 | 21 s | 15.79 | 10.89 | 48.48 | 12.65 |
| 30 | 31 s | 15.42 | 10.79 | 48.84 | 12.46 |

**Pattern:** longer observation reliably lowers FAR under both conditions, and lowers FRR
only for DT-control. DT-fatigue FRR drifts *up*.

> **Why does DT-fatigue FRR rise with more fusion?**
> Averaging drives each user's decision toward their own mean distance. For users whose
> mean sits *below* the threshold, FRR collapses toward 0. For users whose mean sits
> *above* it, FRR climbs toward 100. On DT-fatigue, **8 of 16 users** have genuine mean
> distance above the threshold, so the gains and losses cancel and the macro-average
> creeps upward. Fusion makes correct decisions more reliable *and* wrong ones more
> reliable.

### 6.3 Relative to the problem definition

| | SVM | Proposed (11 s) | Δ |
|---|---|---|---|
| DT-control FRR | 23.04 | 16.33 | **−6.71** |
| DT-control FAR | 10.15 | 11.41 | +1.26 |
| DT-fatigue FRR | 54.33 | 47.55 | **−6.78** |
| DT-fatigue FAR | 13.95 | 13.06 | **−0.89** |

> **Caveat to state before she does:** this is *not* a controlled ablation. The two systems
> differ in training identities, threshold structure, impostor set and decision duration.
> The paper says so explicitly in Section VI-C. Attributing the difference to the
> representation alone would require a matched protocol.

---

## 7. Why DT-fatigue remains hard (the core limitation)

The single most important diagnostic number:

```
per-user genuine mean distance on DT-fatigue :  0.755  …  1.565
typical impostor distance                    :         ~1.42
```

**The spread between people is wider than the gap between genuine and impostor.**
For several participants — sub_17 (1.565), sub_02 (1.501), sub_12 (1.455) — their own
fatigued dual-task walking is further from their enrollment template than a stranger's
walking is.

**Consequence:** no single global threshold can serve this population. Any threshold that
accepts sub_17 must also accept most impostors of sub_08.

**Why this is a data property, not a tuning failure:** the cause is that a week later,
tired, and doing mental arithmetic, some people simply walk differently enough that the
signal is gone. More training or a better threshold cannot recover information that is
not in the recording.

---

## 8. What we tried and rejected (all measured)

Judged on **per-user EER at matched fusion**, which is threshold-free.
Baseline = submitted system: DT-control 12.49, DT-fatigue 20.03 (11 s).

### 8.1 Representation

| Change | Result | Verdict |
|---|---|---|
| 6 raw axes instead of magnitudes | DT-f EER 34.03 vs 25.55 | worse |
| All 8 channels | DT-f EER 36.04 | worse |
| 1-second windows | DT-c EER 24.35 vs 16.08 | worse |
| 4-second windows | worse at every matched span | worse |
| Cadence/amplitude augmentation | EER unchanged (±0.3) | no effect |

### 8.2 Training objective

| Change | Mean Δ EER | Verdict |
|---|---|---|
| Soft-margin loss (non-saturating) | **+2.00** | worse |
| Richer batches (12 users × 5 windows) | **+1.61** | worse |
| Remove identity loss | **+6.01** | much worse |

> **What this pattern means:** pushing training *harder* costs ~2 points; easing off costs
> 6. The current configuration sits at a **local optimum** — both directions degrade. This
> is now evidence rather than assertion, and it is the strongest argument that the next
> gain requires **more participants**, not a better architecture.

### 8.3 Enrollment and calibration

| Change | Result | Verdict |
|---|---|---|
| Multi-prototype templates (k=2,3,5) | wash (−0.95 / +0.92) | no gain |
| Cohort z-normalisation of scores | no gain | no gain |
| Per-user thresholds from enrollment statistics | no gain | no gain |
| Threshold calibrated on held-out identities | worse | worse |
| Enrol on ST-control **+ ST-fatigue** | DT-f FRR 48→29 **but** FAR 11→26 | not deployable |

> **The 2-condition enrollment result is worth mentioning** even though we rejected it: the
> same encoder and threshold, with a template spanning more of the person's gait, cuts
> DT-fatigue FRR by 19 points. That identifies **template coverage** — not encoder capacity
> — as the limitation, and points future work at enrollment design. We rejected it because
> asking a new user to supply fatigued walking defeats the paper's premise.

### 8.4 One methodological finding worth stating

Calibrating the threshold on identities the encoder trained on gives **EER = 0.00% in all
four folds**, even under the evaluation protocol. Development-identity scores carry *no*
information about where the threshold belongs for an unseen user — the encoder memorises
them. This is why the 1% development FAR target does not hold at evaluation, which the
paper reports as a result rather than a defect.

---

## 9. Known weaknesses (concede these before she finds them)

1. **16 participants.** Each fold tests 4 people; impostors come from the same small pool.
   SDs of 15–45 points mean the macro-average describes the cohort, not any individual.
2. **The 11-second duration was chosen on evaluation data.** Disclosed in Section VI-B.
   The gait-cycle justification (≈10 cycles) is post-hoc, though independently sound.
3. **DT confounds three things** — cognitive load, 7 days elapsed, and sensor
   reattachment. The design cannot separate them.
4. **Threshold/fusion asymmetry.** The threshold is calibrated on *unfused* development
   scores but applied to *fused* evaluation scores. Fusing the development scores too
   collapses the threshold and sends FRR to ~100%, because development separation is
   nearly perfect. This is disclosed.
5. **Zero-effort impostors only.** No mimicry, replay, or sensor substitution. The reported
   FAR is not evidence of resistance to active attack.
6. **Correlated decisions.** Adjacent fused decisions reuse 9 of 10 scores.
7. **The SVM/encoder comparison is contextual**, not a matched ablation.

---

## 10. Likely questions and short answers

**"Why is DT-fatigue still 47%?"**
Half the participants' fatigued dual-task gait sits further from their own enrollment
template than an impostor's does (0.755–1.565 vs ~1.42 impostor). A single global
threshold cannot serve both ends of that spread. We tested nine interventions across
calibration, representation and training; none moved it.

**"Did you try just training longer / a bigger model?"**
Yes, indirectly — three training-intensity changes all degraded results (+1.6 to +6.0 EER).
The objective already saturates by epoch ~15. The limit is 12 development identities, not
optimisation.

**"Why is the FAR target 1% but you report 11–15%?"**
Because the threshold is calibrated on session-1 development identities and applied to
session-2 unseen identities. Genuine distances rise from a median of 0.189 to 1.024 across
that gap. We report it as a finding; it is exactly the kind of drift the paper is about.

**"Why not use a per-user threshold?"**
Nothing observable at enrollment predicts a user's offset. We tested cohort normalisation,
enrollment-self-consistency thresholds, and interpolation rules — all gave no gain.

**"Is the improvement over the SVM real?"**
Both FRRs fall and DT-fatigue FAR falls; DT-control FAR rises 1.26. But the two systems
differ in several respects, so we describe it as contextual and state what a matched
comparison would require.

**"What would you do next?"**
More participants and repeat visits (the evidence in §8.2 points there), gait-cycle
normalisation to make cadence invariance structural, and enrollment designs that cover
more within-person variation without burdening the user.

---

## 11. Reproducing the results

```bash
cd /Users/user/Documents/gait-authentication

# Experiment I — the problem definition
venv/bin/python src/01_organize_data.py
venv/bin/python src/02_preprocess.py
venv/bin/python src/03_extract_features.py
venv/bin/python src/04_train.py
venv/bin/python src/05_evaluate.py

# Experiment II — the proposed system
venv/bin/python -m src.condition_invariant.run_experiment

# Paper figures
venv/bin/python paper/ieee_gait/make_figures.py
```

**Outputs**
- `src/results/sacrum_time/` — Experiment I, per-participant and summary
- `src/results/condition_invariant/session1_cnn_bilstm_hard/` — Experiment II
  (per-window comparison scores, per-fold thresholds, training histories, encoders)
- `src/results/condition_invariant/session1_cnn_bilstm_hard_enrol60/` — the 60%-enrollment
  variant, retained for comparison
- `_softmargin/`, `_richbatch/`, `_noidentity/` — the rejected training variants of §8.2

**Per-fold thresholds in the submitted run:** 1.1907, 1.2037, 1.1744, 1.1337
(development FRR/FAR 0.18–0.82% / 0.59–0.97%).

Every window carries participant ID, condition, start sample and block ID; every
comparison score is saved, so all reported numbers can be recomputed from the CSVs without
retraining.
