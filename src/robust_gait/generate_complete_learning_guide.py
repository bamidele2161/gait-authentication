"""Generate a beginner-to-expert learning guide for the proposed verifier."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, Spacer

from src.robust_gait.generate_implementation_guide import (
    BROWN,
    CREAM,
    GOLD,
    GREY,
    INK,
    PALE_BLUE,
    PALE_GREEN,
    PALE_RED,
    GuideDoc,
    bullet,
    p,
    ref,
    section,
    table,
    two_col,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output" / "pdf" / "proposed_solution_complete_learning_guide.pdf"
FIGURES = ROOT / "paper" / "figures"
SIMPLE_ARCH = FIGURES / "proposed_method_simple_architecture.png"
TECHNICAL_ARCH = FIGURES / "proposed_method_architecture.png"


def q(question, answer):
    return [p(question, "Smallx"), p(answer, "Smallx")]


def misconception(wrong, correct):
    return table([
        [p("Common misunderstanding", "Smallx"), p("Correct explanation", "Smallx")],
        [p(wrong, "Smallx"), p(correct, "Smallx")],
    ], widths=[4.85*inch, 4.85*inch], font=8.5)


def build_story():
    s = []

    # Cover
    s += [Spacer(1, 0.45*inch), p("The Proposed Gait Authentication System", "GuideTitle")]
    s += [p("A first-day machine-learning lesson, followed by the full technical implementation", "GuideSubtitle")]
    s += [Spacer(1, 0.30*inch)]
    s += [table([
        [p("What you will learn", "H2x"), p("How data moves from nine body sensors to an accept/reject decision, what every machine-learning term means, why each design choice was made, and where each part lives in the code.")],
        [p("Two explanation levels", "H2x"), p("Every major stage is explained first in simple English and then in technical language suitable for a supervisor or machine-learning researcher.")],
        [p("Important correction", "H2x"), p("The method does not change the signal to 6 Hz or 32 Hz. Six hertz is a filter cutoff used only for stride detection. Mean pooling changes 128 normalized gait-phase points into 32 phase bins, not a 32 Hz signal.")],
        [p("Final goal", "H2x"), p("You should be able to teach the method to a non-expert, defend it to an expert, and point to the exact implementation file.")],
    ], widths=[2.25*inch, 7.55*inch], header=False, font=9.4)]
    s += [Spacer(1, .25*inch), p("Bamidele Akinyemi - implementation learning guide", "GuideSubtitle")]
    s.append(PageBreak())

    # Learning roadmap
    s += [section("Before we begin: the four roles in this experiment")]
    s += [p("Most confusion comes from mixing up development, training, enrolment, and testing. They are related, but they are not the same thing.")]
    roles = [
        ["Role", "Simple meaning", "Who/data is used here"],
        ["Development", "The engineering classroom where system-wide choices are learned.", "12 participants in each outer fold."],
        ["Model training", "Teaching one classifier what 'this user' and 'not this user' look like.", "New user's ST-control positives plus development-user negatives."],
        ["Enrolment", "Registering an authorized new user before authentication.", "That user's complete normal ST-control walk only."],
        ["Testing/authentication", "Asking whether a later walk belongs to the claimed user.", "DT-control and DT-fatigue probes from the four held-out users."],
    ]
    s += [table([[p(x,"Smallx") for x in r] for r in roles], widths=[1.55*inch,4.0*inch,4.25*inch], font=8.5)]
    s += [p("Key idea", "H2x"), p("An evaluation user is unseen during system development, but their normal enrolment is intentionally used to create their personal verifier. That is how biometric authentication works; it is not leakage.", "Callout")]
    s.append(PageBreak())

    # Simple architecture
    s += [section("Simple-level architecture", 1)]
    s += [p("Read this diagram as a story. The top row prepares walking data. The middle row teaches system-wide rules using development people. The bottom row enrols and authenticates a new person.")]
    s += [Image(str(SIMPLE_ARCH), width=9.8*inch, height=6.1*inch)]
    s.append(PageBreak())

    # Technical architecture
    s += [section("Senior-level technical architecture", 2)]
    s += [p("This is the same system written in the language an ML or biometric researcher expects.")]
    s += [Image(str(TECHNICAL_ARCH), width=9.8*inch, height=6.0*inch)]
    s.append(PageBreak())

    # Vocabulary
    s += [section("The signal vocabulary - samples, axes, channels, and strides", 3)]
    vocabulary = [
        ["Word", "Beginner explanation", "Example in this project"],
        ["Sensor/IMU", "A small device that measures movement.", "Sacrum, left shank, right wrist, head, etc."],
        ["Axis", "One direction of movement or rotation.", "AccX or GyrZ."],
        ["Channel", "One continuous stream of measurements.", "Left-foot AccY is one channel."],
        ["Sample", "One measurement taken at one moment.", "At 128 Hz, one channel records 128 samples each second."],
        ["Recording", "A long continuous walking signal.", "One participant's ST-control session."],
        ["Stride", "A full gait cycle, usually from one left-leg mid-swing to the next.", "A variable-length segment retained between 0.7 and 1.8 seconds."],
        ["Feature", "A number given to a model so it can learn a pattern.", "One pooled GyrX value at a particular gait phase."],
        ["Score", "How strongly a model thinks the probe matches a claim.", "Higher logistic decision score means more user-like."],
    ]
    s += [table([[p(x,"Smallx") for x in r] for r in vocabulary], widths=[1.3*inch,4.1*inch,4.4*inch], font=8.1)]
    s += [p("Nine sensors x six axes = 54 channels measured at each original timestamp.", "Callout")]
    s += [ref("src/robust_gait/data.py", "26-47", "sensor locations, raw axes, conditions, and original sample rate")]
    s.append(PageBreak())

    # Sampling and filtering critical
    s += [section("Critical timing lesson: 128 Hz, 6 Hz, 128 phase points, and 32 bins", 4)]
    timing = [
        ["Number", "What it actually means", "What it does NOT mean"],
        ["128 Hz", "The original sensors take 128 measurements per second.", "It is not the number of features."],
        ["6 Hz cutoff", "The stride-detector's low-pass filter keeps slow gait content and suppresses faster wiggles.", "The recordings are not resampled to 6 samples per second."],
        ["128 phase points", "Every detected stride is interpolated onto positions 0 through 100% of one gait cycle.", "After this operation the axis is normalized gait phase, not ordinary seconds at 128 Hz."],
        ["32 phase bins", "Mean pooling averages every four neighboring phase points: 128 / 4 = 32.", "The model input is not described as a 32 Hz signal."],
    ]
    s += [table([[p(x,"Smallx") for x in r] for r in timing], widths=[1.55*inch,4.65*inch,3.6*inch], font=8.3)]
    s += [misconception(
        "'We filtered the data to 6 Hz, resampled it to 128 Hz, and then downsampled it to 32 Hz.'",
        "'The sensors were originally sampled at 128 Hz. A 6 Hz low-pass filter was used only on the left-shank gyro magnitude to locate stride peaks. Each detected stride was then mapped to 128 normalized gait-phase points. Mean pooling converted those 128 phase points to 32 phase bins per axis.'",
    )]
    s += [p("This wording matters. A technical examiner may immediately challenge the method if phase normalization is incorrectly described as ordinary frequency resampling.", "Callout")]
    s.append(PageBreak())

    # Input and QC
    s += [section("Step 1: read and align the nine sensors", 5)]
    s += [two_col([
        p("Simple explanation", "H2x"),
        p("Imagine nine cameras filming the same walk from different body locations. Before combining them, frame 100 from every camera must describe the same moment. The implementation reads all nine IMU files, applies the known recording trim, checks for broken values, and uses their common synchronized duration."),
        p("What enters the next step", "H2x"),
        p("A table with 54 movement columns: six axes from each of nine locations."),
    ], [
        p("Technical explanation", "H2x"),
        bullet("Read GyrX/Y/Z and AccX/Y/Z as finite numeric arrays."),
        bullet("Apply KNOWN_TIME_TRIMS before segmentation."),
        bullet("Use the minimum recording length across sensors."),
        bullet("Concatenate derived magnitude channels and raw-axis channels while preserving a fixed location order."),
        p("Why", "H2x"),
        p("Cross-sensor fusion is meaningful only when channels are temporally aligned. The trim is defined independently of authentication outcomes."),
    ])]
    s += [ref("src/robust_gait/data.py", "21-47, 62-77, 112-128", "location order, raw loading, validity checks, shared length, and stacking")]
    s += [ref("src/utils.py", "49-58", "pre-declared sub_07 ST-control correction")]
    s.append(PageBreak())

    # Magnitude
    s += [section("Step 2: calculate movement magnitude for stride finding", 6)]
    s += [p("Each gyroscope has three rotational directions. The magnitude combines them into one 'overall rotation strength' number:")]
    s += [p("gyro_magnitude = sqrt(GyrX^2 + GyrY^2 + GyrZ^2)", "Codex")]
    s += [two_col([
        p("Numerical example", "H2x"),
        p("If GyrX=3, GyrY=4, and GyrZ=0, then magnitude = sqrt(9+16+0) = 5."),
        p("Why the left shank", "H2x"),
        p("The shank has a strong, repeating mid-swing motion. Its gyro-magnitude peaks make convenient stride landmarks."),
    ], [
        p("What magnitude loses", "H2x"),
        p("Magnitude says how strong movement is, but not its direction. That is acceptable for finding boundaries; the classifier later returns to the six raw directional axes."),
        p("Do not say", "H2x"),
        p("Do not say the final system uses only magnitude features. Magnitude is an internal helper for segmentation and sensor-quality checks."),
    ])]
    s += [ref("src/robust_gait/data.py", "87-103", "magnitude formula, filter, and peak detector")]
    s.append(PageBreak())

    # Filter and peaks
    s += [section("Step 3: smooth the detector signal and find stride peaks", 7)]
    s += [two_col([
        p("What is a low-pass filter?", "H2x"),
        p("A walking signal contains the large repeated gait pattern plus small fast vibrations. A low-pass filter behaves like a smoothing rule: slow changes pass through, while very fast wiggles are reduced."),
        p("Why 6 Hz?", "H2x"),
        p("Normal gait events occur at relatively low frequencies. A 6 Hz cutoff preserves the main gait shape while suppressing high-frequency sensor noise. It is a defensible signal-processing choice, not a new sampling rate."),
    ], [
        p("How peaks are accepted", "H2x"),
        bullet("Two peaks must be at least 0.6 seconds apart."),
        bullet("A peak must stand out by at least 0.5 times the signal standard deviation."),
        bullet("The interval between consecutive peaks must last 0.7 to 1.8 seconds."),
        p("Why these rules", "H2x"),
        p("They reject tiny noise peaks, physically implausible cycles, and accidental double detections."),
    ])]
    s += [p("The filter changes only the temporary left-shank detector signal. It does not replace the raw axes that will be classified.", "Callout")]
    s += [ref("src/robust_gait/data.py", "94-103, 131-139", "Butterworth filter, peak constraints, and stride-duration rule")]
    s.append(PageBreak())

    # resampling
    s += [section("Step 4: turn variable-duration strides into 128 phase points", 8)]
    s += [p("Suppose one stride lasts 1.00 second and another lasts 1.25 seconds. At 128 Hz they contain roughly 128 and 160 original samples. A classifier needs corresponding features to line up.")]
    example = [
        ["Before normalization", "After normalization"],
        ["Stride A: about 128 original samples", "128 points representing 0%, 0.8%, ..., 100% of the stride"],
        ["Stride B: about 160 original samples", "128 points representing the same normalized gait phases"],
    ]
    s += [table([[p(x,"Smallx") for x in r] for r in example], widths=[4.85*inch,4.85*inch], font=8.8)]
    s += [two_col([
        p("Simple explanation", "H2x"),
        p("It is like resizing two photographs to the same width so that the left edge, middle, and right edge can be compared. Here the horizontal axis is one stride rather than pixels."),
    ], [
        p("Technical explanation", "H2x"),
        p("Linear interpolation maps each channel from its original sample positions onto 128 equally spaced positions on the normalized interval [0,1]. The output describes gait phase rather than a fixed number of seconds."),
    ])]
    s += [misconception(
        "'Every stride is resampled to 128 Hz.'",
        "'Every stride is resampled to 128 normalized phase points. Because different strides cover different durations, these points should not be treated as a universal 128 Hz time series.'",
    )]
    s += [ref("src/robust_gait/data.py", "105-109, 131-145", "linear interpolation and fixed-length stride assembly")]
    s.append(PageBreak())

    # pooling
    s += [section("Step 5: mean pooling - what it is and why we use it", 9)]
    s += [p("Mean pooling simply replaces a small group of neighboring values with their average.")]
    pool = [
        ["Four neighboring phase values", "Average", "One pooled value"],
        ["2, 4, 6, 8", "(2+4+6+8)/4", "5"],
        ["10, 9, 11, 10", "(10+9+11+10)/4", "10"],
    ]
    s += [table(pool, widths=[3.6*inch,3.0*inch,3.1*inch], font=9)]
    s += [two_col([
        p("What happens to one sensor", "H2x"),
        bullet("Start: 128 phase points x 6 axes."),
        bullet("Group neighboring phase points four at a time."),
        bullet("128 / 4 = 32 phase bins per axis."),
        bullet("32 bins x 6 axes = 192 model inputs for that sensor."),
    ], [
        p("Why it helps", "H2x"),
        bullet("Reduces tiny point-to-point noise."),
        bullet("Reduces the number of inputs, making a small classifier more stable."),
        bullet("Uses every point in each group, unlike selecting every fourth point."),
        bullet("Preserves coarse gait shape across the whole normalized stride."),
    ])]
    s += [p("Correct wording: '128 normalized phase points were mean-pooled into 32 phase bins per axis.' Do not call the result 32 Hz.", "Callout")]
    s += [ref("src/robust_gait/verifier.py", "105-131", "mean pooling, flattening, and the resulting per-sensor feature vector")]
    s.append(PageBreak())

    # folds
    s += [section("Step 6: split people into development and evaluation folds", 10)]
    s += [p("We are testing whether the system design transfers to people who were not used to choose its general rules.")]
    folds = [
        ["Fold", "12 development people", "4 evaluated people"],
        ["1", "Everyone except the four on the right", "sub_01, sub_06, sub_10, sub_14"],
        ["2", "Everyone except the four on the right", "sub_02, sub_07, sub_11, sub_15"],
        ["3", "Everyone except the four on the right", "sub_03, sub_08, sub_12, sub_17"],
        ["4", "Everyone except the four on the right", "sub_05, sub_09, sub_13, sub_18"],
    ]
    s += [table(folds, widths=[.7*inch,4.4*inch,4.6*inch], font=8.6)]
    s += [two_col([
        p("Simple explanation", "H2x"),
        p("It is like four examinations. In each examination, four students are kept outside while the rules are prepared with the other twelve. Then the frozen rules are tested on the four outside students."),
    ], [
        p("Technical explanation", "H2x"),
        p("Four-fold participant-disjoint cross-validation. Development and evaluation identity sets have an empty intersection within each fold, and every participant appears in the evaluation set exactly once."),
    ])]
    s += [p("This is not one permanent external test cohort: each person is development data in three folds and evaluation data in one fold.", "Callout")]
    s += [ref("src/robust_gait/protocol.py", "24-74", "fold construction, disjointness assertions, and evaluated-once assertion")]
    s.append(PageBreak())

    # background
    s += [section("Step 7: build the background - examples of 'not this user'", 11)]
    s += [two_col([
        p("Simple explanation", "H2x"),
        p("If Bamidele enrols, the model needs examples of Bamidele and examples of people who are not Bamidele. The other development participants provide the 'not Bamidele' class."),
        p("Data used", "H2x"),
        p("Up to 240 ST-control strides and 240 ST-fatigue strides per development participant."),
    ], [
        p("Why include fatigued strangers?", "H2x"),
        p("An impostor can also be tired. Training against only normal strangers may teach the model that fatigue itself is evidence for the genuine user. Development-user fatigue broadens the negative class without asking the new user to enrol while fatigued."),
        p("Technical term", "H2x"),
        p("This is population background modelling for a one-vs-background verifier."),
    ])]
    s += [ref("src/robust_gait/run_experiment.py", "42-43, 85-91", "training conditions and balanced background sampling")]
    s.append(PageBreak())

    # calibration mirror
    s += [section("Step 8: pretend some development people are new users", 12)]
    s += [p("Before touching the four evaluated people, the system must learn score-normalization references and a pass mark. It rehearses the full process using only development participants.")]
    mirror = [
        ["Development rehearsal", "What happens"],
        ["Choose two pseudo-new users", "They temporarily behave like unseen customers."],
        ["Enrol them", "Use their complete ST-control recording."],
        ["Challenge them", "Use their DT-control and DT-fatigue as later session-2 probes."],
        ["Build background", "Use the other ten development participants."],
        ["Repeat", "Each development pair gets a turn as pseudo-new users."],
        ["Learn fixed rules", "Obtain normalization distributions and the minimax threshold."],
    ]
    s += [table([[p(x,"Smallx") for x in r] for r in mirror], widths=[3.0*inch,6.7*inch], font=8.7)]
    s += [p("This means the method knows from other people that session-2 dual-task and fatigue can change gait. It still does not see an evaluation person's DT data during calibration.", "Callout")]
    s += [ref("src/robust_gait/run_experiment.py", "272-301", "pseudo-unseen grouping, normal enrolment, session-2 probes, and development-only streams")]
    s.append(PageBreak())

    # enrollment
    s += [section("Step 9: enrol a genuinely new user using normal walking only", 13)]
    s += [two_col([
        p("What the person provides", "H2x"),
        p("Their complete ST-control recording. They are not asked to exercise, become fatigued, or perform a cognitive task during enrolment."),
        p("What is built", "H2x"),
        p("Up to nine personal binary classifiers - one for each working sensor."),
    ], [
        p("Why this is not leakage", "H2x"),
        p("Authentication requires an authorized reference. Enrolment is intentionally allowed. Leakage would occur if the later DT-control or DT-fatigue probe, or its true label, were used to train the user's model or choose their threshold."),
        p("Deployment analogy", "H2x"),
        p("A phone stores your fingerprint during setup; later fingerprint attempts are tests. Normal gait enrolment plays the same role here."),
    ])]
    s += [ref("src/robust_gait/run_experiment.py", "318-332", "outer-fold calibration, evaluation-user enrolment, session-2 probes, and scoring")]
    s.append(PageBreak())

    # fault gate and augmentation
    s += [section("Step 10: remove an obviously inactive sensor and expand enrolment", 14)]
    s += [two_col([
        p("Sensor fault gate", "H2x"),
        p("The median gyroscope magnitude of each enrolment sensor is compared with the development-population level. If it is below 25% of the reference, that location is dropped for that user."),
        p("Why", "H2x"),
        p("A detached or nearly motionless sensor should not be allowed to distort the combined decision."),
        p("What it does not do", "H2x"),
        p("It does not inspect the later DT probe to choose convenient sensors."),
    ], [
        p("Enrolment augmentation", "H2x"),
        bullet("Keep every real normal stride."),
        bullet("Create four additional copies."),
        bullet("Change motion amplitude by up to +/-20%."),
        bullet("Smoothly shift within-stride timing by up to +/-10%."),
        p("Why", "H2x"),
        p("The personal model sees a small neighborhood around normal gait instead of memorizing exact enrolment waveforms. These are plausible variations, not claimed perfect simulations of fatigue."),
    ])]
    s += [ref("src/robust_gait/run_experiment.py", "157-176, 195-201", "development-referenced sensor-quality gate")]
    s += [ref("src/robust_gait/verifier.py", "42-89, 147-150", "amplitude and timing augmentation applied during fitting")]
    s.append(PageBreak())

    # scaling
    s += [section("Step 11: put feature values on a comparable scale", 15)]
    s += [p("Different axes have different numerical ranges. A feature with values around 500 could dominate a feature with values around 0.5 even when it is not more informative.")]
    s += [p("scaled_value = (value - background_mean) / background_standard_deviation", "Codex")]
    scale_example = [
        ["Background feature statistics", "Probe value", "Scaled probe"],
        ["mean=10, standard deviation=2", "14", "(14-10)/2 = 2"],
        ["mean=1000, standard deviation=100", "1200", "(1200-1000)/100 = 2"],
    ]
    s += [table(scale_example, widths=[3.3*inch,2.2*inch,4.2*inch], font=9)]
    s += [p("Although the raw units differ, both probe values are two background standard deviations above their background mean.")]
    s += [two_col([
        p("Why background-only statistics", "H2x"),
        p("Every claimed user is measured relative to a common development population. The scaler does not depend on the later probe."),
    ], [
        p("Is this the same as S-norm?", "H2x"),
        p("No. Feature standardization happens before each logistic model. S-norm happens later to the model's output score."),
    ])]
    s += [ref("src/robust_gait/verifier.py", "151-169", "background-only feature mean/SD and classifier fit")]
    s.append(PageBreak())

    # logistic regression
    s += [section("Step 12: teach each sensor's logistic-regression specialist", 16)]
    s += [p("A classifier is simply a rule learned from examples. Here each sensor gets its own binary classifier: genuine user versus background people.")]
    s += [two_col([
        p("Training examples", "H2x"),
        bullet("Label 1: the claimed user's real and augmented ST-control strides."),
        bullet("Label 0: ST-control and ST-fatigue strides from development participants."),
        p("What the model learns", "H2x"),
        p("It learns one weight for each of the 192 pooled sensor features. Positive weights support the user; negative weights oppose the user."),
        p("score = bias + w1*x1 + w2*x2 + ... + w192*x192", "Codex"),
    ], [
        p("Why logistic regression", "H2x"),
        p("It is a small, regularized model. The dataset has many strides but only 16 independent people, so a large deep network can memorize development identities. A user-specific linear model directly uses the person's enrolment and is easier to interpret and stabilize."),
        p("L2 regularization", "H2x"),
        p("L2 discourages extremely large weights. Think of it as telling the model: 'use several moderate clues instead of betting everything on one feature.' The strength is controlled by C=0.3."),
        p("Balanced class weights", "H2x"),
        p("They stop the larger negative class from winning simply because it has more examples."),
    ])]
    s += [p("The decision_function output is used as a comparison score. Higher means the stride lies more on the claimed-user side of the learned boundary.", "Callout")]
    s += [ref("src/robust_gait/verifier.py", "134-181", "one model per sensor, labels, logistic regression, regularization, and score calculation")]
    s.append(PageBreak())

    # late fusion
    s += [section("Step 13: let the sensor specialists vote using the median", 17)]
    s += [p("For one probe stride, every working sensor model outputs one score. The system combines them after scoring; this is called late fusion.")]
    votes = [
        ["Nine sensor scores", "Mean", "Median"],
        ["1.2, 1.0, 1.1, 0.9, 1.3, 1.1, 1.0, 1.2, -8.0", "0.09", "1.10"],
    ]
    s += [table(votes, widths=[6.2*inch,1.7*inch,1.7*inch], font=9)]
    s += [two_col([
        p("Simple explanation", "H2x"),
        p("Eight sensors agree near 1.1, but one broken or shifted sensor says -8. The average is badly pulled down. The median takes the middle vote and remains close to the agreement."),
    ], [
        p("Technical explanation", "H2x"),
        p("Median late fusion is a robust estimator with resistance to extreme sensor-score outliers. Independent per-location models also allow the number of active sensors to vary by user."),
    ])]
    s += [ref("src/robust_gait/verifier.py", "173-187", "per-sensor scoring and median/mean late-fusion alternatives")]
    s.append(PageBreak())

    # z norm
    s += [section("Step 14: Z-norm - understand the claimed model's score scale", 18)]
    s += [p("Different personal models may naturally produce different score ranges. A raw score of 2 may be excellent for one model but ordinary for another.")]
    s += [p("Z = (claimed-model score - unseen-impostor mean) / unseen-impostor SD", "Codex")]
    s += [two_col([
        p("Simple explanation", "H2x"),
        p("Ask: 'Compared with strangers scored by this same claimed-user model, how unusual is the current score?' A Z-score of 3 means it is three stranger-standard-deviations above the stranger average."),
    ], [
        p("How leakage is avoided", "H2x"),
        p("Development background identities are rotated into three cohorts. When one cohort estimates impostor mean/SD, its identities are excluded from the negative training data of that corresponding model."),
    ])]
    s += [ref("src/robust_gait/run_experiment.py", "94-136", "rotated Z-norm cohort exclusion, scoring, mean, and standard deviation")]
    s.append(PageBreak())

    # t/s norm
    s += [section("Step 15: T-norm and S-norm - correct probe difficulty", 19)]
    s += [two_col([
        p("T-norm", "H2x"),
        p("The same probe is scored against several development cohort models. Ask: 'Does this probe score highly only for the claimed person, or does it look acceptable to many people?'"),
        p("T = (claim score - cohort-model mean for this probe) / cohort SD", "Codex"),
        p("No true test label is needed. This is test-time score processing."),
    ], [
        p("S-norm", "H2x"),
        p("S-norm combines both corrections:") ,
        p("S = (Z + T) / 2", "Codex"),
        p("Z-norm corrects claimed-model scale. T-norm corrects probe difficulty. Their average gives the normalized comparison score used downstream."),
    ])]
    s += [misconception(
        "'S-norm is feature scaling.'",
        "'Feature standardization scales the 192 classifier inputs. S-norm standardizes the classifier's output score using development cohort behavior.'",
    )]
    s += [ref("src/robust_gait/run_experiment.py", "139-154, 178-220", "T-norm cohort models, per-probe normalization, and S-norm average")]
    s.append(PageBreak())

    # temporal
    s += [section("Step 16: combine ten consecutive strides", 20)]
    s += [p("One stride can be strange because the person turns, stumbles, adjusts clothing, or a sensor shakes. Continuous authentication should not make a serious decision from one moment.")]
    temporal = [
        ["Ten normalized stride scores", "Average decision score"],
        ["1.1, 1.2, 1.0, 1.1, -0.8, 1.2, 1.1, 1.0, 1.2, 1.1", "1.02"],
    ]
    s += [table(temporal, widths=[7.2*inch,2.5*inch], font=9)]
    s += [p("The one unusual -0.8 stride affects the result, but it does not dominate nine consistent strides.")]
    s += [two_col([
        p("Simple explanation", "H2x"),
        p("The system waits for several pieces of evidence before deciding, like listening to ten words rather than judging a speaker from one syllable."),
    ], [
        p("Technical explanation", "H2x"),
        p("A causal moving average fuses ten consecutive stride scores. In this dataset ten strides correspond to approximately 12 seconds. Longer evidence improves stability but increases latency."),
    ])]
    s += [ref("src/robust_gait/run_experiment.py", "77-82", "causal moving-average score fusion")]
    s.append(PageBreak())

    # threshold
    s += [section("Step 17: choose the pass mark with minimax calibration", 21)]
    s += [p("A score is continuous. Authentication needs a pass mark, called a threshold.")]
    s += [p("score >= threshold -> ACCEPT; score < threshold -> REJECT", "Codex")]
    threshold_example = [
        ["Candidate threshold", "DT-C FRR", "DT-C FAR", "DT-F FRR", "DT-F FAR", "Worst error"],
        ["Low", "2%", "30%", "5%", "35%", "35%"],
        ["Middle", "10%", "12%", "14%", "11%", "14%"],
        ["High", "25%", "3%", "32%", "2%", "32%"],
    ]
    s += [table(threshold_example, widths=[2.0*inch,1.35*inch,1.35*inch,1.35*inch,1.35*inch,1.55*inch], font=8.7)]
    s += [p("Minimax chooses the middle candidate because its largest failure, 14%, is smaller than the largest failure of the other candidates.")]
    s += [two_col([
        p("Why not force FAR=1%?", "H2x"),
        p("On this difficult cross-session dataset, a threshold strict enough for 1% FAR can reject most genuine users. Minimax explicitly searches for a balanced worst-case operating point across conditions."),
    ], [
        p("One threshold or many?", "H2x"),
        p("There is one shared threshold for the four evaluation users inside an outer fold and for both DT conditions. Four outer folds produce four numerical thresholds because each has a different development population."),
    ])]
    s += [ref("src/robust_gait/run_experiment.py", "245-269, 272-315", "per-condition macro errors, minimax search, and development calibration")]
    s.append(PageBreak())

    # final auth
    s += [section("Step 18: run one authentication attempt from start to finish", 22)]
    auth = [
        ["Order", "What happens to the later walking probe"],
        ["1", "Load synchronized raw axes from all nine sensors."],
        ["2", "Use filtered left-shank gyro magnitude to find complete stride boundaries."],
        ["3", "Map the stride from each sensor to 128 normalized gait-phase points."],
        ["4", "For each sensor, mean-pool four neighboring phase points into one of 32 phase bins per axis."],
        ["5", "Apply that sensor model's stored background feature mean and SD."],
        ["6", "Ask each working sensor's logistic model for a claimed-user score."],
        ["7", "Take the median of the sensor scores."],
        ["8", "Apply Z-norm and T-norm, then average them as S-norm."],
        ["9", "Average ten consecutive stride scores."],
        ["10", "Compare the result with the frozen fold threshold and accept or reject."],
    ]
    s += [table([[p(x,"Smallx") for x in r] for r in auth], widths=[.75*inch,8.95*inch], font=8.4)]
    s += [p("The probe never retrains the personal logistic models and its true identity never changes the threshold.", "Callout")]
    s += [ref("src/robust_gait/run_experiment.py", "178-220, 318-350", "complete scoring stream, genuine/impostor construction, threshold decision, and metric rows")]
    s.append(PageBreak())

    # metrics
    s += [section("Step 19: understand FRR, FAR, EER, and standard deviation", 23)]
    metrics = [
        ["Metric", "Question", "Simple example"],
        ["FRR", "How often is the real user rejected?", "20 genuine attempts, 3 rejected -> FRR=3/20=15%."],
        ["FAR", "How often is an impostor accepted?", "100 impostor attempts, 8 accepted -> FAR=8/100=8%."],
        ["EER", "At what post-hoc threshold are FAR and FRR approximately equal?", "A threshold-free score-separation summary; its test-selected threshold is not deployed."],
        ["SD", "How differently do participants perform?", "A large SD means the mean hides easy and very difficult people."],
    ]
    s += [table([[p(x,"Smallx") for x in r] for r in metrics], widths=[1.0*inch,4.3*inch,4.4*inch], font=8.5)]
    s += [p("Macro-average", "H2x"), p("First calculate one FRR/FAR/EER per claimed participant. Then average those participant values so a participant with more strides does not dominate the headline result.")]
    s += [p("Operational versus descriptive", "H2x"), p("FRR and FAR use the frozen development minimax threshold. EER is calculated afterward from evaluation genuine/impostor scores to describe separability.")]
    s += [ref("src/robust_gait/run_experiment.py", "333-369", "participant metrics and macro mean/SD")]
    s += [ref("src/robust_gait/protocol.py", "120-143", "EER calculation from genuine and impostor distance distributions")]
    s.append(PageBreak())

    # leakage
    s += [section("Step 20: explain the leakage boundaries honestly", 24)]
    leakage = [
        ["Possible concern", "Leakage?", "Explanation"],
        ["Evaluation user's ST-control trains their personal model", "No", "This is authorized enrolment."],
        ["Evaluation user's DT data trains their model", "No", "DT-control/fatigue are probes only."],
        ["Development users' DT data selects the threshold", "No direct test leakage", "It is allowed development calibration, but makes the method condition-informed."],
        ["Probe is scored by T-norm cohort models", "No", "Test-time processing uses no true probe label."],
        ["Test EER uses evaluation labels", "Descriptive only", "Allowed for reporting score separation; not allowed as the operational threshold."],
        ["Many methods were tried after viewing the same outer results", "Selection-bias risk", "The final cohort is no longer a pristine one-time external test."],
    ]
    s += [table([[p(x,"Smallx") for x in r] for r in leakage], widths=[3.0*inch,1.65*inch,5.05*inch], font=8.1)]
    s += [p("Defensible statement", "H2x")]
    s += [p("'The executable pipelines contain no direct train-test sample overlap within a fold. Results are participant-disjoint cross-validation estimates. Because the same cohort supported iterative method exploration, independent external validation remains necessary.'", "Callout")]
    s += [ref("src/08_test_operating_points.py", "1-6, 46-70", "diagnostic test-selected operating points that must not be presented as frozen deployment thresholds")]
    s.append(PageBreak())

    # Code walkthrough
    s += [section("Where each implementation lives", 25)]
    code = [
        ["Implementation question", "File and main functions"],
        ["Where are sensors and raw axes defined?", "src/robust_gait/data.py - LOCATIONS, SENSOR_COLUMNS"],
        ["Where is the known trim applied?", "src/robust_gait/data.py - _raw; src/utils.py - KNOWN_TIME_TRIMS"],
        ["Where are magnitude and stride peaks calculated?", "src/robust_gait/data.py - _magnitudes, stride_boundaries"],
        ["Where are strides phase-normalized?", "src/robust_gait/data.py - _resample, load_strides"],
        ["Where are 128 points pooled to 32 bins?", "src/robust_gait/verifier.py - stride_features"],
        ["Where is enrolment augmented?", "src/robust_gait/verifier.py - augment_strides"],
        ["Where is logistic regression trained?", "src/robust_gait/verifier.py - UserModel.fit"],
        ["Where are sensor scores fused?", "src/robust_gait/verifier.py - UserModel.score"],
        ["Where are outer folds created?", "src/robust_gait/protocol.py - create_outer_folds"],
        ["Where is the sensor gate?", "src/robust_gait/run_experiment.py - working_locations"],
        ["Where are Z/T/S-norm implemented?", "src/robust_gait/run_experiment.py - _user_scorer, _cohort_scores, score_streams"],
        ["Where is the minimax threshold selected?", "src/robust_gait/run_experiment.py - _minimax_threshold, calibrate"],
        ["Where is final testing done?", "src/robust_gait/run_experiment.py - run"],
    ]
    s += [table([[p(x,"Smallx") for x in r] for r in code], widths=[4.2*inch,5.5*inch], font=8.1)]
    s.append(PageBreak())

    # settings
    s += [section("Exact final settings and what each one means", 26)]
    configs = [
        ["Setting", "Value", "Meaning"],
        ["Locations", "9", "Sacrum, shanks, feet, wrists, head, sternum."],
        ["Channels", "axes", "Use six raw axes per sensor for classification."],
        ["Pooling", "mean, factor 4", "128 normalized phase points become 32 bins per axis."],
        ["Fusion", "late median", "Train separate sensor models and combine their middle score."],
        ["Regularization C", "0.3", "Controls L2-regularized logistic regression."],
        ["Enrolment augmentation", "4 copies", "Four altered copies plus each original stride."],
        ["Amplitude range", "+/-0.2", "Up to 20% dynamic amplitude change."],
        ["Warp range", "+/-0.1", "Smooth within-stride timing change."],
        ["Background strides", "240", "Maximum per development person and condition."],
        ["Score normalization", "S-norm", "Average Z-norm and T-norm."],
        ["Calibration probes", "session2", "Development DT-control and DT-fatigue."],
        ["Operating point", "minimax", "Minimize the largest macro FRR/FAR across conditions."],
        ["Mirror group", "2", "Two development pseudo-new users at a time."],
        ["Temporal fusion", "10 strides", "Approximately 12 seconds per decision."],
    ]
    s += [table([[p(x,"Smallx") for x in r] for r in configs], widths=[2.25*inch,1.65*inch,5.8*inch], font=7.8)]
    s += [ref("src/results/robust_gait/sacrum+left_leg+right_leg+left_wrist+right_wrist+left_foot+right_foot+head+sternum_late-median_C0.3_axes_mean_aug4a0.2w0.1_bg240_snorm_calsession2_minimax_g2_fuse10/config.json", "entire file", "reproducible final configuration for seed 42")]
    s.append(PageBreak())

    # beginner questions
    s += [section("Questions a non-expert may ask", 27)]
    beginner = [
        q("Why does the system need nine sensors?", "Different body locations carry different identity clues. Combining them reduces dependence on one location."),
        q("Does the user need to enrol while tired?", "No. The new user supplies only normal ST-control walking."),
        q("How can it recognize fatigue without the user's fatigue data?", "Other development people provide examples of general condition-related variability, and the user's normal data is mildly augmented. The method does not claim to generate that user's true fatigue gait."),
        q("Is a stride the same as a two-second window?", "No. A stride is a detected biological gait cycle whose original duration can vary."),
        q("Why wait about 12 seconds?", "Ten strides give a more stable decision than one noisy stride."),
        q("What does each sensor model output?", "A numerical score: higher means more evidence that the stride belongs to the claimed user."),
        q("Why can the same score threshold serve several users?", "Feature scaling and S-norm place different user-model scores on a more comparable development-referenced scale."),
        q("Does the system keep learning from test probes?", "No. The models and threshold stay frozen during the reported evaluation."),
    ]
    s += [table([[p("Question","Smallx"),p("Answer","Smallx")]] + beginner, widths=[3.25*inch,6.45*inch], font=8.2)]
    s.append(PageBreak())

    # expert questions
    s += [section("Questions an ML or biometric expert may ask", 28)]
    expert = [
        q("Why one-vs-background logistic regression rather than a population embedding?", "Only 12 independent development identities are available per fold. Per-user regularized models exploit substantial user enrolment while reducing identity-memorization pressure."),
        q("Why use development ST-fatigue as negative data?", "To broaden impostor-condition support while preserving normal-only enrolment for the evaluated identity."),
        q("Does session-2 calibration weaken the unseen-condition claim?", "Yes, the final method is condition-informed, not unseen-domain generalization. The evaluation identities remain unseen, but the types of session-2 conditions are represented among development users."),
        q("Why use a median rather than learned fusion?", "Median fusion is robust, parameter-free, and suitable for a small cohort where a learned fusion model could overfit."),
        q("Why S-norm?", "Z-norm compensates claimed-model score scale; T-norm compensates probe difficulty; S-norm symmetrically combines both."),
        q("Why minimax rather than target FAR?", "The application requires acceptable behavior across two conditions and both error types. Minimax explicitly controls the worst development macro error."),
        q("Are the 10-stride decisions independent?", "No. A moving average produces overlapping decisions. Reported counts are decision counts, but statistical uncertainty should respect participant/recording dependence."),
        q("What is the strongest validity limitation?", "Sixteen participants and repeated method exploration on the same cohort. Independent external validation is needed."),
    ]
    s += [table([[p("Expert question","Smallx"),p("Defensible answer","Smallx")]] + expert, widths=[3.55*inch,6.15*inch], font=7.9)]
    s.append(PageBreak())

    # scripts to say
    s += [section("How to explain the work at three levels", 29)]
    s += [p("Thirty-second explanation for anyone", "H2x")]
    s += [p("'We record walking from nine body locations. We cut the signal into complete strides and line every stride up to the same gait phase. A new user enrols only with normal walking. Each sensor builds a small personal recognition model, and the sensor models vote. We correct the score using patterns learned from other people, combine ten strides for stability, and compare the result with a pass mark learned only from development participants.'", "Callout")]
    s += [p("Two-minute explanation for your supervisor", "H2x")]
    s += [p("'The method uses participant-disjoint four-fold evaluation. Raw six-axis signals from nine synchronized IMUs are segmented into strides using peaks in a 6 Hz low-pass-filtered left-shank gyro magnitude. The filter is only for segmentation. Each variable-duration stride is interpolated to 128 normalized phase points, then mean-pooled in groups of four to 32 phase bins per axis. For each unseen user, complete ST-control is the only enrolment condition. Four amplitude/timing augmentations expand the positive class, while development participants' ST-control and ST-fatigue provide negatives. A balanced L2 logistic model is fitted per working sensor, sensor scores are median-fused, S-normalized, and averaged across ten strides. Development pseudo-unseen session-2 trials select one minimax threshold per fold, which is frozen for DT-control and DT-fatigue evaluation.'")]
    s += [p("One-sentence technical contribution", "H2x")]
    s += [p("'A normal-only-enrolment, participant-disjoint multi-IMU gait verifier combining gait-phase normalization, augmented per-sensor one-vs-background models, sensor-quality gating, robust median fusion, S-norm, temporal evidence accumulation, and development-only session-2 minimax calibration.'", "Callout")]
    s.append(PageBreak())

    # final recap
    s += [section("Final recap: teach it back without looking", 30)]
    recap = [
        ["If asked...", "Your answer should contain..."],
        ["What enters the system?", "Nine synchronized IMUs, six raw axes each, originally sampled at 128 Hz."],
        ["How is walking divided?", "Left-shank gyro magnitude, 6 Hz detector filter, mid-swing peaks, valid 0.7-1.8 s stride intervals."],
        ["What is resampled?", "Each stride to 128 normalized gait-phase points, not 128 Hz."],
        ["What is mean pooling?", "Average every four neighboring phase points, giving 32 phase bins per axis, not 32 Hz."],
        ["What trains a user model?", "User's normal ST-control positives plus development-user ST-control/fatigue negatives."],
        ["How many models?", "Up to nine per enrolled user - one per working sensor."],
        ["How are sensor scores combined?", "Median late fusion."],
        ["Why S-norm?", "Correct model scale and probe difficulty using development cohorts."],
        ["Why ten strides?", "Reduce one-stride noise; approximately 12-second decision."],
        ["How is the threshold chosen?", "Development pseudo-unseen DT trials; minimize worst macro FRR/FAR."],
        ["What is the final validity claim?", "No direct fold-level sample leakage; participant-disjoint CV; external confirmation still needed."],
    ]
    s += [table([[p(x,"Smallx") for x in r] for r in recap], widths=[3.0*inch,6.7*inch], font=8.2)]
    s += [Spacer(1, .18*inch), p("If you can explain every row above in your own words, you understand the implemented method from start to finish.", "Callout")]
    return s


def main():
    if not SIMPLE_ARCH.exists():
        raise FileNotFoundError(SIMPLE_ARCH)
    if not TECHNICAL_ARCH.exists():
        raise FileNotFoundError(TECHNICAL_ARCH)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    GuideDoc(str(OUTPUT)).build(build_story())
    print(OUTPUT)


if __name__ == "__main__":
    main()
