"""Generate a study-guide PDF for the final robust gait implementation."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output" / "pdf" / "proposed_solution_implementation_guide.pdf"
ARCHITECTURE = ROOT / "paper" / "figures" / "proposed_method_architecture.png"

BROWN = colors.HexColor("#492F24")
GOLD = colors.HexColor("#F5B335")
CREAM = colors.HexColor("#FFF8E8")
PALE_BLUE = colors.HexColor("#EAF2F8")
PALE_GREEN = colors.HexColor("#EAF5EE")
PALE_RED = colors.HexColor("#FBEDEE")
INK = colors.HexColor("#202124")
GREY = colors.HexColor("#5F6368")
LIGHT_GREY = colors.HexColor("#ECEFF1")

PAGE_W, PAGE_H = landscape(letter)
MARGIN = 0.55 * inch


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D6CCC2"))
    canvas.line(MARGIN, 0.38 * inch, PAGE_W - MARGIN, 0.38 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(GREY)
    canvas.drawString(MARGIN, 0.20 * inch, "Robust multi-sensor gait verification - implementation study guide")
    canvas.drawRightString(PAGE_W - MARGIN, 0.20 * inch, f"Page {doc.page}")
    canvas.restoreState()


class GuideDoc(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(
            filename,
            pagesize=landscape(letter),
            leftMargin=MARGIN,
            rightMargin=MARGIN,
            topMargin=0.52 * inch,
            bottomMargin=0.50 * inch,
            title="Proposed Multi-Sensor Gait Verification Method",
            author="Bamidele Akinyemi",
            subject="Implementation, architecture, rationale, and code map",
        )
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        self.addPageTemplates(PageTemplate(id="guide", frames=[frame], onPage=footer))


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    "GuideTitle", parent=styles["Title"], fontName="Helvetica-Bold",
    fontSize=27, leading=31, textColor=BROWN, alignment=TA_CENTER,
    spaceAfter=10,
))
styles.add(ParagraphStyle(
    "GuideSubtitle", parent=styles["Normal"], fontName="Helvetica",
    fontSize=13, leading=18, textColor=GREY, alignment=TA_CENTER,
))
styles.add(ParagraphStyle(
    "H1x", parent=styles["Heading1"], fontName="Helvetica-Bold",
    fontSize=19, leading=22, textColor=BROWN, spaceAfter=10,
))
styles.add(ParagraphStyle(
    "H2x", parent=styles["Heading2"], fontName="Helvetica-Bold",
    fontSize=12.5, leading=15, textColor=BROWN, spaceBefore=7, spaceAfter=5,
))
styles.add(ParagraphStyle(
    "Bodyx", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=9.4, leading=13.2, textColor=INK, spaceAfter=5,
))
styles.add(ParagraphStyle(
    "Smallx", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=8.1, leading=10.7, textColor=INK,
))
styles.add(ParagraphStyle(
    "Bulletx", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=9.2, leading=12.7, leftIndent=14, firstLineIndent=-8,
    bulletIndent=4, textColor=INK, spaceAfter=3,
))
styles.add(ParagraphStyle(
    "Codex", parent=styles["Code"], fontName="Courier",
    fontSize=7.7, leading=10.2, leftIndent=7, rightIndent=7,
    borderColor=colors.HexColor("#D8DEE4"), borderWidth=0.6,
    borderPadding=6, backColor=colors.HexColor("#F6F8FA"), spaceAfter=6,
))
styles.add(ParagraphStyle(
    "Callout", parent=styles["BodyText"], fontName="Helvetica-Bold",
    fontSize=10.2, leading=14, textColor=BROWN, alignment=TA_LEFT,
    borderColor=GOLD, borderWidth=1.2, borderPadding=8,
    backColor=CREAM, spaceBefore=5, spaceAfter=7,
))
styles.add(ParagraphStyle(
    "Ref", parent=styles["BodyText"], fontName="Courier",
    fontSize=7.6, leading=10, textColor=colors.HexColor("#305B78"),
    spaceBefore=3, spaceAfter=5,
))


def p(text, style="Bodyx"):
    return Paragraph(text, styles[style])


def bullet(text):
    return Paragraph(f"- {text}", styles["Bulletx"])


def ref(path, lines, purpose):
    return p(f"CODE: {path} | lines {lines} | {purpose}", "Ref")


def section(title, number=None):
    label = f"{number}. {title}" if number is not None else title
    return p(label, "H1x")


def table(data, widths=None, header=True, font=8.2):
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font),
        ("LEADING", (0, 0), (-1, -1), font + 2.3),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#C7CDD1")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, colors.HexColor("#F8F9FA")]),
    ]
    if header:
        commands.extend([
            ("BACKGROUND", (0, 0), (-1, 0), BROWN),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ])
    t.setStyle(TableStyle(commands))
    return t


def two_col(left, right, ratio=(1, 1)):
    total = PAGE_W - 2 * MARGIN
    widths = [total * ratio[0] / sum(ratio) - 7, total * ratio[1] / sum(ratio) - 7]
    t = Table([[left, right]], colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return t


def build_story():
    story = []

    # Cover.
    story += [Spacer(1, 0.55 * inch), p("Proposed Multi-Sensor Gait Verification Method", "GuideTitle")]
    story += [p("Architecture, implementation flow, design rationale, equations, and source-code map", "GuideSubtitle")]
    story += [Spacer(1, 0.35 * inch)]
    cover_data = [
        [p("Research objective", "H2x"), p("Authenticate a newly enrolled person across a one-week session change, cognitive dual task, and fatigue, while requiring only normal ST-control walking during enrolment.")],
        [p("Final input", "H2x"), p("Nine synchronized IMUs, six raw axes per sensor, 54 channels at 128 Hz.")],
        [p("Final verifier", "H2x"), p("One regularized logistic-regression model per user and working sensor, followed by median late fusion, S-norm, temporal fusion, and a development-calibrated minimax threshold.")],
        [p("Evaluation protocol", "H2x"), p("Four participant-disjoint outer folds: 12 development participants and 4 unseen evaluation participants per fold.")],
    ]
    story += [table(cover_data, widths=[2.1 * inch, 7.7 * inch], header=False, font=9.4)]
    story += [Spacer(1, 0.32 * inch), p(
        "The method is not one new classifier. Its contribution is the complete verification pipeline: phase-aligned multi-location gait representation, normal-only user enrolment, robust sensor fusion, cohort score normalization, temporal evidence accumulation, and condition-aware development calibration without using the evaluation users' DT recordings.",
        "Callout",
    )]
    story += [Spacer(1, 0.25 * inch), p("Prepared from the executable implementation in src/robust_gait", "GuideSubtitle")]
    story.append(PageBreak())

    # Reading map.
    story += [section("How to read this guide")]
    story += [p("The pipeline has three conceptual stages. Keeping these stages separate makes the implementation much easier to explain.")]
    data = [
        ["Stage", "Main question", "What happens"],
        ["A. Shared preprocessing", "How is a comparable gait unit created?", "Synchronize nine sensors, detect strides, normalize every stride to the same length, and retain raw directional axes."],
        ["B. Development and calibration", "How are fixed system rules learned without touching evaluation identities?", "Development participants provide impostor background, cohort normalization, pseudo-unseen trials, and one minimax threshold per outer fold."],
        ["C. Enrolment and authentication", "How is a new person registered and verified?", "Use only the person's ST-control data, train per-sensor user models, score session-2 probes, fuse evidence, and compare with the frozen threshold."],
    ]
    story += [table([[p(x, "Smallx") for x in row] for row in data], widths=[1.75*inch, 3.0*inch, 5.0*inch], font=8.5)]
    story += [Spacer(1, 0.15*inch), p("Important terminology", "H2x")]
    glossary = [
        ["Development participant", "May define model settings, background distributions, normalization cohorts, and thresholds."],
        ["Evaluation participant", "Held out from development. Provides ST-control enrolment only after the system has been fixed."],
        ["Enrolment", "Authorized reference data supplied by the claimed user. It is allowed to train that user's verifier."],
        ["Probe", "A later gait sample presented for authentication."],
        ["Genuine attempt", "Claimed identity equals the person who produced the probe."],
        ["Impostor attempt", "Claimed identity differs from the probe owner."],
    ]
    story += [table([[p(a,"Smallx"), p(b,"Smallx")] for a,b in glossary], widths=[2.2*inch, 7.55*inch], header=False, font=8.4)]
    story.append(PageBreak())

    # Architecture full page.
    story += [section("Complete architecture and implementation flow", 1)]
    story += [p("Follow the top row from left to right, then the development lane, and finally the unseen-user enrolment and verification lane.")]
    if ARCHITECTURE.exists():
        story += [Image(str(ARCHITECTURE), width=9.9*inch, height=6.06*inch)]
    story.append(PageBreak())

    # Data and QC.
    story += [section("Input recordings and quality control", 2)]
    left = [
        p("What is done", "H2x"),
        bullet("Read the sacrum, left/right shanks, left/right feet, left/right wrists, head, and sternum recordings."),
        bullet("Retain GyrX, GyrY, GyrZ, AccX, AccY, and AccZ from every sensor."),
        bullet("Apply the pre-declared sub_07 ST-control time trim."),
        bullet("Reject recordings containing non-finite sensor samples."),
        bullet("Use the shortest synchronized recording length across sensors."),
        p("Result", "H2x"),
        p("At each timestamp the system has 9 x 6 = 54 raw directional measurements."),
    ]
    right = [
        p("Why it is done", "H2x"),
        bullet("Multiple body locations provide complementary identity evidence. A fatigue-related change at one location need not destroy evidence at every location."),
        bullet("A shared length ensures that a sacrum sample is combined with wrist, leg, foot, head, and sternum samples from the same instant."),
        bullet("The sub_07 correction is recording-based and defined independently of authentication performance, avoiding an outcome-driven exclusion."),
        p("Key distinction", "H2x"),
        p("The final classifier uses raw axes. Accelerometer/gyroscope magnitudes are still derived internally for stride detection and sensor-quality checks, but they are not the final classifier input."),
    ]
    story += [two_col(left, right)]
    story += [ref("src/robust_gait/data.py", "21-47, 62-77, 112-145", "sensor list, raw loading, validation, synchronization, and stride cache")]
    story += [ref("src/utils.py", "49-58", "pre-specified sub_07 recording trim")]
    story.append(PageBreak())

    # Strides.
    story += [section("Stride detection and gait-cycle normalization", 3)]
    story += [two_col([
        p("How stride boundaries are found", "H2x"),
        p("The gyroscope magnitude of the left shank is calculated:") ,
        p("g(t) = sqrt(GyrX(t)^2 + GyrY(t)^2 + GyrZ(t)^2)", "Codex"),
        bullet("Apply a fourth-order, zero-phase Butterworth low-pass filter with a 6 Hz cutoff."),
        bullet("Detect mid-swing peaks separated by at least 0.6 seconds."),
        bullet("Require peak prominence of at least 0.5 times the signal standard deviation."),
        bullet("Treat consecutive peaks as one stride and retain durations from 0.7 to 1.8 seconds."),
    ], [
        p("Why this is better than arbitrary windows", "H2x"),
        bullet("Every accepted example represents one complete stride rather than a random fragment."),
        bullet("Each stride begins at approximately the same biomechanical phase."),
        bullet("The same boundaries are applied to all nine sensors, preserving cross-body coordination."),
        p("Duration normalization", "H2x"),
        p("Valid strides have different sample counts because walking speed changes. Linear interpolation maps each stride to exactly 128 samples. This reduces cadence-related length variation while retaining waveform shape."),
    ])]
    story += [p("Important: the left-shank magnitude selects the boundaries; the raw 54-axis data inside those boundaries remains available for classification.", "Callout")]
    story += [ref("src/robust_gait/data.py", "88-109", "magnitude calculation, filtering, peak detection, and interpolation")]
    story += [ref("src/robust_gait/data.py", "121-145", "shared segmentation and fixed-length stride construction")]
    story.append(PageBreak())

    # Features.
    story += [section("Per-sensor raw-axis feature representation", 4)]
    data = [
        ["Operation", "Shape", "Meaning"],
        ["One normalized stride", "128 x 54", "128 gait-phase samples across 9 sensors and 6 raw axes."],
        ["Select one sensor", "128 x 6", "Directional accelerometer and gyroscope profiles for one body location."],
        ["Mean-pool every 4 samples", "32 x 6", "Box-filtered, downsampled gait profile."],
        ["Flatten", "192 values", "Feature vector used by that sensor's user-specific classifier."],
        ["Repeat", "Up to 9 vectors", "One independent vector and model per working sensor."],
    ]
    story += [table([[p(x,"Smallx") for x in row] for row in data], widths=[2.25*inch, 1.45*inch, 6.05*inch], font=8.6)]
    story += [two_col([
        p("Why raw axes", "H2x"),
        p("Raw axes retain direction: forward/backward motion, lateral motion, rotation direction, body asymmetry, and coordination. Magnitudes are orientation-resistant but discard this directional information."),
        p("Why separate sensors", "H2x"),
        p("A single concatenated classifier can be dominated by a shifted or faulty device. Separate classifiers allow a bad sensor to be removed or outvoted during late fusion."),
    ], [
        p("Why mean pooling", "H2x"),
        p("Averaging each block of four samples suppresses high-frequency noise before decimation, reduces dimensionality, and uses every input sample. Selecting every fourth point would risk aliasing and discard information."),
        p("Final dimensionality", "H2x"),
        p("The system does not train one 1,728-feature classifier. It trains up to nine 192-feature classifiers and combines their scores."),
    ])]
    story += [ref("src/robust_gait/verifier.py", "92-131", "channel selection, mean pooling, flattening, and optional derivative support")]
    story.append(PageBreak())

    # Protocol.
    story += [section("Participant-disjoint outer protocol", 5)]
    folds = [
        ["Fold", "Development participants", "Evaluation participants"],
        ["1", "12", "sub_01, sub_06, sub_10, sub_14"],
        ["2", "12", "sub_02, sub_07, sub_11, sub_15"],
        ["3", "12", "sub_03, sub_08, sub_12, sub_17"],
        ["4", "12", "sub_05, sub_09, sub_13, sub_18"],
    ]
    story += [table(folds, widths=[0.75*inch, 2.2*inch, 6.8*inch], font=8.8)]
    story += [two_col([
        p("What the 12 development participants may do", "H2x"),
        bullet("Supply ST-control and ST-fatigue background negatives."),
        bullet("Supply score-normalization cohort identities."),
        bullet("Act as pseudo-unseen users for threshold calibration."),
        bullet("Supply development DT-control and DT-fatigue probes for the final session-2 calibration mirror."),
    ], [
        p("What the 4 evaluation participants may do", "H2x"),
        bullet("Supply their complete ST-control recording for personal enrolment."),
        bullet("Supply DT-control and DT-fatigue only after calibration is frozen."),
        bullet("Never determine background statistics, cohort distributions, hyperparameters, or thresholds within their evaluation fold."),
    ])]
    story += [p("Each person is evaluated once, but appears as a development participant in the other folds. This is participant-level cross-validation, not one permanently untouched external test cohort.", "Callout")]
    story += [ref("src/robust_gait/protocol.py", "24-74", "deterministic outer folds and disjointness assertions")]
    story += [ref("src/robust_gait/run_experiment.py", "318-350", "fold execution, enrolment, probes, and evaluation metrics")]
    story.append(PageBreak())

    # Development background.
    story += [section("Development background and pseudo-unseen calibration", 6)]
    story += [two_col([
        p("Background population", "H2x"),
        p("For every selected background participant, the system samples up to 240 strides from both ST-control and ST-fatigue."),
        p("background = {other development users' ST-control and ST-fatigue strides}", "Codex"),
        p("Why", "H2x"),
        p("A verifier trained only against normal impostors may accept another person's fatigued gait. Including fatigue in the negative population teaches what non-target variability looks like without requiring the new user to provide fatigued enrolment."),
    ], [
        p("Pseudo-unseen mirror", "H2x"),
        p("The 12 development participants are arranged into groups of two. For each group:") ,
        bullet("The pair enrols using full ST-control."),
        bullet("Their DT-control and DT-fatigue recordings are probes."),
        bullet("The other 10 development participants are background identities."),
        p("Why", "H2x"),
        p("This recreates the deployment situation before the outer evaluation users are scored: normal enrolment followed by session-2 authentication one week later."),
    ])]
    story += [p("Development DT data influences calibration, but evaluation-user DT data does not. The method is therefore condition-informed across people, not condition-blind.", "Callout")]
    story += [ref("src/robust_gait/run_experiment.py", "42-43, 85-91", "training/evaluation condition definitions and background sampling")]
    story += [ref("src/robust_gait/run_experiment.py", "272-301", "pseudo-unseen groups, full enrolment, and session-2 calibration probes")]
    story.append(PageBreak())

    # Enrollment QC + augmentation.
    story += [section("Normal-only enrolment, sensor gate, and augmentation", 7)]
    story += [two_col([
        p("Normal-only enrolment", "H2x"),
        p("A new evaluation user provides the complete ST-control recording. Their real ST-fatigue, DT-control, and DT-fatigue data are not used to fit their models."),
        p("Sensor fault gate", "H2x"),
        p("For each location, the median enrolment gyroscope magnitude is compared with the development-population median. A location below 25% of that reference is dropped for that user."),
        p("Why", "H2x"),
        p("A detached or nearly inactive sensor should not contribute an extreme score. The gate uses enrolment and development data only, never the later DT probe."),
    ], [
        p("Positive augmentation", "H2x"),
        p("Each genuine enrolment stride is retained and accompanied by four synthetic variants:") ,
        bullet("Per-sensor dynamic amplitude factor sampled within 1 +/- 0.20."),
        bullet("Smooth monotonic timing warp with amplitude up to +/- 0.10."),
        bullet("Accelerometer axes scale around their stride mean; gyroscope axes scale directly."),
        p("Why", "H2x"),
        p("The augmented samples enlarge the accepted neighborhood around normal gait, representing reasonable strength and timing variation without using that person's actual fatigue or dual-task recordings."),
    ])]
    story += [ref("src/robust_gait/run_experiment.py", "157-176, 195-208", "development-referenced sensor gate and scorer construction")]
    story += [ref("src/robust_gait/verifier.py", "42-89, 142-151", "amplitude/timing augmentation and its use during fitting")]
    story.append(PageBreak())

    # User model.
    story += [section("Per-user, per-sensor classifier training", 8)]
    story += [p("For each working sensor, one binary logistic-regression model is trained for the claimed user.")]
    model_table = [
        ["Element", "Implementation"],
        ["Positive class", "Claimed user's original and augmented ST-control strides."],
        ["Negative class", "Sampled ST-control and ST-fatigue strides from development identities."],
        ["Features", "192 pooled raw-axis values for one sensor."],
        ["Scaling", "Mean and standard deviation calculated from negative background features only."],
        ["Classifier", "L2-regularized logistic regression, C = 0.3, balanced class weights."],
        ["Output", "Decision-function score; a larger value means more evidence for the claimed user."],
    ]
    story += [table([[p(x,"Smallx") for x in row] for row in model_table], widths=[2.1*inch, 7.65*inch], font=8.8)]
    story += [two_col([
        p("Background-only standardization", "H2x"),
        p("x_scaled = (x - mean_background) / std_background", "Codex"),
        p("Using the background as the reference keeps feature scales less dependent on the particular enrolled user and applies the same stored transformation to later probes."),
    ], [
        p("Why logistic regression", "H2x"),
        p("The dataset contains many correlated strides but only 16 independent people. A small regularized classifier uses the new user's substantial enrolment recording directly and is less likely than a deep population encoder to memorize the limited development identities."),
        p("Balanced class weights prevent the larger negative background from dominating the objective."),
    ])]
    story += [ref("src/robust_gait/verifier.py", "134-181", "model groups, background scaling, logistic fitting, and probe scoring")]
    story.append(PageBreak())

    # Sensor fusion.
    story += [section("Sensor scoring and robust late fusion", 9)]
    story += [p("A probe stride is transformed by every working sensor model. With nine working devices, the claimed user produces nine decision scores for that stride.")]
    story += [p("score_fused(t) = median(score_sacrum(t), ..., score_sternum(t))", "Codex")]
    example = [
        ["Sensor scores", "Mean", "Median"],
        ["1.2, 1.0, 1.1, 0.9, 1.3, 1.1, 1.0, 1.2, -8.0", "0.09", "1.10"],
    ]
    story += [table(example, widths=[6.0*inch, 1.7*inch, 1.7*inch], font=9)]
    story += [two_col([
        p("Why late fusion", "H2x"),
        p("Each location learns independently. The system can drop a failed sensor and still use all remaining models. This would be difficult if all 54 channels entered one indivisible classifier."),
    ], [
        p("Why the median", "H2x"),
        p("A reattached, shifted, or faulty sensor can generate an extreme score. The mean is pulled toward that outlier; the median preserves the consensus of the body locations."),
    ])]
    story += [ref("src/robust_gait/verifier.py", "173-187", "per-sensor decision scores and median combination")]
    story.append(PageBreak())

    # S norm.
    story += [section("Cohort score normalization: Z-norm, T-norm, and S-norm", 10)]
    rows = [
        ["Method", "Question answered", "Calculation"],
        ["Z-norm", "Is this score unusual for this claimed-user model compared with unseen impostor cohorts?", "z = (claimed score - cohort impostor mean) / cohort impostor SD"],
        ["T-norm", "Is this probe specifically similar to the claim, or does it score highly against many cohort models?", "t = (claimed score - same-probe cohort-model mean) / same-probe cohort SD"],
        ["S-norm", "How can model-specific and probe-specific shifts both be reduced?", "s = 0.5 x (z + t)"],
    ]
    story += [table([[p(x,"Smallx") for x in row] for row in rows], widths=[1.15*inch, 5.0*inch, 3.6*inch], font=8.1)]
    story += [p("Z-norm implementation detail", "H2x")]
    story += [p("Background identities are rotated into three cohorts. A cohort identity is excluded from the model's negative training before its scores estimate that model's impostor mean and standard deviation.")]
    story += [p("T-norm implementation detail", "H2x")]
    story += [p("The same probe is scored by development cohort models. These scores estimate whether that probe is globally easy to accept. Probe labels are not needed for this test-time normalization.")]
    story += [p("Why normalization matters: raw decision values from different user models need not share the same numerical scale. S-norm makes the later shared threshold more meaningful.", "Callout")]
    story += [ref("src/robust_gait/run_experiment.py", "94-136", "rotated Z-norm models and cohort statistics")]
    story += [ref("src/robust_gait/run_experiment.py", "139-154, 178-220", "T-norm cohort scoring and final S-norm combination")]
    story.append(PageBreak())

    # temporal and threshold.
    story += [section("Temporal fusion and minimax threshold", 11)]
    story += [two_col([
        p("Temporal fusion", "H2x"),
        p("Ten consecutive S-normalized stride scores are averaged:") ,
        p("score_10(t) = (1/10) x sum(score(t-i)), i = 0...9", "Codex"),
        p("Ten strides correspond to approximately 12 seconds of walking. One unusual stride caused by a stumble, turn, or sensor disturbance is diluted by the surrounding evidence."),
        p("Trade-off", "H2x"),
        p("Longer fusion improves stability but increases decision delay and creates fewer decisions per recording."),
    ], [
        p("Minimax calibration", "H2x"),
        p("For each candidate threshold, development pseudo-users provide four macro errors:"),
        bullet("DT-control FRR"), bullet("DT-control FAR"),
        bullet("DT-fatigue FRR"), bullet("DT-fatigue FAR"),
        p("loss(threshold) = max(the four errors)", "Codex"),
        p("The chosen threshold minimizes this worst error. It avoids selecting a threshold that looks excellent for one condition while failing badly in another."),
    ])]
    story += [p("One numerical threshold is selected per outer fold because each fold has a different 12-person development population. The same frozen threshold is then applied to all four evaluation users and both DT conditions inside that fold.", "Callout")]
    story += [ref("src/robust_gait/run_experiment.py", "77-82", "causal moving-average fusion")]
    story += [ref("src/robust_gait/run_experiment.py", "245-269, 302-315", "minimax objective and operating-point selection")]
    story.append(PageBreak())

    # Evaluation.
    story += [section("Final authentication and metric calculation", 12)]
    story += [p("After all development operations are frozen, each evaluation user is enrolled and tested.")]
    sequence = [
        ["Step", "Authentication action"],
        ["1", "Load the claimed user's ST-control enrolment and train their per-sensor models against the 12 development users."],
        ["2", "Load each evaluation participant's DT-control or DT-fatigue recording as a probe stream."],
        ["3", "Calculate sensor scores, median late fusion, S-norm, and ten-stride temporal fusion."],
        ["4", "Accept if the normalized score is at least the frozen fold threshold; otherwise reject."],
        ["5", "For one claimed user, their own stream is genuine; the other three evaluation users' streams are impostor attempts."],
        ["6", "Calculate participant-level FRR, FAR, and EER, then macro-average so every participant has equal weight."],
    ]
    story += [table([[p(x,"Smallx") for x in row] for row in sequence], widths=[0.7*inch, 9.05*inch], font=8.7)]
    story += [two_col([
        p("FRR", "H2x"), p("Fraction of genuine decisions rejected. Lower FRR means fewer legitimate-user interruptions."),
        p("FAR", "H2x"), p("Fraction of impostor decisions accepted. Lower FAR means stronger security."),
    ], [
        p("EER", "H2x"), p("Post-hoc score-separation measure at the point where FAR and FRR are approximately equal. Its test threshold is descriptive and is not the deployed threshold."),
        p("Standard deviation", "H2x"), p("Shows how strongly performance differs between participants; a low mean can conceal severe individual failures."),
    ])]
    story += [ref("src/robust_gait/run_experiment.py", "318-369", "final scoring, genuine/impostor construction, metrics, macro summary, and saved outputs")]
    story += [ref("src/robust_gait/protocol.py", "77-143", "target-FAR selector and threshold-free EER calculation")]
    story.append(PageBreak())

    # Leakage.
    story += [section("Leakage boundaries and what the claims mean", 13)]
    checks = [
        ["Question", "Answer", "Reason"],
        ["Does an evaluation user's DT data train their model?", "No", "Only their ST-control enrolment is positive training data."],
        ["Does an evaluation identity enter the development background?", "No", "Outer-fold development and evaluation identities are disjoint."],
        ["Do evaluation labels select the operational threshold?", "No", "Minimax selection uses development pseudo-users."],
        ["Does development DT data influence the method?", "Yes", "It calibrates the threshold for the expected session-2 conditions."],
        ["Does T-norm process the probe?", "Yes", "It uses cohort scores at test time without using the probe's identity label."],
        ["Is using ST-control from the evaluation user leakage?", "No", "That is authorized biometric enrolment, not test probing."],
        ["Is EER computed from evaluation labels?", "Yes", "EER is reported as a post-hoc descriptive separation metric; it does not set FRR/FAR decisions."],
    ]
    story += [table([[p(x,"Smallx") for x in row] for row in checks], widths=[3.2*inch, 0.8*inch, 5.75*inch], font=8.2)]
    story += [p("Research-level limitation", "H2x")]
    story += [p("Many candidate methods and settings were compared using the same cross-validation participants. The code contains no direct train-test sample leakage, but repeated inspection of evaluation performance introduces adaptive test-set or model-selection bias. The results should be described as participant-disjoint cross-validation, with confirmation on a new cohort or external dataset left for future validation.", "Callout")]
    story += [ref("src/08_test_operating_points.py", "1-6, 46-70, 113-124", "explicitly post-hoc test-selected thresholds; diagnostic only")]
    story.append(PageBreak())

    # Code map.
    story += [section("Source-code map", 14)]
    code_map = [
        ["File", "Responsibility"],
        ["src/robust_gait/data.py", "Loads nine raw IMUs, applies known trims, derives magnitudes, detects strides, resamples cycles, and maintains the stride cache."],
        ["src/robust_gait/verifier.py", "Defines augmentation, raw-axis selection, mean pooling, background feature scaling, per-sensor logistic-regression fitting, and late score fusion."],
        ["src/robust_gait/protocol.py", "Creates participant-disjoint folds and implements target-FAR and EER calculations."],
        ["src/robust_gait/run_experiment.py", "Builds development backgrounds, score-normalization cohorts, sensor gates, calibration mirrors, minimax thresholds, evaluation trials, and result summaries."],
        ["src/robust_gait/select_config.py", "Development-mirror configuration search utility. It must not use outer evaluation scores as its selection criterion."],
        ["src/robust_gait/plot_architecture.py", "Reproducibly generates the architecture diagram used in this guide."],
        ["src/utils.py", "Contains the known sub_07 recording-level trim used by the shared data-loading logic."],
        ["src/results/robust_gait/.../config.json", "Records the exact final settings used for each seed."],
        ["src/results/robust_gait/.../participant_metrics.csv", "Stores fold, participant, condition, FRR, FAR, EER, threshold, and comparison counts."],
    ]
    story += [table([[p(x,"Smallx") for x in row] for row in code_map], widths=[3.1*inch, 6.65*inch], font=8.2)]
    story += [Spacer(1, 0.15*inch), p("Exact final configuration", "H2x")]
    story += [p(
        "9 locations | late fusion | median sensor combination | raw axes | mean pooling | C=0.3 | four augmented copies | amplitude range=0.2 | warp range=0.1 | 240 background strides per participant and condition | S-norm | session-2 calibration probes | minimax operating point | pseudo-unseen group size=2 | ten-stride temporal fusion",
        "Codex",
    )]
    story.append(PageBreak())

    # Oral defense.
    story += [section("Questions you should be able to answer", 15)]
    qa = [
        ["Question", "Short defensible answer"],
        ["Why nine sensors?", "They provide complementary identity evidence across the body and reduce dependence on one unstable location."],
        ["Why raw axes rather than magnitudes?", "Raw axes preserve directional and coordination information. Magnitudes are still used for segmentation and quality checks."],
        ["Why stride segmentation?", "It compares phase-aligned complete gait cycles instead of arbitrary temporal fragments."],
        ["Why logistic regression?", "There are many correlated strides but few independent people. A regularized per-user model directly uses the new user's enrolment without requiring a population encoder to generalize from only 12 development identities."],
        ["Why train separate sensor models?", "It supports sensor-quality gating and robust late fusion; one poor location cannot dominate the entire representation."],
        ["Why median fusion?", "The median is resistant to extreme scores caused by reattachment, shift, or sensor failure."],
        ["Why use fatigue among negatives?", "It broadens the background impostor population while preserving normal-only enrolment for the new user."],
        ["Why S-norm?", "It corrects both claimed-model score-scale differences and probe-specific score shifts."],
        ["Why ten strides?", "Temporal averaging suppresses isolated stride noise, trading approximately 12 seconds of delay for greater stability."],
        ["Why minimax calibration?", "It selects the threshold with the smallest worst macro error across both session-2 conditions and both error types."],
        ["Is development DT use leakage?", "No, because it belongs only to development identities. It makes the method condition-informed, while outer evaluation users remain unseen during calibration."],
        ["What is the novelty?", "The defensible contribution is the integrated normal-only-enrolment, multi-sensor, condition-robust verification pipeline and protocol, not a claim that logistic regression or S-norm is individually new."],
    ]
    story += [table([[p(x,"Smallx") for x in row] for row in qa], widths=[2.65*inch, 7.1*inch], font=7.7)]
    story.append(PageBreak())

    # Final summary.
    story += [section("The complete method in one explanation", 16)]
    story += [p(
        "The proposed system begins by synchronizing nine body-worn IMUs and detecting complete gait strides from the filtered left-shank gyroscope magnitude. Every valid stride is resampled to 128 points so recordings with different cadence can be compared at a common gait phase. The final classifier retains the six directional accelerometer and gyroscope axes from each sensor, mean-pools them to 32 points, and treats every sensor separately.",
    )]
    story += [p(
        "Within each participant-disjoint fold, 12 development participants define the negative background, score-normalization cohorts, and operating threshold. Their ST-control and ST-fatigue strides represent impostor variability. Groups of two development participants then simulate unseen users: they enrol using full ST-control and are probed using their own DT-control and DT-fatigue data. These development trials produce S-normalized scores and a minimax threshold chosen to limit the worst FRR or FAR across both conditions.",
    )]
    story += [p(
        "Each of the four outer evaluation users is enrolled using only their complete ST-control recording. A development-referenced gate removes an obviously inactive sensor, and amplitude/timing augmentation expands the user's normal enrolment variation. One balanced, L2-regularized logistic-regression verifier is trained per working sensor against the development background. At authentication, a session-2 probe is scored by every sensor model, the median provides robust late fusion, S-norm places the score on a cohort-referenced scale, and ten consecutive stride scores are averaged. The resulting value is compared with the frozen fold threshold to accept or reject the claim.",
    )]
    story += [p(
        "The core research idea is therefore not simply 'use nine sensors.' It is to combine phase-aligned body-wide information, user-specific normal-only enrolment, robust handling of unreliable sensors, population-based score correction, and condition-aware threshold calibration while keeping the evaluated user's later DT recordings outside training and calibration.",
        "Callout",
    )]
    story += [Spacer(1, 0.25*inch), p("End of implementation guide", "GuideSubtitle")]
    return story


def main():
    if not ARCHITECTURE.exists():
        raise FileNotFoundError(f"Architecture image not found: {ARCHITECTURE}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    GuideDoc(str(OUTPUT)).build(build_story())
    print(OUTPUT)


if __name__ == "__main__":
    main()
