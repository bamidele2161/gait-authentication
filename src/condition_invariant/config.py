"""Fixed paths and data facts for the condition-invariant experiment.

This experiment reuses the existing six-channel sacrum IMU windows, but the
model receives only two derived channels:

- AccMagnitude = sqrt(AccX^2 + AccY^2 + AccZ^2)
- GyrMagnitude = sqrt(GyrX^2 + GyrY^2 + GyrZ^2)

No preprocessing files need to be regenerated.
"""

from pathlib import Path


# Directory containing this package: src/condition_invariant/
PACKAGE_DIR = Path(__file__).resolve().parent

# Existing project directory: src/
SRC_DIR = PACKAGE_DIR.parent

# Reuse the raw sacrum windows created by src/02_preprocess.py.
WINDOWS_DIR = (
    SRC_DIR
    / "data"
    / "processed"
    / "windows"
    / "sacrum_time"
)

# Experiment output directories.
MODELS_DIR = (
    SRC_DIR
    / "models"
    / "condition_invariant"
)

RESULTS_DIR = (
    SRC_DIR
    / "results"
    / "condition_invariant"
)


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

CONDITIONS = (
    "st_control",
    "st_fatigue",
    "dt_control",
    "dt_fatigue",
)

# Session 1 supplies representation learning and threshold development.
TRAINING_CONDITIONS = (
    "st_control",
    "st_fatigue",
)

# Session 2 was recorded seven days later and is final test data only.
EVALUATION_CONDITIONS = (
    "dt_control",
    "dt_fatigue",
)


# ---------------------------------------------------------------------------
# Sensor configuration
# ---------------------------------------------------------------------------

# Original channel order in the already-saved NumPy windows.
STORED_SENSOR_CHANNELS = (
    "GyrX",
    "GyrY",
    "GyrZ",
    "AccX",
    "AccY",
    "AccZ",
)

# Features supplied to the model in this experiment.
SENSOR_CHANNELS = (
    "AccMagnitude",
    "GyrMagnitude",
)


# ---------------------------------------------------------------------------
# Window properties
# ---------------------------------------------------------------------------

SAMPLING_RATE_HZ = 128

WINDOW_SECONDS = 2

WINDOW_SAMPLES = (
    SAMPLING_RATE_HZ
    * WINDOW_SECONDS
)

WINDOW_STEP_SAMPLES = 128

WINDOW_OVERLAP = (
    1
    - (
        WINDOW_STEP_SAMPLES
        / WINDOW_SAMPLES
    )
)

NUMBER_OF_CHANNELS = len(
    SENSOR_CHANNELS
)

EXPECTED_WINDOW_SHAPE = (
    WINDOW_SAMPLES,
    NUMBER_OF_CHANNELS,
)


def ensure_output_directories() -> None:
    """Create the experiment's model and result directories."""

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )