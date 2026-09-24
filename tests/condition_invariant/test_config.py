"""Checks for the condition-invariant experiment configuration."""

from src.condition_invariant import config


def test_fixed_window_properties() -> None:
    assert config.SAMPLING_RATE_HZ == 128
    assert config.WINDOW_SECONDS == 2
    assert config.WINDOW_SAMPLES == 256
    assert config.WINDOW_STEP_SAMPLES == 128
    assert config.WINDOW_OVERLAP == 0.5
    assert config.EXPECTED_WINDOW_SHAPE == (256, 6)


def test_all_condition_input_directories_exist() -> None:
    assert config.WINDOWS_DIR.is_dir()
    for condition in config.CONDITIONS:
        assert (config.WINDOWS_DIR / condition).is_dir()


def test_every_condition_has_all_sixteen_window_files() -> None:
    for condition in config.CONDITIONS:
        window_files = list((config.WINDOWS_DIR / condition).glob("*_windows.npy"))
        assert len(window_files) == 16


def test_output_directories_are_separate_from_baseline() -> None:
    config.ensure_output_directories()
    assert config.MODELS_DIR.is_dir()
    assert config.RESULTS_DIR.is_dir()
    assert config.MODELS_DIR.name == "condition_invariant"
    assert config.RESULTS_DIR.name == "condition_invariant"
