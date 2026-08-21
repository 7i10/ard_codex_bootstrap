from pathlib import Path

from ard.analysis.ert_cw_margin_calibration import _config, _quantiles


def test_margin_calibration_config_and_quantiles_are_frozen() -> None:
    config = _config(Path("configs/analysis/ert_cw_margin_calibration_v1.yaml"))
    assert set(config["runs"]) == {"L2", "L4"}
    assert _quantiles([0.0, 1.0, 2.0, 3.0]) == {"q25": 0.75, "q50": 1.5, "q75": 2.25}
