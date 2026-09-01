from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_i100_transfer_manifest_uses_inclusive_epoch_114_endpoint(tmp_path: Path) -> None:
    script = Path(__file__).parents[2] / "scripts/build_i100_transfer_manifest.py"
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "status": "complete_no_update",
                "beta_advce": 0.1,
                "margin_coefficient": 0.2,
                "margin_floor": 0.1,
                "margin_cap": 0.3,
            }
        ),
        encoding="utf-8",
    )
    config1 = tmp_path / "dev1.yaml"
    config2 = tmp_path / "dev2.yaml"
    mask1 = tmp_path / "mask1.json"
    mask2 = tmp_path / "mask2.json"
    for path in (config1, config2, mask1, mask2):
        path.write_text("placeholder\n", encoding="utf-8")
    output = tmp_path / "manifest.json"
    command = [
        sys.executable,
        str(script),
        "--output",
        str(output),
        "--source-sha",
        "a" * 40,
        "--root",
        str(tmp_path / "root"),
        "--repo",
        str(tmp_path / "repo"),
        "--calibration",
        str(calibration),
        "--dev1-config",
        str(config1),
        "--dev2-config",
        str(config2),
        "--dev1-mask",
        str(mask1),
        "--dev2-mask",
        str(mask2),
    ]
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    manifest = json.loads(output.read_text(encoding="utf-8"))
    train_jobs = [job for job in manifest["jobs"] if job["job_id"].startswith("train-")]
    assert len(train_jobs) == 8
    for job in train_jobs:
        command = job["command"]
        assert command[command.index("--epochs") + 1] == "115"
        assert command[command.index("--horizon-epochs") + 1 : command.index("--run-namespace")] == [
            "104",
            "109",
            "114",
        ]
