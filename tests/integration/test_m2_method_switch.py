from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.t3


def _method(name: str) -> dict[str, object]:
    if name == "pgd_at":
        return {"name": name, "attack": {"loss": "ce"}}
    target = "student_clean" if name == "trades" else "teacher_clean"
    method: dict[str, object] = {
        "name": name,
        "attack": {"loss": "kl", "kl_target": target},
    }
    if name == "trades":
        method["trades_beta"] = 2.0
    if name == "rslad_entropy":
        method.update({"entropy_gamma": 1.0})
    return method


def test_one_epoch_synthetic_method_switch_smoke(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    for name in ("pgd_at", "trades", "rslad", "rslad_entropy", "rslad_student", "rslad_joint"):
        output = tmp_path / name
        data: dict[str, object] = {
            "tier": "smoke",
            "seed": 23,
            "dataset": {"name": "synthetic_cifar", "num_samples": 8, "num_classes": 2, "image_size": 4, "seed": 23},
            "student": {"architecture": "fixture_cnn", "num_classes": 2},
            "method": {
                **_method(name),
                "attack": {
                    **_method(name)["attack"],
                    "epsilon": "1/255",
                    "step_size": "1/255",
                    "steps": 1,
                    "random_start": False,
                },
            },
            "training": {"epochs": 1, "batch_size": 4, "learning_rate": 0.02, "device": "cpu"},
            "output_dir": str(output),
        }
        if name.startswith("rslad"):
            data["teacher"] = {"source": "fixture", "architecture": "fixture_cnn", "num_classes": 2}
        config = tmp_path / f"{name}.yaml"
        config.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, "-m", "ard.cli.train", "--config", str(config)],
            cwd=root,
            env=environment,
            text=True,
            capture_output=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert (output / "best.pt").is_file() and (output / "last.pt").is_file()
