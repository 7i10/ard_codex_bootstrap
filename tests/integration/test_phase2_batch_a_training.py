"""Plan 0103 Phase 2 batch A end to end through ``ard.cli.train`` (CPU, synthetic smoke protocol).

Mixed batch, AWP (with a warm-up epoch) and SGD's norm/bias weight-decay exclusion each train two
epochs and write their option into the resolved config and their observability into the epoch rows
(resume exactness is tested at Trainer level in tests/unit/test_phase2_batch_a.py); split BN refuses
the BatchNorm-free fixture student before training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from ard.config import load_config
from tests.integration.test_synthetic_training import config_data, run_cli

pytestmark = pytest.mark.t3

ROOT = Path(__file__).resolve().parents[2]


def _write(tmp_path: Path, name: str, **sections: dict[str, Any]) -> tuple[Path, Path]:
    output = tmp_path / name
    data = config_data(output)
    data["dataset"]["num_samples"] = 16
    data["method"]["attack"]["random_start"] = True
    for section, values in sections.items():
        data[section] = {**data[section], **values}
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path, output


def _rows(output: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (output / "epoch-metrics.jsonl").read_text(encoding="utf-8").splitlines()]


def _train(path: Path) -> None:
    completed = run_cli(ROOT, "--config", str(path))
    assert completed.returncode == 0, completed.stderr[-4000:]


def test_mixed_batch_trains_and_reports_its_branches(tmp_path: Path) -> None:
    path, output = _write(
        tmp_path, "mixed", method={"mixed_batch": {"adversarial_fraction": 0.5, "adversarial_weight": 0.3}}
    )
    _train(path)
    resolved = load_config(output / "resolved_config.yaml")
    assert resolved.method.mixed_batch is not None and resolved.method.mixed_batch.adversarial_weight == 0.3
    rows = _rows(output)
    assert len(rows) == 2
    for row in rows:
        # 12 training examples (16 * 0.75) in batches of 4: 2 of every 4 attacked.
        assert row["train_mixed_batch_adversarial_examples"] == row["train_mixed_batch_clean_examples"] == 6.0
        assert 0.0 <= row["train_robust_accuracy"] <= 1.0


def test_awp_warms_up_then_runs(tmp_path: Path) -> None:
    path, output = _write(tmp_path, "awp", method={"awp": {"gamma": 0.01, "warmup_epochs": 1}})
    _train(path)
    assert [row["train_awp_active"] for row in _rows(output)] == [0.0, 1.0]
    assert load_config(output / "resolved_config.yaml").method.awp is not None


def test_sgd_exclusion_builds_a_zero_decay_group(tmp_path: Path) -> None:
    path, output = _write(
        tmp_path,
        "exclude",
        optimizer={"weight_decay": 5e-4, "exclude_norm_bias_from_weight_decay": True},
    )
    _train(path)
    payload = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    groups = payload["optimizer"]["param_groups"]
    assert [group["weight_decay"] for group in groups] == [5e-4, 0.0]
    # fixture_cnn: conv weight + linear weight decay; their two biases do not.
    assert [len(group["params"]) for group in groups] == [2, 2]
    plain_path, plain_output = _write(tmp_path, "plain", optimizer={"weight_decay": 5e-4})
    _train(plain_path)
    plain = torch.load(plain_output / "last.pt", map_location="cpu", weights_only=False)
    assert [group["weight_decay"] for group in plain["optimizer"]["param_groups"]] == [5e-4]


def test_split_bn_refuses_a_student_without_batchnorm(tmp_path: Path) -> None:
    path, output = _write(
        tmp_path,
        "split",
        method={"mixed_batch": {"adversarial_fraction": 0.5, "adversarial_weight": 0.3, "split_batchnorm": True}},
    )
    completed = run_cli(ROOT, "--config", str(path))
    assert completed.returncode != 0
    assert "requires a student with BatchNorm layers" in completed.stderr
    assert not (output / "last.pt").exists()
