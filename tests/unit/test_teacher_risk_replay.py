from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ard.analysis import teacher_risk_replay
from ard.analysis.signal_audit import CheckpointInventory
from ard.analysis.teacher_risk_replay import load_historical_student, replay_rows, replay_source_hashes
from ard.attacks import LinfPGD
from ard.cli.replay_teacher_risk import main as replay_teacher_risk_main
from ard.config.schema import AttackConfig
from ard.data import IndexedBatch
from ard.engine.checkpoint import REQUIRED_KEYS

pytestmark = pytest.mark.t2


def _model(classes: int = 3) -> nn.Module:
    torch.manual_seed(9)
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, classes))


def _batches() -> tuple[IndexedBatch, ...]:
    torch.manual_seed(4)
    return (
        IndexedBatch(
            images=torch.rand(2, 3, 4, 4),
            labels=torch.tensor([1, 2]),
            sample_ids=torch.tensor([7, 3]),
        ),
        IndexedBatch(
            images=torch.rand(1, 3, 4, 4),
            labels=torch.tensor([0]),
            sample_ids=torch.tensor([11]),
        ),
    )


def test_teacher_risk_replay_uses_deterministic_kl_pgd_and_frozen_teacher_gradients() -> None:
    student, teacher = _model(), _model()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    attack = LinfPGD(
        AttackConfig(
            loss="kl",
            kl_target="teacher_clean",
            epsilon="8/255",
            step_size="2/255",
            steps=10,
            random_start=True,
            student_mode="eval",
            teacher_mode="eval",
        )
    )
    first = replay_rows(
        student=student,
        teacher=teacher,
        loader=_batches(),
        attack=attack,
        device=torch.device("cpu"),
        attack_seed_base=17,
    )
    second = replay_rows(
        student=student,
        teacher=teacher,
        loader=_batches(),
        attack=attack,
        device=torch.device("cpu"),
        attack_seed_base=17,
    )
    assert first.rows == second.rows
    assert [row["sample_id"] for row in first.rows] == [7, 3, 11]
    assert first.max_abs_delta <= 8 / 255 + 1e-7
    for row in first.rows:
        assert row["namespace"] == "train"
        assert row["teacher_risk"] == pytest.approx(1.0 - row["teacher_entropy"] / math.log(3), abs=1e-7)
        assert 0.0 <= row["teacher_risk"] <= 1.0
        assert row["teacher_correct"] is (row["teacher_prediction"] == row["class_id"])
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in teacher.parameters())


def test_replay_source_provenance_hashes_both_entry_points() -> None:
    hashes = replay_source_hashes()
    assert set(hashes) == {"analysis_module", "cli_module"}
    assert all(len(value) == 64 for value in hashes.values())


def test_historical_loader_accepts_explicit_saved_raw_config_hash_without_changing_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"fixture")
    checkpoint = CheckpointInventory(
        "run", "model", ("last",), 0, str(path), "a" * 64, 4, False, 0, "raw" * 21 + "r", "b" * 40
    )
    captured: dict[str, object] = {}

    def validate(_path: Path, *, expected_config_hash: str) -> dict[str, object]:
        captured["hash"] = expected_config_hash
        return {**{key: None for key in REQUIRED_KEYS}, "tracker_run_id": "run", "epoch": 4}

    monkeypatch.setattr(teacher_risk_replay, "sha256_file", lambda _: "a" * 64)
    monkeypatch.setattr(teacher_risk_replay, "resolved_config_dict", lambda _: {"current": "defaults"})
    monkeypatch.setattr(teacher_risk_replay, "config_digest", lambda _: "reserialized-defaults")
    monkeypatch.setattr(
        teacher_risk_replay,
        "validate_checkpoint_lineage",
        validate,
    )
    monkeypatch.setattr(teacher_risk_replay, "build_student", lambda *_args, **_kwargs: nn.Identity())
    monkeypatch.setattr(teacher_risk_replay, "load_saved_student_checkpoint", lambda *_args: None)

    model, _ = load_historical_student(
        checkpoint,
        config=SimpleNamespace(student=object(), tier="smoke"),
        device=torch.device("cpu"),
        expected_config_hash=checkpoint.config_hash,
    )

    assert isinstance(model, nn.Identity)
    assert captured["hash"] == checkpoint.config_hash


def test_replay_cli_rejects_batch_size_drift_before_reading_artifacts(tmp_path: Path) -> None:
    config = tmp_path / "audit.json"
    config.write_text(json.dumps({"replay_batch_size": 128}), encoding="utf-8")
    with pytest.raises(ValueError, match="replay_batch_size"):
        replay_teacher_risk_main(
            [
                "--config",
                str(config),
                "--output",
                str(tmp_path / "replay.json"),
                "--device",
                "cpu",
                "--batch-size",
                "64",
            ]
        )


def test_replay_cli_rejects_device_type_and_distributed_world_size_before_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "audit.json"
    config.write_text(json.dumps({"replay_batch_size": 128, "replay_device_type": "cuda"}), encoding="utf-8")
    args = [
        "--config",
        str(config),
        "--output",
        str(tmp_path / "replay.json"),
        "--device",
        "cpu",
        "--batch-size",
        "128",
    ]
    with pytest.raises(ValueError, match="device type"):
        replay_teacher_risk_main(args)
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="WORLD_SIZE=1"):
        replay_teacher_risk_main(args)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    with pytest.raises(ValueError, match="initialized distributed"):
        replay_teacher_risk_main(args)
