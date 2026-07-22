from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch
import yaml

from ard.config import load_config

pytestmark = pytest.mark.t3


def config_data(output: Path) -> dict:
    return {
        "schema_version": 2,
        "protocol": {"id": "synthetic_smoke_v2"},
        "tier": "smoke",
        "seeds": {
            k: 8
            for k in (
                "split",
                "model_init",
                "data_order",
                "augmentation",
                "train_attack",
                "evaluation_attack",
                "qualitative_panel",
            )
        },
        "dataset": {"name": "synthetic_cifar", "num_samples": 8, "num_classes": 3, "image_size": 4, "seed": 8},
        "student": {"architecture": "fixture_cnn", "num_classes": 3},
        "method": {
            "id": "pgd_at",
            "version": 1,
            "attack": {
                "epsilon": "1/255",
                "step_size": "1/255",
                "steps": 1,
                "random_start": False,
            },
        },
        "optimizer": {"id": "sgd", "learning_rate": 0.02, "momentum": 0.9, "weight_decay": 0.0, "nesterov": False},
        "scheduler": {"id": "identity", "milestones": [], "gamma": 1.0, "step_at": "epoch_end"},
        "training": {"epochs": 2, "per_rank_batch_size": 4, "global_batch_size": 4, "device": "cpu"},
        "output_dir": str(output),
        "tracker_run_id": "smoke-local",
    }


def two_rank_config_data(output: Path) -> dict:
    data = config_data(output)
    data["training"]["global_batch_size"] = data["training"]["per_rank_batch_size"] * 2
    return data


def run_cli(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    return subprocess.run(
        [sys.executable, "-m", "ard.cli.train", *arguments],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
    )


def run_torchrun(
    root: Path,
    *arguments: str,
    script: str = "torchrun_delayed_train.py",
    environment_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    environment.update(environment_overrides or {})
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(root / "tests" / "integration" / script),
            *arguments,
        ],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
    )


def test_train_cli_dry_run_overrides_and_two_epoch_synthetic_e2e(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    config_path = tmp_path / "experiment.yaml"
    output = tmp_path / "run"
    config_path.write_text(yaml.safe_dump(config_data(output), sort_keys=False), encoding="utf-8")
    dry_output = tmp_path / "dry"
    dry = run_cli(root, "--config", str(config_path), "--output", str(dry_output), "--dry-run", "seeds.model_init=21")
    assert dry.returncode == 0, dry.stderr
    assert load_config(dry_output / "resolved_config.yaml").seed == 21

    trained = run_cli(root, "--config", str(config_path))
    assert trained.returncode == 0, trained.stderr
    assert (output / "resolved_config.yaml").is_file()
    assert (output / "best.pt").is_file()
    assert (output / "last.pt").is_file()


def test_cli_rejects_output_collision_and_test_split_before_any_write(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    collision = tmp_path / "collision"
    collision.mkdir()
    sentinel = collision / "keep.txt"
    sentinel.write_text("do not overwrite", encoding="utf-8")
    collision_config = tmp_path / "collision.yaml"
    collision_config.write_text(yaml.safe_dump(config_data(collision), sort_keys=False), encoding="utf-8")
    result = run_cli(root, "--config", str(collision_config), "--dry-run")
    assert result.returncode != 0
    assert "refusing to overwrite" in result.stderr
    assert sentinel.read_text(encoding="utf-8") == "do not overwrite"

    test_output = tmp_path / "test-output"
    test_data = config_data(test_output)
    test_data["dataset"]["split"] = "test"
    test_config = tmp_path / "test.yaml"
    test_config.write_text(yaml.safe_dump(test_data, sort_keys=False), encoding="utf-8")
    result = run_cli(root, "--config", str(test_config), "--dry-run")
    assert result.returncode != 0
    assert "official train split" in result.stderr
    assert not test_output.exists()


def test_cli_rejects_invalid_production_tracking_before_output_write(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "production-output"
    data = config_data(output)
    data["tier"] = "production"
    data["dataset"] = {"name": "cifar10", "root": str(tmp_path / "data"), "num_classes": 10}
    data["student"] = {
        "architecture": "resnet18_cifar",
        "num_classes": 10,
        "normalization": {"profile": "cifar10_standard"},
    }
    config_path = tmp_path / "production.yaml"
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    result = run_cli(root, "--config", str(config_path), "--dry-run")

    assert result.returncode != 0
    assert "production requires non-disabled tracking" in result.stderr
    assert not output.exists()


def test_two_process_gloo_coordinates_fresh_output_and_collision_error(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "distributed-output"
    config_path = tmp_path / "distributed.yaml"
    config_path.write_text(yaml.safe_dump(two_rank_config_data(output), sort_keys=False), encoding="utf-8")

    fresh = run_torchrun(root, "--config", str(config_path), "--dry-run")
    assert fresh.returncode == 0, fresh.stderr
    resolved = output / "resolved_config.yaml"
    assert resolved.is_file()
    before = resolved.read_bytes()
    assert sorted(path.name for path in output.iterdir()) == ["resolved_config.yaml"]

    collision = run_torchrun(root, "--config", str(config_path), "--dry-run")
    assert collision.returncode != 0
    assert "rank-zero output guard failed (FileExistsError)" in collision.stderr
    assert resolved.read_bytes() == before
    assert sorted(path.name for path in output.iterdir()) == ["resolved_config.yaml"]


def test_two_process_gloo_checkpoint_success_and_rank_zero_failure_propagation(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    success_output = tmp_path / "checkpoint-success"
    success_data = two_rank_config_data(success_output)
    success_data["training"]["epochs"] = 1
    success_config = tmp_path / "checkpoint-success.yaml"
    success_config.write_text(yaml.safe_dump(success_data, sort_keys=False), encoding="utf-8")

    success = run_torchrun(root, "--config", str(success_config))
    assert success.returncode == 0, success.stderr
    best = success_output / "best.pt"
    last = success_output / "last.pt"
    assert best.is_file() and last.is_file() and best != last
    assert torch.load(best, map_location="cpu", weights_only=False)["world_size"] == 2
    assert torch.load(last, map_location="cpu", weights_only=False)["world_size"] == 2

    failure_output = tmp_path / "checkpoint-failure"
    failure_data = two_rank_config_data(failure_output)
    failure_data["training"]["epochs"] = 1
    failure_config = tmp_path / "checkpoint-failure.yaml"
    failure_config.write_text(yaml.safe_dump(failure_data, sort_keys=False), encoding="utf-8")
    failed = run_torchrun(
        root,
        "--config",
        str(failure_config),
        script="torchrun_checkpoint_failure.py",
    )
    assert failed.returncode != 0
    assert "ARD_CHECKPOINT_FAILURE_RANK=0: rank-zero checkpoint write last.pt failed (OSError)" in failed.stderr
    assert "ARD_CHECKPOINT_FAILURE_RANK=1: rank-zero checkpoint write last.pt failed (OSError)" in failed.stderr
    assert (failure_output / "resolved_config.yaml").is_file()
    assert not (failure_output / "last.pt").exists()
    assert not (failure_output / "best.pt").exists()


def test_two_process_gloo_sample_statistics_are_rank_zero_only_and_fail_coherently(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    success_output = tmp_path / "sample-stats-success"
    success_data = two_rank_config_data(success_output)
    success_data["dataset"]["num_classes"] = 2
    success_data["student"]["num_classes"] = 2
    success_data["training"]["epochs"] = 1
    success_config = tmp_path / "sample-stats-success.yaml"
    success_config.write_text(yaml.safe_dump(success_data, sort_keys=False), encoding="utf-8")

    success = run_torchrun(
        root,
        "--config",
        str(success_config),
        script="torchrun_sample_stats.py",
    )
    assert success.returncode == 0, success.stderr
    stats = success_output / "sample-stats-train.parquet"
    assert stats.read_bytes()[:4] == b"PAR1"
    assert pq.read_table(stats).column("sample_id").to_pylist() == [0, 1, 2, 4, 5, 7]

    failure_output = tmp_path / "sample-stats-failure"
    failure_data = two_rank_config_data(failure_output)
    failure_data["dataset"]["num_classes"] = 2
    failure_data["student"]["num_classes"] = 2
    failure_data["training"]["epochs"] = 1
    failure_config = tmp_path / "sample-stats-failure.yaml"
    failure_config.write_text(yaml.safe_dump(failure_data, sort_keys=False), encoding="utf-8")
    failed = run_torchrun(
        root,
        "--config",
        str(failure_config),
        script="torchrun_sample_stats.py",
        environment_overrides={"ARD_INJECT_SAMPLE_STATS_FAILURE": "1"},
    )
    assert failed.returncode != 0
    expected = "rank-zero sample statistics write failed (OSError): injected sample statistics failure"
    assert f"ARD_SAMPLE_STATS_FAILURE_RANK=0: {expected}" in failed.stderr
    assert f"ARD_SAMPLE_STATS_FAILURE_RANK=1: {expected}" in failed.stderr
    assert not (failure_output / "sample-stats-train.parquet").exists()
    bundle = failure_output / "run-bundle"
    manifest = yaml.safe_load((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert not (bundle / "completion.json").exists()
