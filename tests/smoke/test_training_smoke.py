"""Bounded synthetic CPU/CUDA training smokes for the M5 handoff gate."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

pytestmark = [pytest.mark.t3, pytest.mark.smoke]


def smoke_config(output: Path, *, device: str, world_size: int = 1) -> dict[str, object]:
    """The fixed one-epoch fixture deliberately cannot become a real run."""
    return {
        "schema_version": 2,
        "protocol": {"id": "synthetic_smoke_v2"},
        "tier": "smoke",
        "seeds": {
            k: 41
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
        "dataset": {"name": "synthetic_cifar", "num_samples": 8, "num_classes": 3, "image_size": 4, "seed": 41},
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
        "training": {
            "epochs": 1,
            "per_rank_batch_size": 4,
            "global_batch_size": 4 * world_size,
            "device": device,
        },
        "output_dir": str(output),
        "tracker_run_id": "m5-smoke",
    }


def write_config(tmp_path: Path, *, device: str, world_size: int = 1) -> tuple[Path, Path]:
    output = tmp_path / "run"
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(smoke_config(output, device=device, world_size=world_size), sort_keys=False),
        encoding="utf-8",
    )
    return config_path, output


def training_environment(root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    return environment


def assert_train_succeeds(command: list[str], *, root: Path, output: Path) -> None:
    completed = subprocess.run(
        command, cwd=root, env=training_environment(root), text=True, capture_output=True, timeout=60
    )
    assert completed.returncode == 0, completed.stderr
    assert (output / "best.pt").is_file()
    assert (output / "last.pt").is_file()


def test_bounded_synthetic_cpu_training_smoke(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    config_path, output = write_config(tmp_path, device="cpu")
    assert_train_succeeds(
        [sys.executable, "-m", "ard.cli.train", "--config", str(config_path)], root=root, output=output
    )


def test_two_rank_smoke_config_declares_effective_global_batch(tmp_path: Path) -> None:
    config = smoke_config(tmp_path / "run", device="cuda", world_size=2)
    training = config["training"]
    assert isinstance(training, dict)
    assert training["per_rank_batch_size"] == 4
    assert training["global_batch_size"] == 8


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_bounded_synthetic_single_cuda_training_smoke(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    config_path, output = write_config(tmp_path, device="cuda")
    assert_train_succeeds(
        [sys.executable, "-m", "ard.cli.train", "--config", str(config_path)], root=root, output=output
    )


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices are unavailable")
def test_bounded_synthetic_two_cuda_ddp_training_smoke(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    config_path, output = write_config(tmp_path, device="cuda", world_size=2)
    assert_train_succeeds(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(root / "tests" / "smoke" / "torchrun_cuda_train.py"),
            "--config",
            str(config_path),
        ],
        root=root,
        output=output,
    )
