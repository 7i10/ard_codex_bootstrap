"""``training.relu_surrogate_gradient`` end to end on CPU (SAT smooth ReLU backward, 2026-10-15).

The checked-in ``imagenet_mobilenetv4_pgd_at_phase2_smoothgrad_opt.yaml`` (scope optimizer), shrunk to a tiny
on-disk ImageNet fixture and the ReLU ``fixture_cnn`` student, deterministic on CPU, runs through the real
``ard.cli.train`` and ``ard.cli.evaluate``:

* last.pt differs from the same-seed run without the option (the option changes the training updates);
* one epoch, an injected crash, then ``--resume`` to two epochs gives last.pt bitwise equal to an
  uninterrupted two-epoch run;
* the evaluation record's training_protocol_identity carries the option, and the checkpoint loads strictly
  into the plain ReLU architecture.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from PIL import Image
from torch import nn

from ard.config.schema import DatasetConfig, ModelConfig
from ard.data import build_dataset
from ard.models import build_student

pytestmark = pytest.mark.t3

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "scientific" / "imagenet_mobilenetv4_pgd_at_phase2_smoothgrad_opt.yaml"

# Runs ard.cli.train, crashing right after the first epoch's checkpoints are written.
_CRASH_AFTER_FIRST_EPOCH = """
import sys
from ard.engine import Trainer

original_fit = Trainer.fit

def fit(self, *args, **kwargs):
    inner = kwargs.get("on_epoch_end")

    def on_epoch_end(metrics, improved):
        if inner is not None:
            inner(metrics, improved)
        raise RuntimeError("injected crash after the first epoch")

    kwargs["on_epoch_end"] = on_epoch_end
    return original_fit(self, *args, **kwargs)

Trainer.fit = fit
from ard.cli.train import main
raise SystemExit(main(sys.argv[1:]))
"""


def _imagenet_fixture(root: Path) -> None:
    for split in ("train", "val"):
        for class_index, wnid in enumerate(("n001", "n002")):
            class_dir = root / split / wnid
            class_dir.mkdir(parents=True)
            for image_index in range(6):
                generator = torch.Generator().manual_seed(
                    100 * class_index + image_index + (7 if split == "val" else 0)
                )
                pixels = torch.rand(40, 48, 3, generator=generator)
                Image.fromarray((pixels * 255).to(torch.uint8).numpy()).save(class_dir / f"{wnid}_{image_index}.JPEG")


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run(
        [sys.executable, *args], cwd=ROOT, env=environment, text=True, capture_output=True, check=False
    )


def _tiny(*, data_root: Path, output: Path, run_id: str, surrogate: bool) -> dict[str, Any]:
    """The checked-in config with scale, tier, tracking and the CPU determinism setting shrunk."""
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert raw["training"]["relu_surrogate_gradient"] == {"beta": 10.0, "scope": "optimizer"}
    for key, split in (("dataset", "train"), ("evaluation", "val")):
        block = raw[key] if key == "dataset" else raw[key]["dataset"]
        observed = build_dataset(
            DatasetConfig(name="imagenet", root=data_root, split=split, num_classes=2, image_size=32)
        ).content_identity
        assert observed is not None
        block.update(root=str(data_root), num_classes=2, image_size=32, content_sha256=observed["observed_sha256"])
    raw["tier"] = "smoke"
    raw["student"].update(architecture="fixture_cnn", num_classes=2, pretrained=False)
    raw["seeds"] = dict.fromkeys(raw["seeds"], 3)
    raw["training"].update(
        epochs=2,
        per_rank_batch_size=4,
        global_batch_size=4,
        num_workers=0,
        device="cpu",
        deterministic=True,
        cudnn_benchmark=False,
        validation_fraction=0.5,
        train_probe_size=2,
        selection_subset_size=2,
    )
    if not surrogate:
        del raw["training"]["relu_surrogate_gradient"]
    raw["scheduler"].update(milestones=[1], warmup_epochs=1)
    raw["tracking"] = {
        "mode": "offline",
        "project": "ard-test",
        "run_id": run_id,
        "group": raw["tracking"]["group"],
        "panel_size": 2,
        "panel_interval_epochs": 1,
        "artifact_retention": "full",
    }
    raw["evaluation"].update(panel_size=2, autoattack_batch_size=4)
    raw["output_dir"] = str(output)
    return raw


def _config(tmp_path: Path, name: str, *, data_root: Path, surrogate: bool) -> tuple[Path, Path]:
    output = tmp_path / name
    config = tmp_path / f"{name}.yaml"
    raw = _tiny(data_root=data_root, output=output, run_id=f"offline-{name}", surrogate=surrogate)
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return config, output


def _train(config: Path) -> None:
    trained = _run(["-m", "ard.cli.train", "--config", str(config)])
    assert trained.returncode == 0, trained.stderr


def _model(output: Path) -> dict[str, torch.Tensor]:
    return torch.load(output / "last.pt", map_location="cpu", weights_only=False)["model"]


def test_surrogate_cli_changes_training_resumes_exactly_and_evaluates_the_plain_model(tmp_path: Path) -> None:
    data_root = tmp_path / "imagenet"
    _imagenet_fixture(data_root)
    plain_config, plain_out = _config(tmp_path, "plain", data_root=data_root, surrogate=False)
    full_config, full_out = _config(tmp_path, "surrogate", data_root=data_root, surrogate=True)
    resumed_config, resumed_out = _config(tmp_path, "resumed", data_root=data_root, surrogate=True)
    _train(plain_config)
    _train(full_config)

    # (a) The option changes the trained weights (same seed, same data, same everything else).
    plain, full = _model(plain_out), _model(full_out)
    assert list(plain) == list(full)
    assert any(not torch.equal(plain[key], full[key]) for key in plain)

    # (b) Crash after epoch 1, resume from last.pt: bitwise the uninterrupted run.
    crashed = _run(["-c", _CRASH_AFTER_FIRST_EPOCH, "--config", str(resumed_config)])
    assert crashed.returncode != 0 and "injected crash after the first epoch" in crashed.stderr
    first_leg = torch.load(resumed_out / "last.pt", map_location="cpu", weights_only=False)
    assert first_leg["epoch"] == 0
    resumed = _run(
        [
            "-m",
            "ard.cli.train",
            "--config",
            str(resumed_out / "resolved_config.yaml"),
            "--resume",
            str(resumed_out / "last.pt"),
        ]
    )
    assert resumed.returncode == 0, resumed.stderr
    resumed_model = _model(resumed_out)
    assert list(resumed_model) == list(full)
    for key, value in full.items():
        assert torch.equal(resumed_model[key], value), key

    # (c) Evaluation records the option and builds the plain ReLU architecture, which loads last.pt strictly.
    for output, expected in ((plain_out, None), (full_out, {"beta": 10.0, "scope": "optimizer"})):
        evaluated = _run(
            [
                "-m",
                "ard.cli.evaluate",
                "--config",
                str(output / "resolved_config.yaml"),
                "--checkpoint-dir",
                str(output),
            ]
        )
        assert evaluated.returncode == 0, evaluated.stderr
        results = json.loads((output / "evaluation" / "evaluation-results.json").read_text(encoding="utf-8"))
        assert sorted(item["checkpoint_alias"] for item in results) == ["best", "last"]
        for item in results:
            assert item["training_protocol_identity"].get("relu_surrogate_gradient") == expected
    plain_student = build_student(ModelConfig(architecture="fixture_cnn", num_classes=2), tier="smoke")
    plain_student.load_state_dict(full, strict=True)
    assert sum(type(module) is nn.ReLU for module in plain_student.modules()) == 1
