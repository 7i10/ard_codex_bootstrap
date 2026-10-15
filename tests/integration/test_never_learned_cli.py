"""Never-learned guard through the real ``ard.cli.train`` (debugging 0030), CPU.

A student whose output is forced to a constant must stop after epoch 1 through
the ordinary failure path: non-zero exit, run-bundle manifest ``failed``, no
``completion.json``, the application-failure error marker, and a
``never-learned.json`` with the reason. Epochs 0 and 1 stay recorded.
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

from ard.config.schema import DatasetConfig
from ard.data import build_dataset

pytestmark = pytest.mark.t3

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "scientific" / "imagenet_mobilenetv4_pgd_at_random_init_lr0025.yaml"

# Forces every student forward to all-zero logits (input-independent), then runs the CLI.
# argv[1] == "crash-after-epoch-0" additionally raises right after epoch 0's
# checkpoints and metric row (as tests/integration/test_relu_surrogate_gradient_training.py).
_CONSTANT_TRAIN = """
import sys
import ard.cli.train as train
from ard.engine import Trainer

original = train.build_student

def build_student(*args, **kwargs):
    model = original(*args, **kwargs)
    model.register_forward_hook(lambda module, inputs, output: output * 0.0)
    return model

train.build_student = build_student
if sys.argv[1] == "crash-after-epoch-0":
    original_fit = Trainer.fit

    def fit(self, *args, **kwargs):
        inner = kwargs.get("on_epoch_end")

        def on_epoch_end(metrics, improved):
            if inner is not None:
                inner(metrics, improved)
            raise RuntimeError("injected crash after epoch 0")

        kwargs["on_epoch_end"] = on_epoch_end
        return original_fit(self, *args, **kwargs)

    Trainer.fit = fit
raise SystemExit(train.main(sys.argv[2:]))
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


def _tiny(*, data_root: Path, output: Path, subset: int | None = None) -> dict[str, Any]:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
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
        epochs=4,
        per_rank_batch_size=4,
        global_batch_size=4,
        num_workers=0,
        device="cpu",
        validation_fraction=0.5,
        train_probe_size=2,
    )
    if subset is not None:
        raw["training"]["selection_subset_size"] = subset
    raw["scheduler"].update(milestones=[1], warmup_epochs=1)
    raw["tracking"] = {
        "mode": "offline",
        "project": "ard-test",
        "run_id": "offline-never-learned",
        "group": raw["tracking"]["group"],
        "panel_size": 2,
        "panel_interval_epochs": 1,
        "artifact_retention": "full",
    }
    raw["evaluation"].update(panel_size=2, autoattack_batch_size=4)
    raw["output_dir"] = str(output)
    return raw


def _setup(tmp_path: Path, *, subset: int | None = None) -> tuple[Path, Path]:
    data_root = tmp_path / "imagenet"
    _imagenet_fixture(data_root)
    output = tmp_path / "run"
    config = tmp_path / "run.yaml"
    raw = _tiny(data_root=data_root, output=output, subset=subset)
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return output, config


def _train(mode: str, *args: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-c", _CONSTANT_TRAIN, mode, *args],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _assert_never_learned_failure(output: Path, completed: subprocess.CompletedProcess[str], *, prefix: str) -> None:
    assert completed.returncode != 0
    assert "NeverLearnedError" in completed.stderr, completed.stderr[-4000:]
    bundle = output / "run-bundle"
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert not (bundle / "completion.json").exists()
    assert (bundle / "error-marker.txt").read_text(encoding="utf-8") == "application failure recorded\n"
    marker = json.loads((output / "never-learned.json").read_text(encoding="utf-8"))
    assert set(marker["observations"]) == {"0", "1"}
    assert all(item["logit_std_over_images"] == 0.0 for item in marker["observations"].values())
    rows = [json.loads(line) for line in (output / "epoch-metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["epoch"] for row in rows] == [0, 1]
    assert all(row[f"{prefix}_distinct_predictions"] == 1.0 for row in rows)
    assert all(row[f"{prefix}_logit_std_over_images"] == 0.0 for row in rows)
    assert torch.load(output / "last.pt", map_location="cpu", weights_only=False)["epoch"] == 1


def test_constant_output_cli_run_fails_after_epoch_one_and_refuses_resume(tmp_path: Path) -> None:
    output, config = _setup(tmp_path)
    _assert_never_learned_failure(output, _train("plain", "--config", str(config)), prefix="val")
    # Review P2-3: the stopped run cannot be resumed past the guard.
    resumed = _train("plain", "--config", str(config), "--resume", str(output / "last.pt"))
    assert resumed.returncode != 0
    assert "refusing --resume: the never-learned guard stopped this run" in resumed.stderr, resumed.stderr[-4000:]


def test_constant_output_with_selection_subset_fires_on_the_val_subset_prefix(tmp_path: Path) -> None:
    output, config = _setup(tmp_path, subset=2)
    _assert_never_learned_failure(output, _train("plain", "--config", str(config)), prefix="val_subset")


def test_crash_after_epoch_zero_then_resume_fires_at_epoch_one(tmp_path: Path) -> None:
    output, config = _setup(tmp_path)
    crashed = _train("crash-after-epoch-0", "--config", str(config))
    assert crashed.returncode != 0 and "injected crash after epoch 0" in crashed.stderr, crashed.stderr[-4000:]
    assert not (output / "never-learned.json").exists()
    assert torch.load(output / "last.pt", map_location="cpu", weights_only=False)["epoch"] == 0
    resumed = _train("plain", "--config", str(config), "--resume", str(output / "last.pt"))
    # Epoch 0's values come from the persisted row; epoch 1 completes the pair.
    _assert_never_learned_failure(output, resumed, prefix="val")
