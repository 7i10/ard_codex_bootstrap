"""Plan 0103 two-stage low-resolution PGD-AT, end to end on CPU.

The two checked-in stage configs, shrunk to a tiny on-disk ImageNet fixture
(fixture student, 32px evaluation / 16px stage-1 training, 2 epochs each),
run through the real ``ard.cli.train`` and ``ard.cli.evaluate``:

* stage 1 trains and selects at the reduced resolution, and its official
  evaluation runs at the full resolution;
* stage 2 refuses a wrong digest before writing anything, then starts from
  exactly stage 1's ``last.pt`` student weights with a fresh optimizer,
  scheduler and step counter, and records the init lineage in its run bundle
  and its evaluation identity.
"""

from __future__ import annotations

import hashlib
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

from ard.config import load_config
from ard.config.schema import DatasetConfig
from ard.data import build_dataset

pytestmark = pytest.mark.t3

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "scientific"
STAGE1 = CONFIG_DIR / "imagenet_mobilenetv4_twostage_stage1_112_pgd1.yaml"
STAGE2 = CONFIG_DIR / "imagenet_mobilenetv4_twostage_stage2_224_pgd3_ft.yaml"

# Runs ard.cli.train after wrapping Trainer.fit to snapshot the state it starts from.
_SNAPSHOT_TRAIN = """
import sys
import torch
from ard.engine import Trainer
from ard.engine.distributed import unwrap_model

original_fit = Trainer.fit

def fit(self, *args, **kwargs):
    torch.save(
        {
            "model": {key: value.clone() for key, value in unwrap_model(self.model).state_dict().items()},
            "optimizer_state_entries": len(self.optimizer.state),
            "learning_rates": [group["lr"] for group in self.optimizer.param_groups],
            "scheduler_last_epoch": self.scheduler.last_epoch,
            "global_step": self.global_step,
            "start_epoch": kwargs.get("start_epoch"),
            # Item shapes only: indexing a view draws from no global RNG.
            "image_shapes": {
                "train": tuple(args[0].dataset[0][0].shape),
                "validation": tuple(kwargs["validation_loader"].dataset[0][0].shape),
                "probe": tuple(kwargs["probe_loader"].dataset[0][0].shape),
            },
        },
        sys.argv[1],
    )
    return original_fit(self, *args, **kwargs)

Trainer.fit = fit
from ard.cli.train import main
raise SystemExit(main(sys.argv[2:]))
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


def _environment(extra: dict[str, str]) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment.update(extra)
    return environment


def _run(args: list[str], environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args], cwd=ROOT, env=environment, text=True, capture_output=True, check=False
    )


def _tiny(path: Path, *, data_root: Path, output: Path, run_id: str) -> dict[str, Any]:
    """The checked-in stage config with only scale, tier and tracking shrunk."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key, split in (("dataset", "train"), ("evaluation", "val")):
        block = raw[key] if key == "dataset" else raw[key]["dataset"]
        observed = build_dataset(
            DatasetConfig(name="imagenet", root=data_root, split=split, num_classes=2, image_size=32)
        ).content_identity
        assert observed is not None
        block.update(root=str(data_root), num_classes=2, image_size=32, content_sha256=observed["observed_sha256"])
    raw["tier"] = "smoke"
    raw["student"].update(architecture="fixture_cnn", num_classes=2)
    raw["seeds"] = dict.fromkeys(raw["seeds"], 3)
    raw["training"].update(
        epochs=2,
        per_rank_batch_size=4,
        global_batch_size=4,
        num_workers=0,
        device="cpu",
        validation_fraction=0.34,
        train_probe_size=2,
    )
    if "train_image_size" in raw["training"]:
        raw["training"]["train_image_size"] = 16
    raw["scheduler"].update(milestones=[1], warmup_epochs=1)
    raw["tracking"] = {
        "mode": "offline",
        "project": "ard-test",
        "run_id": run_id,
        "group": raw["tracking"]["group"],
        "panel_size": 2,
        "panel_interval_epochs": 1,
        # Local artifact records, so a terminal no-op resume can validate them.
        "artifact_retention": "full",
    }
    raw["evaluation"].update(panel_size=2, autoattack_batch_size=4)
    raw["output_dir"] = str(output)
    return raw


def _write(path: Path, raw: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_two_stage_cli_trains_stage1_small_then_fine_tunes_stage2_from_its_last_checkpoint(tmp_path: Path) -> None:
    data_root = tmp_path / "imagenet"
    _imagenet_fixture(data_root)
    environment = _environment({})

    # ---- stage 1: 16px training/selection, 32px official evaluation
    out1 = tmp_path / "stage1"
    raw1 = _tiny(STAGE1, data_root=data_root, output=out1, run_id="offline-stage1")
    config1 = _write(tmp_path / "stage1.yaml", raw1)
    snapshot1 = tmp_path / "stage1-start.pt"
    trained = _run(["-c", _SNAPSHOT_TRAIN, str(snapshot1), "--config", str(config1)], environment)
    assert trained.returncode == 0, trained.stderr
    # Training crop, in-training validation and train probe all at 16px.
    assert torch.load(snapshot1, weights_only=False)["image_shapes"] == dict.fromkeys(
        ("train", "validation", "probe"), (3, 16, 16)
    )
    resolved1 = load_config(out1 / "resolved_config.yaml")
    assert resolved1.training.train_image_size == 16 and resolved1.dataset.image_size == 32
    assert resolved1.method.attack is not None and resolved1.method.attack.steps == 1
    manifest1 = json.loads((out1 / "run-bundle" / "manifest.json").read_text(encoding="utf-8"))
    assert "init_lineage" not in manifest1
    last1 = out1 / "last.pt"
    stage1_weights = torch.load(last1, map_location="cpu", weights_only=False)["model"]

    evaluated = _run(
        ["-m", "ard.cli.evaluate", "--config", str(out1 / "resolved_config.yaml"), "--checkpoint-dir", str(out1)],
        environment,
    )
    assert evaluated.returncode == 0, evaluated.stderr
    results1 = json.loads((out1 / "evaluation" / "evaluation-results.json").read_text(encoding="utf-8"))
    assert sorted(item["checkpoint_alias"] for item in results1) == ["best", "last"]
    for item in results1:
        assert item["dataset_identity"]["image_size"] == 32
        assert item["training_protocol_identity"]["train_image_size"] == 16
        assert "init_checkpoint_sha256" not in item["training_protocol_identity"]

    # ---- stage 2: refuses a wrong digest before writing anything
    digest = hashlib.sha256(last1.read_bytes()).hexdigest()
    out2 = tmp_path / "stage2"
    raw2 = _tiny(STAGE2, data_root=data_root, output=out2, run_id="offline-stage2")
    assert raw2["training"]["init_checkpoint"] == {
        "path": "${ARD_STAGE1_CHECKPOINT}",
        "sha256": "${ARD_STAGE1_CHECKPOINT_SHA256}",
    }
    config2 = _write(tmp_path / "stage2.yaml", raw2)
    wrong = _environment({"ARD_STAGE1_CHECKPOINT": str(last1), "ARD_STAGE1_CHECKPOINT_SHA256": "0" * 64})
    refused = _run(["-m", "ard.cli.train", "--config", str(config2), "--dry-run"], wrong)
    assert refused.returncode != 0 and "SHA-256 mismatch" in refused.stderr
    assert not out2.exists()

    # ---- stage 2: exact stage-1 weights, fresh optimizer/scheduler/step
    right = _environment({"ARD_STAGE1_CHECKPOINT": str(last1), "ARD_STAGE1_CHECKPOINT_SHA256": digest})
    snapshot2 = tmp_path / "stage2-start.pt"
    trained = _run(["-c", _SNAPSHOT_TRAIN, str(snapshot2), "--config", str(config2)], right)
    assert trained.returncode == 0, trained.stderr
    start = torch.load(snapshot2, map_location="cpu", weights_only=False)
    assert start["model"].keys() == stage1_weights.keys()
    for key, value in stage1_weights.items():
        assert torch.equal(start["model"][key], value), key
    assert start["image_shapes"] == dict.fromkeys(("train", "validation", "probe"), (3, 32, 32))
    assert start["optimizer_state_entries"] == 0
    assert start["global_step"] == 0 and start["start_epoch"] == 0 and start["scheduler_last_epoch"] == 0
    # warmup_multistep, 1 warmup epoch: epoch 0 runs at the configured peak 0.0025.
    assert start["learning_rates"] == [pytest.approx(0.0025)]
    resolved2 = load_config(out2 / "resolved_config.yaml")
    assert resolved2.training.init_checkpoint is not None
    assert resolved2.training.init_checkpoint.path == last1 and resolved2.training.init_checkpoint.sha256 == digest
    assert resolved2.training.train_image_size is None
    manifest2 = json.loads((out2 / "run-bundle" / "manifest.json").read_text(encoding="utf-8"))
    lineage = manifest2["init_lineage"]
    assert lineage["kind"] == "student_init_checkpoint_v1"
    assert (lineage["path"], lineage["sha256"]) == (str(last1), digest)
    assert lineage["source_tracker_run_id"] == "offline-stage1"
    assert lineage["source_config_hash"] == manifest1["config_hash"]
    assert (lineage["source_epoch"], lineage["source_epochs"], lineage["source_train_image_size"]) == (1, 2, 16)
    assert manifest2["status"] in {"completed", "sync_pending"}

    # An epoch-boundary resume never re-loads the init file (the resumed
    # weights win) but requires the recorded lineage; here a terminal no-op.
    resumed = _run(["-m", "ard.cli.train", "--config", str(config2), "--resume", str(out2 / "last.pt")], right)
    assert resumed.returncode == 0, resumed.stderr
    assert json.loads((out2 / "run-bundle" / "manifest.json").read_text(encoding="utf-8"))["init_lineage"] == lineage
    manifest_path = out2 / "run-bundle" / "manifest.json"
    original_manifest = manifest_path.read_text(encoding="utf-8")
    stripped = json.loads(original_manifest)
    del stripped["init_lineage"]
    manifest_path.write_text(json.dumps(stripped), encoding="utf-8")
    refused = _run(["-m", "ard.cli.train", "--config", str(config2), "--resume", str(out2 / "last.pt")], right)
    assert refused.returncode != 0 and "lacks the configured init checkpoint lineage" in refused.stderr
    manifest_path.write_text(original_manifest, encoding="utf-8")

    evaluated = _run(
        ["-m", "ard.cli.evaluate", "--config", str(out2 / "resolved_config.yaml"), "--checkpoint-dir", str(out2)],
        right,
    )
    assert evaluated.returncode == 0, evaluated.stderr
    results2 = json.loads((out2 / "evaluation" / "evaluation-results.json").read_text(encoding="utf-8"))
    for item in results2:
        assert item["dataset_identity"]["image_size"] == 32
        assert item["training_protocol_identity"]["init_checkpoint_sha256"] == digest
        assert "train_image_size" not in item["training_protocol_identity"]
        # Both stages are selected/evaluated under the same threat identity.
        assert item["threat_hash"] == results1[0]["threat_hash"]
