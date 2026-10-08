"""``dataset.imagenet_augmentation`` end to end on CPU (plan 0103 Phase 2 batch C).

A checked-in Phase-1 single-stage config, shrunk to a tiny on-disk ImageNet fixture, runs
through the real ``ard.cli.train`` for two epochs with the standard augmentation and with
``idbh_weak_nocropshift``, then through ``ard.cli.evaluate``:

* both runs train and evaluate; the IDBH run's config hash differs;
* the IDBH run's ``training_protocol_identity`` carries ``imagenet_augmentation`` and the
  standard run's does not;
* ``summarize_checkpoint_groups`` pools each arm with itself but refuses to pool the IDBH
  rows with the standard rows (same protocol id, so the identity entry is the only guard).
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

from ard.analysis.aggregate import summarize_checkpoint_groups
from ard.config.schema import DatasetConfig
from ard.data import build_dataset

pytestmark = pytest.mark.t3

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "scientific" / "imagenet_mobilenetv4_pgd_at_random_init_lr0025.yaml"


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
    return subprocess.run(
        [sys.executable, *args], cwd=ROOT, env=environment, text=True, capture_output=True, check=False
    )


def _tiny(*, data_root: Path, output: Path, run_id: str, augmentation: str | None) -> dict[str, Any]:
    """The checked-in config with only scale, tier and tracking shrunk."""
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    for key, split in (("dataset", "train"), ("evaluation", "val")):
        block = raw[key] if key == "dataset" else raw[key]["dataset"]
        observed = build_dataset(
            DatasetConfig(name="imagenet", root=data_root, split=split, num_classes=2, image_size=32)
        ).content_identity
        assert observed is not None
        block.update(root=str(data_root), num_classes=2, image_size=32, content_sha256=observed["observed_sha256"])
    if augmentation is not None:
        raw["dataset"]["imagenet_augmentation"] = augmentation
    raw["tier"] = "smoke"
    raw["student"].update(architecture="fixture_cnn", num_classes=2, pretrained=False)
    raw["seeds"] = dict.fromkeys(raw["seeds"], 3)
    raw["training"].update(
        epochs=2,
        per_rank_batch_size=4,
        global_batch_size=4,
        num_workers=0,
        device="cpu",
        validation_fraction=0.5,
        train_probe_size=2,
    )
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


def _train_and_evaluate(
    tmp_path: Path, name: str, *, data_root: Path, augmentation: str | None
) -> list[dict[str, Any]]:
    output = tmp_path / name
    config = tmp_path / f"{name}.yaml"
    config.write_text(
        yaml.safe_dump(
            _tiny(data_root=data_root, output=output, run_id=f"offline-{name}", augmentation=augmentation),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    trained = _run(["-m", "ard.cli.train", "--config", str(config)])
    assert trained.returncode == 0, trained.stderr
    resolved = yaml.safe_load((output / "resolved_config.yaml").read_text(encoding="utf-8"))
    if augmentation is None:
        assert "imagenet_augmentation" not in resolved["dataset"]
    else:
        assert resolved["dataset"]["imagenet_augmentation"] == augmentation
    evaluated = _run(
        ["-m", "ard.cli.evaluate", "--config", str(output / "resolved_config.yaml"), "--checkpoint-dir", str(output)]
    )
    assert evaluated.returncode == 0, evaluated.stderr
    results = json.loads((output / "evaluation" / "evaluation-results.json").read_text(encoding="utf-8"))
    assert sorted(item["checkpoint_alias"] for item in results) == ["best", "last"]
    return results


def test_idbh_run_trains_evaluates_and_never_pools_with_a_standard_run(tmp_path: Path) -> None:
    data_root = tmp_path / "imagenet"
    _imagenet_fixture(data_root)
    standard = _train_and_evaluate(tmp_path, "standard", data_root=data_root, augmentation=None)
    idbh = _train_and_evaluate(tmp_path, "idbh", data_root=data_root, augmentation="idbh_weak_nocropshift")

    for item in standard:
        assert "imagenet_augmentation" not in item["training_protocol_identity"]
    for item in idbh:
        assert item["training_protocol_identity"]["imagenet_augmentation"] == "idbh_weak_nocropshift"
    assert {item["config_hash"] for item in standard}.isdisjoint({item["config_hash"] for item in idbh})
    # Same scientific contract: the protocol id is shared, so only the identity entry separates the arms.
    assert {item["training_protocol_identity"]["id"] for item in standard + idbh} == {
        standard[0]["training_protocol_identity"]["id"]
    }

    for rows in (standard, idbh):
        summary = summarize_checkpoint_groups(rows, metric="pgd_accuracy")
        assert set(summary) == {"best", "last"}
    with pytest.raises(ValueError, match="cannot aggregate mixed experiment identities"):
        summarize_checkpoint_groups(standard + idbh, metric="pgd_accuracy")
    # Positive control: the imagenet_augmentation entry is what separates the arms.
    stripped = [
        {
            **item,
            "training_protocol_identity": {
                key: value
                for key, value in item["training_protocol_identity"].items()
                if key != "imagenet_augmentation"
            },
        }
        for item in idbh
    ]
    assert set(summarize_checkpoint_groups(standard + stripped, metric="pgd_accuracy")) == {"best", "last"}
