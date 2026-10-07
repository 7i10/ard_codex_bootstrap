"""``training.selection_subset_size`` end to end on CPU (plan 0103 Phase 2).

A checked-in Phase-1 single-stage config, shrunk to a tiny on-disk ImageNet fixture, runs
through the real ``ard.cli.train`` twice -- without and with the option -- and then through
``ard.cli.evaluate``:

* the training partition, the held-out split and the trained last.pt are identical;
* per-epoch selection reads a fixed subset of the held-out split, labelled in every
  epoch row, the selection metadata and the run summary;
* the final epoch adds full-split numbers for best and last under distinct keys;
* the official evaluation is unchanged apart from the pooling identity entry.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
import torch
import yaml
from PIL import Image

from ard.config.schema import DatasetConfig
from ard.data import build_dataset

pytestmark = pytest.mark.t3

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "scientific" / "imagenet_mobilenetv4_pgd_at_random_init_lr0025.yaml"

# Runs ard.cli.train after wrapping Trainer.fit to record the source IDs of every loader.
_SNAPSHOT_TRAIN = """
import json
import sys
from ard.engine import Trainer

original_fit = Trainer.fit

def fit(self, *args, **kwargs):
    full = kwargs.get("full_validation_loader")
    with open(sys.argv[1], "w", encoding="utf-8") as handle:
        json.dump(
            {
                "train": list(args[0].dataset.indices),
                "validation": list(kwargs["validation_loader"].dataset.indices),
                "full_validation": None if full is None else list(full.dataset.indices),
                "probe": list(kwargs["probe_loader"].dataset.indices),
            },
            handle,
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


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    return environment


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args], cwd=ROOT, env=_environment(), text=True, capture_output=True, check=False
    )


def _tiny(*, data_root: Path, output: Path, run_id: str, subset: int | None) -> dict[str, Any]:
    """The checked-in config with only scale, tier and tracking shrunk."""
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
        epochs=2,
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
        "run_id": run_id,
        "group": raw["tracking"]["group"],
        "panel_size": 2,
        "panel_interval_epochs": 1,
        "artifact_retention": "full",
    }
    raw["evaluation"].update(panel_size=2, autoattack_batch_size=4)
    raw["output_dir"] = str(output)
    return raw


def _train(tmp_path: Path, name: str, *, data_root: Path, subset: int | None) -> tuple[Path, dict[str, Any]]:
    output = tmp_path / name
    config = tmp_path / f"{name}.yaml"
    raw = _tiny(data_root=data_root, output=output, run_id=f"offline-{name}", subset=subset)
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    snapshot = tmp_path / f"{name}-loaders.json"
    trained = _run(["-c", _SNAPSHOT_TRAIN, str(snapshot), "--config", str(config)])
    assert trained.returncode == 0, trained.stderr
    return output, json.loads(snapshot.read_text(encoding="utf-8"))


def test_selection_subset_cli_trains_identically_selects_on_the_subset_and_reports_full_best_and_last(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "imagenet"
    _imagenet_fixture(data_root)
    plain_out, plain = _train(tmp_path, "plain", data_root=data_root, subset=None)
    subset_out, subset = _train(tmp_path, "subset", data_root=data_root, subset=2)

    # Training partition, held-out split and probe unchanged; selection reads a fixed subset of it.
    assert subset["train"] == plain["train"] and subset["probe"] == plain["probe"]
    assert plain["full_validation"] is None and subset["full_validation"] == plain["validation"]
    assert len(plain["validation"]) == 6
    assert len(subset["validation"]) == 2 and set(subset["validation"]) < set(plain["validation"])
    # Training itself is identical.
    plain_last = torch.load(plain_out / "last.pt", map_location="cpu", weights_only=False)
    subset_last = torch.load(subset_out / "last.pt", map_location="cpu", weights_only=False)
    for key, value in plain_last["model"].items():
        assert torch.equal(subset_last["model"][key], value), key
    assert subset_last["config_hash"] != plain_last["config_hash"]

    # Labelled everywhere: selection metadata, epoch rows and the run summary.
    record = subset_last["selection_metadata"]["selection_subset"]
    assert (record["size"], record["full_split_size"], record["sampling"]) == (2, 6, "class_stratified")
    assert record["source"] == "held_out_validation_split"
    full = subset_last["selection_metadata"]["full_split_final"]
    assert full["num_examples"] == 6 and full["last_epoch"] == 1
    best_payload = torch.load(subset_out / "best.pt", map_location="cpu", weights_only=False)
    assert best_payload["epoch"] == full["best_epoch"]
    assert best_payload["selection_metadata"]["selection_subset"] == record
    rows = pq.read_table(subset_out / "epoch-metrics.parquet").to_pylist()
    assert [row["val_selection_subset_size"] for row in rows] == [2, 2]
    assert {row["val_selection_subset_sha256"] for row in rows} == {record["ids_sha256"]}
    assert rows[0]["val_full_pgd_accuracy"] is None and rows[0]["best_val_full_pgd_accuracy"] is None
    assert rows[1]["val_full_pgd_accuracy"] == full["last_pgd_accuracy"]
    assert rows[1]["best_val_full_pgd_accuracy"] == full["best_pgd_accuracy"]
    assert rows[1]["val_full_num_examples"] == 6
    plain_rows = pq.read_table(plain_out / "epoch-metrics.parquet").to_pylist()
    assert not any(key.startswith(("val_full", "best_val_full", "val_selection")) for key in plain_rows[0])
    summary = json.loads((subset_out / "run-bundle" / "manifest.json").read_text(encoding="utf-8"))["summary"]
    assert summary["selection_subset_size"] == 2 and summary["selection_subset_sha256"] == record["ids_sha256"]
    assert summary["val_full_num_examples"] == 6
    assert summary["best_full_epoch"] == full["best_epoch"]
    assert (summary["best_full_clean_accuracy"], summary["best_full_pgd_accuracy"]) == (
        full["best_clean_accuracy"],
        full["best_pgd_accuracy"],
    )
    assert (summary["last_full_clean_accuracy"], summary["last_full_pgd_accuracy"]) == (
        full["last_clean_accuracy"],
        full["last_pgd_accuracy"],
    )
    assert summary["robust_overfit_gap_full"] == pytest.approx(full["best_pgd_accuracy"] - full["last_pgd_accuracy"])
    plain_summary = json.loads((plain_out / "run-bundle" / "manifest.json").read_text(encoding="utf-8"))["summary"]
    assert not any("full" in key or "selection_subset" in key for key in plain_summary)

    # Official evaluation: unchanged apart from the pooling identity entry.
    for output, expected in ((plain_out, None), (subset_out, 2)):
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
            assert item["training_protocol_identity"].get("selection_subset_size") == expected
