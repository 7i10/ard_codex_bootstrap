"""scripts/evaluate_external_checkpoint.py --dataset imagenet (plan 0103).

A foreign ImageNet checkpoint is only a fair external anchor if the script
measures it exactly as ``ard.cli.evaluate`` measures this project's own run with
the same scientific config: same split and manifest pin, same loader batches,
same seeded PGD, same AutoAttack subset, seed, batch and epsilon.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch
import yaml
from PIL import Image

from ard.attacks import LinfPGD
from ard.cli.evaluate import _autoattack_loader, evaluation_loader
from ard.config import load_config
from ard.config.schema import DatasetConfig
from ard.data import build_dataset
from ard.evaluation import evaluate_loaded_model
from ard.models import build_student

pytestmark = pytest.mark.t1

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "evaluate_external_checkpoint.py"
PROTOCOL = ROOT / "configs" / "scientific" / "imagenet_efficientnet_b0_pgd_at.yaml"
CLASSES = ("n001", "n002", "n003")


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evaluate_external_checkpoint", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _imagenet_fixture(root: Path) -> None:
    for split in ("train", "val"):
        for class_index, wnid in enumerate(CLASSES):
            class_dir = root / split / wnid
            class_dir.mkdir(parents=True)
            for image_index in range(3):
                pixels = torch.rand(40, 48, 3, generator=torch.Generator().manual_seed(class_index * 10 + image_index))
                Image.fromarray((pixels * 255).to(torch.uint8).numpy()).save(class_dir / f"{wnid}_{image_index}.JPEG")


def _protocol(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, pin: bool = True) -> Path:
    data_root = tmp_path / "imagenet"
    _imagenet_fixture(data_root)
    for name, value in {
        "ARD_IMAGENET_ROOT": str(data_root),
        "ARD_NUM_WORKERS": "0",
        "ARD_SEED": "0",
        "ARD_RUN_ID": "external-eval-test",
        "ARD_JOB_OUTPUT_DIR": str(tmp_path / "unused"),
        "WANDB_ENTITY": "test-entity",
        "WANDB_PROJECT": "test-project",
    }.items():
        monkeypatch.setenv(name, value)
    raw = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    for key, split in (("dataset", "train"), ("evaluation", "val")):
        block = raw[key] if key == "dataset" else raw[key]["dataset"]
        observed = build_dataset(
            DatasetConfig(name="imagenet", root=data_root, split=split, num_classes=3, image_size=32)
        ).content_identity
        assert observed is not None
        block.update(num_classes=3, image_size=32, content_sha256=observed["observed_sha256"] if pin else None)
    raw["student"]["num_classes"] = 3
    if not pin:
        # Production configs cannot even load without a pin; a dev config can,
        # so this reaches the script's own second guard.
        raw["tier"] = "dev"
    # Several batches, so batch order and per-batch random starts matter.
    raw["training"].update(per_rank_batch_size=4, global_batch_size=4, device="cpu")
    raw["evaluation"].update(autoattack_batch_size=3, panel_size=4)
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def _foreign_checkpoint(tmp_path: Path, protocol: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    config = load_config(protocol)
    torch.manual_seed(11)
    model = build_student(config.student.model_copy(update={"pretrained": False}), tier=config.tier)
    state = {name: value.clone() for name, value in model.model.state_dict().items()}
    path = tmp_path / "foreign.pth"
    torch.save(state, path)
    return path, state


def _run(module: ModuleType, argv: list[str], monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setattr(sys, "argv", ["evaluate_external_checkpoint.py", *argv])
    return int(module.main())


def test_imagenet_external_evaluation_measures_like_our_own_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _protocol(tmp_path, monkeypatch)
    checkpoint, state = _foreign_checkpoint(tmp_path, protocol)
    module = _script()
    recorded: dict[str, Any] = {}

    def fake_autoattack(**kwargs: Any) -> dict[str, Any]:
        recorded.update(kwargs)
        return {"autoattack_accuracy": 0.25}

    monkeypatch.setattr(module, "run_autoattack", fake_autoattack)
    output = tmp_path / "out"
    argv = [
        "--dataset",
        "imagenet",
        "--checkpoint",
        str(checkpoint),
        "--protocol-config",
        str(protocol),
        "--autoattack-sample-count",
        "5",
        "--output",
        str(output),
        "--source",
        "unit-test fixture",
        "--device",
        "cpu",
        "--published-clean-accuracy",
        "0.5",
        "--allow-autoattack",
    ]
    assert _run(module, argv, monkeypatch) == 0
    record = json.loads((output / "result.json").read_text(encoding="utf-8"))

    # Reference: exactly what ard.cli.evaluate does for our own checkpoint.
    config = load_config(protocol)
    assert config.method.selection_attack is not None
    dataset = build_dataset(config.evaluation.dataset)
    reference_model = build_student(config.student.model_copy(update={"pretrained": False}), tier=config.tier)
    reference_model.model.load_state_dict(state, strict=True)
    reference = evaluate_loaded_model(
        model=reference_model,
        result_name="reference",
        artifact_stem="reference",
        loader=evaluation_loader(dataset, seed=config.evaluation.seed, batch_size=4, num_workers=0),
        attack=LinfPGD(config.method.selection_attack),
        device=torch.device("cpu"),
        seed=config.evaluation.seed,
        output_dir=tmp_path / "reference",
        panel_size=4,
    )
    assert record["count"] == reference.count == 9
    assert record["clean_accuracy"] == reference.clean_accuracy
    assert record["pgd_accuracy"] == reference.pgd_accuracy
    assert record["clean_difference_pp"] == pytest.approx((reference.clean_accuracy - 0.5) * 100.0)
    assert record["threat_model"] == config.method.selection_attack.identity()
    assert record["threat_model"]["steps"] == 10
    assert record["dataset_identity"]["verification"] == "computed-and-matched"
    assert record["lineage"].startswith("FOREIGN")
    assert "NOT an official test" in record["evaluation_kind"]
    assert record["checkpoint_state_key"] == "<root>"
    assert set(record["code"]) == {"source_git_sha", "dirty", "status"}
    assert (output / "sample-stats-external.parquet").is_file()

    # AutoAttack: the same seeded subset, seed, batch and epsilon as our runs.
    subset = list(_autoattack_loader(dataset, sample_count=5, seed=0, batch_size=4, num_workers=0))
    expected_images = torch.cat([batch.images for batch in subset])
    assert torch.equal(recorded["images"], expected_images)
    assert torch.equal(recorded["labels"], torch.cat([batch.labels for batch in subset]))
    assert record["autoattack"]["sample_ids"] == [int(i) for batch in subset for i in batch.sample_ids.tolist()]
    assert recorded["seed"] == 0
    assert recorded["batch_size"] == 3
    assert recorded["epsilon"] == pytest.approx(4 / 255)
    assert recorded["norm"] == "linf"
    assert recorded["model"].training is False

    # A second run into the same directory must not overwrite the first.
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        _run(module, argv, monkeypatch)


def test_imagenet_external_evaluation_refuses_cifar_flags_and_unpinned_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _script()
    base = ["--dataset", "imagenet", "--checkpoint", "x.pth", "--output", str(tmp_path / "o"), "--source", "s"]
    with pytest.raises(SystemExit) as refused:
        _run(module, [*base, "--batch-size", "64", "--allow-autoattack"], monkeypatch)
    assert refused.value.code == 2

    protocol = _protocol(tmp_path, monkeypatch, pin=False)
    checkpoint, _ = _foreign_checkpoint(tmp_path, protocol)
    argv = [
        "--dataset",
        "imagenet",
        "--checkpoint",
        str(checkpoint),
        "--protocol-config",
        str(protocol),
        "--autoattack-sample-count",
        "5",
        "--output",
        str(tmp_path / "unpinned"),
        "--source",
        "s",
        "--device",
        "cpu",
        "--allow-autoattack",
    ]
    with pytest.raises(SystemExit, match="pinned by content_sha256"):
        _run(module, argv, monkeypatch)
