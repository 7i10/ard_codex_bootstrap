"""End-to-end ``ard.cli.train`` + ``ard.cli.evaluate`` in soft-label-bank mode on a CPU fixture.

Builds a bank with ``ard.cli.build_soft_label_bank.build_banks``, trains plain
``rslad`` from it (no teacher forward at all, crop keys checked every step),
and checks the evaluation record carries the distillation protocol identity.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from ard.cli import evaluate as evaluate_cli
from ard.cli import train as train_cli
from ard.config.schema import ExperimentConfig
from ard.data.datasets import ImageNetDataset

_SPEC = importlib.util.spec_from_file_location(
    "soft_label_bank_fixtures", Path(__file__).resolve().parents[1] / "unit" / "test_soft_label_bank.py"
)
assert _SPEC is not None and _SPEC.loader is not None
fixtures: Any = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fixtures)


def test_train_and_evaluate_from_bank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The fixture teacher block is a 4-class stand-in for the 1000-class Salman profile; the
    # profile check itself is covered in tests/unit/test_imagenet_teacher_registry.py.
    monkeypatch.setattr(train_cli, "validate_imagenet_teacher_config", lambda config: None)
    root = tmp_path / "imagenet"
    fixtures._make_imagenet(root)
    shutil.copytree(root / "train", root / "val")
    train_sha = ImageNetDataset(root, "train", image_size=16).content_identity["observed_sha256"]
    val_sha = ImageNetDataset(root, "val", image_size=16).content_identity["observed_sha256"]

    def with_sha(config: ExperimentConfig) -> ExperimentConfig:
        dumped = config.model_dump(mode="json")
        dumped["dataset"]["content_sha256"] = train_sha
        return ExperimentConfig.model_validate(dumped)

    digest = fixtures._build(
        with_sha(fixtures._config(root)), fixtures._teacher(), tmp_path / "banks", top_k=2, epochs=[0, 1]
    )
    payload = with_sha(fixtures._bank_config(root, tmp_path / "banks", digest, top_k=2)).model_dump(mode="json")
    payload["output_dir"] = str(tmp_path / "out")
    payload["training"]["device"] = "cpu"
    (tmp_path / "train.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")
    assert train_cli.main(["--config", str(tmp_path / "train.yaml")]) == 0
    rows = [json.loads(line) for line in (tmp_path / "out" / "epoch-metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and all(row["train_teacher_clean_forward_calls"] == 0.0 for row in rows)

    evaluation = {
        "evaluation": {
            "dataset": {
                "name": "imagenet",
                "root": str(root),
                "split": "val",
                "num_classes": 4,
                "image_size": 16,
                "content_sha256": val_sha,
            },
            "attack": {"loss": "ce", "epsilon": "4/255", "step_size": "8/765", "steps": 2, "random_start": True},
        }
    }
    (tmp_path / "eval.yaml").write_text(yaml.safe_dump(evaluation), encoding="utf-8")
    assert evaluate_cli.main(["--config", str(tmp_path / "eval.yaml"), "--checkpoint-dir", str(tmp_path / "out")]) == 0
    record = json.loads((tmp_path / "out" / "evaluation" / "evaluation-results.json").read_text())
    for result in record:
        assert result["training_protocol_identity"]["distillation"] == {
            "target_source": "soft_label_bank",
            "teacher_registry_id": "salman2020_resnet50_linf_eps4",
            "teacher_checkpoint_sha256": "9e62216975c111b8081481533b6142dec34be69abe5d90b54d7f860015c263db",
            "bank_storage": "ard-soft-label-bank-v1/top_k_marginal_smoothing_v1/fp16",
            "bank_top_k": 2,
        }
        assert result["distillation_lineage"]["bank_manifest_sha256"] == digest
