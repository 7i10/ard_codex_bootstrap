from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parents[2]))
from ard.config import load_config
from tools.internal.schedule_control.prepare_inputs import PrepareInputsError, prepare_inputs

pytest.importorskip("pyarrow")
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402


def _inputs(tmp_path: Path) -> dict[str, Path | str]:
    n = 45_000
    records = {str(i * 2 + 1): {"true_label": i % 10} for i in range(n)}
    state = {"format_version": 3, "pending": [], "records": records}
    config_raw = {
        "schema_version": 2,
        "protocol": {"id": "controlled_cifar10_r18_v1"},
        "tier": "dev",
        "seeds": {
            "split": 20260722,
            "model_init": 1,
            "data_order": 2,
            "augmentation": 3,
            "train_attack": 4,
            "evaluation_attack": 0,
            "qualitative_panel": 5,
        },
        "dataset": {"name": "cifar10", "root": str(tmp_path / "cifar"), "num_classes": 10},
        "student": {
            "architecture": "saad_resnet18_cifar_v1",
            "num_classes": 10,
            "normalization": {"profile": "cifar10_raw_identity"},
        },
        "teacher": {
            "source": "robustbench",
            "architecture": "robustbench_wide_resnet",
            "num_classes": 10,
            "normalization": {"profile": "robustbench_cifar10_bartoldson_embedded"},
            "preprocessing_owner": "model_embedded",
            "checkpoint": str(tmp_path / "teacher.pt"),
            "checkpoint_sha256": "b" * 64,
            "registry_id": "bartoldson2024_adversarial_wrn94_16",
        },
        "method": {
            "id": "rslad",
            "version": 1,
            "attack": {"loss": "kl", "kl_target": "teacher_clean"},
            "selection_attack": {"loss": "ce", "steps": 20},
        },
        "optimizer": {"id": "sgd", "learning_rate": 0.1, "momentum": 0.9, "weight_decay": 0.0005, "nesterov": False},
        "scheduler": {"id": "multistep", "milestones": [100, 150], "gamma": 0.1, "step_at": "epoch_end"},
        "training": {"epochs": 200, "per_rank_batch_size": 128, "global_batch_size": 128, "validation_fraction": 0.1},
        "output_dir": str(tmp_path / "parent"),
    }
    config = tmp_path / "resolved.yaml"
    config.write_text(yaml.safe_dump(config_raw, sort_keys=False), encoding="utf-8")
    import ard.engine.checkpoint as checkpoint_mod

    config_hash = checkpoint_mod.config_digest(config_raw)
    checkpoint = tmp_path / "epoch79.pt"
    torch.save(
        {"epoch": 79, "sample_state": state, "tracker_run_id": "parent", "config_hash": config_hash, "world_size": 1},
        checkpoint,
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "run_id": "parent",
                "config_hash": config_hash,
                "git": {"sha": "a" * 40, "dirty": False},
                "teacher": {"checkpoint_sha256": "b" * 64},
            }
        ),
        encoding="utf-8",
    )
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    epochs = [39, 79]
    ids = [i * 2 + 1 for i in range(n)]
    table = pa.table(
        {
            "namespace": ["train"] * (n * len(epochs)),
            "sample_id": ids * len(epochs),
            "class_id": [i % 10 for i in range(n)] * len(epochs),
            "epoch": [epoch for epoch in epochs for _ in range(n)],
            "observation_schema_version": [2] * (n * len(epochs)),
        }
    )
    feature = tmp_path / "feature.parquet"
    pq.write_table(table, feature)
    lineage = tmp_path / "lineage.json"
    lineage.write_text(
        json.dumps(
            {
                "checkpoints": [{"epoch": 79, "sha256": checkpoint_sha, "path": str(checkpoint)}],
                "feature_observations_sha256": hashlib.sha256(feature.read_bytes()).hexdigest(),
                "run_id": "parent",
                "config_hash": config_hash,
                "scientific_git_sha": "a" * 40,
                "teacher": {"checkpoint_sha256": "b" * 64},
                "observation_schema_version": 2,
            }
        ),
        encoding="utf-8",
    )
    return {
        "parent_checkpoint": checkpoint,
        "parent_config": config,
        "parent_manifest": manifest,
        "feature_observations": feature,
        "feature_lineage": lineage,
        "config_hash": config_hash,
    }


def test_prepare_inputs_binds_sparse_epoch79_parent(tmp_path: Path) -> None:
    values = _inputs(tmp_path)
    output = tmp_path / "prepared"
    spec = prepare_inputs(
        **{key: value for key, value in values.items() if key != "config_hash"},
        artifact_name="model-parent-last",
        artifact_version="v3",
        artifact_digest="digest3",
        output_dir=output,
        child_output_dir=tmp_path / "child",
        child_run_id="child-run",
    )
    assert spec.is_file()
    import yaml

    parent = yaml.safe_load(spec.read_text())["parent"]
    for key in ("train_partition_manifest", "artifact_attestation", "artifact_inventory"):
        assert Path(parent[key]).is_file()
    child = load_config(output / "child-config.yaml")
    assert child.protocol.id == "controlled_cifar10_r18_delayed_multistep_v1"
    assert child.scheduler.milestones == (120, 170)
    assert child.tracking.run_id == "child-run"
    payload = json.loads((output / "artifact-inventory.json").read_text())
    assert payload["artifact"]["version"] == "v3"
    assert len(json.loads((output / "train-partition.json").read_text())["ids_labels"]) == 45_000
    assert (output / "artifact-attestation.json").is_file()


def test_prepare_inputs_rejects_checkpoint_lineage_drift(tmp_path: Path) -> None:
    values = _inputs(tmp_path)
    values["feature_lineage"].write_text(
        json.dumps({"checkpoints": [{"epoch": 79, "sha256": "0" * 64}]}), encoding="utf-8"
    )
    with pytest.raises(PrepareInputsError, match="exact epoch-79"):
        prepare_inputs(
            **{key: value for key, value in values.items() if key != "config_hash"},
            artifact_name="model-parent-last",
            artifact_version="v3",
            artifact_digest="digest3",
            output_dir=tmp_path / "prepared",
            child_output_dir=tmp_path / "child",
            child_run_id="child-run",
        )


def test_prepare_inputs_rejects_feature_observation_hash_drift(tmp_path: Path) -> None:
    values = _inputs(tmp_path)
    feature = values["feature_observations"]
    assert isinstance(feature, Path)
    feature.write_bytes(feature.read_bytes() + b"drift")
    with pytest.raises(PrepareInputsError, match="feature observations bytes"):
        prepare_inputs(
            **{key: value for key, value in values.items() if key != "config_hash"},
            artifact_name="model-parent-last",
            artifact_version="v3",
            artifact_digest="digest3",
            output_dir=tmp_path / "prepared",
            child_output_dir=tmp_path / "child",
            child_run_id="child-run",
        )
