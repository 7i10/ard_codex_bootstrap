from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ard.config import ExperimentConfig, load_config
from tools.internal.history_routing_v2.prepare_inputs import ARMS, _arm_raw


def _source(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "schema_version": 2,
            "protocol": {"id": "controlled_cifar10_r18_v1"},
            "tier": "dev",
            "seeds": {
                name: (20260722 if name == "split" else 0)
                for name in (
                    "split",
                    "model_init",
                    "data_order",
                    "augmentation",
                    "train_attack",
                    "evaluation_attack",
                    "qualitative_panel",
                )
            },
            "dataset": {"name": "cifar10", "root": str(tmp_path / "data"), "num_classes": 10},
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
                "checkpoint_sha256": "e" * 64,
                "registry_id": "bartoldson2024_adversarial_wrn94_16",
            },
            "method": {
                "id": "rslad",
                "version": 1,
                "attack": {"loss": "kl", "kl_target": "teacher_clean"},
                "selection_attack": {"loss": "ce", "steps": 20},
            },
            "optimizer": {
                "id": "sgd",
                "learning_rate": 0.1,
                "momentum": 0.9,
                "weight_decay": 0.0005,
                "nesterov": False,
            },
            "scheduler": {"id": "multistep", "milestones": [100, 150], "gamma": 0.1, "step_at": "epoch_end"},
            "training": {
                "epochs": 200,
                "per_rank_batch_size": 128,
                "global_batch_size": 128,
                "validation_fraction": 0.1,
            },
            "observation": {"profile": "teacher_response"},
            "output_dir": str(tmp_path / "parent"),
            "intervention": None,
        }
    )


def test_prepare_arm_configs_are_unique_and_hash_bound(tmp_path: Path) -> None:
    source = _source(tmp_path)
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    bundle = tmp_path / "bundle.json"
    bundle.write_text("{}\n", encoding="utf-8")
    parent = {
        "checkpoint_sha256": "a" * 64,
        "raw_config_sha256": "b" * 64,
        "git_sha": "c" * 40,
        "epoch": 39,
        "world_size": 1,
        "teacher_checkpoint_sha256": "e" * 64,
        "sample_state_records": 45000,
        "sample_state_sha256": "d" * 64,
        "train_partition_manifest": str(tmp_path / "partition.json"),
        "train_partition_manifest_sha256": "f" * 64,
        "train_partition_ids_labels_sha256": "1" * 64,
        "artifact_attestation": str(tmp_path / "attestation.json"),
        "artifact_attestation_sha256": "2" * 64,
        "artifact_inventory": str(tmp_path / "inventory.json"),
        "artifact_inventory_sha256": "3" * 64,
    }
    configs: dict[str, Path] = {}
    for arm, route, kind, anchor_correct in ARMS:
        mask = mask_dir / f"{route}-{kind}.json"
        mask.write_text(
            json.dumps(
                {
                    "selected_ids_sha256": "4" * 64,
                    "selected_count": 1,
                    "selected_class_counts": {"0": 1},
                    "provenance": {
                        "source": "online_history_epoch39_v2"
                        if kind == "history"
                        else "class_state_count_matched_random_epoch39_v2",
                        "approved_selector_spec_sha256": "5" * 64 if kind == "history" else None,
                        "selector_spec_path": str(bundle) if kind == "history" else None,
                        "parent_checkpoint_sha256": "a" * 64,
                        "parent_sample_state_sha256": "d" * 64,
                        "route": route,
                        "anchor_robust_correct": anchor_correct,
                        "random_seed": 7 if kind == "random" else None,
                        "generator": "sha256_rank" if kind == "random" else None,
                        "generator_version": "v1" if kind == "random" else None,
                        "reference_history_mask_sha256": "6" * 64 if kind == "random" else None,
                        "reference_selected_count": 1 if kind == "random" else None,
                        "reference_selected_class_counts": {"0": 1} if kind == "random" else None,
                        "reference_history_selector_spec_sha256": "5" * 64 if kind == "random" else None,
                    },
                }
            ),
            encoding="utf-8",
        )
        raw = _arm_raw(
            source=source,
            arm=arm,
            route=route,
            selector_kind="online_history" if kind == "history" else "class_state_count_matched_random",
            anchor_correct=anchor_correct,
            parent=parent,
            mask_path=mask,
            bundle_path=bundle,
            output_root=tmp_path / "screen",
            entity="entity",
            project="project",
            group="group",
            parent_run_id="parent",
        )
        path = tmp_path / f"{arm}.yaml"
        import yaml

        path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        loaded = load_config(path)
        configs[arm] = path
        assert loaded.intervention is not None and loaded.intervention.arm == arm
        assert loaded.protocol.id == "controlled_cifar10_r18_delayed_multistep_v1"
        assert tuple(loaded.scheduler.milestones) == (120, 170)
        assert loaded.tracking.mode == "online"
        assert loaded.tracking.run_id == f"h2-parent-{arm.lower()}"
        assert loaded.intervention.selector_bundle_sha256 == hashlib.sha256(bundle.read_bytes()).hexdigest()
        assert loaded.intervention.mask is not None
        assert loaded.intervention.mask.sha256 == hashlib.sha256(mask.read_bytes()).hexdigest()
    assert len({load_config(path).tracking.run_id for path in configs.values()}) == 4
    assert len({str(load_config(path).output_dir) for path in configs.values()}) == 4
