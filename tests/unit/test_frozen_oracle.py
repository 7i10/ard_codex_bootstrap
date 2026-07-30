from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from ard.analysis.frozen_oracle import (
    FrozenOracleError,
    build_frozen_oracle_manifests,
    load_frozen_risk_lookup,
    md5_base64_file,
    sha256_file,
    validate_wandb_checkpoint_inventory,
    write_frozen_oracle_manifests,
)
from ard.config.loader import resolved_config_dict
from ard.config.schema import ExperimentConfig
from ard.engine import config_digest
from ard.engine.trainer import Trainer
from ard.targets import UniformSofteningTeacherTargetPolicy

pytestmark = pytest.mark.t2


def _source_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "schema_version": 2,
            "protocol": {"id": "controlled_cifar10_r18_v1"},
            "tier": "dev",
            "seeds": {
                "split": 20260722,
                "model_init": 0,
                "data_order": 0,
                "augmentation": 0,
                "train_attack": 0,
                "evaluation_attack": 0,
                "qualitative_panel": 0,
            },
            "dataset": {"name": "cifar10", "root": str(tmp_path / "cifar"), "num_classes": 10},
            "student": {
                "architecture": "saad_resnet18_cifar_v1",
                "num_classes": 10,
                "preprocessing_owner": "student_adapter",
                "normalization": {"profile": "cifar10_raw_identity"},
            },
            "teacher": {
                "source": "robustbench",
                "architecture": "robustbench_dm_wide_resnet",
                "num_classes": 10,
                "normalization": {"profile": "robustbench_cifar10_bartoldson_embedded"},
                "preprocessing_owner": "model_embedded",
                "checkpoint": str(tmp_path / "bart.pt"),
                "checkpoint_sha256": "e" * 64,
                "registry_id": "bartoldson2024_adversarial_wrn94_16",
            },
            "method": {
                "id": "rslad",
                "version": 1,
                "attack": {"loss": "kl", "kl_target": "teacher_clean"},
                "selection_attack": {
                    "loss": "ce",
                    "steps": 20,
                    "epsilon": "8/255",
                    "step_size": "2/255",
                    "random_start": True,
                },
            },
            "optimizer": {
                "id": "sgd",
                "learning_rate": 0.1,
                "momentum": 0.9,
                "weight_decay": 0.0005,
                "nesterov": False,
            },
            "scheduler": {
                "id": "multistep",
                "milestones": [100, 150],
                "gamma": 0.1,
                "step_at": "epoch_end",
            },
            "training": {
                "epochs": 200,
                "per_rank_batch_size": 128,
                "global_batch_size": 128,
                "validation_fraction": 0.1,
            },
            "output_dir": str(tmp_path / "source-output"),
        }
    )


def _checkpoint(*, config_hash: str, epoch: int, records: dict[str, dict[str, object]]) -> dict[str, object]:
    return {
        "format_version": 1,
        "epoch": epoch,
        "epoch_boundary": "end",
        "model": {},
        "optimizer": {},
        "scheduler": {},
        "scaler": None,
        "rng": [],
        "sampler_epoch": [],
        "sampler_state": [],
        "sample_state": {"ema_decay": 0.9, "records": records, "pending": []},
        "global_step": epoch + 1,
        "best_metric": 0.0,
        "selection_metadata": {},
        "tracker_run_id": "source-run",
        "config_hash": config_hash,
        "world_size": 1,
    }


def _inputs(tmp_path: Path) -> tuple[ExperimentConfig, Path, Path, Path, dict[int, int]]:
    config = _source_config(tmp_path)
    labels = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 2}
    historical = {str(sample_id): {"previous_robust_correct": sample_id not in {2, 5}} for sample_id in labels}
    final = {str(sample_id): {"previous_robust_correct": sample_id not in {1, 2, 5}} for sample_id in labels}
    config_hash = config_digest(resolved_config_dict(config))
    historical_path, final_path = tmp_path / "epoch99.pt", tmp_path / "epoch199.pt"
    torch.save(_checkpoint(config_hash=config_hash, epoch=99, records=historical), historical_path)
    torch.save(_checkpoint(config_hash=config_hash, epoch=199, records=final), final_path)
    manifest = tmp_path / "source-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "run_id": "source-run",
                "config_hash": config_hash,
                "git": {"sha": "a" * 40},
                "teacher": {"checkpoint_sha256": "e" * 64},
            }
        ),
        encoding="utf-8",
    )
    return config, manifest, historical_path, final_path, labels


def _replay(*, checkpoint: Path, epoch: int, config_hash: str, correctness: dict[int, bool]) -> dict[str, object]:
    return {
        "checkpoint_sha256": sha256_file(checkpoint),
        "epoch": epoch,
        "tracker_run_id": "source-run",
        "config_sha256": config_hash,
        "correctness": correctness,
        "correctness_sha256": hashlib.sha256(
            json.dumps([[key, correctness[key]] for key in sorted(correctness)], separators=(",", ":")).encode()
        ).hexdigest(),
        "replay_protocol": {
            "input_view": "raw_unaugmented_train_partition",
            "attack_seed_base": 123,
            "seed_formula": "attack_seed_base + 1000003 * batch_index",
            "batch_size": 2,
            "device_type": "cuda",
        },
        "max_abs_delta": 8 / 255,
    }


def _inventory(*, historical: Path, final: Path, config_hash: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": "source-run",
        "artifact_name": "model-source-run-last",
        "config_sha256": config_hash,
        "scientific_git_sha": "a" * 40,
        "checkpoints": {
            "historical": {
                "epoch": 99,
                "version": "v19",
                "artifact_digest": "1" * 32,
                "file_name": "last.pt",
                "file_md5": md5_base64_file(historical),
                "size": historical.stat().st_size,
                "checkpoint_sha256": sha256_file(historical),
            },
            "final": {
                "epoch": 199,
                "version": "v39",
                "artifact_digest": "2" * 32,
                "file_name": "last.pt",
                "file_md5": md5_base64_file(final),
                "size": final.stat().st_size,
                "checkpoint_sha256": sha256_file(final),
            },
        },
    }


def _manifests(
    tmp_path: Path,
) -> tuple[dict[str, dict[str, object]], dict[int, int]]:
    config, source_manifest, historical, final, labels = _inputs(tmp_path)
    config_hash = config_digest(resolved_config_dict(config))
    historical_correct = {sample_id: sample_id not in {2, 5} for sample_id in labels}
    final_correct = {sample_id: sample_id not in {1, 2, 5} for sample_id in labels}
    inventory = validate_wandb_checkpoint_inventory(
        _inventory(historical=historical, final=final, config_hash=config_hash),
        historical_checkpoint=historical,
        final_checkpoint=final,
        source_config_hash=config_hash,
        source_run_id="source-run",
        source_scientific_git_sha="a" * 40,
    )
    return (
        build_frozen_oracle_manifests(
            source_config=config,
            source_manifest=source_manifest,
            historical_replay=_replay(
                checkpoint=historical, epoch=99, config_hash=config_hash, correctness=historical_correct
            ),
            final_replay=_replay(checkpoint=final, epoch=199, config_hash=config_hash, correctness=final_correct),
            labels=labels,
            builder_git={"sha": "b" * 40, "dirty": False},
            wandb_checkpoint_inventory=inventory,
        ),
        labels,
    )


def test_builds_binary_train_only_oracle_and_deterministic_class_matched_controls(tmp_path: Path) -> None:
    first, labels = _manifests(tmp_path)
    second, _ = _manifests(tmp_path)
    assert first == second
    oracle = first["oracle"]
    assert oracle["selected_count"] == 3
    assert oracle["selected_class_counts"] == {"0": 1, "1": 1, "2": 1}
    selected = {row["sample_id"] for row in oracle["rows"] if row["risk"]}
    assert selected == {1, 2, 5}
    assert {row["transition"] for row in oracle["rows"] if row["risk"]} == {"future_forgetting", "persistent_failure"}
    assert {row["transition"] for row in oracle["rows"] if not row["risk"]} <= {
        "stable_correct",
        "recovered",
        "future_forgetting",
        "persistent_failure",
    }
    for row in oracle["rows"]:
        expected_transition = {
            (True, True): "stable_correct",
            (True, False): "future_forgetting",
            (False, True): "recovered",
            (False, False): "persistent_failure",
        }[(row["source_historical_robust_correct"], row["source_final_robust_correct"])]
        assert row["transition"] == expected_transition
    assert oracle["dataset"]["train_namespace_sha256"]
    assert oracle["source"]["replays"]["historical"]["epoch"] == 99
    assert oracle["source"]["replays"]["final"]["epoch"] == 199
    for key in ("control-1", "control-2", "control-3"):
        assert first[key]["selected_class_counts"] == oracle["selected_class_counts"]
        assert sum(row["risk"] for row in first[key]["rows"]) == 3
        assert first[key]["assignment"]["kind"] == "class_matched_random"


def test_lookup_rejects_official_test_namespace_missing_ids_and_byte_drift(tmp_path: Path) -> None:
    manifests, labels = _manifests(tmp_path)
    hashes = write_frozen_oracle_manifests(tmp_path / "masks", manifests)
    path = tmp_path / "masks" / "oracle.json"
    lookup = load_frozen_risk_lookup(
        path,
        expected_sha256=hashes["oracle"],
        expected_dataset_name="cifar10",
        expected_num_classes=10,
        expected_train_labels=labels,
        expected_attack_identity=_source_config(tmp_path).method.attack.identity(),
        expected_teacher_checkpoint_sha256="e" * 64,
    )
    values = lookup.values(torch.tensor([0, 1, 5]), device=torch.device("cpu"), dtype=torch.float32)
    assert torch.equal(values, torch.tensor([0.0, 1.0, 1.0]))
    altered = json.loads(path.read_text(encoding="utf-8"))
    altered["rows"][0]["namespace"] = "test"
    path.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(FrozenOracleError, match="SHA-256"):
        load_frozen_risk_lookup(
            path,
            expected_sha256=hashes["oracle"],
            expected_dataset_name="cifar10",
            expected_num_classes=10,
            expected_train_labels=labels,
            expected_attack_identity=_source_config(tmp_path).method.attack.identity(),
            expected_teacher_checkpoint_sha256="e" * 64,
        )
    altered_hash = sha256_file(path)
    with pytest.raises(FrozenOracleError, match="training namespace"):
        load_frozen_risk_lookup(
            path,
            expected_sha256=altered_hash,
            expected_dataset_name="cifar10",
            expected_num_classes=10,
            expected_train_labels=labels,
            expected_attack_identity=_source_config(tmp_path).method.attack.identity(),
            expected_teacher_checkpoint_sha256="e" * 64,
        )


def test_lookup_and_target_policy_are_exact_for_binary_risk_and_padding(tmp_path: Path) -> None:
    manifests, labels = _manifests(tmp_path)
    hashes = write_frozen_oracle_manifests(tmp_path / "masks", manifests)
    lookup = load_frozen_risk_lookup(
        tmp_path / "masks" / "oracle.json",
        expected_sha256=hashes["oracle"],
        expected_dataset_name="cifar10",
        expected_num_classes=10,
        expected_train_labels=labels,
        expected_attack_identity=_source_config(tmp_path).method.attack.identity(),
        expected_teacher_checkpoint_sha256="e" * 64,
    )
    trainer = object.__new__(Trainer)
    trainer.frozen_risk_lookup = lookup
    from ard.data import IndexedBatch

    batch = IndexedBatch(torch.zeros(3, 3, 1, 1), torch.tensor([0, 0, 2]), torch.tensor([0, 1, 5]))
    weights = Trainer._policy_weights(
        trainer,
        batch=batch,
        adversarial=batch.images,
        logits=torch.zeros(3, 3),
        valid_mask=torch.tensor([True, True, False]),
        student_signals={},
    )
    assert weights is not None
    assert torch.equal(weights.kd_weight, torch.tensor([1.0, 1.0, 0.0]))
    assert torch.equal(weights.hard_weight, torch.zeros(3))
    assert torch.equal(weights.joint_risk, torch.tensor([0.0, 1.0, 0.0]))
    teacher_logits = torch.tensor([[4.0, 1.0, -1.0], [4.0, 1.0, -1.0]])
    output = UniformSofteningTeacherTargetPolicy(rho_max=0.5)(
        teacher_logits=teacher_logits, risk=torch.tensor([0.0, 1.0]), temperature=1.0
    )
    expected = torch.softmax(teacher_logits, dim=1)
    assert torch.equal(output.probabilities[0], expected[0])
    assert output.rho.tolist() == [0.0, 0.5]
    assert not output.probabilities.requires_grad


def test_schema_requires_hash_target_and_guarded_tier() -> None:
    target = {
        "id": "teacher_target_uniform_mix",
        "version": 1,
        "risk_transform": "identity",
        "mixing": "uniform",
        "apply_to": "adversarial_student_kd",
        "rho_max": 0.5,
    }
    valid = {
        "id": "rslad_frozen_oracle_softening",
        "version": 1,
        "attack": {"loss": "kl", "kl_target": "teacher_clean"},
        "selection_attack": {"loss": "ce", "steps": 20},
        "target_policy": target,
        "frozen_oracle_manifest": "mask.json",
        "frozen_oracle_manifest_sha256": "f" * 64,
    }
    base = _source_config(Path("/tmp")).model_dump(mode="python")
    assert ExperimentConfig.model_validate({**base, "method": valid}).method.id == valid["id"]
    missing = copy.deepcopy(valid)
    missing.pop("frozen_oracle_manifest_sha256")
    with pytest.raises(ValidationError, match="requires frozen_oracle_manifest"):
        ExperimentConfig.model_validate({**base, "method": missing})


def test_builder_rejects_dirty_or_unaddressable_git_identity(tmp_path: Path) -> None:
    config, source_manifest, historical, final, labels = _inputs(tmp_path)
    config_hash = config_digest(resolved_config_dict(config))
    historical_correct = {sample_id: sample_id not in {2, 5} for sample_id in labels}
    final_correct = {sample_id: sample_id not in {1, 2, 5} for sample_id in labels}
    with pytest.raises(FrozenOracleError, match="must be clean"):
        build_frozen_oracle_manifests(
            source_config=config,
            source_manifest=source_manifest,
            historical_replay=_replay(
                checkpoint=historical, epoch=99, config_hash=config_hash, correctness=historical_correct
            ),
            final_replay=_replay(checkpoint=final, epoch=199, config_hash=config_hash, correctness=final_correct),
            labels=labels,
            builder_git={"sha": "b" * 40, "dirty": True},
            wandb_checkpoint_inventory=validate_wandb_checkpoint_inventory(
                _inventory(historical=historical, final=final, config_hash=config_hash),
                historical_checkpoint=historical,
                final_checkpoint=final,
                source_config_hash=config_hash,
                source_run_id="source-run",
                source_scientific_git_sha="a" * 40,
            ),
        )


def test_builder_rejects_replay_protocol_or_linf_drift(tmp_path: Path) -> None:
    config, source_manifest, historical, final, labels = _inputs(tmp_path)
    config_hash = config_digest(resolved_config_dict(config))
    historical_correct = {sample_id: sample_id not in {2, 5} for sample_id in labels}
    final_correct = {sample_id: sample_id not in {1, 2, 5} for sample_id in labels}
    historical_replay = _replay(
        checkpoint=historical, epoch=99, config_hash=config_hash, correctness=historical_correct
    )
    final_replay = _replay(checkpoint=final, epoch=199, config_hash=config_hash, correctness=final_correct)
    inventory = validate_wandb_checkpoint_inventory(
        _inventory(historical=historical, final=final, config_hash=config_hash),
        historical_checkpoint=historical,
        final_checkpoint=final,
        source_config_hash=config_hash,
        source_run_id="source-run",
        source_scientific_git_sha="a" * 40,
    )
    drifted_protocol = copy.deepcopy(final_replay)
    drifted_protocol["replay_protocol"]["attack_seed_base"] = 124
    with pytest.raises(FrozenOracleError, match="protocols must be exactly identical"):
        build_frozen_oracle_manifests(
            source_config=config,
            source_manifest=source_manifest,
            historical_replay=historical_replay,
            final_replay=drifted_protocol,
            labels=labels,
            builder_git={"sha": "b" * 40, "dirty": False},
            wandb_checkpoint_inventory=inventory,
        )
    exceeded_bound = copy.deepcopy(final_replay)
    exceeded_bound["max_abs_delta"] = 9 / 255
    with pytest.raises(FrozenOracleError, match="exceeds the configured"):
        build_frozen_oracle_manifests(
            source_config=config,
            source_manifest=source_manifest,
            historical_replay=historical_replay,
            final_replay=exceeded_bound,
            labels=labels,
            builder_git={"sha": "b" * 40, "dirty": False},
            wandb_checkpoint_inventory=inventory,
        )


def test_wandb_checkpoint_inventory_rejects_version_or_byte_drift(tmp_path: Path) -> None:
    config, _, historical, final, _ = _inputs(tmp_path)
    config_hash = config_digest(resolved_config_dict(config))
    inventory = _inventory(historical=historical, final=final, config_hash=config_hash)
    validated = validate_wandb_checkpoint_inventory(
        inventory,
        historical_checkpoint=historical,
        final_checkpoint=final,
        source_config_hash=config_hash,
        source_run_id="source-run",
        source_scientific_git_sha="a" * 40,
    )
    assert validated["checkpoints"]["historical"]["version"] == "v19"
    drifted = copy.deepcopy(inventory)
    drifted["checkpoints"]["historical"]["version"] = "v18"
    with pytest.raises(FrozenOracleError, match="bytes/version"):
        validate_wandb_checkpoint_inventory(
            drifted,
            historical_checkpoint=historical,
            final_checkpoint=final,
            source_config_hash=config_hash,
            source_run_id="source-run",
            source_scientific_git_sha="a" * 40,
        )


def test_manifest_hash_changes_method_config_and_resume_identity(tmp_path: Path) -> None:
    base = _source_config(tmp_path).model_dump(mode="python")
    target = {
        "id": "teacher_target_uniform_mix",
        "version": 1,
        "risk_transform": "identity",
        "mixing": "uniform",
        "apply_to": "adversarial_student_kd",
        "rho_max": 0.5,
    }

    def digest(mask_sha: str) -> str:
        method = {
            "id": "rslad_frozen_oracle_softening",
            "version": 1,
            "attack": {"loss": "kl", "kl_target": "teacher_clean"},
            "selection_attack": {"loss": "ce", "steps": 20},
            "target_policy": target,
            "frozen_oracle_manifest": str(tmp_path / "mask.json"),
            "frozen_oracle_manifest_sha256": mask_sha,
        }
        config = ExperimentConfig.model_validate({**base, "method": method})
        return config_digest(resolved_config_dict(config))

    assert digest("1" * 64) != digest("2" * 64)
