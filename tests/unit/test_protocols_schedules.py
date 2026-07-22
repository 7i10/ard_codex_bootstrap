from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn
from torch.optim import SGD

from ard.cli import train as train_cli
from ard.config.schema import ExperimentConfig, MethodConfig, SchedulerConfig
from ard.protocols import ensure_local_trainable, get_protocol
from ard.schedules import build_scheduler

pytestmark = pytest.mark.t1


def _controlled_config() -> dict[str, object]:
    return {
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
        "dataset": {
            "name": "cifar10",
            "root": "/tmp/cifar",
            "split": "train",
            "download": False,
            "num_classes": 10,
            "image_size": 32,
        },
        "student": {
            "architecture": "saad_resnet18_cifar_v1",
            "num_classes": 10,
            "preprocessing_owner": "student_adapter",
            "normalization": {"profile": "cifar10_raw_identity"},
        },
        "method": {
            "id": "rslad",
            "version": 1,
            "attack": {
                "loss": "kl",
                "kl_target": "teacher_clean",
                "steps": 10,
                "epsilon": "8/255",
                "step_size": "2/255",
                "student_mode": "eval",
                "teacher_mode": "eval",
            },
            "selection_attack": {
                "loss": "ce",
                "steps": 20,
                "epsilon": "8/255",
                "step_size": "2/255",
                "student_mode": "eval",
                "teacher_mode": "eval",
            },
        },
        "teacher": {
            "source": "fixture",
            "architecture": "fixture_cnn",
            "num_classes": 10,
            "normalization": {"profile": "cifar10_standard"},
        },
        "optimizer": {"id": "sgd", "learning_rate": 0.1, "momentum": 0.9, "weight_decay": 5e-4, "nesterov": False},
        "scheduler": {"id": "multistep", "milestones": [100, 150], "gamma": 0.1, "step_at": "epoch_end"},
        "training": {
            "epochs": 200,
            "per_rank_batch_size": 128,
            "global_batch_size": 128,
            "validation_fraction": 0.1,
            "deterministic": True,
        },
        "evaluation": {"seed": 0},
    }


def test_controlled_protocol_keeps_distinct_valid_train_and_selection_attacks() -> None:
    config = ExperimentConfig.model_validate(_controlled_config())
    assert config.method.attack.steps == 10
    assert config.method.selection_attack is not None and config.method.selection_attack.steps == 20
    assert config.method.attack.epsilon == config.method.selection_attack.epsilon == "8/255"
    assert config.method.attack.step_size == config.method.selection_attack.step_size == "2/255"
    assert config.method.attack.loss == "kl"
    assert config.method.selection_attack.loss == "ce"


def _pilot_config() -> dict[str, object]:
    data = _controlled_config()
    data["protocol"] = {"id": "controlled_cifar10_r18_pilot_v1"}
    data["tier"] = "pilot"
    training = data["training"]
    assert isinstance(training, dict)
    training["epochs"] = 5
    data["teacher"] = {
        "source": "robustbench",
        "registry_id": "chen2021_ltd_wrn34_10",
        "architecture": "robustbench_wide_resnet",
        "num_classes": 10,
        "normalization": {"profile": "cifar10_standard"},
        "preprocessing_owner": "teacher_adapter",
        "checkpoint": "teacher_cache/robustbench/Chen2021LTD_WRN34_10.pt",
        "checkpoint_sha256": "a" * 64,
    }
    data["tracking"] = {
        "mode": "offline_sync",
        "project": "ard-single-teacher",
        "entity": "shunsuke-n-waseda-university",
        "group": "chen-cifar10-r18-controlled",
    }
    return data


def test_controlled_pilot_protocol_is_five_epoch_and_requires_pilot_tier() -> None:
    config = ExperimentConfig.model_validate(_pilot_config())
    assert config.protocol.id == "controlled_cifar10_r18_pilot_v1"
    assert config.training.epochs == 5
    assert config.training.batchnorm_mode == "local_per_rank"
    invalid = _pilot_config()
    invalid["tier"] = "production"
    with pytest.raises(ValueError, match="requires tier=pilot"):
        ExperimentConfig.model_validate(invalid)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        ("training.epochs", 4, "pilot_v1 contract violation"),
        ("dataset.name", "synthetic_cifar", "forbid synthetic"),
        ("student.architecture", "fixture_cnn", "forbid synthetic"),
        ("tracking.mode", "disabled", "require online or offline_sync"),
        ("tracking.entity", None, "requires tracking.project"),
        ("tracking.group", None, "requires tracking.project"),
    ),
)
def test_controlled_pilot_contract_rejects_noncanonical_or_untracked_inputs(
    path: str, value: object, message: str
) -> None:
    data = _pilot_config()
    _set_path(data, path, value)
    with pytest.raises(ValueError, match=message):
        ExperimentConfig.model_validate(data)


def _set_path(data: dict[str, object], path: str, value: object) -> None:
    target = data
    parts = path.split(".")
    for part in parts[:-1]:
        nested = target[part]
        assert isinstance(nested, dict)
        target = nested
    target[parts[-1]] = value


@pytest.mark.parametrize(
    ("path", "value"),
    (
        ("protocol.id", "synthetic_smoke_v2"),
        ("dataset.name", "synthetic_cifar"),
        ("dataset.split", "test"),
        ("dataset.download", True),
        ("dataset.num_classes", 2),
        ("dataset.image_size", 16),
        ("student.architecture", "fixture_cnn"),
        ("student.num_classes", 2),
        ("student.preprocessing_owner", "not_student_adapter"),
        ("student.normalization.profile", "cifar10_standard"),
        ("training.epochs", 199),
        ("training.global_batch_size", 64),
        ("training.validation_fraction", 0.2),
        ("training.deterministic", False),
        ("seeds.split", 0),
        ("seeds.evaluation_attack", 1),
        ("evaluation.seed", 1),
        ("optimizer.learning_rate", 0.01),
        ("optimizer.momentum", 0.0),
        ("optimizer.weight_decay", 0.0),
        ("optimizer.nesterov", True),
        ("scheduler.id", "identity"),
        ("scheduler.milestones", []),
        ("scheduler.gamma", 0.5),
        ("method.attack.loss", "ce"),
        ("method.attack.kl_target", "student_clean"),
        ("method.attack.temperature", 2.0),
        ("method.attack.temperature_squared", False),
        ("method.attack.steps", 9),
        ("method.attack.epsilon", "4/255"),
        ("method.attack.step_size", "1/255"),
        ("method.attack.random_start", False),
        ("method.attack.student_mode", "train"),
        ("method.attack.teacher_mode", "train"),
        ("method.selection_attack.loss", "kl"),
        ("method.selection_attack.kl_target", "teacher_clean"),
        ("method.selection_attack.steps", 10),
        ("method.selection_attack.epsilon", "4/255"),
        ("method.selection_attack.step_size", "1/255"),
        ("method.selection_attack.random_start", False),
        ("method.selection_attack.temperature", 2.0),
        ("method.selection_attack.temperature_squared", False),
        ("method.selection_attack.student_mode", "train"),
        ("method.selection_attack.teacher_mode", "train"),
    ),
)
def test_controlled_protocol_rejects_each_independent_frozen_field(path: str, value: object) -> None:
    data = copy.deepcopy(_controlled_config())
    _set_path(data, path, value)
    with pytest.raises(ValueError):
        ExperimentConfig.model_validate(data)


def test_synthetic_protocol_cannot_relabel_real_repro_or_production_runs() -> None:
    data = _controlled_config()
    data["protocol"] = {"id": "synthetic_smoke_v2"}
    for tier in ("repro", "production"):
        candidate = copy.deepcopy(data)
        candidate["tier"] = tier
        candidate["teacher"] = {
            "source": "checkpoint",
            "architecture": "fixture_cnn",
            "num_classes": 10,
            "checkpoint": "/tmp/teacher.pt",
            "checkpoint_sha256": "a" * 64,
        }
        candidate["tracking"] = {
            "mode": "offline_sync",
            "project": "protocol-test",
            "entity": "protocol-test",
            "group": "protocol-test",
        }
        with pytest.raises(ValueError, match="synthetic_smoke_v2 contract violation"):
            ExperimentConfig.model_validate(candidate)


def test_audit_protocols_are_rejected_by_train_cli_before_any_training(tmp_path: Path) -> None:
    for protocol_id in ("saad_paper_reproduction_v1", "saad_code_295121c_audit_v1"):
        data = {
            "schema_version": 2,
            "protocol": {"id": protocol_id},
            "tier": "dev",
            "seeds": {
                name: 0
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
            "dataset": {"name": "synthetic_cifar", "num_samples": 4, "num_classes": 2},
            "student": {"architecture": "fixture_cnn", "num_classes": 2},
            "method": {"id": "pgd_at", "version": 1, "attack": {"steps": 1}},
            "optimizer": {"id": "sgd", "learning_rate": 0.01, "momentum": 0.0, "weight_decay": 0.0, "nesterov": False},
            "scheduler": {"id": "identity", "milestones": [], "gamma": 1.0, "step_at": "epoch_end"},
            "training": {"epochs": 1, "per_rank_batch_size": 2, "global_batch_size": 2},
            "output_dir": str(tmp_path / protocol_id),
        }
        path = tmp_path / f"{protocol_id}.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        with pytest.raises(ValueError, match="not a controlled|audit-only"):
            train_cli.main(["--config", str(path), "--dry-run"])


def test_controlled_protocol_audit_records_remain_not_locally_trainable() -> None:
    for protocol_id in ("saad_paper_reproduction_v1", "saad_code_295121c_audit_v1"):
        assert not get_protocol(protocol_id).runnable_locally
        with pytest.raises(ValueError):
            ensure_local_trainable(protocol_id)


def test_paper_and_code_audit_metadata_are_independent_of_controlled_protocol() -> None:
    paper = get_protocol("saad_paper_reproduction_v1").metadata
    assert paper["train_set"] == "official_full_50k"
    assert paper["validation"] == "none"
    assert paper["checkpoint_lifecycle"] == "last_or_published"
    assert paper["optimizer"] == {"id": "sgd", "weight_decay": 5e-4}
    assert paper["attack"] == {"algorithm": "pgd", "steps": 10, "epsilon": "8/255", "step_size": "2/255"}
    assert "validation_fraction" not in paper and "split_seed" not in paper

    code = get_protocol("saad_code_295121c_audit_v1").metadata
    assert code["optimizer"] == {"id": "sgd", "weight_decay": 2e-4}
    assert code["source_attack_call"] == {"step_size": "8/255", "epsilon": "2/255"}
    assert code["test_each_epoch"] is True
    assert code["swa"] == {"enabled": True, "start_epoch": 95}
    assert code["parallelism"] == "DataParallel"
    assert "train_attack" not in code and "selection_attack" not in code


def test_selection_ce_does_not_inherit_kl_only_temperature_or_step_count() -> None:
    method = MethodConfig(
        id="rslad",
        version=1,
        attack={"loss": "kl", "kl_target": "teacher_clean", "steps": 10, "temperature": 4.0},
        selection_attack={"loss": "ce", "steps": 20, "temperature": 1.0, "temperature_squared": False},
    )
    assert method.selection_attack is not None and method.selection_attack.steps == 20


def _lr_after_completed_epochs(completed: int) -> tuple[float, dict[str, object]]:
    parameter = nn.Parameter(torch.ones(()))
    optimizer = SGD([parameter], lr=0.1)
    scheduler = build_scheduler(
        optimizer, SchedulerConfig(id="multistep", milestones=(100, 150), gamma=0.1, step_at="epoch_end")
    )
    for _ in range(completed):
        optimizer.step()
        scheduler.step()
    return optimizer.param_groups[0]["lr"], scheduler.state_dict()


def test_epoch_end_multistep_boundaries_and_state_dict_resume_are_exact() -> None:
    observed = {epoch: _lr_after_completed_epochs(epoch)[0] for epoch in (0, 99, 100, 149, 150)}
    assert observed == pytest.approx({0: 0.1, 99: 0.1, 100: 0.01, 149: 0.01, 150: 0.001}, abs=0, rel=1e-15)
    for boundary in (99, 100, 149, 150):
        resumed_parameter = nn.Parameter(torch.ones(()))
        resumed_optimizer = SGD([resumed_parameter], lr=0.1)
        resumed = build_scheduler(
            resumed_optimizer, SchedulerConfig(id="multistep", milestones=(100, 150), gamma=0.1, step_at="epoch_end")
        )
        _, state = _lr_after_completed_epochs(boundary)
        resumed_optimizer.param_groups[0]["lr"] = observed[boundary]
        resumed.load_state_dict(state)
        uninterrupted_lr, uninterrupted_state = _lr_after_completed_epochs(boundary + 1)
        resumed_optimizer.step()
        resumed.step()
        assert resumed_optimizer.param_groups[0]["lr"] == uninterrupted_lr
        assert resumed.state_dict() == uninterrupted_state
