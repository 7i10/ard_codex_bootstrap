from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn
from torch.optim import SGD

from ard.analysis.intervention_fork import build_parent_artifact_attestation
from ard.analysis.schedule_control_fork import (
    ScheduleControlForkError,
    _validate_scheduler_parent,
    create_schedule_control_fork,
)
from ard.cli import train as train_cli
from ard.config import ExperimentConfig, load_config
from ard.config.loader import resolved_config_dict
from ard.data import EpochShuffleSampler
from ard.engine.checkpoint import config_digest, load_checkpoint, save_checkpoint, validate_resume_checkpoint
from ard.schedules import build_scheduler
from ard.state import SampleRecord

pytestmark = pytest.mark.t2


def _equal(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _raw_config(tmp_path: Path) -> dict[str, object]:
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
            "checkpoint_sha256": "e" * 64,
            "registry_id": "bartoldson2024_adversarial_wrn94_16",
        },
        "method": {
            "id": "rslad",
            "version": 1,
            "attack": {"loss": "kl", "kl_target": "teacher_clean"},
            "selection_attack": {"loss": "ce", "steps": 20},
        },
        "optimizer": {"id": "sgd", "learning_rate": 0.1, "momentum": 0.9, "weight_decay": 5e-4, "nesterov": False},
        "scheduler": {"id": "multistep", "milestones": [100, 150], "gamma": 0.1, "step_at": "epoch_end"},
        "training": {"epochs": 200, "per_rank_batch_size": 128, "global_batch_size": 128, "validation_fraction": 0.1},
        "observation": {"profile": "teacher_response"},
        "output_dir": str(tmp_path / "parent"),
    }


def _sample_state() -> dict[str, object]:
    record = asdict(SampleRecord(0.0, 80, 80, True, 0, 28_159, true_label=0))
    return {
        "format_version": 3,
        "ema_decay": 0.9,
        "records": {str(index): record for index in range(45_000)},
        "pending": [],
        "next_order": 0,
    }


def _advance(optimizer: SGD, scheduler: object, epochs: int) -> None:
    for _ in range(epochs):
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()  # type: ignore[union-attr]


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, Path, dict[str, object]]:
    parent = ExperimentConfig.model_validate(_raw_config(tmp_path))
    raw = resolved_config_dict(parent)
    config_hash = config_digest(raw)
    resolved = tmp_path / "parent-resolved.yaml"
    resolved.write_text(yaml.safe_dump(raw), encoding="utf-8")
    torch.manual_seed(31)
    model = nn.Linear(2, 2)
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = build_scheduler(optimizer, parent.scheduler)
    _advance(optimizer, scheduler, 80)
    assert scheduler.state_dict()["last_epoch"] == 80
    state = _sample_state()
    sampler = EpochShuffleSampler(45_000, seed=2)
    sampler.set_epoch(79)
    checkpoint = tmp_path / "parent.pt"
    save_checkpoint(
        checkpoint,
        epoch=79,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        sampler=sampler,
        sample_state=state,
        global_step=28_160,
        best_metric=0.7,
        selection_metadata={"metric": "val_pgd_accuracy", "selected_epoch": 40},
        tracker_run_id="parent-run",
        config_hash=config_hash,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["rng"][0]["torch_cuda"] = [torch.tensor([0], dtype=torch.uint8)]
    torch.save(payload, checkpoint)
    rows = [[index, 0] for index in range(45_000)]
    rows_hash = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    partition = tmp_path / "partition.json"
    partition.write_text(
        json.dumps({"schema_version": 1, "namespace": "train", "ids_labels": rows, "ids_labels_sha256": rows_hash}),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "config_hash": config_hash,
                "run_id": "parent-run",
                "git": {"sha": "a" * 40},
                "teacher": {"checkpoint_sha256": "e" * 64},
            }
        ),
        encoding="utf-8",
    )
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "artifact": {
                    "name": "model-parent-last",
                    "version": "v15",
                    "digest": "d" * 32,
                    "checkpoint_sha256": checkpoint_hash,
                }
            }
        ),
        encoding="utf-8",
    )
    attestation = tmp_path / "attestation.json"
    attestation.write_text(
        json.dumps(
            build_parent_artifact_attestation(
                parent_manifest=manifest, artifact_inventory=inventory, checkpoint=checkpoint
            ),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    fields = {
        "checkpoint_sha256": checkpoint_hash,
        "raw_config_sha256": config_hash,
        "git_sha": "a" * 40,
        "epoch": 79,
        "world_size": 1,
        "teacher_checkpoint_sha256": "e" * 64,
        "sample_state_records": 45_000,
        "sample_state_sha256": hashlib.sha256(
            json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
        "train_partition_manifest": str(partition),
        "train_partition_manifest_sha256": hashlib.sha256(partition.read_bytes()).hexdigest(),
        "train_partition_ids_labels_sha256": rows_hash,
        "artifact_attestation": str(attestation),
        "artifact_attestation_sha256": hashlib.sha256(attestation.read_bytes()).hexdigest(),
        "artifact_inventory": str(inventory),
        "artifact_inventory_sha256": hashlib.sha256(inventory.read_bytes()).hexdigest(),
    }
    return checkpoint, resolved, manifest, inventory, attestation, partition, fields


def _child_config(tmp_path: Path, raw: dict[str, object], fields: dict[str, object]) -> Path:
    child = copy.deepcopy(raw)
    child["output_dir"] = str(tmp_path / "control")
    child["protocol"] = {"id": "controlled_cifar10_r18_delayed_multistep_v1"}
    child["scheduler"] = {"id": "multistep", "milestones": [120, 170], "gamma": 0.1, "step_at": "epoch_end"}
    path = tmp_path / "control.yaml"
    path.write_text(yaml.safe_dump(child), encoding="utf-8")
    return path


def _spec(tmp_path: Path, fields: dict[str, object]) -> Path:
    path = tmp_path / "schedule-control-spec.yaml"
    path.write_text(
        yaml.safe_dump({"schema_version": 1, "kind": "delayed_multistep_schedule_control_v1", "parent": fields}),
        encoding="utf-8",
    )
    return path


def test_epoch79_schedule_control_preserves_state_replaces_only_future_milestones_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, resolved, manifest, inventory, attestation, _, fields = _inputs(tmp_path)
    raw = _raw_config(tmp_path)
    child_path = _child_config(tmp_path, raw, fields)
    spec = _spec(tmp_path, fields)
    created = create_schedule_control_fork(
        parent_checkpoint=checkpoint,
        parent_resolved_config=resolved,
        parent_manifest=manifest,
        artifact_inventory=inventory,
        artifact_attestation=attestation,
        spec_path=spec,
        child_config_path=child_path,
        root=Path.cwd(),
        git_state_collector=lambda _: {"sha": "b" * 40, "dirty": False},
    )
    parent = torch.load(checkpoint, map_location="cpu", weights_only=False)
    child = torch.load(created, map_location="cpu", weights_only=False)
    for key in ("model", "optimizer", "scaler", "rng", "sampler_epoch", "sampler_state", "sample_state", "global_step"):
        assert _equal(parent[key], child[key]), key
    assert child["scheduler"]["milestones"] == Counter({120: 1, 170: 1})
    for key in ("last_epoch", "_step_count", "base_lrs", "_last_lr", "gamma"):
        assert child["scheduler"][key] == parent["scheduler"][key], key
    assert child["best_metric"] == float("-inf")
    assert child["selection_metadata"]["scope"] == "post_fork_best"
    assert child["fork_lineage"]["kind"] == "delayed_multistep_schedule_control_v1"
    assert child["fork_lineage"]["parent_epoch"] == 79
    assert child["fork_lineage"]["parent_tracker_run_id"] == parent["tracker_run_id"]
    child_config = load_config(child_path)
    child_hash = config_digest(resolved_config_dict(child_config))
    validate_resume_checkpoint(created, expected_config_hash=child_hash)
    with pytest.raises(ValueError, match="config hash"):
        validate_resume_checkpoint(
            created, expected_config_hash=config_digest(resolved_config_dict(ExperimentConfig.model_validate(raw)))
        )
    # Training recognizes no schedule-control-specific mode: its existing
    # intervention guard allows this ordinary exact-child config, while the
    # standard checkpoint hash rejects the parent config above.
    train_cli._validate_intervention_resume(created, child_config, config_hash=child_hash)
    monkeypatch.setattr(train_cli, "collect_git_state", lambda _: {"sha": "b" * 40, "dirty": False})
    with pytest.raises(ValueError, match="requires a registered"):
        train_cli._validate_required_fork_resume(None, child_config, config_hash=child_hash)
    train_cli._validate_required_fork_resume(created, child_config, config_hash=child_hash)
    one_epoch = tmp_path / "one-epoch.pt"
    one_epoch_payload = copy.deepcopy(child)
    one_epoch_payload["epoch"] = 80
    torch.save(one_epoch_payload, one_epoch)
    train_cli._validate_required_fork_resume(one_epoch, child_config, config_hash=child_hash)
    for key, value, message in (
        ("kind", "wrong", "matching"),
        ("child_config_sha256", "0" * 64, "child config"),
        ("child_tracker_run_id", "wrong-run", "tracker run ID"),
    ):
        invalid = copy.deepcopy(child)
        invalid["fork_lineage"][key] = value
        path = tmp_path / f"invalid-{key}.pt"
        torch.save(invalid, path)
        with pytest.raises(ValueError, match=message):
            train_cli._validate_required_fork_resume(path, child_config, config_hash=child_hash)
    monkeypatch.setattr(train_cli, "collect_git_state", lambda _: {"sha": "c" * 40, "dirty": False})
    with pytest.raises(ValueError, match="clean Git SHA"):
        train_cli._validate_required_fork_resume(created, child_config, config_hash=child_hash)
    parent_config = ExperimentConfig.model_validate(raw)
    with pytest.raises(ValueError, match="ordinary protocols"):
        train_cli._validate_required_fork_resume(
            created, parent_config, config_hash=config_digest(resolved_config_dict(parent_config))
        )

    # The transformed state is exactly the uninterrupted delayed schedule at
    # this epoch boundary.  Epoch 80 starts at 0.1; the first decay appears
    # only after 120 completed epochs.
    def components() -> tuple[nn.Linear, SGD, object, EpochShuffleSampler]:
        torch.manual_seed(31)
        model = nn.Linear(2, 2)
        optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        config = child_config.scheduler
        return model, optimizer, build_scheduler(optimizer, config), EpochShuffleSampler(45_000, seed=2)

    uninterrupted_model, uninterrupted_optimizer, uninterrupted_scheduler, _ = components()
    _advance(uninterrupted_optimizer, uninterrupted_scheduler, 80)
    fork_model, fork_optimizer, fork_scheduler, fork_sampler = components()
    state = load_checkpoint(
        created,
        model=fork_model,
        optimizer=fork_optimizer,
        scheduler=fork_scheduler,
        scaler=None,
        sampler=fork_sampler,
        expected_config_hash=child_hash,
        device=torch.device("cpu"),
    )
    assert state.next_epoch == 80
    assert fork_optimizer.param_groups[0]["lr"] == 0.1
    assert _equal(uninterrupted_scheduler.state_dict(), fork_scheduler.state_dict())
    assert _equal(uninterrupted_optimizer.state_dict(), fork_optimizer.state_dict())
    _advance(fork_optimizer, fork_scheduler, 40)
    assert fork_scheduler.get_last_lr() == pytest.approx([0.01], abs=1e-15)

    # Two independent exact resumes from the child checkpoint produce the
    # same first post-fork optimizer/scheduler transition.
    left_model, left_optimizer, left_scheduler, left_sampler = components()
    right_model, right_optimizer, right_scheduler, right_sampler = components()
    load_checkpoint(
        created,
        model=left_model,
        optimizer=left_optimizer,
        scheduler=left_scheduler,
        scaler=None,
        sampler=left_sampler,
        expected_config_hash=child_hash,
        device=torch.device("cpu"),
    )
    load_checkpoint(
        created,
        model=right_model,
        optimizer=right_optimizer,
        scheduler=right_scheduler,
        scaler=None,
        sampler=right_sampler,
        expected_config_hash=child_hash,
        device=torch.device("cpu"),
    )
    inputs = torch.tensor([[0.1, -0.2], [0.3, 0.4]])
    for model, optimizer, scheduler in (
        (left_model, left_optimizer, left_scheduler),
        (right_model, right_optimizer, right_scheduler),
    ):
        optimizer.zero_grad(set_to_none=True)
        model(inputs).square().mean().backward()
        optimizer.step()
        scheduler.step()
    assert _equal(left_model.state_dict(), right_model.state_dict())
    assert _equal(left_optimizer.state_dict(), right_optimizer.state_dict())
    assert _equal(left_scheduler.state_dict(), right_scheduler.state_dict())


def test_schedule_control_rejects_non_allowlisted_delta_and_non_predecay_parent(tmp_path: Path) -> None:
    checkpoint, resolved, manifest, inventory, attestation, _, fields = _inputs(tmp_path)
    raw = _raw_config(tmp_path)
    child_path = _child_config(tmp_path, raw, fields)
    spec = _spec(tmp_path, fields)
    child = yaml.safe_load(child_path.read_text(encoding="utf-8"))
    child["method"]["temperature"] = 2.0
    child_path.write_text(yaml.safe_dump(child), encoding="utf-8")
    with pytest.raises(ScheduleControlForkError, match="outside output/tracking"):
        create_schedule_control_fork(
            parent_checkpoint=checkpoint,
            parent_resolved_config=resolved,
            parent_manifest=manifest,
            artifact_inventory=inventory,
            artifact_attestation=attestation,
            spec_path=spec,
            child_config_path=child_path,
            root=Path.cwd(),
            git_state_collector=lambda _: {"sha": "b" * 40, "dirty": False},
        )
    child["method"].pop("temperature")
    child_path.write_text(yaml.safe_dump(child), encoding="utf-8")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["scheduler"]["last_epoch"] = 100
    with pytest.raises(ScheduleControlForkError, match="pre-decay"):
        _validate_scheduler_parent(payload)
