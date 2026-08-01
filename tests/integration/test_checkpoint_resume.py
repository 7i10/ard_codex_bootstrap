from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Sized
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
import yaml
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from ard.analysis.intervention_fork import build_parent_artifact_attestation, create_intervention_forks
from ard.attacks import LinfPGD
from ard.attacks.base import AttackResult
from ard.config import ExperimentConfig
from ard.config.loader import resolved_config_dict
from ard.config.schema import AttackConfig, ModelConfig, SchedulerConfig
from ard.data import (
    EpochShuffleSampler,
    EpochSourceTransform,
    IndexedBatch,
    IndexedDataset,
    SyntheticCIFAR,
    collate_indexed,
    stratified_train_validation_split,
)
from ard.engine.checkpoint import REQUIRED_KEYS, config_digest, load_checkpoint, save_checkpoint
from ard.engine.trainer import Trainer
from ard.models import build_student
from ard.objectives import ObjectiveTerms, PGDATObjective, RSLADObjective
from ard.policies import EntropyOnlyPolicy, RSLADBaselinePolicy, selected_ids_sha256
from ard.schedules import build_scheduler
from ard.state import SampleRecord, SampleStateStore
from ard.tracking import NullTracker, coordinated_tracker_action
from ard.tracking.diagnostics import TrainingDiagnostics


def test_training_diagnostics_are_observational_for_full_checkpoint_state(tmp_path: Path) -> None:
    import random

    import numpy as np

    def equal(left: object, right: object) -> bool:
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            return torch.equal(left, right)
        if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
            return np.array_equal(left, right)
        if isinstance(left, dict) and isinstance(right, dict):
            return left.keys() == right.keys() and all(equal(left[key], right[key]) for key in left)
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
        return left == right

    plain = make_trainer(tmp_path / "plain-diagnostics")
    observed = make_trainer(tmp_path / "observed-diagnostics")
    loader_a, validation_a, _ = make_loaders()
    loader_b, validation_b, _ = make_loaders()
    observed.diagnostics = TrainingDiagnostics.for_ids(list(range(8)), seed=4, size=2)
    torch.manual_seed(991)
    np.random.seed(991)
    random.seed(991)
    plain.fit(loader_a, validation_loader=validation_a, epochs=1)
    torch.manual_seed(991)
    np.random.seed(991)
    random.seed(991)
    observed.fit(loader_b, validation_loader=validation_b, epochs=1)
    for name in ("best.pt", "last.pt"):
        first = torch.load(tmp_path / "plain-diagnostics" / name, map_location="cpu", weights_only=False)
        second = torch.load(tmp_path / "observed-diagnostics" / name, map_location="cpu", weights_only=False)
        assert REQUIRED_KEYS.issubset(first) and REQUIRED_KEYS.issubset(second)
        for key in REQUIRED_KEYS:
            assert equal(first[key], second[key]), key


def test_teacher_response_observation_is_exact_for_optimization_rng_and_checkpoint_state(tmp_path: Path) -> None:
    import random

    import numpy as np

    class IdentityAttack:
        def generate(self, request: object) -> AttackResult:
            inputs = request.inputs  # type: ignore[attr-defined]
            return AttackResult(inputs, torch.zeros_like(inputs), (), 0.0)

    def make(output: Path, *, observed: bool) -> Trainer:
        torch.manual_seed(123)
        student = build_student(ModelConfig(architecture="fixture_cnn", num_classes=3), tier="smoke")
        torch.manual_seed(456)
        teacher = build_student(ModelConfig(architecture="fixture_cnn", num_classes=3), tier="smoke")
        optimizer = SGD(student.parameters(), lr=0.03, momentum=0.9)
        return Trainer(
            model=student,
            teacher=teacher,
            optimizer=optimizer,
            scheduler=StepLR(optimizer, step_size=1, gamma=0.8),
            scaler=None,
            attack=IdentityAttack(),  # type: ignore[arg-type]
            selection_attack=IdentityAttack(),  # type: ignore[arg-type]
            objective=RSLADObjective(),
            policy=RSLADBaselinePolicy(),
            sample_store=SampleStateStore(ema_decay=0.9) if observed else None,
            observation_profile="teacher_response" if observed else "off",
            diagnostics=TrainingDiagnostics.for_ids(list(range(8)), seed=4, size=0, mode="summary"),
            device=torch.device("cpu"),
            output_dir=output,
            config_hash="logging-only-parity",
            seed=4,
            tracker_run_id="logging-only-parity",
        )

    def equal(left: object, right: object) -> bool:
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            return torch.equal(left, right)
        if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
            return np.array_equal(left, right)
        if isinstance(left, dict) and isinstance(right, dict):
            return left.keys() == right.keys() and all(equal(left[key], right[key]) for key in left)
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
        return left == right

    plain = make(tmp_path / "plain-rslad", observed=False)
    observed = make(tmp_path / "logging-only", observed=True)
    loader_a, validation_a, _ = make_loaders()
    loader_b, validation_b, _ = make_loaders()
    torch.manual_seed(991)
    np.random.seed(991)
    random.seed(991)
    plain.fit(loader_a, validation_loader=validation_a, epochs=2)
    torch.manual_seed(991)
    np.random.seed(991)
    random.seed(991)
    observed.fit(loader_b, validation_loader=validation_b, epochs=2)
    first = torch.load(tmp_path / "plain-rslad" / "last.pt", map_location="cpu", weights_only=False)
    second = torch.load(tmp_path / "logging-only" / "last.pt", map_location="cpu", weights_only=False)
    for key in REQUIRED_KEYS - {"sample_state"}:
        assert equal(first[key], second[key]), key
    assert first["sample_state"] == {}
    assert second["sample_state"]["format_version"] == 3
    assert len(second["sample_state"]["records"]) == len(cast(Sized, loader_b.dataset))
    for record in second["sample_state"]["records"].values():
        assert record["seen"] == 2
        assert record["teacher_clean_entropy"] is not None
        assert record["teacher_adversarial_entropy"] is not None
        assert record["teacher_clean_true_probability"] is not None
        assert record["teacher_adversarial_max_wrong_probability"] is not None
        assert record["teacher_clean_to_adversarial_margin_response"] is not None
        assert record["teacher_clean_to_adversarial_js_response"] is not None
        assert record["margin_mean"] is not None
        assert record["margin_m2"] is not None
        assert record["longest_correct_streak"] >= record["current_correct_streak"]
    assert all(parameter.grad is None for parameter in observed.teacher.parameters())


def test_teacher_response_reuses_one_adversarial_forward_for_entropy_policy(tmp_path: Path) -> None:
    class IdentityAttack:
        def generate(self, request: object) -> AttackResult:
            inputs = request.inputs  # type: ignore[attr-defined]
            return AttackResult(inputs, torch.zeros_like(inputs), (), 0.0)

    class CountingTeacher(nn.Module):
        def __init__(self, module: nn.Module) -> None:
            super().__init__()
            self.module = module
            self.calls = 0

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return self.module(inputs)

    torch.manual_seed(123)
    student = build_student(ModelConfig(architecture="fixture_cnn", num_classes=3), tier="smoke")
    torch.manual_seed(456)
    teacher = CountingTeacher(build_student(ModelConfig(architecture="fixture_cnn", num_classes=3), tier="smoke"))
    optimizer = SGD(student.parameters(), lr=0.03, momentum=0.9)
    trainer = Trainer(
        model=student,
        teacher=teacher,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        attack=IdentityAttack(),  # type: ignore[arg-type]
        selection_attack=IdentityAttack(),  # type: ignore[arg-type]
        objective=RSLADObjective(),
        policy=EntropyOnlyPolicy(),
        sample_store=SampleStateStore(ema_decay=0.9),
        observation_profile="teacher_response",
        device=torch.device("cpu"),
        output_dir=tmp_path,
        config_hash="teacher-forward-reuse",
        seed=4,
    )
    loader, _, _ = make_loaders()
    trainer.train_epoch(loader)
    # RSLAD needs one clean teacher target; teacher_response supplies exactly
    # one adversarial response, reused by entropy policy rather than recomputed.
    assert teacher.calls == 2 * len(loader)
    assert all(parameter.grad is None for parameter in teacher.parameters())


@pytest.mark.gpu
def test_teacher_response_cuda_parity_with_random_start_pgd(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    import random

    import numpy as np

    device = torch.device("cuda:0")

    def make(output: Path, *, observed: bool) -> Trainer:
        torch.manual_seed(123)
        student = build_student(ModelConfig(architecture="fixture_cnn", num_classes=3), tier="smoke")
        torch.manual_seed(456)
        teacher = build_student(ModelConfig(architecture="fixture_cnn", num_classes=3), tier="smoke")
        optimizer = SGD(student.parameters(), lr=0.03, momentum=0.9)
        return Trainer(
            model=student,
            teacher=teacher,
            optimizer=optimizer,
            scheduler=StepLR(optimizer, step_size=1, gamma=0.8),
            scaler=None,
            attack=LinfPGD(
                AttackConfig(
                    loss="kl",
                    kl_target="teacher_clean",
                    epsilon="1/255",
                    step_size="1/255",
                    steps=1,
                    random_start=True,
                )
            ),
            selection_attack=LinfPGD(
                AttackConfig(
                    loss="ce",
                    epsilon="1/255",
                    step_size="1/255",
                    steps=1,
                    random_start=True,
                    student_mode="eval",
                    teacher_mode="eval",
                )
            ),
            objective=RSLADObjective(),
            policy=RSLADBaselinePolicy(),
            sample_store=SampleStateStore(ema_decay=0.9) if observed else None,
            observation_profile="teacher_response" if observed else "off",
            diagnostics=TrainingDiagnostics.for_ids(list(range(8)), seed=4, size=0, mode="summary"),
            device=device,
            output_dir=output,
            config_hash="logging-only-cuda-parity",
            seed=4,
            evaluation_attack_seed=9,
            tracker_run_id="logging-only-cuda-parity",
        )

    def equal(left: object, right: object) -> bool:
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            return torch.equal(left, right)
        if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
            return np.array_equal(left, right)
        if isinstance(left, dict) and isinstance(right, dict):
            return left.keys() == right.keys() and all(equal(left[key], right[key]) for key in left)
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
        return left == right

    plain = make(tmp_path / "plain-rslad-cuda", observed=False)
    observed = make(tmp_path / "logging-only-cuda", observed=True)
    loader_a, validation_a, _ = make_loaders()
    loader_b, validation_b, _ = make_loaders()
    for trainer, loader, validation in (
        (plain, loader_a, validation_a),
        (observed, loader_b, validation_b),
    ):
        torch.manual_seed(991)
        torch.cuda.manual_seed_all(991)
        np.random.seed(991)
        random.seed(991)
        trainer.fit(loader, validation_loader=validation, epochs=1)
    first = torch.load(tmp_path / "plain-rslad-cuda" / "last.pt", map_location="cpu", weights_only=False)
    second = torch.load(tmp_path / "logging-only-cuda" / "last.pt", map_location="cpu", weights_only=False)
    for key in REQUIRED_KEYS - {"sample_state"}:
        assert equal(first[key], second[key]), key
    assert second["sample_state"]["format_version"] == 3
    assert all(parameter.grad is None for parameter in observed.teacher.parameters())


pytestmark = pytest.mark.t3


def make_loaders(seed: int = 4) -> tuple[DataLoader, DataLoader, EpochShuffleSampler]:
    dataset = IndexedDataset(SyntheticCIFAR(size=8, num_classes=3, image_size=4, seed=seed))
    train_dataset, validation_dataset = stratified_train_validation_split(dataset, validation_fraction=0.25, seed=seed)
    sampler = EpochShuffleSampler(len(train_dataset), seed=seed)
    validation_sampler = EpochShuffleSampler(len(validation_dataset), seed=seed, shuffle=False)
    loader = DataLoader(train_dataset, batch_size=4, sampler=sampler, collate_fn=collate_indexed)
    validation_loader = DataLoader(
        validation_dataset, batch_size=4, sampler=validation_sampler, collate_fn=collate_indexed
    )
    return loader, validation_loader, sampler


def make_trainer(output: Path, *, seed: int = 4) -> Trainer:
    torch.manual_seed(123)
    model = build_student(ModelConfig(architecture="fixture_cnn", num_classes=3), tier="smoke")
    optimizer = SGD(model.parameters(), lr=0.03, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.8)
    return Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        attack=LinfPGD(AttackConfig(epsilon="1/255", step_size="1/255", steps=1, random_start=True)),
        selection_attack=LinfPGD(
            AttackConfig(
                epsilon="1/255",
                step_size="1/255",
                steps=1,
                random_start=True,
                student_mode="eval",
                teacher_mode="eval",
            )
        ),
        objective=PGDATObjective(),
        device=torch.device("cpu"),
        output_dir=output,
        config_hash="a" * 64,
        seed=seed,
        tracker_run_id="offline-fixture",
    )


def test_checkpoint_is_complete_and_best_last_are_distinct(tmp_path: Path) -> None:
    trainer = make_trainer(tmp_path)
    trainer.sample_state = {"placeholder_version": 1}
    loader, validation_loader, _ = make_loaders()
    callback_metrics: list[dict[str, float]] = []
    history = trainer.fit(
        loader,
        validation_loader=validation_loader,
        epochs=2,
        on_epoch_end=lambda metrics, _: callback_metrics.append(dict(metrics)),
    )
    best, last = tmp_path / "best.pt", tmp_path / "last.pt"
    assert best.is_file() and last.is_file() and best != last
    payload = torch.load(last, map_location="cpu", weights_only=False)
    assert REQUIRED_KEYS.issubset(payload)
    assert payload["epoch"] == 1 and payload["epoch_boundary"] == "end"
    assert payload["sample_state"] == {"placeholder_version": 1}
    assert payload["tracker_run_id"] == "offline-fixture"
    assert payload["world_size"] == 1
    assert set(history[0]) == {
        "train_loss",
        "train_clean_accuracy",
        "train_robust_accuracy",
        "train_valid_examples",
        "train_seconds",
        "train_images_per_second",
        "train_cuda_peak_allocated_bytes",
        "train_cuda_peak_reserved_bytes",
        "train_teacher_clean_forward_calls",
        "train_teacher_adversarial_forward_calls",
        "val_clean_accuracy",
        "val_pgd_accuracy",
    }
    assert history[0]["train_valid_examples"] == float(len(cast(Sized, loader.dataset)))
    assert history[0]["train_teacher_clean_forward_calls"] == 0.0
    assert history[0]["train_teacher_adversarial_forward_calls"] == 0.0
    assert history[0]["train_cuda_peak_allocated_bytes"] == 0.0
    assert history[0]["train_cuda_peak_reserved_bytes"] == 0.0
    assert callback_metrics == history
    assert payload["selection_metadata"]["metric"] == "val_pgd_accuracy"
    assert payload["selection_metadata"]["tie_break"] == "earliest_epoch"


def test_epoch_boundary_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    uninterrupted = make_trainer(tmp_path / "full")
    full_loader, full_validation_loader, _ = make_loaders()
    uninterrupted_history = uninterrupted.fit(full_loader, validation_loader=full_validation_loader, epochs=2)

    first_leg = make_trainer(tmp_path / "resumed")
    first_loader, first_validation_loader, _ = make_loaders()
    first_leg_history = first_leg.fit(first_loader, validation_loader=first_validation_loader, epochs=1)
    resumed = make_trainer(tmp_path / "resumed")
    resumed_loader, resumed_validation_loader, resumed_sampler = make_loaders()
    state = resumed.resume(tmp_path / "resumed" / "last.pt", sampler=resumed_sampler)
    assert state.next_epoch == 1
    resumed_history = resumed.fit(
        resumed_loader, validation_loader=resumed_validation_loader, epochs=2, start_epoch=state.next_epoch
    )

    for name, expected in uninterrupted.model.state_dict().items():
        assert torch.equal(expected, resumed.model.state_dict()[name]), name
    assert uninterrupted.global_step == resumed.global_step
    assert uninterrupted.best_metric == resumed.best_metric
    deterministic_metrics = (
        "train_loss",
        "train_clean_accuracy",
        "train_robust_accuracy",
        "train_valid_examples",
        "train_teacher_clean_forward_calls",
        "train_teacher_adversarial_forward_calls",
        "val_clean_accuracy",
        "val_pgd_accuracy",
    )
    combined_history = first_leg_history + resumed_history
    for uninterrupted_epoch, resumed_epoch in zip(uninterrupted_history, combined_history, strict=True):
        assert {key: uninterrupted_epoch[key] for key in deterministic_metrics} == {
            key: resumed_epoch[key] for key in deterministic_metrics
        }
        assert uninterrupted_epoch["train_valid_examples"] == float(len(cast(Sized, full_loader.dataset)))
        assert uninterrupted_epoch["train_teacher_clean_forward_calls"] == 0.0
        assert uninterrupted_epoch["train_teacher_adversarial_forward_calls"] == 0.0


def test_fork_lineage_survives_one_epoch_and_strict_resume_without_relaxing_checkpoint_identity(tmp_path: Path) -> None:
    lineage = {"kind": "common_state_intervention_v1", "screen_id": "a" * 64, "post_fork_best_scope": True}
    first = make_trainer(tmp_path / "fork", seed=9)
    first.fork_lineage = dict(lineage)
    loader, validation_loader, _ = make_loaders()
    first.fit(loader, validation_loader=validation_loader, epochs=1)
    initial = torch.load(tmp_path / "fork" / "last.pt", map_location="cpu", weights_only=False)
    assert initial["fork_lineage"] == lineage

    resumed = make_trainer(tmp_path / "fork", seed=9)
    resumed_loader, resumed_validation_loader, resumed_sampler = make_loaders()
    state = resumed.resume(tmp_path / "fork" / "last.pt", sampler=resumed_sampler)
    assert state.fork_lineage == lineage and resumed.fork_lineage == lineage
    resumed.fit(resumed_loader, validation_loader=resumed_validation_loader, epochs=2, start_epoch=state.next_epoch)
    final = torch.load(tmp_path / "fork" / "last.pt", map_location="cpu", weights_only=False)
    assert final["fork_lineage"] == lineage


def test_c_fork_continuation_is_exact_parity_for_one_optimizer_epoch(tmp_path: Path) -> None:
    def make(output: Path, *, config_hash: str, tracker_run_id: str) -> Trainer:
        torch.manual_seed(31)
        student = build_student(ModelConfig(architecture="fixture_cnn", num_classes=3), tier="smoke")
        torch.manual_seed(37)
        teacher = build_student(ModelConfig(architecture="fixture_cnn", num_classes=3), tier="smoke")
        optimizer = SGD(student.parameters(), lr=0.03, momentum=0.9)
        return Trainer(
            model=student,
            teacher=teacher,
            optimizer=optimizer,
            scheduler=StepLR(optimizer, step_size=1),
            scaler=None,
            attack=LinfPGD(
                AttackConfig(
                    loss="kl",
                    kl_target="teacher_clean",
                    epsilon="1/255",
                    step_size="1/255",
                    steps=1,
                    random_start=True,
                )
            ),
            selection_attack=LinfPGD(
                AttackConfig(
                    loss="ce",
                    epsilon="1/255",
                    step_size="1/255",
                    steps=1,
                    random_start=True,
                    student_mode="eval",
                    teacher_mode="eval",
                )
            ),
            objective=RSLADObjective(),
            policy=RSLADBaselinePolicy(),
            device=torch.device("cpu"),
            output_dir=output,
            config_hash=config_hash,
            seed=9,
            tracker_run_id=tracker_run_id,
        )

    # The parent config is deliberately the registered controlled protocol,
    # while the executable model is a tiny CPU fixture.  Fork construction
    # validates its immutable control-plane state; continuation below proves
    # that the returned C checkpoint itself retains executable state exactly.
    raw: dict[str, object] = {
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
        "optimizer": {"id": "sgd", "learning_rate": 0.1, "momentum": 0.9, "weight_decay": 0.0005, "nesterov": False},
        "scheduler": {"id": "multistep", "milestones": [100, 150], "gamma": 0.1, "step_at": "epoch_end"},
        "training": {"epochs": 200, "per_rank_batch_size": 128, "global_batch_size": 128, "validation_fraction": 0.1},
        "observation": {"profile": "teacher_response"},
        "output_dir": str(tmp_path / "parent"),
        "intervention": None,
    }
    parent = ExperimentConfig.model_validate(raw)
    raw = resolved_config_dict(parent)
    raw_hash = config_digest(raw)
    parent_config = tmp_path / "parent.yaml"
    parent_config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    loader_a, validation_a, sampler_a = make_loaders(seed=0)
    loader_b, validation_b, sampler_b = make_loaders(seed=0)
    parent_checkpoint = tmp_path / "parent.pt"
    parent_trainer = make(tmp_path / "parent-runtime", config_hash=raw_hash, tracker_run_id="parent-run")
    record = asdict(
        SampleRecord(
            0.0,
            100,
            0,
            False,
            0,
            0,
            true_label=0,
            teacher_clean_entropy=0.1,
            teacher_clean_true_probability=0.8,
            teacher_clean_max_wrong_probability=0.1,
            teacher_clean_prediction=0,
            teacher_clean_correct=True,
            teacher_adversarial_entropy=0.2,
            teacher_adversarial_true_probability=0.7,
            teacher_adversarial_max_wrong_probability=0.2,
            teacher_adversarial_prediction=0,
            teacher_adversarial_correct=True,
            teacher_clean_to_adversarial_margin_response=-0.2,
            teacher_clean_to_adversarial_js_response=0.01,
            history_statistics_complete=True,
        )
    )
    sample_state = {
        "format_version": 3,
        "ema_decay": 0.9,
        "records": {str(index): copy.deepcopy(record) for index in range(45_000)},
        "pending": [],
        "next_order": 0,
    }
    common = dict(
        epoch=99,
        model=parent_trainer.model,
        optimizer=parent_trainer.optimizer,
        scheduler=parent_trainer.scheduler,
        scaler=None,
        sampler=sampler_a,
        sample_state=sample_state,
        global_step=35_200,
        best_metric=0.7,
        selection_metadata={"metric": "val_pgd_accuracy", "selected_epoch": 42},
        tracker_run_id="parent-run",
        config_hash=raw_hash,
    )
    save_checkpoint(parent_checkpoint, **common)
    parent_payload = torch.load(parent_checkpoint, map_location="cpu", weights_only=False)
    parent_payload["rng"][0]["torch_cuda"] = [torch.tensor([0], dtype=torch.uint8)]
    parent_payload["sampler_epoch"] = [99]
    parent_payload["sampler_state"] = [{"epoch": 99, "seed": 0, "rank": 0, "world_size": 1, "shuffle": True}]
    torch.save(parent_payload, parent_checkpoint)
    parent_sha = hashlib.sha256(parent_checkpoint.read_bytes()).hexdigest()
    partition_rows = [[index, 0] for index in range(45_000)]
    partition_digest = hashlib.sha256(
        json.dumps(partition_rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    partition = tmp_path / "partition.json"
    partition.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "namespace": "train",
                "ids_labels": partition_rows,
                "ids_labels_sha256": partition_digest,
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "config_hash": raw_hash,
                "run_id": "parent-run",
                "git": {"sha": "a" * 40},
                "teacher": {"checkpoint_sha256": "e" * 64},
            }
        ),
        encoding="utf-8",
    )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "artifact": {
                    "name": "model-parent-run-last",
                    "version": "v19",
                    "digest": "d" * 32,
                    "checkpoint_sha256": parent_sha,
                }
            }
        ),
        encoding="utf-8",
    )
    attestation = tmp_path / "attestation.json"
    attestation.write_text(
        json.dumps(
            build_parent_artifact_attestation(
                parent_manifest=manifest, artifact_inventory=inventory, checkpoint=parent_checkpoint
            ),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    parent_fields = {
        "sample_state_sha256": hashlib.sha256(
            json.dumps(sample_state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
        "train_partition_manifest": str(partition),
        "train_partition_manifest_sha256": hashlib.sha256(partition.read_bytes()).hexdigest(),
        "train_partition_ids_labels_sha256": partition_digest,
        "artifact_attestation": str(attestation),
        "artifact_attestation_sha256": hashlib.sha256(attestation.read_bytes()).hexdigest(),
        "artifact_inventory": str(inventory),
        "artifact_inventory_sha256": hashlib.sha256(inventory.read_bytes()).hexdigest(),
    }
    selector_spec = tmp_path / "selector.json"
    selector_spec.write_text(
        json.dumps(
            {
                "confirmatory_design_sha256": "a0a7fe0e70fcc8aaf519440012900c7bd8e6db92a8f0143d06892fca1146dd38",
                "predictor_spec_sha256": "d653d9ef08cfa94976a0e3279166b47543d16f3eaadb69810769470b77838c12",
                "seed0_report_sha256": "d44ee166f8866b77067ebd07757d394a060242c9cf1cdc5d4513f127897981f8",
                "seed0_lineage_sha256": "9b6ea091dc9ed4ff81bb579bf05d6650ac8e6d4ab6104981c446f29069e4a64e",
                "anchor_epoch": 99,
                "input_namespace": "train_sample_state_only",
                "coefficients_sha256": "a" * 64,
                "preprocessing_sha256": "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    selector_sha = hashlib.sha256(selector_spec.read_bytes()).hexdigest()

    def mask(path: Path, sample_id: int, provenance: dict[str, object]) -> dict[str, object]:
        payload = {
            "schema_version": 1,
            "namespace": "train",
            "num_classes": 10,
            "selected_ids": [sample_id],
            "selected_ids_sha256": selected_ids_sha256((sample_id,)),
            "selected_count": 1,
            "selected_class_counts": {"0": 1},
            "provenance": provenance,
        }
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "selected_ids_sha256": payload["selected_ids_sha256"],
            "selected_count": 1,
            "selected_class_counts": {"0": 1},
            "provenance": provenance,
        }

    history = mask(
        tmp_path / "history.json",
        0,
        {
            "source": "seed0_bartoldson_frozen_predictor",
            "approved_selector_spec_sha256": selector_sha,
            "selector_spec_path": str(selector_spec),
            "parent_checkpoint_sha256": parent_sha,
            "parent_sample_state_sha256": parent_fields["sample_state_sha256"],
            "random_seed": None,
            "generator": None,
            "generator_version": None,
            "reference_history_mask_sha256": None,
            "reference_selected_count": None,
            "reference_selected_class_counts": None,
            "reference_history_selector_spec_sha256": None,
        },
    )
    random = mask(
        tmp_path / "random.json",
        1,
        {
            "source": "class_matched_random",
            "approved_selector_spec_sha256": None,
            "selector_spec_path": None,
            "parent_checkpoint_sha256": parent_sha,
            "parent_sample_state_sha256": parent_fields["sample_state_sha256"],
            "random_seed": 17,
            "generator": "numpy_pcg64",
            "generator_version": "1",
            "reference_history_mask_sha256": history["sha256"],
            "reference_selected_count": 1,
            "reference_selected_class_counts": {"0": 1},
            "reference_history_selector_spec_sha256": selector_sha,
        },
    )
    arms: list[Path] = []
    for arm, selector, kind, arm_mask in (
        ("C", "none", "ordinary_rslad", None),
        ("HS", "student_history", "uniform_target_softening", history),
        ("RS", "class_matched_random", "uniform_target_softening", random),
        ("HD", "student_history", "adversarial_kd_downweight", history),
        ("RD", "class_matched_random", "adversarial_kd_downweight", random),
    ):
        child = copy.deepcopy(raw)
        child["output_dir"] = str(tmp_path / "screen" / arm)
        child["intervention"] = {
            "arm": arm,
            "selector": selector,
            "kind": kind,
            "parent": {
                "checkpoint_sha256": parent_sha,
                "raw_config_sha256": raw_hash,
                "git_sha": "a" * 40,
                "epoch": 99,
                "world_size": 1,
                "teacher_checkpoint_sha256": "e" * 64,
                "sample_state_records": 45_000,
                **parent_fields,
            },
            "mask": arm_mask,
        }
        path = tmp_path / f"{arm}.yaml"
        path.write_text(yaml.safe_dump(child), encoding="utf-8")
        arms.append(path)
    created = create_intervention_forks(
        parent_checkpoint=parent_checkpoint,
        parent_resolved_config=parent_config,
        parent_manifest=manifest,
        arm_config_paths=arms,
        root=Path.cwd(),
        git_state_collector=lambda _root: {"sha": "b" * 40, "dirty": False},
    )
    # The returned C checkpoint, rather than a hand-written copy, is resumed.
    c_payload = torch.load(created["C"], map_location="cpu", weights_only=False)
    ordinary_path = tmp_path / "ordinary" / "last.pt"
    ordinary_path.parent.mkdir()
    ordinary_payload = copy.deepcopy(c_payload)
    ordinary_payload.pop("fork_lineage")
    torch.save(ordinary_payload, ordinary_path)
    ordinary = make(
        tmp_path / "ordinary",
        config_hash=str(c_payload["config_hash"]),
        tracker_run_id=str(c_payload["tracker_run_id"]),
    )
    c_fork = make(
        created["C"].parent, config_hash=str(c_payload["config_hash"]), tracker_run_id=str(c_payload["tracker_run_id"])
    )
    ordinary.resume(ordinary_path, sampler=sampler_a)
    ordinary.fit(loader_a, validation_loader=validation_a, epochs=101, start_epoch=100)
    c_fork.resume(created["C"], sampler=sampler_b)
    c_fork.fit(loader_b, validation_loader=validation_b, epochs=101, start_epoch=100)
    left = torch.load(tmp_path / "ordinary" / "last.pt", map_location="cpu", weights_only=False)
    right = torch.load(created["C"], map_location="cpu", weights_only=False)
    for key in REQUIRED_KEYS:
        assert _state_equal(left[key], right[key]), key
    assert right["fork_lineage"]["kind"] == "common_state_intervention_v1"


def _advance_optimizer_and_schedule(optimizer: SGD, scheduler: object, *, completed_epochs: int) -> None:
    for _ in range(completed_epochs):
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()  # type: ignore[union-attr]


def _state_equal(left: object, right: object) -> bool:
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_state_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_state_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


@pytest.mark.parametrize("completed_epochs", (99, 100, 149, 150))
def test_multistep_optimizer_scheduler_checkpoint_roundtrip_matches_uninterrupted(
    tmp_path: Path, completed_epochs: int
) -> None:
    schedule = SchedulerConfig(id="multistep", milestones=(100, 150), gamma=0.1, step_at="epoch_end")

    def components() -> tuple[nn.Linear, SGD, object, EpochShuffleSampler]:
        torch.manual_seed(71)
        model = nn.Linear(2, 2)
        optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
        return model, optimizer, build_scheduler(optimizer, schedule), EpochShuffleSampler(2, seed=3)

    uninterrupted_model, uninterrupted_optimizer, uninterrupted_scheduler, _ = components()
    _advance_optimizer_and_schedule(
        uninterrupted_optimizer, uninterrupted_scheduler, completed_epochs=completed_epochs + 1
    )

    saved_model, saved_optimizer, saved_scheduler, saved_sampler = components()
    _advance_optimizer_and_schedule(saved_optimizer, saved_scheduler, completed_epochs=completed_epochs)
    saved_sampler.set_epoch(completed_epochs)
    checkpoint = tmp_path / f"boundary-{completed_epochs}.pt"
    save_checkpoint(
        checkpoint,
        epoch=completed_epochs - 1,
        model=saved_model,
        optimizer=saved_optimizer,
        scheduler=saved_scheduler,
        scaler=None,
        sampler=saved_sampler,
        sample_state={},
        global_step=completed_epochs,
        best_metric=0.0,
        selection_metadata={},
        tracker_run_id="scheduler-roundtrip",
        config_hash="a" * 64,
    )

    resumed_model, resumed_optimizer, resumed_scheduler, resumed_sampler = components()
    state = load_checkpoint(
        checkpoint,
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        scaler=None,
        sampler=resumed_sampler,
        expected_config_hash="a" * 64,
        device=torch.device("cpu"),
    )
    assert state.next_epoch == completed_epochs
    assert _state_equal(resumed_optimizer.state_dict(), saved_optimizer.state_dict())
    assert resumed_scheduler.state_dict() == saved_scheduler.state_dict()  # type: ignore[union-attr]
    assert resumed_optimizer.param_groups[0]["lr"] == saved_optimizer.param_groups[0]["lr"]

    _advance_optimizer_and_schedule(resumed_optimizer, resumed_scheduler, completed_epochs=1)
    assert _state_equal(resumed_model.state_dict(), uninterrupted_model.state_dict())
    assert _state_equal(resumed_optimizer.state_dict(), uninterrupted_optimizer.state_dict())
    assert resumed_scheduler.state_dict() == uninterrupted_scheduler.state_dict()  # type: ignore[union-attr]
    assert resumed_optimizer.param_groups[0]["lr"] == uninterrupted_optimizer.param_groups[0]["lr"]


def test_epoch_keyed_augmentation_view_at_resumed_epoch_matches_uninterrupted() -> None:
    raw = SyntheticCIFAR(size=8, num_classes=2, image_size=32, seed=13)
    uninterrupted = IndexedDataset(raw, EpochSourceTransform(augmentation_seed=17))
    resumed = IndexedDataset(raw, EpochSourceTransform(augmentation_seed=17))
    uninterrupted.set_epoch(100)
    resumed.set_epoch(100)
    for source_id in range(len(raw)):
        first, _, first_id = uninterrupted[source_id]
        second, _, second_id = resumed[source_id]
        assert first_id == second_id == source_id
        assert torch.equal(first, second)


def test_rng_consuming_recording_callback_is_scientifically_observational(tmp_path: Path) -> None:
    plain = make_trainer(tmp_path / "plain")
    recorded = make_trainer(tmp_path / "recorded")
    loader_a, validation_a, _ = make_loaders()
    loader_b, validation_b, _ = make_loaders()
    plain.fit(loader_a, validation_loader=validation_a, epochs=2)
    tracker = NullTracker("recording-only")

    def callback(_: object, __: bool) -> None:
        def consume(_: object) -> None:
            import random

            random.random()
            torch.rand(31)

        coordinated_tracker_action(tracker, phase="recording parity", action=consume)

    recorded.fit(loader_b, validation_loader=validation_b, epochs=2, on_epoch_end=callback)
    for key, value in plain.model.state_dict().items():
        assert torch.equal(value, recorded.model.state_dict()[key])
    for name in ("best.pt", "last.pt"):
        first = torch.load(tmp_path / "plain" / name, map_location="cpu", weights_only=False)
        second = torch.load(tmp_path / "recorded" / name, map_location="cpu", weights_only=False)
        for key in (
            "model",
            "optimizer",
            "scheduler",
            "best_metric",
            "global_step",
            "selection_metadata",
            "sample_state",
        ):
            assert (
                first[key] == second[key]
                if not isinstance(first[key], dict)
                else first[key].keys() == second[key].keys()
            )


def test_checkpoint_resume_restores_student_sample_store_exactly(tmp_path: Path) -> None:
    trainer = make_trainer(tmp_path)
    trainer.sample_store = SampleStateStore(ema_decay=0.9)
    trainer.sample_state = trainer.sample_store.state_dict()
    loader, validation_loader, _ = make_loaders()
    trainer.fit(loader, validation_loader=validation_loader, epochs=1)
    expected = trainer.sample_store.state_dict()
    assert expected["records"] and expected["pending"] == []

    resumed = make_trainer(tmp_path)
    resumed.sample_store = SampleStateStore(ema_decay=0.9)
    _, _, sampler = make_loaders()
    resumed.resume(tmp_path / "last.pt", sampler=sampler)
    assert resumed.sample_store.state_dict() == expected


def test_resume_rejects_world_size_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = make_trainer(tmp_path)
    loader, validation_loader, _ = make_loaders()
    trainer.fit(loader, validation_loader=validation_loader, epochs=1)
    target = make_trainer(tmp_path)
    _, _, sampler = make_loaders()
    monkeypatch.setattr("ard.engine.checkpoint.get_world_size", lambda: 2)
    with pytest.raises(ValueError, match="world size"):
        target.resume(tmp_path / "last.pt", sampler=sampler)


@pytest.mark.parametrize("world_size", (0, -1, True, "1"))
def test_resume_rejects_invalid_checkpoint_world_size(tmp_path: Path, world_size: object) -> None:
    trainer = make_trainer(tmp_path)
    loader, validation_loader, _ = make_loaders()
    trainer.fit(loader, validation_loader=validation_loader, epochs=1)
    checkpoint = tmp_path / "last.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["world_size"] = world_size
    torch.save(payload, checkpoint)
    target = make_trainer(tmp_path)
    _, _, sampler = make_loaders()

    with pytest.raises(ValueError, match="positive integer"):
        target.resume(checkpoint, sampler=sampler)


def test_best_selection_uses_post_update_validation_and_keeps_earliest_tie(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = make_trainer(tmp_path)
    trainer.scheduler = None
    loader, validation_loader, _ = make_loaders()
    train_results = iter(
        [
            {"loss": 0.0, "clean_accuracy": 0.0, "robust_accuracy": 1.0},
            {"loss": 0.0, "clean_accuracy": 0.0, "robust_accuracy": 0.0},
        ]
    )
    validation_results = iter(
        [
            {"clean_accuracy": 0.5, "pgd_accuracy": 0.25},
            {"clean_accuracy": 0.5, "pgd_accuracy": 0.25},
        ]
    )
    monkeypatch.setattr(trainer, "train_epoch", lambda _: next(train_results))
    monkeypatch.setattr(trainer, "validate_epoch", lambda _: next(validation_results))
    trainer.fit(loader, validation_loader=validation_loader, epochs=2)
    best = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=False)
    last = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    assert best["epoch"] == 0
    assert last["selection_metadata"]["selected_epoch"] == 0
    assert trainer.best_metric == pytest.approx(0.25)


def test_padded_rows_are_excluded_from_training_loss_and_accuracy(tmp_path: Path) -> None:
    class IdentityAttack:
        def generate(self, request):
            return AttackResult(request.inputs, torch.zeros_like(request.inputs), (), 0.0)

    class LabelObjective:
        def __call__(
            self, *, student_logits: torch.Tensor, labels: torch.Tensor, teacher_logits=None
        ) -> ObjectiveTerms:
            hard = student_logits[:, 0] * 0 + labels.to(torch.float32) + 1
            return ObjectiveTerms(hard, torch.zeros_like(hard), torch.zeros_like(hard))

    trainer = make_trainer(tmp_path)
    trainer.attack = IdentityAttack()
    trainer.objective = LabelObjective()
    for parameter in trainer.model.parameters():
        parameter.data.zero_()
    batch = IndexedBatch(
        images=torch.rand(4, 3, 4, 4),
        labels=torch.tensor([0, 1, 0, 1]),
        sample_ids=torch.tensor([0, 1, 0, 1]),
        state_update_mask=torch.tensor([True, True, False, False]),
        multiplicity=torch.tensor([2, 2, 2, 2]),
    )
    metrics = trainer.train_epoch([batch])
    assert {key: metrics[key] for key in ("loss", "clean_accuracy", "robust_accuracy")} == {
        "loss": pytest.approx(1.5),
        "clean_accuracy": pytest.approx(0.5),
        "robust_accuracy": pytest.approx(0.5),
    }
    assert metrics["valid_examples"] == 2.0
    assert metrics["teacher_clean_forward_calls"] == 0.0
    assert metrics["cuda_peak_allocated_bytes"] == 0.0
    assert metrics["cuda_peak_reserved_bytes"] == 0.0
    assert metrics["seconds"] > 0.0
    assert metrics["images_per_second"] == pytest.approx(2.0 / metrics["seconds"])


def test_validation_attack_preserves_batchnorm_state_and_modes(tmp_path: Path) -> None:
    model = nn.Sequential(
        nn.BatchNorm2d(3),
        nn.Flatten(),
        nn.Linear(3 * 4 * 4, 3),
    )
    optimizer = SGD(model.parameters(), lr=0.01)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        attack=LinfPGD(
            AttackConfig(
                epsilon="1/255",
                step_size="1/255",
                steps=1,
                random_start=False,
                student_mode="train",
            )
        ),
        selection_attack=LinfPGD(
            AttackConfig(
                epsilon="1/255",
                step_size="1/255",
                steps=1,
                random_start=False,
                student_mode="eval",
                teacher_mode="eval",
            )
        ),
        objective=PGDATObjective(),
        device=torch.device("cpu"),
        output_dir=tmp_path,
        config_hash="b" * 64,
        seed=5,
    )
    trainer.model.train()
    before_state = {name: value.detach().clone() for name, value in trainer.model.state_dict().items()}
    before_modes = {name: module.training for name, module in trainer.model.named_modules()}
    batch = IndexedBatch(
        images=torch.rand(2, 3, 4, 4),
        labels=torch.tensor([0, 1]),
        sample_ids=torch.tensor([0, 1]),
        state_update_mask=torch.tensor([True, True]),
        multiplicity=torch.ones(2, dtype=torch.long),
    )

    trainer.validate_epoch([batch])

    assert before_modes == {name: module.training for name, module in trainer.model.named_modules()}
    for name, expected in before_state.items():
        assert torch.equal(expected, trainer.model.state_dict()[name]), name


def test_validation_random_stream_advances_repeats_and_separates_ranks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingAttack:
        def __init__(self) -> None:
            self.records: list[torch.Tensor] = []

        def generate(self, request):
            draw = torch.rand(request.inputs.shape, generator=request.generator, device=request.inputs.device)
            self.records.append(draw.detach().clone())
            return AttackResult(request.inputs, torch.zeros_like(request.inputs), (), 0.0)

    batch = IndexedBatch(
        images=torch.rand(2, 3, 4, 4),
        labels=torch.tensor([0, 1]),
        sample_ids=torch.tensor([0, 1]),
        state_update_mask=torch.tensor([True, True]),
        multiplicity=torch.ones(2, dtype=torch.long),
    )

    def sequence(rank: int) -> list[torch.Tensor]:
        monkeypatch.setattr("ard.engine.trainer.get_rank", lambda: rank)
        trainer = make_trainer(tmp_path / f"rank-{rank}")
        recorder = RecordingAttack()
        trainer.selection_attack = recorder
        trainer.global_step = 7
        trainer.validate_epoch([batch, batch])
        return recorder.records

    rank_zero = sequence(0)
    repeated_rank_zero = sequence(0)
    rank_one = sequence(1)
    assert not torch.equal(rank_zero[0], rank_zero[1])
    assert all(torch.equal(left, right) for left, right in zip(rank_zero, repeated_rank_zero, strict=True))
    assert not torch.equal(rank_zero[0], rank_one[0])
