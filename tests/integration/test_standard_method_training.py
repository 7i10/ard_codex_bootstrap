"""Plan 0103 option A: method ``standard`` trains on clean batches, no attack.

* The Trainer without a training attack never generates one (no attack call,
  no training-attack generator) and feeds the clean batch to the objective --
  it is exactly the PGD-AT path with a training attack returning the clean batch.
* Per-step metrics that assume an adversarial batch are absent or honestly
  renamed; validation still reports clean and PGD accuracy.
* A small end-to-end ``ard.cli.train`` + ``ard.cli.evaluate`` run.
* Existing methods are bit-identical to the pre-change Trainer (commit
  bf37707, loaded from git into this process), using the same-process
  ``_fingerprint`` differential of ``test_step_sync_free_parity``.
"""

from __future__ import annotations

import hashlib
import importlib.util
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

import ard.engine.trainer as trainer_module
import tests.integration.test_step_sync_free_parity as parity
from ard.attacks import AttackGenerator, AttackRequest, AttackResult, LinfPGD
from ard.config import load_config
from ard.engine.trainer import Trainer
from ard.models import build_student
from ard.objectives import PGDATObjective
from ard.state import SampleStateStore

pytestmark = pytest.mark.t3

ROOT = Path(__file__).resolve().parents[2]
PRE_CHANGE_COMMIT = "bf37707"


def _digest(value: object) -> str:
    digest = hashlib.sha256()
    parity._canonical(value, digest)
    return digest.hexdigest()


def _standard_trainer(output: Path, **extra: Any) -> Trainer:
    torch.manual_seed(123)
    student = build_student(parity.FIXTURE, tier="smoke")
    optimizer = torch.optim.SGD(student.parameters(), lr=0.05, momentum=0.9)
    return Trainer(
        model=student,
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8),
        scaler=None,
        attack=None,
        selection_attack=parity._selection_attack(),
        objective=PGDATObjective(),
        device=torch.device("cpu"),
        output_dir=output,
        config_hash="c" * 64,
        seed=11,
        tracker_run_id="standard",
        **extra,
    )


class _IdentityAttack(AttackGenerator):
    """A training "attack" that returns the clean batch and touches no RNG."""

    def generate(self, request: AttackRequest) -> AttackResult:
        zeros = torch.zeros_like(request.inputs)
        return AttackResult(adversarial=request.inputs, initial_delta=zeros, step_losses=(), max_abs_delta=0.0)


def _identity_attack_pgd_at_trainer(output: Path) -> Trainer:
    trainer = _standard_trainer(output)
    # Through pgd_at's own attack branch, with a perturbation of exactly zero.
    trainer.attack = _IdentityAttack()
    return trainer


def test_standard_trainer_makes_no_training_attack_call_and_trains_on_clean_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = _standard_trainer(tmp_path / "run")
    generate_calls: list[LinfPGD] = []
    original_generate = LinfPGD.generate

    def counting_generate(self: LinfPGD, request: Any) -> Any:
        generate_calls.append(self)
        return original_generate(self, request)

    monkeypatch.setattr(LinfPGD, "generate", counting_generate)

    def forbidden_generator(self: Trainer) -> torch.Generator:
        raise AssertionError("a training-attack generator was constructed")

    monkeypatch.setattr(Trainer, "_attack_generator", forbidden_generator)
    seen: list[tuple[torch.Tensor, torch.Tensor]] = []
    original_call = PGDATObjective.__call__

    def recording_call(self: PGDATObjective, **inputs: Any) -> Any:
        seen.append((inputs["student_logits"].detach().clone(), inputs["labels"].clone()))
        return original_call(self, **inputs)

    monkeypatch.setattr(PGDATObjective, "__call__", recording_call)
    forwards: list[torch.Tensor] = []
    trainer.model.register_forward_pre_hook(
        lambda module, args: forwards.append(args[0].detach().clone()) if module.training else None
    )
    loader, validation_loader = parity._loaders()
    batches = [batch.images for batch in loader]
    metrics = trainer.train_epoch(loader)
    assert generate_calls == []
    assert len(seen) == len(batches) == trainer.global_step
    # The only train-mode forward per step is the objective's input: the
    # clean batch itself.
    assert len(forwards) == len(batches)
    for forward, images in zip(forwards, batches, strict=True):
        assert torch.equal(forward, images)
    # Honest per-step metrics: no adversarial batch, so no "robust" keys.
    assert "robust_accuracy" not in metrics and "robust_accuracy_eval_mode" not in metrics
    assert 0.0 <= metrics["clean_accuracy_train_mode"] <= 1.0
    assert metrics["attack_epsilon"] == metrics["attack_step_size"] == 0.0
    assert "clean_accuracy" in metrics  # post-step eval-mode clean accuracy (step_diagnostics)
    # Validation still runs the selection attack.
    validation = trainer.validate_epoch(validation_loader)
    assert generate_calls and all(attack is trainer.selection_attack for attack in generate_calls)
    assert set(validation) == {"clean_accuracy", "pgd_accuracy"}


def test_standard_equals_pgd_at_whose_training_attack_returns_the_clean_batch(tmp_path: Path) -> None:
    """Same objective, same data order, same RNG streams: standard is the
    PGD-AT code path with the training input replaced by the clean batch.
    Model, optimizer, RNG and selection state must be bit-identical; only the
    metric names differ."""
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        loaders = parity._loaders()
        standard = _standard_trainer(tmp_path / "standard")
        standard_history = standard.fit(loaders[0], validation_loader=loaders[1], epochs=2)
        loaders = parity._loaders()
        zero = _identity_attack_pgd_at_trainer(tmp_path / "zero")
        zero_history = zero.fit(loaders[0], validation_loader=loaders[1], epochs=2)
    finally:
        torch.use_deterministic_algorithms(deterministic)
    for name in ("last.pt", "best.pt"):
        left = torch.load(tmp_path / "standard" / name, map_location="cpu", weights_only=False)
        right = torch.load(tmp_path / "zero" / name, map_location="cpu", weights_only=False)
        for key in ("model", "optimizer", "rng", "global_step", "best_metric", "selection_metadata"):
            assert _digest(left[key]) == _digest(right[key]), (name, key)
    for standard_row, zero_row in zip(standard_history, zero_history, strict=True):
        assert "train_robust_accuracy" not in standard_row
        assert "train_robust_accuracy_eval_mode" not in standard_row
        assert "train_robust_overtakes_clean" not in standard_row
        assert standard_row["train_clean_accuracy_train_mode"] == zero_row["train_robust_accuracy"]
        for key in ("train_loss", "train_clean_accuracy", "val_clean_accuracy", "val_pgd_accuracy"):
            assert standard_row[key] == zero_row[key], key


def test_standard_step_diagnostics_off_drops_only_the_post_step_metrics(tmp_path: Path) -> None:
    trainer = _standard_trainer(tmp_path / "run", step_diagnostics=False)
    loader, validation_loader = parity._loaders()
    row = trainer.fit(loader, validation_loader=validation_loader, epochs=1)[0]
    assert "train_clean_accuracy" not in row and "train_robust_accuracy" not in row
    assert "train_clean_accuracy_train_mode" in row and "val_pgd_accuracy" in row


@pytest.mark.parametrize(
    "extra",
    [
        {"sample_store": SampleStateStore(ema_decay=0.9)},
        {"epsilon_warmup_epochs": 1},
        {"observation_profile": "student_history", "sample_store": SampleStateStore(ema_decay=0.9)},
    ],
)
def test_attack_free_trainer_refuses_attack_dependent_options(tmp_path: Path, extra: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="training without an attack"):
        _standard_trainer(tmp_path / "run", **extra)


def test_attack_free_trainer_refuses_a_teacher(tmp_path: Path) -> None:
    teacher = build_student(parity.FIXTURE, tier="smoke")
    with pytest.raises(ValueError, match="teacher"):
        _standard_trainer(tmp_path / "run", teacher=teacher)


def _run(module: str, *args: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", module, *args], cwd=ROOT, env=environment, text=True, capture_output=True
    )


def _cli_config(output: Path) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "protocol": {"id": "synthetic_smoke_v2"},
        "tier": "smoke",
        "seeds": dict.fromkeys(
            (
                "split",
                "model_init",
                "data_order",
                "augmentation",
                "train_attack",
                "evaluation_attack",
                "qualitative_panel",
            ),
            4,
        ),
        "dataset": {"name": "synthetic_cifar", "num_samples": 8, "num_classes": 2, "image_size": 4},
        "student": {"architecture": "fixture_cnn", "num_classes": 2},
        "method": {
            "id": "standard",
            "version": 1,
            "selection_attack": {
                "loss": "ce",
                "epsilon": "1/255",
                "step_size": "1/255",
                "steps": 2,
                "random_start": True,
                "student_mode": "eval",
                "teacher_mode": "eval",
            },
        },
        "optimizer": {"id": "sgd", "learning_rate": 0.01, "momentum": 0.9, "weight_decay": 0.0, "nesterov": False},
        "scheduler": {"id": "identity", "milestones": [], "gamma": 1.0, "step_at": "epoch_end"},
        "training": {
            "epochs": 2,
            "per_rank_batch_size": 2,
            "global_batch_size": 2,
            "device": "cpu",
            "train_probe_size": 2,
        },
        "tracking": {
            "mode": "offline",
            "project": "ard-test",
            "run_id": "offline-standard",
            "group": "fixture-standard",
            "panel_size": 2,
            "panel_interval_epochs": 1,
        },
        "evaluation": {
            "dataset": {
                "name": "synthetic_cifar",
                "split": "test",
                "num_samples": 8,
                "num_classes": 2,
                "image_size": 4,
                "seed": 4,
            },
            "checkpoints": "both",
            "panel_size": 2,
        },
        "output_dir": str(output),
    }


def test_standard_cli_trains_and_evaluates_end_to_end(tmp_path: Path) -> None:
    output = tmp_path / "train"
    raw = _cli_config(output)
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    trained = _run("ard.cli.train", "--config", str(config_path))
    assert trained.returncode == 0, trained.stderr
    assert (output / "best.pt").is_file() and (output / "last.pt").is_file()
    resolved = load_config(output / "resolved_config.yaml")
    assert resolved.method.id == "standard" and resolved.method.attack is None
    metrics_files = sorted(output.rglob("epoch-metrics.parquet"))
    assert metrics_files, sorted(str(path.relative_to(output)) for path in output.rglob("*"))
    rows = pq.read_table(metrics_files[0]).to_pylist()
    assert [row["epoch"] for row in rows] == [0, 1]
    for row in rows:
        assert row.get("train_robust_accuracy") is None
        assert row.get("train_robust_accuracy_eval_mode") is None
        assert row["train_clean_accuracy_train_mode"] is not None
        assert row["val_clean_accuracy"] is not None and row["val_pgd_accuracy"] is not None
        assert row["train_probe_clean_accuracy"] is not None and row["train_probe_pgd_accuracy"] is not None
        assert row["train_attack_epsilon"] == 0.0
    panels = sorted(output.rglob("panel-epoch-*.jsonl"))
    assert panels
    for line in panels[0].read_text(encoding="utf-8").splitlines():
        panel_row = json.loads(line)
        assert panel_row["adversarial_image"] is None and panel_row["perturbation_visualization"] is None
        assert panel_row["student_adv_prediction"] is None and panel_row["robust_correct"] is None
        assert panel_row["clean_image"] is not None

    evaluation_path = tmp_path / "evaluation.yaml"
    evaluation_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    evaluated = _run("ard.cli.evaluate", "--config", str(evaluation_path), "--checkpoint-dir", str(output))
    assert evaluated.returncode == 0, evaluated.stderr


def test_standard_cli_refuses_a_training_attack_override(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(_cli_config(tmp_path / "train")), encoding="utf-8")
    refused = _run("ard.cli.train", "--config", str(config_path), "--dry-run", "method.attack.epsilon=1/255")
    assert refused.returncode != 0
    assert "no training attack" in refused.stderr
    assert not (tmp_path / "train").exists()


def _pre_change_trainer_class() -> type:
    try:
        source = subprocess.run(
            ["git", "show", f"{PRE_CHANGE_COMMIT}:src/ard/engine/trainer.py"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        pytest.skip(f"pre-change trainer {PRE_CHANGE_COMMIT} unavailable from git: {error}")
    name = "ard.engine._trainer_pre_standard"
    spec = importlib.util.spec_from_loader(name, loader=None)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "ard.engine"
    sys.modules[name] = module
    try:
        exec(compile(source, f"{PRE_CHANGE_COMMIT}:src/ard/engine/trainer.py", "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module.Trainer


@pytest.mark.parametrize("method", parity.METHODS)
def test_existing_methods_are_bit_identical_to_the_pre_standard_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    old_trainer = _pre_change_trainer_class()
    assert old_trainer is not trainer_module.Trainer
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        new = parity._fingerprint(tmp_path / "new", method, device=torch.device("cpu"), loaders=parity._loaders())
        with monkeypatch.context() as patch:
            patch.setattr(parity, "Trainer", old_trainer)
            old = parity._fingerprint(tmp_path / "old", method, device=torch.device("cpu"), loaders=parity._loaders())
    finally:
        torch.use_deterministic_algorithms(deterministic)
    assert set(new) == set(old)
    assert "last.pt:rng" in new and "best.pt:model" in new
    assert new == old
