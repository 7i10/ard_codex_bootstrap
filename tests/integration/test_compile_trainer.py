"""``training.compile``: the compiled wrapper never reaches saved state.

Fixture-scale, CPU-only. ``torch.compile`` runs with ``backend="aot_eager"``
so the tests exercise dynamo + AOTAutograd (the wrapper, its ``_orig_mod.``
key prefix, the traced backward) without paying for inductor's C++ codegen.
The property under test -- which module checkpoints, EMA copies and resume
see -- does not depend on the backend.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from ard.analysis import summarize_checkpoint_groups
from ard.attacks import LinfPGD
from ard.cli import evaluate as evaluate_cli
from ard.cli import train as train_cli
from ard.config.schema import AttackConfig, ModelConfig
from ard.data import (
    EpochShuffleSampler,
    IndexedDataset,
    SyntheticCIFAR,
    collate_indexed,
    stratified_train_validation_split,
)
from ard.engine.distributed import unwrap_model
from ard.engine.trainer import Trainer
from ard.evaluation.saved_checkpoint import load_saved_student_checkpoint
from ard.models import build_student
from ard.objectives import PGDATObjective

pytestmark = pytest.mark.t3

FIXTURE = ModelConfig(architecture="fixture_cnn", num_classes=3)


def _loaders(seed: int = 5) -> tuple[DataLoader, DataLoader, EpochShuffleSampler]:
    dataset = IndexedDataset(SyntheticCIFAR(size=8, num_classes=3, image_size=4, seed=seed))
    train_dataset, validation_dataset = stratified_train_validation_split(dataset, validation_fraction=0.25, seed=seed)
    sampler = EpochShuffleSampler(len(train_dataset), seed=seed)
    validation_sampler = EpochShuffleSampler(len(validation_dataset), seed=seed, shuffle=False)
    loader = DataLoader(train_dataset, batch_size=4, sampler=sampler, collate_fn=collate_indexed)
    validation_loader = DataLoader(
        validation_dataset, batch_size=4, sampler=validation_sampler, collate_fn=collate_indexed
    )
    return loader, validation_loader, sampler


def _trainer(output: Path, *, compiled: bool) -> Trainer:
    torch.manual_seed(123)
    model: nn.Module = build_student(FIXTURE, tier="smoke")
    optimizer = SGD(model.parameters(), lr=0.03, momentum=0.9)
    if compiled:
        # Same order as ard.cli.train: optimizer parameters are the original
        # module's parameters either way.
        model = torch.compile(model, backend="aot_eager")
    return Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=StepLR(optimizer, step_size=1, gamma=0.8),
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
        config_hash="c" * 64,
        seed=5,
        tracker_run_id="offline-compile-fixture",
        # An EMA copy proves deepcopy(unwrap_model(...)) sees the original.
        weight_ema_decay=0.5,
    )


def _payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert isinstance(payload, dict)
    return payload


def test_unwrap_model_strips_the_compile_wrapper() -> None:
    model = build_student(FIXTURE, tier="smoke")
    compiled = torch.compile(model, backend="aot_eager")
    assert any(key.startswith("_orig_mod.") for key in compiled.state_dict())
    assert unwrap_model(compiled) is model
    assert unwrap_model(model) is model


def test_compiled_checkpoints_match_eager_keys_load_strictly_and_resume(tmp_path: Path) -> None:
    torch._dynamo.reset()
    eager = _trainer(tmp_path / "eager", compiled=False)
    eager_loader, eager_validation, _ = _loaders()
    eager.fit(eager_loader, validation_loader=eager_validation, epochs=2)

    compiled = _trainer(tmp_path / "compiled", compiled=True)
    assert type(compiled.model).__name__ == "OptimizedModule"
    compiled_loader, compiled_validation, _ = _loaders()
    compiled_history = compiled.fit(compiled_loader, validation_loader=compiled_validation, epochs=2)
    assert all(torch.isfinite(torch.tensor(row["train_loss"])) for row in compiled_history)

    for name in ("last.pt", "best.pt", "best-ema.pt"):
        eager_payload = _payload(tmp_path / "eager" / name)
        compiled_payload = _payload(tmp_path / "compiled" / name)
        assert list(compiled_payload["model"]) == list(eager_payload["model"]), name
        assert list(compiled_payload["ema"]) == list(eager_payload["ema"]), name
        assert not any("_orig_mod" in key for key in compiled_payload["model"]), name
        # Exactly the loader ard.cli.evaluate uses, into an uncompiled model.
        fresh = build_student(FIXTURE, tier="smoke")
        load_saved_student_checkpoint(tmp_path / "compiled" / name, fresh)
        for key, value in compiled_payload["model"].items():
            assert torch.equal(fresh.state_dict()[key], value), (name, key)
        ema_fresh = build_student(FIXTURE, tier="smoke")
        ema_fresh.load_state_dict(compiled_payload["ema"], strict=True)
    # The EMA shadow is a plain module, never a second compiled wrapper.
    assert compiled.ema_model is not None
    assert type(compiled.ema_model).__name__ != "OptimizedModule"

    # Resume: a compiled trainer restores from a compiled run's epoch-0
    # boundary and reproduces the uninterrupted compiled run exactly.
    torch._dynamo.reset()
    first_leg = _trainer(tmp_path / "resumed", compiled=True)
    first_loader, first_validation, _ = _loaders()
    first_leg.fit(first_loader, validation_loader=first_validation, epochs=1)
    resumed = _trainer(tmp_path / "resumed", compiled=True)
    resumed_loader, resumed_validation, resumed_sampler = _loaders()
    state = resumed.resume(tmp_path / "resumed" / "last.pt", sampler=resumed_sampler)
    assert state.next_epoch == 1
    resumed.fit(resumed_loader, validation_loader=resumed_validation, epochs=2, start_epoch=state.next_epoch)
    expected = unwrap_model(compiled.model).state_dict()
    observed = unwrap_model(resumed.model).state_dict()
    assert list(observed) == list(expected)
    for key, value in expected.items():
        assert torch.equal(value, observed[key]), key
    assert resumed.global_step == compiled.global_step

    # An eager trainer resumes from a compiled run's checkpoint as well.
    eager_resume = _trainer(tmp_path / "resumed", compiled=False)
    _, _, eager_resume_sampler = _loaders()
    assert eager_resume.resume(tmp_path / "resumed" / "last.pt", sampler=eager_resume_sampler).next_epoch == 2


def _cli_config(output: Path, *, compile_student: bool = False, cudnn_benchmark: bool = False) -> dict[str, Any]:
    seeds = (
        "split",
        "model_init",
        "data_order",
        "augmentation",
        "train_attack",
        "evaluation_attack",
        "qualitative_panel",
    )
    return {
        "schema_version": 2,
        "protocol": {"id": "synthetic_smoke_v2"},
        "tier": "smoke",
        "seeds": dict.fromkeys(seeds, 8),
        "dataset": {"name": "synthetic_cifar", "num_samples": 8, "num_classes": 3, "image_size": 4, "seed": 8},
        "student": {"architecture": "fixture_cnn", "num_classes": 3},
        "method": {
            "id": "pgd_at",
            "version": 1,
            "attack": {"epsilon": "1/255", "step_size": "1/255", "steps": 1, "random_start": False},
        },
        "optimizer": {"id": "sgd", "learning_rate": 0.02, "momentum": 0.9, "weight_decay": 0.0, "nesterov": False},
        "scheduler": {"id": "identity", "milestones": [], "gamma": 1.0, "step_at": "epoch_end"},
        "training": {
            "epochs": 1,
            "per_rank_batch_size": 4,
            "global_batch_size": 4,
            "device": "cpu",
            # Both throughput options require deterministic=false.
            "deterministic": not (compile_student or cudnn_benchmark),
            "compile": compile_student,
            "cudnn_benchmark": cudnn_benchmark,
        },
        "evaluation": {
            "dataset": {
                "name": "synthetic_cifar",
                "split": "test",
                "num_samples": 8,
                "num_classes": 3,
                "image_size": 4,
                "seed": 8,
            },
            "checkpoints": "both",
        },
        "output_dir": str(output),
        "tracker_run_id": "smoke-local",
    }


@pytest.fixture
def restore_global_torch_flags() -> Iterator[None]:
    """ard.cli.train mutates process-global switches; keep other tests unaffected."""
    import torch._dynamo.config as dynamo_config

    deterministic = torch.are_deterministic_algorithms_enabled()
    benchmark = torch.backends.cudnn.benchmark
    fail_on_recompile = dynamo_config.fail_on_recompile_limit_hit
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(deterministic)
        torch.backends.cudnn.benchmark = benchmark
        dynamo_config.fail_on_recompile_limit_hit = fail_on_recompile
        torch._dynamo.reset()


def _write_config(tmp_path: Path, name: str, data: dict[str, Any]) -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _aot_eager_compile(monkeypatch: pytest.MonkeyPatch, calls: list[nn.Module]) -> None:
    original_compile = torch.compile

    def recording_compile(model: nn.Module, **kwargs: Any) -> Callable[..., Any]:
        assert kwargs == {}, "ard.cli.train must use torch.compile's default mode"
        calls.append(model)
        return original_compile(model, backend="aot_eager")

    monkeypatch.setattr(torch, "compile", recording_compile)


def test_train_cli_compile_flag_wraps_student_and_saves_plain_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_global_torch_flags: None
) -> None:
    import torch._dynamo.config as dynamo_config

    calls: list[nn.Module] = []
    _aot_eager_compile(monkeypatch, calls)
    monkeypatch.chdir(tmp_path)
    dynamo_config.fail_on_recompile_limit_hit = False
    keys: dict[bool, list[str]] = {}
    for compile_student in (False, True):
        torch._dynamo.reset()
        output = tmp_path / f"compile-{compile_student}"
        config_path = _write_config(
            tmp_path, f"compile-{compile_student}", _cli_config(output, compile_student=compile_student)
        )
        assert train_cli.main(["--config", str(config_path)]) == 0
        resolved = yaml.safe_load((output / "resolved_config.yaml").read_text(encoding="utf-8"))
        assert resolved["training"]["compile"] is compile_student
        assert resolved["training"]["cudnn_benchmark"] is False
        keys[compile_student] = list(_payload(output / "last.pt")["model"])
        load_saved_student_checkpoint(output / "last.pt", build_student(FIXTURE, tier="smoke"))
        assert torch.backends.cudnn.benchmark is False
        # A compiled run can never silently fall back to eager at the
        # recompile limit; an eager run leaves dynamo's setting alone.
        assert dynamo_config.fail_on_recompile_limit_hit is compile_student
    assert len(calls) == 1 and type(calls[0]).__name__ != "OptimizedModule"
    assert keys[True] == keys[False]
    assert not any("_orig_mod" in key for key in keys[True])


def test_recompile_limit_is_a_hard_error_once_fail_on_recompile_is_set(restore_global_torch_flags: None) -> None:
    import torch._dynamo.config as dynamo_config

    torch._dynamo.reset()
    dynamo_config.fail_on_recompile_limit_hit = True
    with dynamo_config.patch(recompile_limit=1):
        compiled = torch.compile(lambda x, scale: x * scale, backend="eager")
        compiled(torch.ones(2), 2)
        with pytest.raises(torch._dynamo.exc.FailOnRecompileLimitHit):
            compiled(torch.ones(2), 3)


def test_train_cli_cudnn_benchmark_flag_enables_autotuning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_global_torch_flags: None
) -> None:
    monkeypatch.chdir(tmp_path)
    torch.backends.cudnn.benchmark = False
    output = tmp_path / "benchmark"
    config_path = _write_config(tmp_path, "benchmark", _cli_config(output, cudnn_benchmark=True))
    assert train_cli.main(["--config", str(config_path)]) == 0
    assert torch.backends.cudnn.benchmark is True


def test_compiled_run_evaluation_identity_is_never_pooled_with_eager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_global_torch_flags: None
) -> None:
    calls: list[nn.Module] = []
    _aot_eager_compile(monkeypatch, calls)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "compiled-run"
    config_path = _write_config(tmp_path, "compiled-run", _cli_config(output, compile_student=True))
    assert train_cli.main(["--config", str(config_path)]) == 0
    evaluation_output = tmp_path / "compiled-evaluation"
    arguments = ["--config", str(config_path), "--checkpoint-dir", str(output), "--output", str(evaluation_output)]
    assert evaluate_cli.main(arguments) == 0
    rows = json.loads((evaluation_output / "evaluation-results.json").read_text(encoding="utf-8"))
    assert {row["checkpoint_alias"] for row in rows} == {"best", "last"}
    assert all(row["training_protocol_identity"]["compile"] is True for row in rows)
    assert all("cudnn_benchmark" not in row["training_protocol_identity"] for row in rows)

    def twin(row: dict[str, Any], *, keep_compile: bool) -> dict[str, Any]:
        copy = json.loads(json.dumps(row))
        copy["train_run_id"] = f"{row['train_run_id']}-twin"
        copy["training_seeds"]["model_init"] += 1
        if not keep_compile:
            del copy["training_protocol_identity"]["compile"]
        return copy

    # Control: a second compiled run with another seed pools normally ...
    pooled = summarize_checkpoint_groups(rows + [twin(row, keep_compile=True) for row in rows], metric="pgd_accuracy")
    assert {alias: summary["count"] for alias, summary in pooled.items()} == {"best": 2, "last": 2}
    # ... but an otherwise identical eager run is refused.
    with pytest.raises(ValueError, match="cannot aggregate mixed experiment identities"):
        summarize_checkpoint_groups(rows + [twin(row, keep_compile=False) for row in rows], metric="pgd_accuracy")


def test_train_cli_rejects_compile_under_ddp_before_any_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train_cli, "initialize_from_env", lambda device: (torch.device("cpu"), True))
    monkeypatch.setattr(train_cli, "teardown", lambda: None)
    output = tmp_path / "ddp"
    config_path = _write_config(tmp_path, "ddp", _cli_config(output, compile_student=True))
    with pytest.raises(ValueError, match="training.compile is only validated for single-process"):
        train_cli.main(["--config", str(config_path)])
    assert not output.exists()


def test_train_cli_rejects_compile_when_inductor_would_drop_device_asserts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch._inductor.config as inductor_config

    monkeypatch.setattr(inductor_config, "assert_indirect_indexing", False)
    output = tmp_path / "no-asserts"
    config_path = _write_config(tmp_path, "no-asserts", _cli_config(output, compile_student=True))
    with pytest.raises(ValueError, match="assert_indirect_indexing=True"):
        train_cli.main(["--config", str(config_path)])
    assert not output.exists()
