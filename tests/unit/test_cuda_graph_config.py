"""``training.cuda_graph`` (plan 0105): default off, byte-identical configs, fail-closed scope,
the stale-graph fingerprint, and an unchanged ``LinfPGD.generate`` around the new sync-free core.

The GPU parity and guard differentials live in
``tests/integration/test_cuda_graph_training_step.py``.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from torch import nn
from torch.optim import SGD

from ard.attacks import AttackRequest, LinfPGD
from ard.cli.evaluate import _throughput_protocol_identity
from ard.config import load_config
from ard.config.loader import _expand_environment, resolved_config_dict
from ard.config.schema import AttackConfig, ExperimentConfig, TrainingConfig, reject_throughput_options
from ard.engine.checkpoint import config_digest
from ard.engine.cuda_graph import momentum_buffers_ready, optimizer_fingerprint
from ard.engine.trainer import Trainer
from ard.objectives import PGDATObjective

pytestmark = pytest.mark.t1

ROOT = Path(__file__).resolve().parents[2]
# The commit this change is based on: the pre-change schema and attack.
PRE_CHANGE_COMMIT = "4fd5ceb"
STAGE1 = ROOT / "configs" / "scientific" / "imagenet_mobilenetv4_twostage_stage1_112_pgd1_resized_s256.yaml"


def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Every environment variable a repository config interpolates (as in test_imagenet_loader_speedups)."""
    teacher_checkpoint = tmp_path / "teacher.pt"
    teacher_checkpoint.write_bytes(b"fixture checkpoint")
    for key, value in {
        "ARD_TEACHER_CHEN2021_LTD_WRN34_10_CHECKPOINT": str(teacher_checkpoint),
        "ARD_TEACHER_CHEN2021_LTD_WRN34_10_CHECKPOINT_SHA256": "a" * 64,
        "ARD_TEACHER_BARTOLDSON2024_ADVERSARIAL_WRN94_16_CHECKPOINT": str(teacher_checkpoint),
        "ARD_TEACHER_BARTOLDSON2024_ADVERSARIAL_WRN94_16_CHECKPOINT_SHA256": "b" * 64,
        "ARD_PER_RANK_BATCH_SIZE": "128",
        "ARD_DEVICE": "cpu",
        "ARD_OUTPUT_ROOT": str(tmp_path / "outputs"),
        "ARD_STAGEWISE_SWITCH_EPOCH": "100",
        "ARD_STAGEWISE_LATE_POLICY": "crop_re",
        "WANDB_GROUP_STAGEWISE": "stagewise-group",
        "WANDB_GROUP_CHEN": "chen-group",
        "WANDB_GROUP_BARTOLDSON": "bartoldson-group",
        "WANDB_GROUP_BARTOLDSON_ORACLE": "bartoldson-oracle-group",
        "ARD_FROZEN_ORACLE_MANIFEST": str(tmp_path / "frozen-oracle.json"),
        "ARD_FROZEN_ORACLE_MANIFEST_SHA256": "c" * 64,
        "ARD_SEED": "7",
        "ARD_IMAGENET_ROOT": str(tmp_path / "imagenet"),
        "ARD_IMAGENET_TRAIN_ROOT": str(tmp_path / "imagenet_train_derived"),
        "ARD_IMAGENET100_ROOT": str(tmp_path / "imagenet100"),
        "ARD_CIFAR10_ROOT": str(tmp_path / "cifar10"),
        "ARD_NUM_WORKERS": "0",
        "ARD_JOB_OUTPUT_DIR": str(tmp_path / "job-output"),
        "ARD_RUN_ID": "config-test-run",
        "WANDB_ENTITY": "entity",
        "WANDB_PROJECT": "project",
        "ARD_STAGE1_CHECKPOINT": str(tmp_path / "stage1" / "last.pt"),
        "ARD_STAGE1_CHECKPOINT_SHA256": "d" * 64,
    }.items():
        monkeypatch.setenv(key, value)


def _pre_change_module(relative: str, name: str, package: str) -> types.ModuleType:
    try:
        source = subprocess.run(
            ["git", "show", f"{PRE_CHANGE_COMMIT}:{relative}"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:  # pragma: no cover - shallow clones
        pytest.skip(f"pre-change {relative} at {PRE_CHANGE_COMMIT} unavailable from git: {error}")
    module = types.ModuleType(name)
    module.__package__ = package
    module.__file__ = f"{PRE_CHANGE_COMMIT}:{relative}"
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


# =========================================================================== config


def test_default_off_and_unserialized() -> None:
    training = TrainingConfig(per_rank_batch_size=4, global_batch_size=4)
    assert training.cuda_graph is False
    assert "cuda_graph" not in json.loads(training.model_dump_json())


def test_existing_configs_serialize_exactly_as_under_the_pre_change_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _env(monkeypatch, tmp_path)
    old_schema = _pre_change_module("src/ard/config/schema.py", "_ard_schema_pre_cuda_graph", "ard.config")
    paths = [
        path
        for directory in ("experiments", "pilot", "production", "scientific")
        for path in sorted((ROOT / "configs" / directory).glob("*.yaml"))
    ]
    assert len(paths) > 20
    for path in paths:
        expanded = _expand_environment(yaml.safe_load(path.read_text(encoding="utf-8")))
        new = ExperimentConfig.model_validate(expanded)
        old = old_schema.ExperimentConfig.model_validate(expanded)
        assert resolved_config_dict(new) == json.loads(old.model_dump_json()), path.name
        assert config_digest(resolved_config_dict(new)) == config_digest(json.loads(old.model_dump_json()))


def _stage1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    _env(monkeypatch, tmp_path)
    return resolved_config_dict(load_config(STAGE1))


def _with(raw: dict[str, Any], **sections: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(raw)
    for section, values in sections.items():
        updated[section] = {**updated[section], **values}
    return updated


def test_stage1_config_accepts_cuda_graph_and_records_it_in_the_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _stage1(monkeypatch, tmp_path)
    enabled = ExperimentConfig.model_validate(_with(raw, training={"cuda_graph": True}))
    assert enabled.training.cuda_graph is True
    assert resolved_config_dict(enabled)["training"]["cuda_graph"] is True
    assert config_digest(resolved_config_dict(enabled)) != config_digest(raw)
    # A saved resolved config reloads to the same setting.
    assert ExperimentConfig.model_validate(resolved_config_dict(enabled)).training.cuda_graph is True


@pytest.mark.parametrize(
    ("training", "reason"),
    [
        ({"device": "auto"}, "training.device=cuda"),
        ({"deterministic": False}, "training.deterministic=true"),
        ({"amp": True}, "training.amp=false"),
        ({"deterministic": False, "compile": True}, "training.compile=false"),
        ({"deterministic": False, "cudnn_benchmark": True}, "training.cudnn_benchmark=false"),
        ({"step_diagnostics": True}, "training.step_diagnostics=false"),
        ({"global_batch_size": 256}, "world size 1"),
        ({"epsilon_warmup_epochs": 5}, "training.epsilon_warmup_epochs unset"),
        ({"weight_ema_decay": 0.999}, "training.weight_ema_decay unset"),
    ],
)
def test_training_scope_fails_closed(training: dict[str, Any], reason: str) -> None:
    base = {
        "epochs": 10,
        "per_rank_batch_size": 128,
        "global_batch_size": 128,
        "device": "cuda",
        "step_diagnostics": False,
    }
    TrainingConfig(**base, cuda_graph=True)
    with pytest.raises(ValueError, match="training.cuda_graph=true requires") as refused:
        TrainingConfig(**{**base, **training}, cuda_graph=True)
    assert reason in str(refused.value)


@pytest.mark.parametrize(
    ("sections", "reason"),
    [
        ({"optimizer": {"id": "adamw", "momentum": None, "nesterov": None, "beta1": 0.9, "beta2": 0.999}}, "sgd"),
        ({"attack": {"random_start_keying": "sample_keyed_v1"}}, "random_start_keying=batch"),
        ({"attack": {"trace_step_losses": True}}, "trace_step_losses=false"),
        ({"attack": {"epsilon": "0", "step_size": "1/255"}}, "0 < epsilon"),
        ({"attack": {"epsilon": "2/255", "step_size": "4/255"}}, "step_size <= epsilon"),
    ],
)
def test_experiment_scope_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sections: dict[str, Any], reason: str
) -> None:
    raw = _with(_stage1(monkeypatch, tmp_path), training={"cuda_graph": True})
    attack = sections.pop("attack", None)
    updated = _with(raw, **sections)
    if attack is not None:
        updated["method"]["attack"] = {**updated["method"]["attack"], **attack}
        for key in ("epsilon_value", "step_size_value"):
            if key.split("_value")[0] in attack:
                updated["method"]["attack"].pop(key, None)
        if "epsilon" in attack:
            # Selection and evaluation must keep the training threat model.
            updated["method"]["selection_step_size_independent"] = False
            for container, key in ((updated["method"], "selection_attack"), (updated["evaluation"], "attack")):
                container[key] = {**container[key], "epsilon": attack["epsilon"], "step_size": attack["step_size"]}
                container[key].pop("epsilon_value", None)
                container[key].pop("step_size_value", None)
    ExperimentConfig.model_validate(raw)
    try:
        without_graph = copy.deepcopy(updated)
        without_graph["training"].pop("cuda_graph")
        ExperimentConfig.model_validate(without_graph)
    except ValueError as error:  # pragma: no cover - the fixture edit itself must be valid
        pytest.fail(f"fixture edit is invalid even without cuda_graph: {error}")
    with pytest.raises(ValueError, match="training.cuda_graph=true requires") as refused:
        ExperimentConfig.model_validate(updated)
    assert reason in str(refused.value)


def test_runtimes_other_than_the_trainer_cli_refuse_cuda_graph() -> None:
    training = TrainingConfig(
        per_rank_batch_size=4, global_batch_size=4, device="cuda", step_diagnostics=False, cuda_graph=True
    )
    with pytest.raises(ValueError) as refused:
        reject_throughput_options(training, runtime="fixture")
    assert str(refused.value) == (
        "fixture does not implement training.step_diagnostics=false, training.cuda_graph=true; run it with "
        "training.step_diagnostics=true, training.cuda_graph=false (only ard.cli.train applies these options)"
    )


def test_cuda_graph_is_not_part_of_the_pooling_identity() -> None:
    """Bitwise identical to eager by the tested contract (plan 0105), like step_diagnostics."""
    base = {"per_rank_batch_size": 4, "global_batch_size": 4, "device": "cuda", "step_diagnostics": False}
    assert _throughput_protocol_identity(TrainingConfig(**base, cuda_graph=True)) == {}
    assert _throughput_protocol_identity(TrainingConfig(**base)) == {}


# =========================================================================== trainer scope


def _trainer_kwargs(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    model = nn.Sequential(nn.Flatten(), nn.Linear(12, 3))
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
    attack = LinfPGD(AttackConfig(epsilon="4/255", step_size="4/255", steps=1))
    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": None,
        "scaler": None,
        "attack": attack,
        "selection_attack": attack,
        "objective": PGDATObjective(),
        "device": torch.device("cpu"),
        "output_dir": tmp_path,
        "config_hash": "c" * 64,
        "seed": 0,
        "step_diagnostics": False,
        "cuda_graph": True,
        **overrides,
    }


def test_trainer_refuses_cuda_graph_outside_its_scope(tmp_path: Path) -> None:
    Trainer(**{**_trainer_kwargs(tmp_path), "cuda_graph": False})
    with pytest.raises(ValueError, match="cuda_graph requires") as refused:
        Trainer(**_trainer_kwargs(tmp_path, step_diagnostics=True, weight_ema_decay=0.99))
    message = str(refused.value)
    for reason in ("a CUDA device", "step_diagnostics=False", "no EMA model"):
        assert reason in message
    with pytest.raises(ValueError, match="a CE, batch-keyed"):
        Trainer(
            **_trainer_kwargs(
                tmp_path,
                attack=LinfPGD(AttackConfig(epsilon="4/255", step_size="4/255", random_start_keying="sample_keyed_v1")),
            )
        )


# =========================================================================== fingerprint


def _stepped_sgd(**group: Any) -> tuple[nn.Module, SGD]:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Linear(4, 2))
    optimizer = SGD(model.parameters(), **{"lr": 0.1, "momentum": 0.9, "weight_decay": 1e-4, **group})
    model(torch.randn(5, 3)).sum().backward()
    optimizer.step()
    return model, optimizer


@pytest.mark.parametrize(
    "mutate",
    [
        lambda model, opt: opt.param_groups[0].__setitem__("lr", 0.05),
        lambda model, opt: opt.param_groups[0].__setitem__("momentum", 0.8),
        lambda model, opt: opt.param_groups[0].__setitem__("weight_decay", 0.0),
        lambda model, opt: opt.param_groups[0].__setitem__("nesterov", True),
        lambda model, opt: opt.param_groups[0].__setitem__("dampening", 0.1),
        lambda model, opt: opt.load_state_dict(copy.deepcopy(opt.state_dict())),
        lambda model, opt: model.eval(),
        lambda model, opt: model[1].register_buffer("running_mean", torch.zeros(4)),
        lambda model, opt: opt.add_param_group({"params": [nn.Parameter(torch.zeros(1))]}),
    ],
)
def test_fingerprint_detects_every_change_a_replay_would_ignore(mutate: Any) -> None:
    model, optimizer = _stepped_sgd()
    before = optimizer_fingerprint(optimizer, model)
    assert optimizer_fingerprint(optimizer, model) == before
    mutate(model, optimizer)
    assert optimizer_fingerprint(optimizer, model) != before


def test_fingerprint_ignores_in_place_value_updates_and_refuses_tensor_hyperparameters() -> None:
    model, optimizer = _stepped_sgd()
    before = optimizer_fingerprint(optimizer, model)
    # What a replay itself does: in-place updates of the same tensors.
    model(torch.randn(5, 3)).sum().backward()
    optimizer.step()
    optimizer.load_state_dict(optimizer.state_dict())  # same tensors, re-attached
    assert optimizer_fingerprint(optimizer, model) == before
    optimizer.param_groups[0]["lr"] = torch.tensor(0.1)
    with pytest.raises(RuntimeError, match="Python-scalar optimizer 'lr'"):
        optimizer_fingerprint(optimizer, model)


def test_momentum_buffers_ready_only_after_the_first_update() -> None:
    model = nn.Linear(3, 2)
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
    assert not momentum_buffers_ready(optimizer)
    model(torch.randn(4, 3)).sum().backward()
    optimizer.step()
    assert momentum_buffers_ready(optimizer)
    assert momentum_buffers_ready(SGD(nn.Linear(3, 2).parameters(), lr=0.1, momentum=0.0))


# =========================================================================== attack core


def _attack_cases() -> list[tuple[AttackConfig, dict[str, Any]]]:
    return [
        (AttackConfig(epsilon="8/255", step_size="2/255", steps=3), {}),
        (AttackConfig(epsilon="8/255", step_size="2/255", steps=3, random_start=False), {}),
        (AttackConfig(epsilon="8/255", step_size="2/255", steps=3, trace_step_losses=True), {"capture_step": 2}),
        (
            AttackConfig(epsilon="8/255", step_size="2/255", steps=2, random_start_keying="sample_keyed_v1"),
            {"source_ids": torch.arange(6), "epoch": 3, "attack_seed": 7},
        ),
        (
            AttackConfig(epsilon="8/255", step_size="2/255", steps=2),
            {
                "epsilon_override": torch.tensor([0.0, 4 / 255, 8 / 255, 8 / 255, 2 / 255, 0.0]),
                "step_size_override": torch.tensor([0.0, 1 / 255, 2 / 255, 2 / 255, 1 / 255, 0.0]),
            },
        ),
        (
            AttackConfig(
                epsilon="8/255", step_size="2/255", steps=2, loss="kl", kl_target="student_clean", student_mode="train"
            ),
            {},
        ),
    ]


def _result_fields(result: Any) -> dict[str, Any]:
    return {
        "adversarial": result.adversarial,
        "initial_delta": result.initial_delta,
        "step_losses": result.step_losses,
        "max_abs_delta": float(result.max_abs_delta).hex(),
        "captured": result.captured_adversarial,
    }


def _same(first: Any, second: Any) -> bool:
    if isinstance(first, torch.Tensor):
        return isinstance(second, torch.Tensor) and torch.equal(first, second)
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(_same(first[key], second[key]) for key in first)
    if isinstance(first, tuple):
        return len(first) == len(second) and all(_same(a, b) for a, b in zip(first, second, strict=True))
    return first == second


def test_generate_is_bit_identical_to_the_pre_change_attack() -> None:
    old_pgd = _pre_change_module("src/ard/attacks/pgd.py", "ard.attacks._pgd_pre_cuda_graph", "ard.attacks")
    torch.manual_seed(0)
    student = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(), nn.Flatten(), nn.Linear(4 * 16, 3)
    )
    images = torch.rand(6, 3, 4, 4)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    for config, extra in _attack_cases():
        results = []
        for attack in (LinfPGD(config), old_pgd.LinfPGD(config)):
            student.train()
            request = AttackRequest(
                inputs=images, labels=labels, student=student, generator=torch.Generator().manual_seed(5), **extra
            )
            results.append(_result_fields(attack.generate(request)))
            assert student.training  # mode restored
        assert _same(results[0], results[1]), config


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_perturb_core_never_synchronizes_with_the_host() -> None:
    device = torch.device("cuda")
    torch.manual_seed(0)
    student = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(), nn.Flatten(), nn.Linear(64, 3))
    student = student.to(device)
    attack = LinfPGD(AttackConfig(epsilon="8/255", step_size="2/255", steps=3))
    request = AttackRequest(
        inputs=torch.rand(6, 3, 4, 4, device=device),
        labels=torch.zeros(6, dtype=torch.long, device=device),
        student=student,
    )
    noise = torch.empty_like(request.inputs).uniform_(-1.0, 1.0)
    epsilon, step_size = attack.budgets(request)
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        attack.perturb(request, epsilon=epsilon, step_size=step_size, unit_noise=noise)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
