"""training.relu_surrogate_gradient: ReLU forward, SAT smooth (softplus) backward (human-approved 2026-10-15).

Xie et al. 2020, "Smooth Adversarial Training" (arXiv:2006.14536), Sec. 3: parametric softplus
f(alpha, x) = (1/alpha) log(1 + exp(alpha x)), derivative sigmoid(alpha x), alpha = 10. Pins: forward bitwise
ReLU; inactive backward bitwise nn.ReLU's; active backward the closed form; the Trainer turns it on only
around the training attack (scope attack_and_optimizer) and the training loss's backward; selection attacks
see true gradients; config hash / identity unchanged when absent; refusals.
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
import torch.nn.functional as F
import yaml
from torch import nn
from torch.optim import SGD

from ard.attacks import AttackRequest, LinfPGD
from ard.cli.evaluate import _surrogate_gradient_protocol_identity
from ard.config.loader import _expand_environment, resolved_config_dict
from ard.config.schema import (
    AttackConfig,
    ExperimentConfig,
    ReluSurrogateGradientConfig,
    reject_throughput_options,
)
from ard.data import IndexedBatch
from ard.engine.checkpoint import config_digest
from ard.engine.surrogate_gradient import ReluSurrogateGradient, SurrogateGradientReLU, install_relu_surrogate
from ard.engine.trainer import Trainer
from ard.models import build_architecture
from ard.objectives import PGDATObjective

pytestmark = pytest.mark.t1

ROOT = Path(__file__).resolve().parents[2]
# The commit this change is based on: the pre-change schema and configs.
PRE_CHANGE_COMMIT = "9738f19"
SYNTHETIC = ROOT / "configs" / "experiments" / "synthetic_pgd_at.yaml"
BETA = 10.0


# =========================================================================== the autograd Function


def _inputs(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    values = torch.randn(4, 3, 5, 5, generator=generator, dtype=dtype) * 0.5
    values.view(-1)[:4] = torch.tensor([0.0, -0.0, 1e-3, -1e-3], dtype=dtype)  # the kink and its neighbours
    return values


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_forward_is_bitwise_relu(active: bool, requires_grad: bool) -> None:
    switch = ReluSurrogateGradient(BETA)
    module = SurrogateGradientReLU(switch)
    values = _inputs().requires_grad_(requires_grad)
    if active:
        with switch.active():
            output = module(values)
    else:
        output = module(values)
    reference = nn.ReLU()(values.detach())
    assert torch.equal(output.detach(), reference)
    assert torch.equal(output.detach().view(torch.int32), reference.view(torch.int32))  # bit patterns, -0.0 too


def test_inactive_backward_is_bitwise_nn_relu() -> None:
    switch = ReluSurrogateGradient(BETA)
    values = _inputs()
    upstream = torch.randn(values.shape, generator=torch.Generator().manual_seed(1))
    surrogate_input = values.clone().requires_grad_(True)
    SurrogateGradientReLU(switch)(surrogate_input).backward(upstream)
    relu_input = values.clone().requires_grad_(True)
    nn.ReLU()(relu_input).backward(upstream)
    assert surrogate_input.grad is not None and relu_input.grad is not None
    assert torch.equal(surrogate_input.grad.view(torch.int32), relu_input.grad.view(torch.int32))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_active_backward_is_the_parametric_softplus_derivative(dtype: torch.dtype) -> None:
    switch = ReluSurrogateGradient(BETA)
    values = _inputs(dtype)
    upstream = torch.randn(values.shape, generator=torch.Generator().manual_seed(1), dtype=dtype)
    leaf = values.clone().requires_grad_(True)
    output = SurrogateGradientReLU(switch)(leaf)
    with switch.active():
        output.backward(upstream)
    assert leaf.grad is not None
    # SAT Eqn. 3, written independently: d/dx (1/a) log(1 + exp(a x)) = 1 / (1 + exp(-a x)).
    torch.testing.assert_close(leaf.grad, upstream / (1 + torch.exp(-BETA * values)))
    # ... which is the derivative of torch's Softplus(beta=10) (threshold above any |beta x| here).
    softplus_leaf = values.clone().requires_grad_(True)
    nn.Softplus(beta=BETA, threshold=1e4)(softplus_leaf).backward(upstream)
    torch.testing.assert_close(leaf.grad, softplus_leaf.grad)
    assert not torch.equal(leaf.grad, torch.where(values > 0, upstream, torch.zeros_like(upstream)))


def test_switch_is_read_at_backward_time_and_restored() -> None:
    switch = ReluSurrogateGradient(BETA)
    assert not switch.is_active
    with pytest.raises(RuntimeError, match="boom"), switch.active():
        assert switch.is_active
        raise RuntimeError("boom")
    assert not switch.is_active
    # A graph recorded while active but differentiated after the block uses the true gradient.
    leaf = _inputs().requires_grad_(True)
    with switch.active():
        output = SurrogateGradientReLU(switch)(leaf)
    output.sum().backward()
    assert leaf.grad is not None
    assert torch.equal(leaf.grad, (leaf.detach() > 0).to(leaf.dtype))
    with pytest.raises(ValueError, match="finite and positive"):
        ReluSurrogateGradient(0.0)


def _plain_relu_call_count(model: nn.Module, images: torch.Tensor) -> int:
    from torch.overrides import TorchFunctionMode

    relu_functions = {F.relu, torch.relu, torch.relu_, torch.nn.functional.relu_}

    class Count(TorchFunctionMode):
        calls = 0

        def __torch_function__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
            if func in relu_functions:
                Count.calls += 1
            return func(*args, **(kwargs or {}))

    with torch.no_grad(), Count():
        model(images)
    return Count.calls


def test_install_replaces_every_relu_of_mobilenetv4_small_and_keeps_the_state_dict() -> None:
    torch.manual_seed(0)
    model = build_architecture("mobilenetv4_conv_small_imagenet", num_classes=10).eval()
    relu_count = sum(type(module) is nn.ReLU for module in model.modules())
    images = torch.rand(2, 3, 64, 64)
    # Every ReLU the forward applies is a module (no functional relu the replacement would miss).
    assert _plain_relu_call_count(model, images) == relu_count == 30
    plain_state = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.no_grad():
        plain_logits = model(images)
    switch = ReluSurrogateGradient(BETA)
    assert install_relu_surrogate(model, switch) == relu_count
    assert not any(type(module) is nn.ReLU for module in model.modules())
    assert sum(isinstance(module, SurrogateGradientReLU) for module in model.modules()) == relu_count
    state = model.state_dict()
    assert list(state) == list(plain_state) and all(torch.equal(state[key], plain_state[key]) for key in state)
    # A checkpoint of the installed student loads strictly into the plain architecture (what evaluation builds).
    plain = build_architecture("mobilenetv4_conv_small_imagenet", num_classes=10).eval()
    plain.load_state_dict(state, strict=True)
    with torch.no_grad():
        assert torch.equal(model(images), plain_logits)
        assert torch.equal(plain(images), plain_logits)
    # Inactive: input gradient bitwise the plain model's; active: different.
    leaf = images.clone().requires_grad_(True)
    model(leaf).sum().backward()
    plain_leaf = images.clone().requires_grad_(True)
    plain(plain_leaf).sum().backward()
    assert leaf.grad is not None and torch.equal(leaf.grad, plain_leaf.grad)
    active_leaf = images.clone().requires_grad_(True)
    with switch.active():
        model(active_leaf).sum().backward()
    assert not torch.equal(active_leaf.grad, plain_leaf.grad)


def test_install_refuses_a_student_without_relu() -> None:
    with pytest.raises(ValueError, match="requires a student with nn.ReLU"):
        install_relu_surrogate(
            build_architecture("mobilenetv4_conv_small_silu_imagenet", num_classes=10), ReluSurrogateGradient(BETA)
        )


# =========================================================================== Trainer scope


_ATTACK = AttackConfig(epsilon="8/255", step_size="2/255", steps=2, random_start=True)
_SEED = 5


def _relu_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.BatchNorm2d(4),
        nn.ReLU(inplace=True),
        nn.Conv2d(4, 4, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 3),
    )


def _batch(size: int = 6, seed: int = 1) -> IndexedBatch:
    generator = torch.Generator().manual_seed(seed)
    return IndexedBatch(
        images=torch.rand(size, 3, 4, 4, generator=generator),
        labels=torch.randint(0, 3, (size,), generator=generator),
        sample_ids=torch.arange(size),
    )


def _trainer(model: nn.Module, tmp_path: Path, **kwargs: Any) -> Trainer:
    return Trainer(
        model=model,
        optimizer=SGD(model.parameters(), lr=0.1),
        scheduler=None,
        scaler=None,
        attack=LinfPGD(_ATTACK),
        selection_attack=LinfPGD(_ATTACK),
        objective=PGDATObjective(),
        device=torch.device("cpu"),
        output_dir=tmp_path,
        config_hash="e" * 64,
        seed=_SEED,
        step_diagnostics=False,
        **kwargs,
    )


def _spied_step(trainer: Trainer, batch: IndexedBatch) -> dict[str, Any]:
    """One training step; record the training attack's input gradient (in whatever switch state the
    Trainer called it), its adversarial output, and the weight gradients the optimizer steps with."""
    record: dict[str, Any] = {}
    attack = trainer.attack
    assert attack is not None
    original_generate = attack.generate

    def generate(request: AttackRequest) -> Any:
        student = request.student
        mode = student.training
        student.eval()
        probe = request.inputs.detach().clone().requires_grad_(True)
        (gradient,) = torch.autograd.grad(F.cross_entropy(student(probe), request.labels), probe)
        student.train(mode)
        for parameter in student.parameters():
            parameter.grad = None
        record["attack_input_gradient"] = gradient.detach().clone()
        record["attack_switch_active"] = trainer._relu_surrogate is not None and trainer._relu_surrogate.is_active
        result = original_generate(request)
        record["adversarial"] = result.adversarial.detach().clone()
        return result

    attack.generate = generate  # type: ignore[method-assign]
    original_step = trainer.optimizer.step

    def step(*args: Any, **kwargs: Any) -> Any:
        record["weight_gradients"] = [parameter.grad.detach().clone() for parameter in trainer.model.parameters()]
        return original_step(*args, **kwargs)

    trainer.optimizer.step = step  # type: ignore[method-assign]
    trainer.train_epoch([batch])  # type: ignore[arg-type]
    assert trainer._relu_surrogate is None or not trainer._relu_surrogate.is_active
    return record


def _straight_through_softplus_model(model: nn.Module) -> nn.Module:
    """Independent transcription: forward relu(x) + (s(x) - s(x).detach()) with s = Softplus(beta=10)."""

    class StraightThrough(nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            smooth = F.softplus(inputs, beta=BETA, threshold=1e4)
            return F.relu(inputs).detach() + (smooth - smooth.detach())

    reference = copy.deepcopy(model)
    for parent in list(reference.modules()):
        for name, child in list(parent.named_children()):
            if type(child) is nn.ReLU:
                setattr(parent, name, StraightThrough())
    return reference


@pytest.mark.parametrize("scope", ["optimizer", "attack_and_optimizer"])
def test_trainer_applies_the_surrogate_only_in_its_scope(tmp_path: Path, scope: str) -> None:
    batch = _batch()
    baseline = _spied_step(_trainer(_relu_model(), tmp_path / "baseline"), batch)
    model = _relu_model()
    initial = copy.deepcopy(model)
    trainer = _trainer(model, tmp_path / scope, relu_surrogate_gradient=ReluSurrogateGradientConfig(scope=scope))
    assert trainer._relu_surrogate is not None and trainer._relu_surrogate.beta == BETA
    surrogate = _spied_step(trainer, batch)
    weight_differs = [
        not torch.equal(a, b) for a, b in zip(surrogate["weight_gradients"], baseline["weight_gradients"])
    ]
    assert any(weight_differs)  # the optimizer's gradient is the smooth one in both scopes
    if scope == "optimizer":
        assert surrogate["attack_switch_active"] is False
        assert torch.equal(surrogate["attack_input_gradient"], baseline["attack_input_gradient"])
        assert torch.equal(surrogate["adversarial"], baseline["adversarial"])  # the training attack is untouched
        # The weight gradient equals an independent straight-through transcription on the same x'.
        reference = _straight_through_softplus_model(initial).train()
        loss = F.cross_entropy(reference(surrogate["adversarial"]), batch.labels)
        loss.backward()
        for parameter, recorded in zip(reference.parameters(), surrogate["weight_gradients"], strict=True):
            torch.testing.assert_close(parameter.grad, recorded, rtol=1e-5, atol=1e-6)
    else:
        assert surrogate["attack_switch_active"] is True
        assert not torch.equal(surrogate["attack_input_gradient"], baseline["attack_input_gradient"])
        assert not torch.equal(surrogate["adversarial"], baseline["adversarial"])


@pytest.mark.parametrize("scope", ["optimizer", "attack_and_optimizer"])
def test_selection_attack_sees_true_relu_gradients(tmp_path: Path, scope: str) -> None:
    batch, validation = _batch(), [_batch(5, seed=9)]
    model = _relu_model()
    trainer = _trainer(model, tmp_path / scope, relu_surrogate_gradient=ReluSurrogateGradientConfig(scope=scope))
    trainer.train_epoch([batch])  # type: ignore[arg-type]
    plain = _relu_model()
    plain.load_state_dict(trainer.model.state_dict(), strict=True)
    plain_trainer = _trainer(plain, tmp_path / "plain")
    # The selection generator is keyed on global_step (seed + 1000003 * global_step): same step, same stream.
    plain_trainer.global_step, plain_trainer.current_epoch = trainer.global_step, trainer.current_epoch

    def spy(target: Trainer) -> list[torch.Tensor]:
        seen: list[torch.Tensor] = []
        original = target.selection_attack.generate

        def generate(request: AttackRequest) -> Any:
            assert target._relu_surrogate is None or not target._relu_surrogate.is_active
            result = original(request)
            seen.append(result.adversarial.detach().clone())
            return result

        target.selection_attack.generate = generate  # type: ignore[method-assign]
        return seen

    seen, plain_seen = spy(trainer), spy(plain_trainer)
    metrics = trainer.validate_epoch(validation)  # type: ignore[arg-type]
    plain_metrics = plain_trainer.validate_epoch(validation)  # type: ignore[arg-type]
    assert len(seen) == len(plain_seen) == 1
    assert torch.equal(seen[0], plain_seen[0])
    assert metrics == plain_metrics
    # Checkpoints carry the plain architecture's keys.
    assert list(trainer.model.state_dict()) == list(_relu_model().state_dict())


def test_ema_shadow_keeps_plain_relu(tmp_path: Path) -> None:
    trainer = _trainer(
        _relu_model(),
        tmp_path,
        weight_ema_decay=0.9,
        relu_surrogate_gradient=ReluSurrogateGradientConfig(scope="optimizer"),
    )
    assert trainer.ema_model is not None
    assert sum(type(module) is nn.ReLU for module in trainer.ema_model.modules()) == 2
    assert not any(isinstance(module, SurrogateGradientReLU) for module in trainer.ema_model.modules())


def test_trainer_refusals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = ReluSurrogateGradientConfig(scope="optimizer")
    no_relu = nn.Sequential(nn.Flatten(), nn.Linear(48, 3))
    with pytest.raises(ValueError, match="requires a student with nn.ReLU"):
        _trainer(no_relu, tmp_path, relu_surrogate_gradient=config)
    monkeypatch.setattr("ard.engine.trainer.get_world_size", lambda: 2)
    with pytest.raises(ValueError, match="world size 1"):
        _trainer(_relu_model(), tmp_path, relu_surrogate_gradient=config)
    monkeypatch.undo()
    with pytest.raises(ValueError, match="a training attack for scope=attack_and_optimizer"):
        Trainer(
            model=_relu_model(),
            optimizer=SGD(_relu_model().parameters(), lr=0.1),
            scheduler=None,
            scaler=None,
            attack=None,
            selection_attack=LinfPGD(_ATTACK),
            objective=PGDATObjective(),
            device=torch.device("cpu"),
            output_dir=tmp_path,
            config_hash="e" * 64,
            seed=_SEED,
            relu_surrogate_gradient=ReluSurrogateGradientConfig(scope="attack_and_optimizer"),
        )


# =========================================================================== config


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:  # pragma: no cover - shallow clones
        pytest.skip(f"git {' '.join(args)} unavailable: {error}")


def _pre_change_schema() -> types.ModuleType:
    relative = "src/ard/config/schema.py"
    name = "_ard_schema_pre_relu_surrogate_gradient"
    module = types.ModuleType(name)
    module.__package__ = "ard.config"
    module.__file__ = f"{PRE_CHANGE_COMMIT}:{relative}"
    sys.modules[name] = module
    exec(compile(_git("show", f"{PRE_CHANGE_COMMIT}:{relative}"), module.__file__, "exec"), module.__dict__)
    return module


def _synthetic(**sections: dict[str, Any]) -> dict[str, Any]:
    raw = yaml.safe_load(SYNTHETIC.read_text(encoding="utf-8"))
    for section, values in sections.items():
        raw[section] = {**raw[section], **values}
    return raw


def test_configs_that_existed_before_the_change_serialize_exactly_as_under_the_pre_change_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.unit.test_phase2_batch_a import _env_all

    _env_all(monkeypatch, tmp_path)
    old_schema = _pre_change_schema()
    listed = _git("ls-tree", "-r", "--name-only", PRE_CHANGE_COMMIT, "configs/").splitlines()
    paths = [
        path
        for path in listed
        if path.endswith(".yaml") and path.split("/")[1] in {"experiments", "pilot", "production", "scientific"}
    ]
    assert len(paths) > 20
    for path in paths:
        expanded = _expand_environment(yaml.safe_load(_git("show", f"{PRE_CHANGE_COMMIT}:{path}")))
        new = ExperimentConfig.model_validate(expanded)
        old = old_schema.ExperimentConfig.model_validate(expanded)
        assert "relu_surrogate_gradient" not in resolved_config_dict(new)["training"], path
        assert resolved_config_dict(new) == json.loads(old.model_dump_json()), path
        assert config_digest(resolved_config_dict(new)) == config_digest(json.loads(old.model_dump_json())), path


def test_absent_by_default_present_in_hash_and_identity_when_set() -> None:
    base = ExperimentConfig.model_validate(_synthetic())
    assert base.training.relu_surrogate_gradient is None
    assert "relu_surrogate_gradient" not in resolved_config_dict(base)["training"]
    assert _surrogate_gradient_protocol_identity(base.training) == {}
    digests = {config_digest(resolved_config_dict(base))}
    for scope in ("optimizer", "attack_and_optimizer"):
        enabled = ExperimentConfig.model_validate(_synthetic(training={"relu_surrogate_gradient": {"scope": scope}}))
        resolved = resolved_config_dict(enabled)
        assert resolved["training"]["relu_surrogate_gradient"] == {"beta": 10.0, "scope": scope}
        assert resolved_config_dict(ExperimentConfig.model_validate(resolved)) == resolved
        assert _surrogate_gradient_protocol_identity(enabled.training) == {
            "relu_surrogate_gradient": {"beta": 10.0, "scope": scope}
        }
        digests.add(config_digest(resolved))
    assert len(digests) == 3


@pytest.mark.parametrize(
    ("training", "method", "message"),
    [
        (
            {"device": "cuda", "cuda_graph": True, "step_diagnostics": False},
            {},
            "relu_surrogate_gradient requires training.cuda_graph=false",
        ),
        ({"deterministic": False, "compile": True}, {}, "relu_surrogate_gradient requires training.compile=false"),
        ({"per_rank_batch_size": 2, "global_batch_size": 4}, {}, "world size 1"),
        ({}, None, "method standard with scope=optimizer"),
    ],
)
def test_schema_refusals(training: dict[str, Any], method: dict[str, Any] | None, message: str) -> None:
    surrogate = {"relu_surrogate_gradient": {"scope": "attack_and_optimizer"}}
    raw = _synthetic(training={**training, **surrogate})
    if method is None:
        raw["method"] = {
            "id": "standard",
            "version": 1,
            "selection_attack": {"epsilon": "4/255", "step_size": "1/255", "steps": 2},
        }
    with pytest.raises(ValueError, match=message):
        ExperimentConfig.model_validate(raw)
    for bad in ({"scope": "optimizer", "beta": 0}, {"scope": "attack"}, {"beta": 10.0}):
        with pytest.raises(ValueError):
            ReluSurrogateGradientConfig.model_validate(bad)


def test_other_trainer_builders_refuse_it() -> None:
    enabled = ExperimentConfig.model_validate(_synthetic(training={"relu_surrogate_gradient": {"scope": "optimizer"}}))
    with pytest.raises(ValueError, match=r"fixture does not implement training.relu_surrogate_gradient=<set>"):
        reject_throughput_options(enabled.training, runtime="fixture")
    reject_throughput_options(ExperimentConfig.model_validate(_synthetic()).training, runtime="fixture")


def _experiment(name: str, **sections: dict[str, Any]) -> dict[str, Any]:
    raw = yaml.safe_load((ROOT / "configs" / "experiments" / name).read_text(encoding="utf-8"))
    for section, values in sections.items():
        raw[section] = {**raw.get(section, {}), **values}
    return raw


_SURROGATE = {"relu_surrogate_gradient": {"scope": "optimizer"}}


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (_experiment("synthetic_trades.yaml", training=_SURROGATE), "method pgd_at"),
        (_experiment("synthetic_rslad.yaml", training=_SURROGATE), "method pgd_at"),
        (_experiment("synthetic_rslad.yaml", training=_SURROGATE), "no teacher or distillation"),
        (_experiment("synthetic_rslad_observed.yaml", training=_SURROGATE), "observation.profile=off"),
        (_experiment("synthetic_rslad_student.yaml", training=_SURROGATE), "method pgd_at"),
        (
            _experiment(
                "synthetic_pgd_at.yaml",
                training=_SURROGATE,
                method={"mixed_batch": {"adversarial_fraction": 0.5, "adversarial_weight": 0.3}},
            ),
            "no method.mixed_batch",
        ),
        (_experiment("synthetic_pgd_at.yaml", training=_SURROGATE, method={"awp": {}}), "no method.awp"),
        (_experiment("synthetic_pgd_at.yaml", training={**_SURROGATE, "amp": True}), "training.amp=false"),
        (
            _experiment("synthetic_pgd_at.yaml", training=_SURROGATE, observation={"profile": "student_history"}),
            "observation.profile=off",
        ),
    ],
)
def test_schema_admits_only_the_tested_plain_step(raw: dict[str, Any], message: str) -> None:
    """Allowed = tested (review P2-4): pgd_at / standard, no mixed batch, AWP, ADR, policy, teacher,
    distillation, intervention, observation profile or AMP. (Clean-wrong attack skipping is a Trainer-only
    option that ard.cli.train never sets; the Trainer refuses it.)"""
    with pytest.raises(ValueError, match=message):
        ExperimentConfig.model_validate(raw)


def test_schema_refuses_adr() -> None:
    raw = _synthetic(training=_SURROGATE)
    adr_config = yaml.safe_load((ROOT / "configs" / "scientific" / "cifar10_r18_adr.yaml").read_text(encoding="utf-8"))
    raw["method"] = adr_config["method"]
    with pytest.raises(ValueError, match="relu_surrogate_gradient requires"):
        ExperimentConfig.model_validate(raw)


def test_schema_admits_standard_with_optimizer_scope_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.unit.test_phase2_batch_a import _env_all

    _env_all(monkeypatch, tmp_path)
    for key, value in {"ARD_JOB_OUTPUT_DIR": str(tmp_path / "out"), "ARD_RUN_ID": "run", "WANDB_ENTITY": "e"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("WANDB_PROJECT", "p")
    path = ROOT / "configs" / "scientific" / "imagenet_mobilenetv4_standard_50ep.yaml"
    raw = _expand_environment(yaml.safe_load(path.read_text(encoding="utf-8")))
    raw["training"] = {**raw["training"], "relu_surrogate_gradient": {"scope": "optimizer"}}
    assert ExperimentConfig.model_validate(raw).training.relu_surrogate_gradient is not None
    raw["training"]["relu_surrogate_gradient"] = {"scope": "attack_and_optimizer"}
    with pytest.raises(ValueError, match="method standard with scope=optimizer"):
        ExperimentConfig.model_validate(raw)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"awp": "awp"}, "no mixed_batch or awp"),
        ({"observation_profile": "student_history"}, "observation_profile=off"),
    ],
)
def test_trainer_admits_only_the_tested_plain_step(tmp_path: Path, kwargs: dict[str, Any], message: str) -> None:
    from ard.config.schema import AwpConfig
    from ard.state import SampleStateStore

    if kwargs.get("awp") == "awp":
        kwargs = {"awp": AwpConfig()}
    if "observation_profile" in kwargs:
        kwargs = {**kwargs, "sample_store": SampleStateStore()}
    with pytest.raises(ValueError, match=message):
        _trainer(
            _relu_model(), tmp_path, relu_surrogate_gradient=ReluSurrogateGradientConfig(scope="optimizer"), **kwargs
        )
