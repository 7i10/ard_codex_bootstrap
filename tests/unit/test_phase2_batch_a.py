"""Plan 0103 Phase 2 batch A (human-approved 2026-10-08): mixed clean+adversarial batches with
optional split BatchNorm, AWP, and SGD's norm/bias weight-decay exclusion.

CPU only. Covers: default-off and byte-identical configs (every config of the pre-change commit,
read from git, resolves and hashes exactly as under the pre-change schema), fail-closed scope,
identity, the pure helpers, and one-step Trainer equivalence against independent transcriptions
(Kurakin et al.'s loss; the official AWP code; split BN through a reference model copy).
"""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
import types
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from ard.attacks import AttackRequest, LinfPGD
from ard.cli.train import _adamw_parameter_groups, _weight_decay_parameter_groups
from ard.config.loader import _expand_environment, resolved_config_dict
from ard.config.schema import (
    AttackConfig,
    AwpConfig,
    ExperimentConfig,
    MethodConfig,
    MixedBatchConfig,
    OptimizerConfig,
    reject_phase2_batch_a_options,
)
from ard.data import EpochShuffleSampler, IndexedBatch, IndexedDataset, SyntheticCIFAR, collate_indexed
from ard.engine.awp import AWP_EPS, AdversarialWeightPerturbation
from ard.engine.checkpoint import config_digest
from ard.engine.mixed_batch import AuxiliaryBatchNorm, adversarial_count, example_weights
from ard.engine.trainer import Trainer
from ard.objectives import PGDATObjective
from tests.unit.test_cuda_graph_config import _env

pytestmark = pytest.mark.t1

ROOT = Path(__file__).resolve().parents[2]
# The commit this change is based on: the pre-change schema and configs.
PRE_CHANGE_COMMIT = "922222a"
SYNTHETIC = ROOT / "configs" / "experiments" / "synthetic_pgd_at.yaml"


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:  # pragma: no cover - shallow clones
        pytest.skip(f"git {' '.join(args)} unavailable: {error}")


def _pre_change_schema() -> types.ModuleType:
    relative = "src/ard/config/schema.py"
    name = "_ard_schema_pre_phase2_batch_a"
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


# =========================================================================== config identity


def test_configs_that_existed_before_the_change_serialize_exactly_as_under_the_pre_change_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _env(monkeypatch, tmp_path)
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
        assert resolved_config_dict(new) == json.loads(old.model_dump_json()), path
        assert config_digest(resolved_config_dict(new)) == config_digest(json.loads(old.model_dump_json())), path


def test_defaults_are_off_and_unserialized() -> None:
    config = ExperimentConfig.model_validate(_synthetic())
    assert config.method.mixed_batch is None and config.method.awp is None
    assert config.optimizer.exclude_norm_bias_from_weight_decay is False
    resolved = resolved_config_dict(config)
    assert "mixed_batch" not in resolved["method"] and "awp" not in resolved["method"]
    assert "exclude_norm_bias_from_weight_decay" not in resolved["optimizer"]


@pytest.mark.parametrize(
    "sections",
    [
        {"method": {"mixed_batch": {"adversarial_fraction": 0.5, "adversarial_weight": 0.3}}},
        {"method": {"mixed_batch": {"adversarial_fraction": 0.5, "adversarial_weight": 0.3, "split_batchnorm": True}}},
        {"method": {"awp": {}}},
        {"method": {"awp": {"gamma": 0.005, "warmup_epochs": 0}}},
        {"optimizer": {"exclude_norm_bias_from_weight_decay": True}},
    ],
)
def test_each_option_is_in_the_hash_and_the_recorded_identity(sections: dict[str, Any]) -> None:
    base = ExperimentConfig.model_validate(_synthetic())
    enabled = ExperimentConfig.model_validate(_synthetic(**sections))
    resolved = resolved_config_dict(enabled)
    assert config_digest(resolved) != config_digest(resolved_config_dict(base))
    # ard.cli.evaluate records method.model_dump() as method_identity and optimizer.model_dump()
    # inside training_protocol_identity: the option is in exactly the section it belongs to.
    section, values = next(iter(sections.items()))
    key = next(iter(values))
    dumped = getattr(enabled, section).model_dump(mode="json")
    assert key in dumped and key not in getattr(base, section).model_dump(mode="json")
    # A saved resolved config reloads to the same identity.
    assert resolved_config_dict(ExperimentConfig.model_validate(resolved)) == resolved
    if section == "method" and key == "awp":
        assert dumped["awp"]["gamma"] == values["awp"].get("gamma", 0.01)


def test_awp_defaults_are_the_official_at_awp_code_defaults() -> None:
    # csdongxian/AWP AT_AWP/train_cifar10.py: --awp-gamma 0.01, --awp-warmup 0.
    assert AwpConfig().gamma == 0.01 and AwpConfig().warmup_epochs == 0


@pytest.mark.parametrize(
    ("sections", "message"),
    [
        ({"method": {"id": "trades", "attack": {"loss": "kl", "kl_target": "student_clean"}, "awp": {}}}, "pgd_at"),
        (
            {
                "method": {
                    "mixed_batch": {"adversarial_fraction": 0.5, "adversarial_weight": 0.3},
                    "awp": {},
                }
            },
            "not specified together",
        ),
        ({"method": {"mixed_batch": {"adversarial_fraction": 1.0, "adversarial_weight": 0.3}}}, "less than 1"),
        ({"method": {"mixed_batch": {"adversarial_fraction": 0.0, "adversarial_weight": 0.3}}}, "greater than 0"),
        ({"method": {"mixed_batch": {"adversarial_fraction": 0.5, "adversarial_weight": 0.0}}}, "greater than 0"),
        ({"method": {"mixed_batch": {"adversarial_fraction": 0.5, "adversarial_weight": math.inf}}}, "finite"),
        ({"method": {"awp": {"gamma": 0.0}}}, "greater than 0"),
        ({"method": {"awp": {"warmup_epochs": 1}}}, "smaller than training.epochs"),
        ({"method": {"awp": {}}, "training": {"global_batch_size": 8}}, "world size 1"),
        ({"method": {"awp": {}}, "training": {"amp": True}}, "training.amp"),
        (
            {
                "method": {
                    "mixed_batch": {"adversarial_fraction": 0.5, "adversarial_weight": 0.3, "split_batchnorm": True}
                },
                "training": {"global_batch_size": 8},
            },
            "world size 1",
        ),
        (
            {
                "method": {
                    "mixed_batch": {"adversarial_fraction": 0.5, "adversarial_weight": 0.3, "split_batchnorm": True}
                },
                "training": {"compile": True, "deterministic": False},
            },
            "training.compile",
        ),
        (
            {
                "optimizer": {
                    "id": "adamw",
                    "momentum": None,
                    "nesterov": None,
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "exclude_norm_bias_from_weight_decay": True,
                }
            },
            "only for sgd",
        ),
    ],
)
def test_out_of_scope_combinations_fail_closed(sections: dict[str, Any], message: str) -> None:
    raw = _synthetic()
    for section, values in sections.items():
        if section == "method" and "attack" in values:
            values = {**values, "attack": {**raw["method"]["attack"], **values["attack"]}}
        raw[section] = {**raw[section], **values}
    with pytest.raises(ValueError, match=message):
        ExperimentConfig.model_validate(raw)


def test_mixed_batch_without_split_bn_works_under_ddp_shapes() -> None:
    raw = _synthetic(
        method={"mixed_batch": {"adversarial_fraction": 0.5, "adversarial_weight": 0.3}},
        training={"global_batch_size": 8},
    )
    assert ExperimentConfig.model_validate(raw).method.mixed_batch is not None


@pytest.mark.parametrize(
    "option", [{"mixed_batch": {"adversarial_fraction": 0.5, "adversarial_weight": 0.3}}, {"awp": {}}]
)
def test_standard_method_refuses_both_options(option: dict[str, Any]) -> None:
    selection = {"epsilon": "4/255", "step_size": "1/255", "steps": 2}
    MethodConfig.model_validate({"id": "standard", "version": 1, "selection_attack": selection})
    with pytest.raises(ValueError, match="defined only for method pgd_at"):
        MethodConfig.model_validate({"id": "standard", "version": 1, "selection_attack": selection, **option})


def test_other_trainer_builders_refuse_the_options() -> None:
    reject_phase2_batch_a_options(ExperimentConfig.model_validate(_synthetic()), runtime="fixture")
    enabled = ExperimentConfig.model_validate(
        _synthetic(method={"awp": {}}, optimizer={"exclude_norm_bias_from_weight_decay": True})
    )
    with pytest.raises(ValueError, match="fixture does not implement method.awp, optimizer.exclude_norm_bias"):
        reject_phase2_batch_a_options(enabled, runtime="fixture")


# =========================================================================== optimizer groups


def test_sgd_groups_reuse_the_adamw_split() -> None:
    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.Flatten(), nn.Linear(4, 2))
    frozen = nn.Parameter(torch.zeros(3, 3), requires_grad=False)
    groups = _weight_decay_parameter_groups([*model.parameters(), frozen], weight_decay=5e-4)
    assert [group["weight_decay"] for group in groups] == [5e-4, 0.0]
    decay, no_decay = (group["params"] for group in groups)
    assert [tuple(p.shape) for p in decay] == [(4, 3, 3, 3), (2, 4)]
    assert [tuple(p.shape) for p in no_decay] == [(4,), (4,), (4,), (2,)]  # conv bias, BN weight/bias, linear bias
    assert all(p is not frozen for group in groups for p in group["params"])
    legacy = _adamw_parameter_groups(model, weight_decay=5e-4)
    assert [[id(p) for p in g["params"]] for g in legacy] == [[id(p) for p in g["params"]] for g in groups]
    optimizer = SGD(groups, lr=0.1, momentum=0.9, weight_decay=5e-4)
    assert [group["weight_decay"] for group in optimizer.param_groups] == [5e-4, 0.0]


def test_optimizer_flag_validates_only_for_sgd() -> None:
    OptimizerConfig(
        id="sgd",
        learning_rate=0.1,
        weight_decay=5e-4,
        momentum=0.9,
        nesterov=True,
        exclude_norm_bias_from_weight_decay=True,
    )


# =========================================================================== mixed-batch helpers


@pytest.mark.parametrize(
    ("batch", "fraction", "expected"), [(32, 0.5, 16), (128, 0.5, 64), (7, 0.5, 3), (1, 0.5, 0), (128, 0.3, 38)]
)
def test_adversarial_count_is_the_floor(batch: int, fraction: float, expected: int) -> None:
    assert adversarial_count(batch, fraction) == expected


def test_example_weights_put_lambda_on_the_first_k_positions() -> None:
    weights = example_weights(5, 2, 0.3, device=torch.device("cpu"))
    assert weights.tolist() == pytest.approx([0.3, 0.3, 1.0, 1.0, 1.0])


def test_paper_normalizer_on_the_paper_shape() -> None:
    # m=32, k=16, lambda=0.3: (sum_clean + 0.3 sum_adv) / (16 + 0.3*16).
    torch.manual_seed(0)
    losses = torch.rand(32, dtype=torch.float64)
    weights = example_weights(32, adversarial_count(32, 0.5), 0.3, device=torch.device("cpu"), dtype=torch.float64)
    ours = (losses * weights).sum() / weights.sum()
    paper = (losses[16:].sum() + 0.3 * losses[:16].sum()) / ((32 - 16) + 0.3 * 16)
    assert torch.equal(weights.sum(), torch.tensor(16 + 0.3 * 16, dtype=torch.float64)) or math.isclose(
        float(weights.sum()), 20.8, rel_tol=1e-15
    )
    assert math.isclose(float(ours), float(paper), rel_tol=1e-14)


# =========================================================================== trainer, one step


def _bn_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.BatchNorm2d(4),
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


_ATTACK = AttackConfig(epsilon="8/255", step_size="2/255", steps=2, random_start=True)
_SEED = 5


def _trainer(model: nn.Module, optimizer: torch.optim.Optimizer, tmp_path: Path, **kwargs: Any) -> Trainer:
    return Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        attack=LinfPGD(_ATTACK),
        selection_attack=LinfPGD(_ATTACK),
        objective=PGDATObjective(label_smoothing=kwargs.pop("label_smoothing", 0.0)),
        device=torch.device("cpu"),
        output_dir=tmp_path,
        config_hash="e" * 64,
        seed=_SEED,
        step_diagnostics=kwargs.pop("step_diagnostics", False),
        **kwargs,
    )


def _attack_subset(model: nn.Module, batch: IndexedBatch, count: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(_SEED)  # seed + 1000003 * global_step(0)
    return (
        LinfPGD(_ATTACK)
        .generate(
            AttackRequest(inputs=batch.images[:count], labels=batch.labels[:count], student=model, generator=generator)
        )
        .adversarial
    )


def _state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def test_mixed_batch_step_is_kurakin_loss_on_the_first_k_examples(tmp_path: Path) -> None:
    batch = _batch(6)
    mixed = MixedBatchConfig(adversarial_fraction=0.5, adversarial_weight=0.3)
    model = _bn_model()
    reference = copy.deepcopy(model)
    trainer = _trainer(model, SGD(model.parameters(), lr=0.1), tmp_path, mixed_batch=mixed, step_diagnostics=True)
    seen: list[int] = []
    original = trainer.attack.generate  # type: ignore[union-attr]

    def recording(request: AttackRequest) -> Any:
        seen.append(request.inputs.shape[0])
        assert torch.equal(request.inputs, batch.images[:3])
        return original(request)

    trainer.attack.generate = recording  # type: ignore[union-attr,method-assign]
    metrics = trainer.train_epoch([batch])  # type: ignore[arg-type]
    assert seen == [3]
    # Independent transcription: attack the first k=3, one forward of the mixed batch, paper loss.
    reference.train()
    adversarial = _attack_subset(reference, batch, 3)
    reference.train()
    logits = reference(torch.cat([adversarial, batch.images[3:]]))
    ce = F.cross_entropy(logits, batch.labels, reduction="none")
    loss = (ce[3:].sum() + 0.3 * ce[:3].sum()) / ((6 - 3) + 0.3 * 3)
    optimizer = SGD(reference.parameters(), lr=0.1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    for key, value in _state(reference).items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=1e-6, atol=1e-7, msg=key)
    assert metrics["mixed_batch_adversarial_examples"] == 3.0 and metrics["mixed_batch_clean_examples"] == 3.0
    correct = logits.detach().argmax(1) == batch.labels
    assert metrics["robust_accuracy"] == pytest.approx(float(correct[:3].float().mean()))
    assert metrics["mixed_batch_clean_accuracy_train_mode"] == pytest.approx(float(correct[3:].float().mean()))
    assert metrics["mixed_batch_adversarial_loss"] == pytest.approx(float(ce[:3].mean().detach()), rel=1e-6)
    assert metrics["mixed_batch_clean_loss"] == pytest.approx(float(ce[3:].mean().detach()), rel=1e-6)
    assert 0.0 <= metrics["robust_accuracy_eval_mode"] <= 1.0


def test_a_batch_too_small_for_one_adversarial_example_trains_clean(tmp_path: Path) -> None:
    batch = _batch(1)
    model = _bn_model()
    trainer = _trainer(
        model,
        SGD(model.parameters(), lr=0.1),
        tmp_path,
        mixed_batch=MixedBatchConfig(adversarial_fraction=0.5, adversarial_weight=0.3),
    )
    trainer.attack.generate = lambda request: pytest.fail("no example to attack")  # type: ignore[union-attr,method-assign]
    metrics = trainer.train_epoch([batch])  # type: ignore[arg-type]
    assert metrics["mixed_batch_adversarial_examples"] == 0.0


def test_mixed_batch_with_weight_one_and_all_but_one_attacked_matches_its_definition(tmp_path: Path) -> None:
    # lambda=1 reduces the loss to the plain mean over the mixed batch.
    batch = _batch(4)
    model = _bn_model()
    reference = copy.deepcopy(model)
    trainer = _trainer(
        model,
        SGD(model.parameters(), lr=0.1),
        tmp_path,
        mixed_batch=MixedBatchConfig(adversarial_fraction=0.75, adversarial_weight=1.0),
    )
    trainer.train_epoch([batch])  # type: ignore[arg-type]
    reference.train()
    adversarial = _attack_subset(reference, batch, 3)
    reference.train()
    loss = F.cross_entropy(reference(torch.cat([adversarial, batch.images[3:]])), batch.labels)
    optimizer = SGD(reference.parameters(), lr=0.1)
    loss.backward()
    optimizer.step()
    for key, value in _state(reference).items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=1e-6, atol=1e-7, msg=key)


# --------------------------------------------------------------------------- split BN


def test_auxiliary_bn_refuses_a_model_without_batchnorm() -> None:
    with pytest.raises(ValueError, match="requires a student with BatchNorm layers"):
        AuxiliaryBatchNorm(nn.Sequential(nn.Conv2d(3, 4, 3), nn.LayerNorm([4, 2, 2]), nn.Flatten(), nn.Linear(16, 2)))


def test_auxiliary_bn_forward_is_the_model_with_the_auxiliary_state_substituted() -> None:
    model = _bn_model()
    auxiliary = AuxiliaryBatchNorm(model)
    with torch.no_grad():
        for parameter in auxiliary.parameters():
            parameter.add_(0.25)
    main_before = _state(model)
    reference = copy.deepcopy(model)
    reference[1].load_state_dict(
        {key.split("__", 1)[1]: value for key, value in auxiliary.state_dict().items() if key.startswith("1__")}
        | {name: auxiliary.affine_parameters[f"1__{name}"].detach() for name in ("weight", "bias")}
    )
    inputs = torch.rand(5, 3, 4, 4)
    model.train()
    reference.train()
    output = auxiliary.forward_clean(model, inputs)
    expected = reference(inputs)
    assert torch.equal(output, expected)
    # Running statistics: the auxiliary copy moved, the main BN did not.
    assert all(torch.equal(model.state_dict()[key], value) for key, value in main_before.items())
    assert torch.equal(auxiliary.get_buffer("1__running_mean"), reference[1].running_mean)
    assert int(auxiliary.get_buffer("1__num_batches_tracked")) == 1
    # Gradients reach the auxiliary affine parameters, never the main BN's.
    output.sum().backward()
    assert auxiliary.affine_parameters["1__weight"].grad is not None
    assert model[1].weight.grad is None and model[0].weight.grad is not None


def test_split_bn_step_routes_each_sub_batch_through_its_own_bn(tmp_path: Path) -> None:
    batch = _batch(6)
    model = _bn_model()
    auxiliary = AuxiliaryBatchNorm(model)
    reference = copy.deepcopy(model)
    reference_auxiliary = copy.deepcopy(auxiliary)
    optimizer = SGD([*model.parameters(), *auxiliary.parameters()], lr=0.1)
    trainer = _trainer(
        model,
        optimizer,
        tmp_path,
        mixed_batch=MixedBatchConfig(adversarial_fraction=0.5, adversarial_weight=0.3, split_batchnorm=True),
        auxiliary_batchnorm=auxiliary,
    )
    trainer.train_epoch([batch])  # type: ignore[arg-type]
    reference.train()
    adversarial = _attack_subset(reference, batch, 3)
    reference.train()
    adversarial_logits = reference(adversarial)
    clean_logits = reference_auxiliary.forward_clean(reference, batch.images[3:])
    ce = F.cross_entropy(torch.cat([adversarial_logits, clean_logits]), batch.labels, reduction="none")
    loss = (ce[3:].sum() + 0.3 * ce[:3].sum()) / (3 + 0.3 * 3)
    reference_optimizer = SGD([*reference.parameters(), *reference_auxiliary.parameters()], lr=0.1)
    loss.backward()
    reference_optimizer.step()
    for key, value in _state(reference).items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=1e-6, atol=1e-7, msg=key)
    for key, value in reference_auxiliary.state_dict().items():
        torch.testing.assert_close(auxiliary.state_dict()[key], value, rtol=1e-6, atol=1e-7, msg=key)
    # The main BN saw only the adversarial half: one batch tracked each.
    assert int(model[1].num_batches_tracked) == 1 and int(auxiliary.get_buffer("1__num_batches_tracked")) == 1


def test_split_bn_checkpoint_keeps_model_keys_and_restores_the_auxiliary_state(tmp_path: Path) -> None:
    mixed = MixedBatchConfig(adversarial_fraction=0.5, adversarial_weight=0.3, split_batchnorm=True)

    def build(output: Path) -> tuple[Trainer, AuxiliaryBatchNorm]:
        model = _bn_model()
        auxiliary = AuxiliaryBatchNorm(model)
        optimizer = SGD([*model.parameters(), *auxiliary.parameters()], lr=0.1, momentum=0.9)
        return _trainer(model, optimizer, output, mixed_batch=mixed, auxiliary_batchnorm=auxiliary), auxiliary

    def loader() -> DataLoader:
        dataset = IndexedDataset(SyntheticCIFAR(size=6, num_classes=3, image_size=4, seed=2))
        return DataLoader(dataset, batch_size=6, sampler=EpochShuffleSampler(6, seed=1), collate_fn=collate_indexed)

    trainer, auxiliary = build(tmp_path / "a")
    trainer.fit(loader(), validation_loader=loader(), epochs=1)
    payload = torch.load(tmp_path / "a" / "last.pt", map_location="cpu", weights_only=False)
    # "model" is the main (adversarial-BN) student only: what evaluation loads.
    assert set(payload["model"]) == set(_bn_model().state_dict())
    assert set(payload["auxiliary_batchnorm"]) == set(auxiliary.state_dict())
    resumed, resumed_auxiliary = build(tmp_path / "a")
    resumed.resume(tmp_path / "a" / "last.pt", sampler=None)
    for key, value in auxiliary.state_dict().items():
        assert torch.equal(resumed_auxiliary.state_dict()[key], value), key
    # A checkpoint without the auxiliary state cannot resume a split-BN run.
    del payload["auxiliary_batchnorm"]
    torch.save(payload, tmp_path / "a" / "last.pt")
    fresh, _ = build(tmp_path / "a")
    with pytest.raises(ValueError, match="carries no 'auxiliary_batchnorm' state"):
        fresh.resume(tmp_path / "a" / "last.pt", sampler=None)


def test_trainer_split_bn_scope(tmp_path: Path) -> None:
    model = _bn_model()
    split = MixedBatchConfig(adversarial_fraction=0.5, adversarial_weight=0.3, split_batchnorm=True)
    with pytest.raises(ValueError, match="must be supplied together"):
        _trainer(model, SGD(model.parameters(), lr=0.1), tmp_path, mixed_batch=split)
    with pytest.raises(ValueError, match="an auxiliary BN built from this student"):
        _trainer(
            model,
            SGD(model.parameters(), lr=0.1),
            tmp_path,
            mixed_batch=split,
            auxiliary_batchnorm=AuxiliaryBatchNorm(nn.Sequential(nn.BatchNorm2d(3))),
        )
    with pytest.raises(ValueError, match="auxiliary_batchnorm requires"):
        _trainer(model, SGD(model.parameters(), lr=0.1), tmp_path, auxiliary_batchnorm=AuxiliaryBatchNorm(model))


def test_trainer_refuses_mixed_batch_and_awp_outside_plain_pgd_at(tmp_path: Path) -> None:
    model = _bn_model()
    mixed = MixedBatchConfig(adversarial_fraction=0.5, adversarial_weight=0.3)
    with pytest.raises(ValueError, match="mixed_batch requires a training attack"):
        Trainer(
            model=model,
            optimizer=SGD(model.parameters(), lr=0.1),
            scheduler=None,
            scaler=None,
            attack=None,
            selection_attack=LinfPGD(_ATTACK),
            objective=PGDATObjective(),
            device=torch.device("cpu"),
            output_dir=tmp_path,
            config_hash="e" * 64,
            seed=0,
            mixed_batch=mixed,
        )
    with pytest.raises(ValueError, match="not specified together"):
        _trainer(model, SGD(model.parameters(), lr=0.1), tmp_path, mixed_batch=mixed, awp=AwpConfig())
    with pytest.raises(ValueError, match="no AMP GradScaler for AWP"):
        Trainer(
            model=model,
            optimizer=SGD(model.parameters(), lr=0.1),
            scheduler=None,
            scaler=object(),
            attack=LinfPGD(_ATTACK),
            selection_attack=LinfPGD(_ATTACK),
            objective=PGDATObjective(),
            device=torch.device("cpu"),
            output_dir=tmp_path,
            config_hash="e" * 64,
            seed=0,
            awp=AwpConfig(),
        )


# --------------------------------------------------------------------------- AWP


# Verbatim transcription of csdongxian/AWP AT_AWP/utils_awp.py (the reference, not our code).
_UPSTREAM_EPS = 1e-20


def _upstream_diff_in_weights(model: nn.Module, proxy: nn.Module) -> OrderedDict[str, torch.Tensor]:
    diff_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (_new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if "weight" in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + _UPSTREAM_EPS) * diff_w
    return diff_dict


def _upstream_add_into_weights(model: nn.Module, diff: OrderedDict[str, torch.Tensor], coeff: float = 1.0) -> None:
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])


def _upstream_calc_awp(
    model: nn.Module,
    proxy: nn.Module,
    proxy_optim: torch.optim.Optimizer,
    inputs_adv: torch.Tensor,
    targets: torch.Tensor,
) -> OrderedDict[str, torch.Tensor]:
    proxy.load_state_dict(model.state_dict())
    proxy.train()
    loss = -F.cross_entropy(proxy(inputs_adv), targets)
    proxy_optim.zero_grad()
    loss.backward()
    proxy_optim.step()
    return _upstream_diff_in_weights(model, proxy)


def test_awp_constants_match_upstream() -> None:
    assert AWP_EPS == _UPSTREAM_EPS


def test_awp_direction_matches_the_official_code() -> None:
    model = _bn_model()
    batch = _batch(6)
    ours = AdversarialWeightPerturbation(model, gamma=0.01)
    proxy = copy.deepcopy(model)
    upstream = _upstream_calc_awp(model, proxy, SGD(proxy.parameters(), lr=0.01), batch.images, batch.labels)
    mask = torch.ones(6)
    diff = ours.compute(
        batch.images,
        lambda logits: (PGDATObjective()(student_logits=logits, labels=batch.labels).total * mask).sum() / mask.sum(),
    )
    assert list(diff) == list(upstream) == ["0.weight", "5.weight"]
    for key in diff:
        torch.testing.assert_close(diff[key], upstream[key], rtol=1e-6, atol=1e-9)
    before = _state(model)
    ours.perturb(diff)
    assert not torch.equal(model[0].weight, before["0.weight"])
    torch.testing.assert_close(
        (model[0].weight - before["0.weight"]).norm(), 0.01 * before["0.weight"].norm(), rtol=1e-5, atol=0
    )
    assert torch.equal(model[0].bias, before["0.bias"]) and torch.equal(model[1].weight, before["1.weight"])
    ours.restore(diff)
    torch.testing.assert_close(model[0].weight, before["0.weight"], rtol=0, atol=1e-7)
    # The model's own BN statistics are untouched by the proxy's train-mode forward.
    assert torch.equal(model[1].running_mean, before["1.running_mean"])


def test_awp_step_matches_the_official_training_loop(tmp_path: Path) -> None:
    batch = _batch(6)
    model = _bn_model()
    reference = copy.deepcopy(model)
    trainer = _trainer(model, SGD(model.parameters(), lr=0.1, momentum=0.9), tmp_path, awp=AwpConfig(gamma=0.01))
    metrics = trainer.train_epoch([batch])  # type: ignore[arg-type]
    assert metrics["awp_active"] == 1.0
    # Official AT_AWP/train_cifar10.py order, on our attack's output.
    reference.train()
    adversarial = (
        LinfPGD(_ATTACK)
        .generate(
            AttackRequest(
                inputs=batch.images,
                labels=batch.labels,
                student=reference,
                generator=torch.Generator().manual_seed(_SEED),
            )
        )
        .adversarial
    )
    reference.train()
    proxy = copy.deepcopy(reference)
    awp = _upstream_calc_awp(reference, proxy, SGD(proxy.parameters(), lr=0.01), adversarial, batch.labels)
    _upstream_add_into_weights(reference, awp, coeff=0.01)
    loss = F.cross_entropy(reference(adversarial), batch.labels)
    optimizer = SGD(reference.parameters(), lr=0.1, momentum=0.9)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    _upstream_add_into_weights(reference, awp, coeff=-0.01)
    for key, value in _state(reference).items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=1e-5, atol=1e-7, msg=key)


@pytest.mark.parametrize("option", ["awp", "split_bn"])
def test_resume_at_an_epoch_boundary_reproduces_the_uninterrupted_run(tmp_path: Path, option: str) -> None:
    """AWP keeps no state across steps; split BN restores its auxiliary BN from the checkpoint."""

    def loader() -> DataLoader:
        dataset = IndexedDataset(SyntheticCIFAR(size=8, num_classes=3, image_size=4, seed=2))
        return DataLoader(dataset, batch_size=4, sampler=EpochShuffleSampler(8, seed=1), collate_fn=collate_indexed)

    def build(output: Path) -> Trainer:
        model = _bn_model()
        extra: dict[str, Any] = {}
        parameters = list(model.parameters())
        if option == "awp":
            extra["awp"] = AwpConfig(gamma=0.01, warmup_epochs=1)
        else:
            auxiliary = AuxiliaryBatchNorm(model)
            parameters += list(auxiliary.parameters())
            extra["mixed_batch"] = MixedBatchConfig(
                adversarial_fraction=0.5, adversarial_weight=0.3, split_batchnorm=True
            )
            extra["auxiliary_batchnorm"] = auxiliary
        return _trainer(model, SGD(parameters, lr=0.1, momentum=0.9), output, **extra)

    straight = build(tmp_path / "straight")
    straight.fit(loader(), validation_loader=loader(), epochs=2)
    interrupted = build(tmp_path / "interrupted")
    interrupted.fit(loader(), validation_loader=loader(), epochs=1)
    resumed = build(tmp_path / "interrupted")
    train_loader = loader()
    start = resumed.resume(tmp_path / "interrupted" / "last.pt", sampler=train_loader.sampler).next_epoch
    resumed.fit(train_loader, validation_loader=loader(), epochs=2, start_epoch=start)
    for key in ("model", "optimizer", "auxiliary_batchnorm"):
        a = torch.load(tmp_path / "straight" / "last.pt", map_location="cpu", weights_only=False).get(key)
        b = torch.load(tmp_path / "interrupted" / "last.pt", map_location="cpu", weights_only=False).get(key)
        assert (a is None) == (b is None) == (key == "auxiliary_batchnorm" and option == "awp")
        if key == "model" or (key == "auxiliary_batchnorm" and a is not None):
            for name, value in a.items():
                assert torch.equal(b[name], value), (key, name)


def test_awp_inactive_during_warmup_is_exactly_plain_pgd_at(tmp_path: Path) -> None:
    batch = _batch(6)
    plain_model = _bn_model()
    awp_model = _bn_model()
    plain = _trainer(plain_model, SGD(plain_model.parameters(), lr=0.1), tmp_path / "plain")
    warm = _trainer(awp_model, SGD(awp_model.parameters(), lr=0.1), tmp_path / "awp", awp=AwpConfig(warmup_epochs=1))
    plain_metrics = plain.train_epoch([batch])  # type: ignore[arg-type]
    warm_metrics = warm.train_epoch([batch])  # type: ignore[arg-type]
    assert warm_metrics.pop("awp_active") == 0.0
    for key, value in plain_model.state_dict().items():
        assert torch.equal(awp_model.state_dict()[key], value), key
    assert {k: v for k, v in warm_metrics.items() if k != "seconds" and "per_second" not in k} == {
        k: v for k, v in plain_metrics.items() if k != "seconds" and "per_second" not in k
    }
