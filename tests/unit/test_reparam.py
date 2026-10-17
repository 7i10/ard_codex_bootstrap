"""Plan 0107 structural reparameterization of MobileNetV4-Conv-S (human-approved 2026-10-18).

Pins: where the branches go, that the folded network is exactly the plain ``mobilenetv4_conv_small_imagenet``
(strict load, same modules, same params/MACs), fold equivalence of outputs and input gradients (float64:
exact to rounding; float32: within float32 rounding), a negative control the checks must catch, the registry
/ schema / config contract, and the folded-checkpoint gate of ``ard.cli.evaluate``.

Float32 tolerances. The plain network's own float32 rounding (same weights, float32 vs float64) is about
5e-6 of the largest logit at 224 px, and float32 input gradients differ from float64 by up to a few percent in
single elements (ReLU gates flip at near-zero pre-activations; MobileNetV4's ReLU inputs are mostly small).
Folding cannot be checked more tightly than that in float32, so exactness is asserted in float64 (1e-10, outputs
and input gradients) and float32 is held to its rounding scale.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError
from timm.layers import BatchNormAct2d
from torch import nn

from ard.config.loader import load_config
from ard.config.schema import CUDA_GRAPH_ARCHITECTURES, ModelConfig
from ard.models import build_architecture, build_student
from ard.models.complexity import count_mac_terms, count_parameters
from ard.models.reparam import (
    REPARAM_ARCHITECTURES,
    ReparamConvBnAct,
    build_mobilenetv4_conv_small_reparam,
    deployed_student_config,
    fold_student,
    fold_student_state_dict,
    folded_state_dict,
)

pytestmark = pytest.mark.t1

REPARAM = "mobilenetv4_conv_small_reparam_imagenet"
PLAIN = "mobilenetv4_conv_small_imagenet"
CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "scientific"
REPARAM_CONFIG = "imagenet_mobilenetv4_reparam_pgd_at_phase2_random_lr0025.yaml"
HARDSWISH_CONFIG = "imagenet_mobilenetv4_hardswish_pgd_at_phase2_random_lr0025.yaml"

# Every conv + ReLU-BatchNormAct2d of MobileNetV4-Conv-S:
# (kernel, stride, in, out, groups, scale branch, identity branch)
EXPECTED_POSITIONS = {
    "conv_stem": (3, 2, 3, 32, 1, True, False),
    "blocks.0.0.conv": (3, 2, 32, 32, 1, True, False),
    "blocks.0.1.conv": (1, 1, 32, 32, 1, False, True),
    "blocks.1.0.conv": (3, 2, 32, 96, 1, True, False),
    "blocks.1.1.conv": (1, 1, 96, 64, 1, False, False),
    "blocks.2.0.pw_exp.conv": (1, 1, 64, 192, 1, False, False),
    "blocks.2.0.dw_mid.conv": (5, 2, 192, 192, 192, True, False),
    **{f"blocks.2.{i}.pw_exp.conv": (1, 1, 96, 192, 1, False, False) for i in (1, 2, 3, 4)},
    **{f"blocks.2.{i}.dw_mid.conv": (3, 1, 192, 192, 192, True, True) for i in (1, 2, 3, 4)},
    "blocks.2.5.pw_exp.conv": (1, 1, 96, 384, 1, False, False),
    "blocks.3.0.pw_exp.conv": (1, 1, 96, 576, 1, False, False),
    "blocks.3.0.dw_mid.conv": (3, 2, 576, 576, 576, True, False),
    "blocks.3.1.pw_exp.conv": (1, 1, 128, 512, 1, False, False),
    "blocks.3.1.dw_mid.conv": (5, 1, 512, 512, 512, True, True),
    "blocks.3.2.pw_exp.conv": (1, 1, 128, 512, 1, False, False),
    "blocks.3.2.dw_mid.conv": (5, 1, 512, 512, 512, True, True),
    "blocks.3.3.pw_exp.conv": (1, 1, 128, 384, 1, False, False),
    "blocks.3.3.dw_mid.conv": (5, 1, 384, 384, 384, True, True),
    "blocks.3.4.pw_exp.conv": (1, 1, 128, 512, 1, False, False),
    "blocks.3.4.dw_mid.conv": (3, 1, 512, 512, 512, True, True),
    "blocks.3.5.pw_exp.conv": (1, 1, 128, 512, 1, False, False),
    "blocks.3.5.dw_mid.conv": (3, 1, 512, 512, 512, True, True),
    "blocks.4.0.conv": (1, 1, 128, 960, 1, False, False),
    "conv_head": (1, 1, 960, 1280, 1, False, False),
}
# (params, conv+linear MACs) at 224, 1000 classes (ard.models.complexity).
PLAIN_COMPLEXITY = (3_774_024, 186_005_568)
REPARAM_COMPLEXITY = (5_765_896, 331_985_856)


def _calibrate(model: nn.Module, *, image_size: int, generator: torch.Generator) -> None:
    """Running statistics = the average of a few train-mode batches, as after real training (the init stats
    0/1 make eval-mode activations explode to ~1e8, which is not the regime a trained model is evaluated in)."""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.reset_running_stats()
            module.momentum = None
    model.train()
    with torch.no_grad():
        for _ in range(3):
            model(torch.rand(6, 3, image_size, image_size, generator=generator))
    model.eval()


def _sgd_steps(model: nn.Module, *, image_size: int, generator: torch.Generator, steps: int = 3) -> None:
    """A few real SGD steps with BN in train mode (BN running stats and every branch weight move)."""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.train()
    for _ in range(steps):
        images = torch.rand(6, 3, image_size, image_size, generator=generator)
        labels = torch.randint(0, 1000, (6,), generator=generator)
        loss = nn.functional.cross_entropy(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()


def _folded(model: nn.Module, dtype: torch.dtype) -> nn.Module:
    plain = build_architecture(PLAIN, num_classes=1000).to(dtype)
    plain.load_state_dict(folded_state_dict(model), strict=True)
    return plain.eval()


def _outputs_and_input_gradients(
    model: nn.Module, images: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    images = images.clone().requires_grad_(True)
    logits = model(images)
    (gradient,) = torch.autograd.grad(nn.functional.cross_entropy(logits, labels), images)
    return logits.detach(), gradient


def _relative(reference: torch.Tensor, other: torch.Tensor) -> float:
    return float((reference.double() - other.double()).abs().max() / reference.double().abs().max())


@pytest.fixture(scope="module")
def trained_pair() -> dict[str, nn.Module]:
    """Random init (calibrated BN) and the same model after 3 SGD steps (then calibrated), seed 0."""
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    random_init = build_architecture(REPARAM, num_classes=1000)
    _calibrate(random_init, image_size=96, generator=generator)
    stepped = copy.deepcopy(random_init)
    _sgd_steps(stepped, image_size=96, generator=generator)
    _calibrate(stepped, image_size=96, generator=generator)
    return {"random_init": random_init, "after_sgd": stepped}


def test_branches_sit_exactly_at_every_relu_gated_conv_bn() -> None:
    model = build_architecture(REPARAM, num_classes=1000)
    blocks = {name: module for name, module in model.named_modules() if isinstance(module, ReparamConvBnAct)}
    observed = {
        name: (
            block.conv.kernel_size[0],
            block.conv.stride[0],
            block.conv.in_channels,
            block.conv.out_channels,
            block.conv.groups,
            block.scale is not None,
            block.identity is not None,
        )
        for name, block in blocks.items()
    }
    assert observed == EXPECTED_POSITIONS
    assert len(blocks) == 30
    assert all(len(block.branches) == 1 for block in blocks.values())
    assert all(type(block.act) is nn.ReLU for block in blocks.values())
    # The plain model's activation-free conv+BN (dw_start, pw_proj) are untouched and no ReLU-gated
    # BatchNormAct2d is left anywhere.
    remaining = [module for module in model.modules() if isinstance(module, BatchNormAct2d)]
    assert remaining and all(type(module.act) is nn.Identity for module in remaining)
    assert sum(type(module) is nn.ReLU for module in model.modules()) == 30
    plain = build_architecture(PLAIN, num_classes=1000)
    assert sum(type(module) is nn.ReLU for module in plain.modules()) == 30


def test_extra_branch_count_is_configurable() -> None:
    for extra in (0, 2):
        model = build_mobilenetv4_conv_small_reparam(num_classes=10, extra_branches=extra)
        blocks = [module for module in model.modules() if isinstance(module, ReparamConvBnAct)]
        assert {len(block.branches) for block in blocks} == {extra}
    with pytest.raises(ValueError, match="extra_branches"):
        build_mobilenetv4_conv_small_reparam(num_classes=10, extra_branches=-1)


def test_original_branch_has_the_plain_models_seeded_init() -> None:
    """The plain network is built first, so at one seed the original branch (and every tensor outside the
    positions) is the plain init tensor for tensor; only the extra branches draw new init."""
    torch.manual_seed(0)
    plain = build_architecture(PLAIN, num_classes=1000).state_dict()
    torch.manual_seed(0)
    reparam = build_architecture(REPARAM, num_classes=1000)
    state = reparam.state_dict()
    renamed: dict[str, str] = {}
    for name, block in reparam.named_modules():
        if isinstance(block, ReparamConvBnAct):
            parent = name.rsplit(".", 1)[0] + "." if "." in name else ""
            renamed[f"{name}.weight"] = f"{name}.conv.weight"
            for suffix in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                renamed[f"{parent}{block.deploy_norm_name}.{suffix}"] = f"{name}.bn.{suffix}"
    assert len(renamed) == 30 * 6
    for key, value in plain.items():
        assert torch.equal(state[renamed.get(key, key)], value), key


def test_complexity_training_time_and_folded() -> None:
    reparam = build_architecture(REPARAM, num_classes=1000)
    plain = build_architecture(PLAIN, num_classes=1000)
    assert (count_parameters(plain), count_mac_terms(plain)[0]) == PLAIN_COMPLEXITY
    assert (count_parameters(reparam), *count_mac_terms(reparam)) == (*REPARAM_COMPLEXITY, 0)
    folded = _folded(reparam.eval(), torch.float32)
    assert (count_parameters(folded), *count_mac_terms(folded)) == (*PLAIN_COMPLEXITY, 0)
    assert round(REPARAM_COMPLEXITY[1] / PLAIN_COMPLEXITY[1], 3) == 1.785


def test_folded_state_dict_loads_strictly_and_has_the_plain_structure() -> None:
    reparam = build_architecture(REPARAM, num_classes=1000).eval()
    plain = build_architecture(PLAIN, num_classes=1000)
    folded = folded_state_dict(reparam)
    assert {key: value.shape for key, value in folded.items()} == {
        key: value.shape for key, value in plain.state_dict().items()
    }
    plain.load_state_dict(folded, strict=True)
    config = ModelConfig(architecture=REPARAM, num_classes=1000)
    deployed = fold_student(build_student(config).eval(), config)
    reference = build_student(ModelConfig(architecture=PLAIN, num_classes=1000))
    assert [(name, type(module)) for name, module in deployed.named_modules()] == [
        (name, type(module)) for name, module in reference.named_modules()
    ]
    assert not any(isinstance(module, ReparamConvBnAct) for module in deployed.modules())


@pytest.mark.parametrize("state", ["random_init", "after_sgd"])
@pytest.mark.parametrize("image_size", [64, 224])
def test_fold_is_exact_in_float64(trained_pair: dict[str, nn.Module], state: str, image_size: int) -> None:
    """Outputs and input gradients of the unfolded eval-mode model and the folded plain model agree to
    float64 rounding (measured ~1e-14)."""
    reparam = copy.deepcopy(trained_pair[state]).double().eval()
    plain = _folded(reparam, torch.float64)
    generator = torch.Generator().manual_seed(1)
    images = torch.rand(2, 3, image_size, image_size, generator=generator, dtype=torch.float64)
    labels = torch.randint(0, 1000, (2,), generator=generator)
    reference_logits, reference_gradient = _outputs_and_input_gradients(reparam, images, labels)
    logits, gradient = _outputs_and_input_gradients(plain, images, labels)
    assert _relative(reference_logits, logits) <= 1e-10
    assert _relative(reference_gradient, gradient) <= 1e-10


@pytest.mark.parametrize("state", ["random_init", "after_sgd"])
def test_fold_float32_is_within_float32_rounding(trained_pair: dict[str, nn.Module], state: str) -> None:
    """fp32 (the training/evaluation precision). Logits: within 5e-5 of the largest logit (measured 5e-6 to
    1e-5; the plain net's own fp32-vs-fp64 rounding is ~5e-6). The folded fp32 model is as close to the exact
    (float64) function as the unfolded fp32 model is. Input gradients: cosine >= 0.9999 (measured >= 0.99999)
    and sign agreement (what PGD uses) >= 0.995 (measured >= 0.9994). Single elements can differ by up to ~1% of
    the largest one where a ReLU gate flips; that happens equally between fp32 and fp64 of the same network."""
    reparam = trained_pair[state].eval()
    plain = _folded(reparam, torch.float32)
    exact = copy.deepcopy(reparam).double()
    generator = torch.Generator().manual_seed(2)
    for image_size in (64, 224):
        images = torch.rand(2, 3, image_size, image_size, generator=generator)
        labels = torch.randint(0, 1000, (2,), generator=generator)
        reference_logits, reference_gradient = _outputs_and_input_gradients(reparam, images, labels)
        logits, gradient = _outputs_and_input_gradients(plain, images, labels)
        exact_logits, _ = _outputs_and_input_gradients(exact, images.double(), labels)
        assert _relative(reference_logits, logits) <= 5e-5
        assert _relative(exact_logits, logits) <= 2 * _relative(exact_logits, reference_logits) + 1e-6
        cosine = nn.functional.cosine_similarity(
            reference_gradient.flatten().double(), gradient.flatten().double(), dim=0
        )
        assert float(cosine) >= 0.9999
        assert float((reference_gradient.sign() == gradient.sign()).double().mean()) >= 0.995


def _first_dw_branch(model: nn.Module) -> ReparamConvBnAct:
    return model.get_submodule("blocks.2.1.dw_mid.conv")  # type: ignore[return-value]


@pytest.mark.parametrize("part", ["extra_branch", "scale_branch", "identity_bn", "main_bn_running_var"])
def test_negative_control_one_perturbed_branch_parameter_breaks_equivalence(
    trained_pair: dict[str, nn.Module], part: str
) -> None:
    """Change one parameter of the unfolded model AFTER folding: the float64 check must see it, even for a
    perturbation of 1e-3 in a single element (so a fold that dropped or mis-scaled any branch would fail)."""
    reparam = copy.deepcopy(trained_pair["after_sgd"]).double().eval()
    plain = _folded(reparam, torch.float64)
    block = _first_dw_branch(reparam)
    with torch.no_grad():
        if part == "extra_branch":
            block.branches[0].conv.weight[0, 0, 1, 1] += 1e-3
        elif part == "scale_branch":
            assert block.scale is not None
            block.scale.conv.weight[0, 0, 0, 0] += 1e-3
        elif part == "identity_bn":
            assert block.identity is not None
            block.identity.weight[0] += 1e-3
        else:
            block.bn.running_var[0] *= 1.001
    generator = torch.Generator().manual_seed(3)
    images = torch.rand(2, 3, 64, 64, generator=generator, dtype=torch.float64)
    labels = torch.randint(0, 1000, (2,), generator=generator)
    reference_logits, reference_gradient = _outputs_and_input_gradients(reparam, images, labels)
    logits, gradient = _outputs_and_input_gradients(plain, images, labels)
    assert _relative(reference_logits, logits) > 1e-8
    assert _relative(reference_gradient, gradient) > 1e-8


def test_negative_control_dropping_a_branch_from_the_fold_is_detected(
    trained_pair: dict[str, nn.Module], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fold that forgets the identity branch everywhere must fail the float64 equivalence check."""
    reparam = copy.deepcopy(trained_pair["after_sgd"]).double().eval()
    original = ReparamConvBnAct.folded_kernel_and_bias

    def without_identity(self: ReparamConvBnAct) -> tuple[torch.Tensor, torch.Tensor]:
        identity, self.identity = self.identity, None
        try:
            return original(self)
        finally:
            self.identity = identity

    monkeypatch.setattr(ReparamConvBnAct, "folded_kernel_and_bias", without_identity)
    plain = _folded(reparam, torch.float64)
    images = torch.rand(2, 3, 64, 64, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    with torch.no_grad():
        assert _relative(reparam(images), plain(images)) > 1e-3


def test_fold_student_state_dict_round_trip_and_rng_untouched() -> None:
    config = ModelConfig(architecture=REPARAM, num_classes=10)
    torch.manual_seed(5)
    student = build_student(config)
    _calibrate(student, image_size=64, generator=torch.Generator().manual_seed(5))
    state = student.state_dict()
    rng = torch.get_rng_state()
    folded = fold_student_state_dict(state, config)
    assert torch.equal(torch.get_rng_state(), rng)
    plain = build_student(deployed_student_config(config))
    plain.load_state_dict(folded, strict=True)
    plain.eval()
    images = torch.rand(2, 3, 64, 64, generator=torch.Generator().manual_seed(6))
    with torch.no_grad():
        assert _relative(student(images), plain(images)) <= 5e-5
    assert deployed_student_config(config).architecture == PLAIN
    assert deployed_student_config(config).model_dump() == {**config.model_dump(), "architecture": PLAIN}


def test_folded_state_dict_refuses_a_plain_model() -> None:
    with pytest.raises(ValueError, match="no reparameterized positions"):
        folded_state_dict(build_architecture(PLAIN, num_classes=10))


def test_registry_and_schema_contract() -> None:
    assert REPARAM_ARCHITECTURES[REPARAM].deployed_architecture == PLAIN
    assert REPARAM_ARCHITECTURES[REPARAM].extra_branches == 1
    assert ModelConfig(architecture=REPARAM, num_classes=1000).architecture == REPARAM
    assert REPARAM not in CUDA_GRAPH_ARCHITECTURES
    with pytest.raises(ValueError, match="pretrained=True is not supported"):
        build_architecture(REPARAM, num_classes=1000, pretrained=True)
    with pytest.raises(ValidationError, match="pretrained=True is not supported"):
        ModelConfig(architecture=REPARAM, num_classes=1000, pretrained=True)
    for image_size in (224, 112, 32):
        with torch.no_grad():
            assert build_architecture(REPARAM, num_classes=10).eval()(
                torch.rand(2, 3, image_size, image_size)
            ).shape == (2, 10)


def _set_config_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for key, value in {
        "ARD_SEED": "7",
        "ARD_IMAGENET_ROOT": str(tmp_path / "imagenet"),
        "ARD_NUM_WORKERS": "0",
        "ARD_JOB_OUTPUT_DIR": str(tmp_path / "job-output"),
        "ARD_RUN_ID": "config-test-run",
        "WANDB_ENTITY": "entity",
        "WANDB_PROJECT": "project",
    }.items():
        monkeypatch.setenv(key, value)


def test_reparam_config_changes_only_architecture_and_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_config_env(monkeypatch, tmp_path)
    base = load_config(CONFIG_DIR / HARDSWISH_CONFIG).model_dump(mode="json")
    variant = load_config(CONFIG_DIR / REPARAM_CONFIG).model_dump(mode="json")
    assert variant["student"]["architecture"] == REPARAM and variant["student"]["pretrained"] is False
    assert variant["tracking"]["group"] == "imagenet-mobilenetv4-reparam-pgd-at-phase2" != base["tracking"]["group"]
    assert variant["training"]["deterministic"] is False and variant["training"]["cudnn_benchmark"] is True
    assert "cuda_graph" not in variant["training"]
    header = (CONFIG_DIR / REPARAM_CONFIG).read_text().split("schema_version")[0]
    assert "2026-10-18" in header and HARDSWISH_CONFIG in header and "MobileNetV4-M" in header
    for payload in (base, variant):
        payload["student"]["architecture"] = None
        payload["tracking"]["group"] = None
    assert variant == base


def test_evaluation_gate_accepts_only_matching_folded_checkpoints() -> None:
    """ard.cli.evaluate: a reparameterized student is evaluated only from folded checkpoints of its own
    architecture and the requested weight set; a fold record on any other student is refused; every other
    student gets no record (its evaluation output stays byte-identical)."""
    from types import SimpleNamespace

    from ard.cli.evaluate import _reparameterization_records
    from ard.models.reparam import FOLD_CONTRACT

    def config(architecture: str) -> SimpleNamespace:
        return SimpleNamespace(student=SimpleNamespace(architecture=architecture))

    record = {
        "contract": FOLD_CONTRACT,
        "training_architecture": REPARAM,
        "deployed_architecture": PLAIN,
        "folded_weight_sets": ["model"],
    }
    paths = [Path("last.pt")]
    assert _reparameterization_records(config(REPARAM), paths, [{"reparameterization": record}], weights="model") == [
        record
    ]
    assert _reparameterization_records(config(PLAIN), paths, [{}], weights="model") is None
    with pytest.raises(ValueError, match="ard.cli.fold_reparam"):
        _reparameterization_records(config(REPARAM), paths, [{}], weights="model")
    with pytest.raises(ValueError, match="weight set 'ema'"):
        _reparameterization_records(config(REPARAM), paths, [{"reparameterization": record}], weights="ema")
    for field, value in (("contract", "other"), ("deployed_architecture", REPARAM), ("training_architecture", PLAIN)):
        with pytest.raises(ValueError, match="not a folded"):
            _reparameterization_records(
                config(REPARAM), paths, [{"reparameterization": {**record, field: value}}], weights="model"
            )
    with pytest.raises(ValueError, match="not a reparameterized architecture"):
        _reparameterization_records(config(PLAIN), paths, [{"reparameterization": record}], weights="model")


PRE_CHANGE_COMMIT = "32dcdad"  # master before the reparameterized student id was added to ModelConfig


def test_configs_that_existed_before_the_change_hash_exactly_as_under_the_pre_change_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Adding the architecture id to ModelConfig's Literal changes no existing config's resolved dict or hash."""
    import json
    import subprocess
    import sys
    import types

    import yaml

    from ard.config.loader import _expand_environment, resolved_config_dict
    from ard.config.schema import ExperimentConfig
    from ard.engine.checkpoint import config_digest
    from tests.unit.test_phase2_batch_a import _env_all

    root = Path(__file__).resolve().parents[2]

    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout
        except (OSError, subprocess.CalledProcessError) as error:  # pragma: no cover - shallow clones
            pytest.skip(f"git {' '.join(args)} unavailable: {error}")

    _env_all(monkeypatch, tmp_path)
    relative = "src/ard/config/schema.py"
    name = "_ard_schema_pre_reparam"
    old_schema = types.ModuleType(name)
    old_schema.__package__ = "ard.config"
    old_schema.__file__ = f"{PRE_CHANGE_COMMIT}:{relative}"
    sys.modules[name] = old_schema
    exec(compile(git("show", f"{PRE_CHANGE_COMMIT}:{relative}"), old_schema.__file__, "exec"), old_schema.__dict__)
    listed = git("ls-tree", "-r", "--name-only", PRE_CHANGE_COMMIT, "configs/").splitlines()
    paths = [
        path
        for path in listed
        if path.endswith(".yaml") and path.split("/")[1] in {"experiments", "pilot", "production", "scientific"}
    ]
    assert len(paths) > 20
    for path in paths:
        expanded = _expand_environment(yaml.safe_load(git("show", f"{PRE_CHANGE_COMMIT}:{path}")))
        new = ExperimentConfig.model_validate(expanded)
        old = old_schema.ExperimentConfig.model_validate(expanded)
        assert resolved_config_dict(new) == json.loads(old.model_dump_json()), path
        assert config_digest(resolved_config_dict(new)) == config_digest(json.loads(old.model_dump_json())), path
    with pytest.raises(Exception, match="architecture"):
        old_schema.ModelConfig(architecture=REPARAM, num_classes=1000)  # the id is new: absent before the change
