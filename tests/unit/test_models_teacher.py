from __future__ import annotations

import shutil
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest
import torch
from torch import nn

from ard.attacks import AttackRequest, LinfPGD, teacher_input_gradient
from ard.config.schema import AttackConfig, ExperimentConfig, ModelConfig, NormalizationConfig, TeacherConfig
from ard.models import (
    PixelModel,
    TeacherAdapter,
    TeacherMetadata,
    build_architecture,
    build_student,
    build_teacher,
    normalize_state_dict,
    sha256_file,
    teacher_registry,
)
from ard.models.teacher_registry import FactorySpec, TeacherRegistry, TeacherRegistryError

pytestmark = pytest.mark.t1


def _saad_resnet18_state_shapes() -> dict[str, tuple[int, ...]]:
    """Independent structural specification for the clean-room CIFAR ResNet-18."""
    shapes: dict[str, tuple[int, ...]] = {"conv1.weight": (64, 3, 3, 3)}

    def batch_norm(prefix: str, channels: int) -> None:
        shapes.update(
            {
                f"{prefix}.weight": (channels,),
                f"{prefix}.bias": (channels,),
                f"{prefix}.running_mean": (channels,),
                f"{prefix}.running_var": (channels,),
                f"{prefix}.num_batches_tracked": (),
            }
        )

    batch_norm("bn1", 64)
    for layer, (in_planes, planes) in enumerate(((64, 64), (64, 128), (128, 256), (256, 512)), start=1):
        for block in range(2):
            prefix = f"layer{layer}.{block}"
            block_input = in_planes if block == 0 else planes
            shapes[f"{prefix}.conv1.weight"] = (planes, block_input, 3, 3)
            batch_norm(f"{prefix}.bn1", planes)
            shapes[f"{prefix}.conv2.weight"] = (planes, planes, 3, 3)
            batch_norm(f"{prefix}.bn2", planes)
            if block == 0 and block_input != planes:
                shapes[f"{prefix}.shortcut.0.weight"] = (planes, block_input, 1, 1)
                batch_norm(f"{prefix}.shortcut.1", planes)
    shapes["linear.weight"] = (10, 512)
    shapes["linear.bias"] = (10,)
    return shapes


def test_explicit_cifar_model_registry_and_normalization() -> None:
    normalization = NormalizationConfig(
        profile="custom", mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25), provenance="unit-test"
    )
    adapter = PixelModel(nn.Identity(), normalization)
    pixels = torch.full((1, 3, 2, 2), 0.75)
    assert torch.equal(adapter(pixels), torch.ones_like(pixels))
    resnet = build_student(ModelConfig(architecture="resnet18_cifar", num_classes=7), tier="dev")
    mobile = build_student(ModelConfig(architecture="mobilenet_v2_cifar", num_classes=7), tier="dev")
    assert resnet.model.conv1.kernel_size == (3, 3) and resnet.model.conv1.stride == (1, 1)
    assert isinstance(resnet.model.maxpool, nn.Identity)
    assert mobile.model.features[0][0].stride == (1, 1)


def test_clean_room_saad_resnet18_cifar_contract_is_exact_and_has_no_external_dependency() -> None:
    model = build_architecture("saad_resnet18_cifar_v1", 10)
    assert model(torch.rand(2, 3, 32, 32)).shape == (2, 10)
    assert sum(parameter.numel() for parameter in model.parameters()) == 11_173_962
    actual_shapes = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    assert actual_shapes == _saad_resnet18_state_shapes()
    assert len(actual_shapes) == 122
    assert model.conv1.kernel_size == (3, 3) and model.conv1.bias is None
    assert model.avgpool.kernel_size == 4
    student = build_student(
        ModelConfig(
            architecture="saad_resnet18_cifar_v1",
            num_classes=10,
            normalization=NormalizationConfig(profile="cifar10_raw_identity"),
        ),
        tier="dev",
    )
    pixels = torch.rand(1, 3, 32, 32)
    assert torch.equal(student.normalization(pixels), pixels)


def test_adapter_preprocessing_is_exactly_once_and_profiles_are_independent() -> None:
    class Capture(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inputs: torch.Tensor | None = None

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            self.inputs = inputs.detach().clone()
            return inputs

    student_normalization = NormalizationConfig(
        profile="custom", mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25), provenance="student-unit"
    )
    teacher_normalization = NormalizationConfig(
        profile="custom", mean=(0.25, 0.25, 0.25), std=(0.5, 0.5, 0.5), provenance="teacher-unit"
    )
    pixels = torch.full((1, 3, 2, 2), 0.75)
    student_capture = Capture()
    teacher_capture = Capture()
    student = PixelModel(student_capture, student_normalization)
    teacher = TeacherAdapter(
        teacher_capture,
        TeacherMetadata(
            architecture="fixture_cnn",
            num_classes=3,
            normalization=teacher_normalization,
            checkpoint_sha256="0" * 64,
        ),
    )
    student(pixels)
    teacher(pixels)
    assert student_capture.inputs is not None and teacher_capture.inputs is not None
    assert torch.equal(student_capture.inputs, torch.ones_like(pixels))
    assert torch.equal(teacher_capture.inputs, torch.ones_like(pixels))

    config = ExperimentConfig.model_validate(
        {
            "schema_version": 2,
            "protocol": {"id": "synthetic_smoke_v2"},
            "seeds": {
                key: 0
                for key in (
                    "split",
                    "model_init",
                    "data_order",
                    "augmentation",
                    "train_attack",
                    "evaluation_attack",
                    "qualitative_panel",
                )
            },
            "dataset": {"name": "synthetic_cifar", "num_classes": 3},
            "student": {"architecture": "fixture_cnn", "num_classes": 3, "normalization": {"profile": "fixture_unit"}},
            "teacher": {
                "source": "fixture",
                "architecture": "fixture_cnn",
                "num_classes": 3,
                "normalization": teacher_normalization.model_dump(mode="json"),
            },
            "method": {"id": "rslad", "version": 1, "attack": {"loss": "kl", "kl_target": "teacher_clean", "steps": 1}},
            "optimizer": {"id": "sgd", "learning_rate": 0.01, "momentum": 0.0, "weight_decay": 0.0, "nesterov": False},
            "scheduler": {"id": "identity", "milestones": [], "gamma": 1.0, "step_at": "epoch_end"},
            "training": {"epochs": 1, "per_rank_batch_size": 2, "global_batch_size": 2},
        }
    )
    assert config.teacher is not None and config.teacher.normalization != config.student.normalization


@pytest.mark.parametrize(
    ("config_type", "owner"),
    ((ModelConfig, "model_embedded"), (TeacherConfig, "robustbench_model")),
)
def test_unimplemented_preprocessing_owners_fail_closed(
    config_type: type[ModelConfig] | type[TeacherConfig], owner: str
) -> None:
    with pytest.raises(Exception, match="preprocessing_owner"):
        config_type(preprocessing_owner=owner)


def test_robustbench_registry_is_exact_and_requires_no_model_construction() -> None:
    registry = TeacherRegistry.load(Path(__file__).resolve().parents[2])
    chen = registry.spec("chen2021_ltd_wrn34_10")
    bartoldson = registry.spec("bartoldson2024_adversarial_wrn94_16")
    assert (chen.upstream_model_id, chen.factory.module, chen.factory.symbol, dict(chen.factory.kwargs)) == (
        "Chen2021LTD_WRN34_10",
        "robustbench.model_zoo.architectures.wide_resnet",
        "WideResNet",
        {"depth": 34, "widen_factor": 10, "sub_block1": False},
    )
    assert chen.expected_parameter_count == 46_160_474
    assert chen.preprocessing.owner == "teacher_adapter"
    assert chen.preprocessing.normalization().profile == "cifar10_raw_identity"
    assert (bartoldson.upstream_model_id, bartoldson.factory.module, bartoldson.factory.symbol) == (
        "Bartoldson2024Adversarial_WRN-94-16",
        "robustbench.model_zoo.architectures.dm_wide_resnet",
        "DMWideResNet",
    )
    assert dict(bartoldson.factory.kwargs) == {
        "num_classes": 10,
        "depth": 94,
        "width": 16,
        "activation_fn": "torch.nn.SiLU",
        "mean": [0.4914, 0.4822, 0.4465],
        "std": [0.2471, 0.2435, 0.2616],
    }
    assert bartoldson.expected_parameter_count == 365_915_610
    assert bartoldson.preprocessing.owner == "model_embedded"
    assert bartoldson.preprocessing.normalization().profile == "robustbench_cifar10_bartoldson_embedded"
    for spec in (chen, bartoldson):
        assert spec.checkpoint_status in {"missing", "verified"}
        assert (spec.checkpoint_status == "missing") == (spec.checkpoint_sha256 is None)


def test_robustbench_embedded_preprocessing_is_not_double_applied() -> None:
    class Capture(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inputs: torch.Tensor | None = None

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            self.inputs = inputs.detach().clone()
            return inputs

    capture = Capture()
    metadata = TeacherMetadata(
        architecture="robustbench_dm_wide_resnet",
        num_classes=3,
        normalization=NormalizationConfig(profile="robustbench_cifar10_bartoldson_embedded"),
        checkpoint_sha256="0" * 64,
        preprocessing_owner="model_embedded",
        registry_id="bartoldson2024_adversarial_wrn94_16",
        upstream_model_id="Bartoldson2024Adversarial_WRN-94-16",
    )
    pixels = torch.full((1, 3, 2, 2), 0.75)
    TeacherAdapter(capture, metadata)(pixels)
    assert capture.inputs is not None and torch.equal(capture.inputs, pixels)


def test_checkpoint_prefix_normalization_rejects_collisions_and_strict_load(tmp_path: Path) -> None:
    model = build_architecture("fixture_cnn", 3)
    checkpoint = tmp_path / "teacher.pt"
    torch.save({"state_dict": {f"module.{key}": value for key, value in model.state_dict().items()}}, checkpoint)
    metadata = TeacherMetadata(
        architecture="fixture_cnn",
        num_classes=3,
        normalization=NormalizationConfig(),
        checkpoint_sha256=sha256_file(checkpoint),
    )
    assert isinstance(TeacherAdapter.from_checkpoint(checkpoint, metadata), TeacherAdapter)
    model_wrapper = tmp_path / "model-wrapper.pt"
    torch.save({"model": model.state_dict(), "epoch": 1}, model_wrapper)
    assert isinstance(
        TeacherAdapter.from_checkpoint(
            model_wrapper, metadata.model_copy(update={"checkpoint_sha256": sha256_file(model_wrapper)})
        ),
        TeacherAdapter,
    )
    state_wrapper = tmp_path / "state-wrapper.pt"
    torch.save({"state_dict": model.state_dict(), "epoch": 1}, state_wrapper)
    assert isinstance(
        TeacherAdapter.from_checkpoint(
            state_wrapper, metadata.model_copy(update={"checkpoint_sha256": sha256_file(state_wrapper)})
        ),
        TeacherAdapter,
    )
    assert set(normalize_state_dict({"model": model.state_dict(), "epoch": 1})) == set(model.state_dict())
    assert set(normalize_state_dict({"state_dict": model.state_dict(), "epoch": 1})) == set(model.state_dict())
    with pytest.raises(ValueError, match="collision"):
        normalize_state_dict({"module.x": torch.tensor(1), "x": torch.tensor(2)})
    assert set(normalize_state_dict({"model.x": torch.tensor(1)})) == {"x"}
    with pytest.raises(ValueError, match="ambiguous"):
        normalize_state_dict({"model": model.state_dict(), "state_dict": model.state_dict()})
    invalid = tmp_path / "invalid.pt"
    torch.save({"state_dict": {"module.weight": torch.ones(1)}}, invalid)
    with pytest.raises(RuntimeError, match="Missing key"):
        TeacherAdapter.from_checkpoint(invalid, metadata.model_copy(update={"checkpoint_sha256": sha256_file(invalid)}))


def test_robustbench_missing_hash_fails_before_constructor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    base = replace(TeacherRegistry.load(Path(__file__).resolve().parents[2]), root=tmp_path)
    original = base.spec("chen2021_ltd_wrn34_10")
    spec = replace(original, checkpoint_status="missing", checkpoint_sha256=None)
    registry = replace(base, specs={spec.registry_id: spec})
    checkpoint = tmp_path / spec.checkpoint_path
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"not a checkpoint")
    config = TeacherConfig(
        source="robustbench",
        registry_id=spec.registry_id,
        architecture=spec.architecture,
        num_classes=10,
        normalization=spec.preprocessing.normalization(),
        preprocessing_owner=spec.preprocessing.owner,
        checkpoint=checkpoint,
        checkpoint_sha256="0" * 64,
    )
    monkeypatch.setattr("ard.models.teacher.TeacherRegistry.load", lambda: registry)
    monkeypatch.setattr(TeacherRegistry, "constructor", lambda *_args, **_kwargs: pytest.fail("constructor ran"))
    with pytest.raises(TeacherRegistryError, match="not hash-registered"):
        build_teacher(config, tier="production")


def test_robustbench_config_must_exactly_match_registered_preprocessing_threat_and_path(tmp_path: Path) -> None:
    base = replace(TeacherRegistry.load(Path(__file__).resolve().parents[2]), root=tmp_path)
    original = base.spec("chen2021_ltd_wrn34_10")
    spec = replace(original, checkpoint_status="verified", checkpoint_sha256="0" * 64)
    registry = replace(base, specs={spec.registry_id: spec})
    expected = tmp_path / spec.checkpoint_path
    valid = TeacherConfig(
        source="robustbench",
        registry_id=spec.registry_id,
        architecture=spec.architecture,
        num_classes=10,
        normalization=spec.preprocessing.normalization(),
        preprocessing_owner=spec.preprocessing.owner,
        checkpoint=expected,
        checkpoint_sha256="0" * 64,
    )
    assert registry.validate_config(valid) == spec
    with pytest.raises(TeacherRegistryError, match="does not exactly match"):
        registry.validate_config(valid.model_copy(update={"checkpoint": tmp_path / "other.pt"}))


def test_model_embedded_preprocessing_is_restricted_to_bartoldson_robustbench_teacher(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="restricted to the Bartoldson"):
        TeacherConfig(source="fixture", preprocessing_owner="model_embedded")
    with pytest.raises(ValueError, match="Chen RobustBench teacher"):
        TeacherConfig(
            source="robustbench",
            registry_id="chen2021_ltd_wrn34_10",
            architecture="robustbench_wide_resnet",
            preprocessing_owner="model_embedded",
            checkpoint=tmp_path / "chen.pt",
            checkpoint_sha256="0" * 64,
        )
    bartoldson = TeacherConfig(
        source="robustbench",
        registry_id="bartoldson2024_adversarial_wrn94_16",
        architecture="robustbench_dm_wide_resnet",
        preprocessing_owner="model_embedded",
        normalization=NormalizationConfig(profile="robustbench_cifar10_bartoldson_embedded"),
        checkpoint=tmp_path / "bartoldson.pt",
        checkpoint_sha256="0" * 64,
    )
    assert bartoldson.preprocessing_owner == "model_embedded"
    with pytest.raises(ValueError, match="restricted to the Bartoldson"):
        TeacherAdapter(
            nn.Identity(),
            TeacherMetadata(
                architecture="fixture_cnn",
                num_classes=3,
                normalization=NormalizationConfig(),
                checkpoint_sha256="0" * 64,
                preprocessing_owner="model_embedded",
            ),
        )


def test_robustbench_constructor_dependency_and_kwargs_can_be_tested_without_large_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = TeacherRegistry.load(Path(__file__).resolve().parents[2])
    spec = replace(registry.spec("chen2021_ltd_wrn34_10"), expected_parameter_count=1)
    monkeypatch.setattr(TeacherRegistry, "validate_external", lambda _self: None)

    class OneParameter(nn.Module):
        def __init__(self, **kwargs: object) -> None:
            super().__init__()
            self.kwargs = kwargs
            self.weight = nn.Parameter(torch.ones(1))

    captured: dict[str, object] = {}

    def resolver(factory: FactorySpec) -> type[OneParameter]:
        captured.update(module=factory.module, symbol=factory.symbol)
        return OneParameter

    model = registry.constructor(spec, resolver=resolver)
    assert isinstance(model, OneParameter)
    assert captured == {"module": "robustbench.model_zoo.architectures.wide_resnet", "symbol": "WideResNet"}
    assert model.kwargs == {"depth": 34, "widen_factor": 10, "sub_block1": False}


def test_external_factory_rejects_preloaded_unverified_robustbench_module(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = ModuleType("robustbench")
    fake.__file__ = str(tmp_path / "installed" / "robustbench" / "__init__.py")
    monkeypatch.setitem(sys.modules, "robustbench", fake)
    with pytest.raises(TeacherRegistryError, match="preloaded RobustBench module"):
        teacher_registry._external_factory(
            tmp_path,
            FactorySpec(module="robustbench.model_zoo.architectures.wide_resnet", symbol="WideResNet", kwargs={}),
        )


@pytest.mark.parametrize(
    ("git_values", "license_bytes", "message"),
    (
        ({"remote get-url origin": "wrong://remote"}, b"license", "origin does not match"),
        ({"rev-parse HEAD": "0" * 40}, b"license", "HEAD does not match"),
        ({"status --porcelain --untracked-files=all": " M changed"}, b"license", "checkout is dirty"),
        ({}, b"wrong license", "license evidence"),
    ),
)
def test_robustbench_external_remote_dirty_and_license_rejections(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, git_values: dict[str, str], license_bytes: bytes, message: str
) -> None:
    source_root = Path(__file__).resolve().parents[2]
    shutil.copyfile(source_root / "teachers.lock.yaml", tmp_path / "teachers.lock.yaml")
    shutil.copyfile(source_root / "external.lock.yaml", tmp_path / "external.lock.yaml")
    registry = TeacherRegistry.load(tmp_path)
    checkout = tmp_path / ".external" / "robustbench"
    checkout.mkdir(parents=True)
    (checkout / "LICENSE").write_bytes(license_bytes)

    def fake_git(_cwd: Path, args: list[str]) -> str:
        command = " ".join(args)
        defaults = {
            "rev-parse --is-inside-work-tree": "true",
            "remote get-url origin": registry.repository_url,
            "rev-parse HEAD": registry.repository_commit,
            "status --porcelain --untracked-files=all": "",
        }
        return git_values.get(command, defaults[command])

    monkeypatch.setattr("ard.models.teacher_registry._git", fake_git)
    with pytest.raises(TeacherRegistryError, match=message):
        registry.validate_external()


def test_teacher_checkpoint_hash_strict_load_and_freeze(tmp_path: Path) -> None:
    model = build_architecture("fixture_cnn", 3)
    checkpoint = tmp_path / "teacher.pt"
    torch.save(model.state_dict(), checkpoint)
    metadata = TeacherMetadata(
        architecture="fixture_cnn",
        num_classes=3,
        normalization=NormalizationConfig(),
        checkpoint_sha256=sha256_file(checkpoint),
    )
    teacher = TeacherAdapter.from_checkpoint(checkpoint, metadata)
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert teacher(torch.rand(2, 3, 4, 4)).shape == (2, 3)
    wrong = metadata.model_copy(update={"checkpoint_sha256": "0" * 64})
    with pytest.raises(ValueError, match="hash mismatch"):
        TeacherAdapter.from_checkpoint(checkpoint, wrong)


def test_fixture_teacher_is_deterministic_and_dev_smoke_only() -> None:
    config = TeacherConfig(source="fixture", architecture="fixture_cnn", num_classes=3, fixture_seed=12)
    first = build_teacher(config, tier="dev")
    second = build_teacher(config, tier="smoke")
    for left, right in zip(first.state_dict().values(), second.state_dict().values(), strict=True):
        assert torch.equal(left, right)
    with pytest.raises(ValueError, match="restricted"):
        build_teacher(config, tier="production")


def test_frozen_teacher_stays_eval_through_nested_batchnorm_attack_and_input_gradient() -> None:
    teacher_model = nn.Sequential(
        nn.Conv2d(3, 3, kernel_size=1),
        nn.Sequential(nn.BatchNorm2d(3), nn.ReLU()),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(3, 3),
    )
    metadata = TeacherMetadata(
        architecture="fixture_cnn",
        num_classes=3,
        normalization=NormalizationConfig(),
        checkpoint_sha256="0" * 64,
    )
    teacher = TeacherAdapter(teacher_model, metadata)
    student = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 3))
    batch_norm = teacher.model[1][0]
    running_mean = batch_norm.running_mean.detach().clone()
    teacher.train(True)
    assert not teacher.training and not teacher.model.training and not batch_norm.training
    inputs, labels = torch.rand(2, 3, 4, 4), torch.tensor([0, 1])
    LinfPGD(
        AttackConfig(
            loss="kl",
            kl_target="teacher_clean",
            epsilon="1/255",
            step_size="1/255",
            steps=1,
            teacher_mode="eval",
        )
    ).generate(AttackRequest(inputs=inputs, labels=labels, student=student, teacher=teacher))
    gradient = teacher_input_gradient(teacher, inputs, labels)
    forward_only_pixels = inputs.detach().clone().requires_grad_(True)
    forward_only_logits = teacher.logits(forward_only_pixels, require_input_grad=False)
    assert not forward_only_logits.requires_grad
    with pytest.raises(ValueError, match="requires_grad pixel tensor"):
        teacher.logits(inputs, require_input_grad=True)
    pixels_for_grad = inputs.detach().clone().requires_grad_(True)
    logits = teacher.logits(pixels_for_grad, require_input_grad=True)
    explicit_gradient = torch.autograd.grad(logits.sum(), pixels_for_grad)[0]
    assert torch.isfinite(gradient).all()
    assert torch.isfinite(explicit_gradient).all()
    assert torch.equal(batch_norm.running_mean, running_mean)
    assert not teacher.training and not teacher.model.training and not batch_norm.training
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in teacher.parameters())
