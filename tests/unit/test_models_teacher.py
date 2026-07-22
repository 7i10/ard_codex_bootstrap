from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from ard.attacks import AttackRequest, LinfPGD, teacher_input_gradient
from ard.config.schema import AttackConfig, ModelConfig, NormalizationConfig, TeacherConfig
from ard.models import (
    PixelModel,
    TeacherAdapter,
    TeacherMetadata,
    build_architecture,
    build_student,
    build_teacher,
    sha256_file,
)

pytestmark = pytest.mark.t1


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
            teacher_mode="train",
        )
    ).generate(AttackRequest(inputs=inputs, labels=labels, student=student, teacher=teacher))
    gradient = teacher_input_gradient(teacher, inputs, labels)
    assert torch.isfinite(gradient).all()
    assert torch.equal(batch_norm.running_mean, running_mean)
    assert not teacher.training and not teacher.model.training and not batch_norm.training
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in teacher.parameters())
