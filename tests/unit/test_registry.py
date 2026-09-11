"""Plan 0099: ImageNet-scale architectures registered as available, not selected.

These only prove the new entries build and round-trip a forward pass at their
native resolution, unpatched -- see docs/plans/0099-imagenet-stage0-prep.md
checklist item 3. Picking one for an actual training campaign is a separate,
later, human decision (CLAUDE.md rule 7); nothing here asserts a preference
between them.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torchvision.models import MobileNetV2, MobileNetV3, ResNet

from ard.config.schema import ModelConfig, NormalizationConfig
from ard.models import build_architecture, build_student

pytestmark = pytest.mark.t1


@pytest.mark.parametrize(
    ("architecture", "expected_type"),
    [
        ("resnet18_imagenet", ResNet),
        ("resnet50_imagenet", ResNet),
        ("mobilenet_v2_imagenet", MobileNetV2),
        ("mobilenet_v3_small_imagenet", MobileNetV3),
    ],
)
def test_imagenet_architectures_build_the_expected_torchvision_type(
    architecture: str, expected_type: type[nn.Module]
) -> None:
    model = build_architecture(architecture, num_classes=1000)
    assert isinstance(model, expected_type)


@pytest.mark.parametrize("architecture", ["resnet18_imagenet", "resnet50_imagenet"])
def test_resnet_imagenet_is_the_unpatched_native_stem(architecture: str) -> None:
    """The CIFAR entries patch conv1/maxpool because CIFAR is 32px; ImageNet must not be."""
    model = build_architecture(architecture, num_classes=1000)
    assert isinstance(model, ResNet)
    assert model.conv1.kernel_size == (7, 7)
    assert model.conv1.stride == (2, 2)
    assert isinstance(model.maxpool, nn.MaxPool2d)


def test_mobilenet_v2_imagenet_is_the_unpatched_native_stem() -> None:
    model = build_architecture("mobilenet_v2_imagenet", num_classes=1000)
    assert isinstance(model, MobileNetV2)
    assert model.features[0][0].stride == (2, 2)


@pytest.mark.parametrize(
    "architecture",
    ["resnet18_imagenet", "resnet50_imagenet", "mobilenet_v2_imagenet", "mobilenet_v3_small_imagenet"],
)
def test_imagenet_architectures_round_trip_a_native_resolution_forward_pass(architecture: str) -> None:
    model = build_architecture(architecture, num_classes=1000)
    model.eval()
    images = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(images)
    assert logits.shape == (2, 1000)


def test_build_student_wires_an_imagenet_architecture_through_the_normalization_adapter() -> None:
    config = ModelConfig(
        architecture="resnet50_imagenet", num_classes=10, normalization=NormalizationConfig(profile="imagenet_standard")
    )
    student = build_student(config)
    student.eval()
    pixels = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        logits = student(pixels)
    assert logits.shape == (2, 10)
