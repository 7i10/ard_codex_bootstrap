"""Plan 0099: ImageNet-scale architectures registered as available, not selected.

These only prove the new entries build and round-trip a forward pass at their
native resolution, unpatched -- see docs/plans/0099-imagenet-stage0-prep.md
checklist item 3. Picking one for an actual training campaign is a separate,
later, human decision (CLAUDE.md rule 7); nothing here asserts a preference
between them.
"""

from __future__ import annotations

import pytest
import timm
import torch
from pydantic import ValidationError
from torch import nn
from torchvision import models as torchvision_models
from torchvision.models import MobileNetV2, MobileNetV3, ResNet

from ard.config.schema import ModelConfig, NormalizationConfig
from ard.models import build_architecture, build_student
from ard.models import registry as registry_module

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


def test_convnext_tiny_imagenet_is_timm_not_torchvision() -> None:
    """Registered for the AutoAttack pipeline validation check (Singh/Croce/Hein 2023, arXiv 2303.01870).

    Their released ConvNeXt-T eps=4/255 checkpoint's state_dict uses timm's
    ConvNeXt naming (stem/stages/blocks/conv_dw/gamma/head.norm/head.fc), not
    torchvision's (features/block/layer_scale/classifier) -- confirmed by
    direct key/shape comparison against the downloaded checkpoint. This test
    locks in the resulting library choice so a future refactor toward
    torchvision (matching every other registered architecture) does not
    silently break loading that external checkpoint.
    """
    import timm

    model = build_architecture("convnext_tiny_imagenet", num_classes=1000)
    assert isinstance(model, timm.models.convnext.ConvNeXt)
    assert not isinstance(model, (ResNet, MobileNetV2, MobileNetV3))
    state_dict_keys = set(model.state_dict().keys())
    assert "stem.0.weight" in state_dict_keys
    assert "stages.0.blocks.0.conv_dw.weight" in state_dict_keys
    assert "stages.0.blocks.0.gamma" in state_dict_keys
    assert "head.fc.weight" in state_dict_keys


def test_mobilenetv4_conv_small_imagenet_is_timm_not_torchvision() -> None:
    """Plan 0101: a modern (2024) mobile-scale replacement for
    mobilenet_v3_small_imagenet -- see registry.py's build_architecture
    comment. Confirmed this session: 3,774,024 params (matches the paper's
    3.8M), plain small-stride conv stem (not a patchify stem -- ConvStem,
    Singh/Croce/Hein 2023, is deliberately not applicable here)."""
    model = build_architecture("mobilenetv4_conv_small_imagenet", num_classes=1000)
    assert not isinstance(model, (ResNet, MobileNetV2, MobileNetV3))
    assert sum(p.numel() for p in model.parameters()) == 3_774_024
    state_dict_keys = set(model.state_dict().keys())
    assert "conv_stem.weight" in state_dict_keys


@pytest.mark.parametrize(
    "architecture",
    [
        "resnet18_imagenet",
        "resnet50_imagenet",
        "mobilenet_v2_imagenet",
        "mobilenet_v3_small_imagenet",
        "convnext_tiny_imagenet",
        "mobilenetv4_conv_small_imagenet",
    ],
)
def test_imagenet_architectures_round_trip_a_native_resolution_forward_pass(architecture: str) -> None:
    model = build_architecture(architecture, num_classes=1000)
    model.eval()
    images = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(images)
    assert logits.shape == (2, 1000)


@pytest.mark.parametrize(
    ("architecture", "constructor_name", "weights_enum"),
    [
        ("resnet18_imagenet", "resnet18", "ResNet18_Weights"),
        ("mobilenet_v3_small_imagenet", "mobilenet_v3_small", "MobileNet_V3_Small_Weights"),
    ],
)
def test_pretrained_false_default_still_passes_weights_none(
    monkeypatch: pytest.MonkeyPatch, architecture: str, constructor_name: str, weights_enum: str
) -> None:
    """Plan 0100: pretrained defaults to False and must reproduce today's
    exact behavior (weights=None) for every config that never mentions it --
    same discipline as plan 0098/0099's own default-preserving fields."""
    captured: dict[str, object] = {}
    real_constructor = getattr(torchvision_models, constructor_name)

    def spy(*, weights, num_classes):
        captured["weights"] = weights
        return real_constructor(weights=None, num_classes=num_classes)

    monkeypatch.setattr(registry_module.models, constructor_name, spy)
    build_architecture(architecture, num_classes=1000)
    assert captured["weights"] is None


def test_mobilenetv4_pretrained_false_default_still_passes_pretrained_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """timm counterpart of test_pretrained_false_default_still_passes_weights_none
    above (scientific review P2-4): a hardcoded pretrained=True in the
    registry branch would otherwise be caught by nothing, since every
    other test that reaches this architecture either doesn't set
    pretrained=True or mocks the download away."""
    captured: dict[str, object] = {}
    real_create_model = timm.create_model

    def spy(name: str, *, pretrained: bool, num_classes: int):
        captured["pretrained"] = pretrained
        return real_create_model(name, pretrained=False, num_classes=num_classes)

    monkeypatch.setattr(timm, "create_model", spy)
    build_architecture("mobilenetv4_conv_small_imagenet", num_classes=1000)
    assert captured["pretrained"] is False


@pytest.mark.parametrize(
    ("architecture", "constructor_name", "weights_enum"),
    [
        ("resnet18_imagenet", "resnet18", "ResNet18_Weights"),
        ("mobilenet_v3_small_imagenet", "mobilenet_v3_small", "MobileNet_V3_Small_Weights"),
    ],
)
def test_pretrained_true_requests_the_torchvision_imagenet1k_weights(
    monkeypatch: pytest.MonkeyPatch, architecture: str, constructor_name: str, weights_enum: str
) -> None:
    """No real download in a unit test: substitute a cheap weights=None
    construction whenever pretrained weights are requested, and assert only
    that our own wiring asked for the right weights enum member."""
    captured: dict[str, object] = {}
    real_constructor = getattr(torchvision_models, constructor_name)
    expected_weights = getattr(torchvision_models, weights_enum).IMAGENET1K_V1

    def spy(*, weights):
        captured["weights"] = weights
        return real_constructor(weights=None)

    monkeypatch.setattr(registry_module.models, constructor_name, spy)
    model = build_architecture(architecture, num_classes=1000, pretrained=True)
    assert captured["weights"] is expected_weights
    assert isinstance(model, ResNet if architecture == "resnet18_imagenet" else MobileNetV3)


def test_mobilenetv4_pretrained_true_requests_timm_pretrained_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """No real download in a unit test: substitute a cheap pretrained=False
    construction, and assert only that our own wiring asked timm for the
    right checkpoint tag and pretrained=True -- mirroring
    test_pretrained_true_requests_the_torchvision_imagenet1k_weights's
    torchvision-side pattern, since mobilenetv4_conv_small_imagenet goes
    through timm.create_model instead (registry.py's build_architecture
    comment: timm's own pretrained=True already replaces the head
    correctly, so unlike the two torchvision entries above there is no
    separate _with_replaced_head call to spy on)."""
    captured: dict[str, object] = {}
    real_create_model = timm.create_model

    def spy(name: str, *, pretrained: bool, num_classes: int):
        captured["name"] = name
        captured["pretrained"] = pretrained
        captured["num_classes"] = num_classes
        return real_create_model(name, pretrained=False, num_classes=num_classes)

    monkeypatch.setattr(timm, "create_model", spy)
    model = build_architecture("mobilenetv4_conv_small_imagenet", num_classes=1000, pretrained=True)
    assert captured == {"name": "mobilenetv4_conv_small.e1200_r224_in1k", "pretrained": True, "num_classes": 1000}
    assert not isinstance(model, (ResNet, MobileNetV2, MobileNetV3))


def test_mobilenetv4_pretrained_true_replaces_the_head_for_a_non_1000_class_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No real download in a unit test (scientific review P2-4 -- every
    other test in this module avoids one; a t1 tier test must not depend on
    network access or a warm cache): substitute a cheap pretrained=False
    construction, and assert only that timm's own create_model(pretrained=True,
    num_classes=N) is asked to replace the head in one call, unlike the
    torchvision entries' separate _with_replaced_head."""
    real_create_model = timm.create_model

    def spy(name: str, *, pretrained: bool, num_classes: int):
        del pretrained
        return real_create_model(name, pretrained=False, num_classes=num_classes)

    monkeypatch.setattr(timm, "create_model", spy)
    model = build_architecture("mobilenetv4_conv_small_imagenet", num_classes=5, pretrained=True)
    model.eval()
    with torch.no_grad():
        logits = model(torch.rand(2, 3, 224, 224))
    assert logits.shape == (2, 5)


@pytest.mark.parametrize(
    ("architecture", "constructor_name"),
    [("resnet18_imagenet", "resnet18"), ("mobilenet_v3_small_imagenet", "mobilenet_v3_small")],
)
def test_pretrained_true_replaces_the_head_for_a_non_1000_class_config(
    monkeypatch: pytest.MonkeyPatch, architecture: str, constructor_name: str
) -> None:
    real_constructor = getattr(torchvision_models, constructor_name)
    monkeypatch.setattr(registry_module.models, constructor_name, lambda *, weights: real_constructor(weights=None))

    model = build_architecture(architecture, num_classes=5, pretrained=True)
    model.eval()
    with torch.no_grad():
        logits = model(torch.rand(2, 3, 224, 224))
    assert logits.shape == (2, 5)


@pytest.mark.parametrize("architecture", ["resnet18_imagenet", "mobilenet_v3_small_imagenet"])
def test_pretrained_false_at_a_non_1000_class_count_does_not_consume_extra_rng_or_change_initial_weights(
    architecture: str,
) -> None:
    """Regression test for a real bug scientific review found in plan
    0100's first draft: _with_replaced_head was called on BOTH branches, so
    even pretrained=False (the default) built a second nn.Linear and
    silently consumed extra global RNG draws whenever num_classes != 1000 --
    breaking the "bit-identical to before" guarantee this project's own
    convention requires for a new default-False field."""
    torch.manual_seed(0)
    baseline = build_architecture(architecture, num_classes=5)
    fc_name = "fc" if architecture == "resnet18_imagenet" else "classifier"
    baseline_head = getattr(baseline, fc_name)[-1] if fc_name == "classifier" else baseline.fc

    torch.manual_seed(0)
    current = build_architecture(architecture, num_classes=5, pretrained=False)
    current_head = getattr(current, fc_name)[-1] if fc_name == "classifier" else current.fc

    assert torch.equal(baseline_head.weight, current_head.weight)
    assert torch.equal(baseline_head.bias, current_head.bias)


@pytest.mark.parametrize("architecture", ["resnet50_imagenet", "mobilenet_v2_imagenet", "convnext_tiny_imagenet"])
def test_pretrained_true_is_rejected_for_unsupported_architectures(architecture: str) -> None:
    with pytest.raises(ValueError, match="pretrained=True is not supported"):
        build_architecture(architecture, num_classes=1000, pretrained=True)


def test_model_config_rejects_pretrained_true_for_unsupported_architectures() -> None:
    with pytest.raises(ValidationError, match="pretrained=True is not supported"):
        ModelConfig(architecture="resnet50_imagenet", num_classes=1000, pretrained=True)


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
