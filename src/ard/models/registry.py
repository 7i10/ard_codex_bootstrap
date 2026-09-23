"""Named model variants with normalization explicitly inside their adapters."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from ard.config.schema import ModelConfig, NormalizationConfig


class PixelNormalization(nn.Module):
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, config: NormalizationConfig) -> None:
        super().__init__()
        self.input_domain = config.input_domain
        self.register_buffer("mean", torch.tensor(config.mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(config.std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if not images.is_floating_point():
            raise TypeError("model inputs must be floating-point pixels")
        if images.numel() and (images.detach().amin() < -1e-6 or images.detach().amax() > 1 + 1e-6):
            raise ValueError("model adapter expects pixels in [0, 1]")
        return (images - self.mean.to(images)) / self.std.to(images)


class PixelModel(nn.Module):
    def __init__(self, model: nn.Module, normalization: NormalizationConfig) -> None:
        super().__init__()
        self.normalization = PixelNormalization(normalization)
        self.model = model

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.model(self.normalization(pixels))


class FixtureCNN(nn.Module):
    """Small deterministic architecture restricted to dev/smoke construction paths."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Linear(8, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs).flatten(1))


class SAADBasicBlock(nn.Module):
    """Post-activation basic block specified independently of SAAD source."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, *, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut: nn.Sequential
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        identity = inputs
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        identity = self.shortcut(inputs)
        return self.relu(outputs + identity)


class SAADResNet18CIFAR(nn.Module):
    """Clean-room CIFAR ResNet-18 structural implementation.

    This implementation follows only the architecture specification recorded
    in the protocol; it does not import or reproduce `.external/saad` code.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.in_planes = 64
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=4)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes: int, *, blocks: int, stride: int) -> nn.Sequential:
        layers: list[nn.Module] = [SAADBasicBlock(self.in_planes, planes, stride=stride)]
        self.in_planes = planes
        layers.extend(SAADBasicBlock(self.in_planes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.layer4(outputs)
        outputs = self.avgpool(outputs)
        return self.linear(torch.flatten(outputs, 1))


def _with_replaced_head(model: nn.Module, *, architecture: str, num_classes: int) -> nn.Module:
    """A pretrained torchvision model's final layer is fixed at 1000 classes
    (ImageNet-1k). Standard transfer-learning practice: keep the pretrained
    backbone, replace only the final linear layer when a different class
    count is actually requested (a no-op for this project's real ImageNet-1k
    campaign, needed only for dev/test-scale configs)."""
    if num_classes == 1000:
        return model
    if architecture == "resnet18_imagenet":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if architecture == "mobilenet_v3_small_imagenet":
        final = model.classifier[-1]
        model.classifier[-1] = nn.Linear(final.in_features, num_classes)
        return model
    raise ValueError(f"pretrained head replacement is not defined for architecture: {architecture}")


def build_architecture(architecture: str, num_classes: int, *, pretrained: bool = False) -> nn.Module:
    if pretrained and architecture not in {
        "resnet18_imagenet",
        "mobilenet_v3_small_imagenet",
        "mobilenetv4_conv_small_imagenet",
        "convnextv2_atto_imagenet",
        "xcit_nano_imagenet",
        "ghostnetv2_imagenet",
    }:
        raise ValueError(f"pretrained=True is not supported for architecture: {architecture}")
    if architecture == "saad_resnet18_cifar_v1":
        return SAADResNet18CIFAR(num_classes)
    if architecture in {"torchvision_resnet18_cifar_norm_v1", "resnet18_cifar"}:
        model = models.resnet18(weights=None, num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model
    if architecture == "mobilenet_v2_cifar":
        model = models.mobilenet_v2(weights=None, num_classes=num_classes)
        model.features[0][0].stride = (1, 1)
        return model
    if architecture == "fixture_cnn":
        return FixtureCNN(num_classes)
    if architecture == "resnet18_imagenet":
        # Plain, unpatched torchvision definition -- see resnet50_imagenet's
        # comment. Added post-plan-0099 to match the ADR paper's own
        # ResNet-18 (~11.2M params) as the "larger" reference point
        # docs/MOBILE_ROBUSTNESS_METHOD_PROPOSAL.md's Stage 0 recipe assumes,
        # distinct from resnet50_imagenet (~25.6M params).
        # Plan 0100: pretrained (non-robust ImageNet-1k) initialization,
        # per Singh/Croce/Hein 2023's own recipe -- torchvision's standard
        # weights are an explicitly-sanctioned in-kind substitute (see plan
        # 0100's Recipe table). weights=None (the default) is unaffected.
        if pretrained:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            return _with_replaced_head(model, architecture=architecture, num_classes=num_classes)
        return models.resnet18(weights=None, num_classes=num_classes)
    if architecture == "resnet50_imagenet":
        # Plain, unpatched torchvision definition: the CIFAR entries above
        # patch conv1/maxpool specifically because CIFAR is 32px; ImageNet's
        # native resolution needs the untouched stem. See plan 0099 checklist
        # item 3.
        return models.resnet50(weights=None, num_classes=num_classes)
    if architecture == "mobilenet_v2_imagenet":
        return models.mobilenet_v2(weights=None, num_classes=num_classes)
    if architecture == "mobilenet_v3_small_imagenet":
        # Plan 0100: pretrained initialization, see resnet18_imagenet's comment.
        if pretrained:
            model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            return _with_replaced_head(model, architecture=architecture, num_classes=num_classes)
        return models.mobilenet_v3_small(weights=None, num_classes=num_classes)
    if architecture == "mobilenetv4_conv_small_imagenet":
        # Plan 0101: MobileNetV4-Conv-Small (Howard et al./Google,
        # arXiv:2404.10518, 2024), 3.77M params -- a modern (2024) recipe
        # replacement for mobilenet_v3_small_imagenet's 2019-era checkpoint,
        # adopted specifically to raise the achievable clean-accuracy
        # ceiling before adversarial fine-tuning (plan 0100 found the old
        # checkpoint's post-AT clean accuracy, ~46%, well below what a
        # modern mobile-scale recipe should support). Unlike
        # resnet18_imagenet/mobilenet_v3_small_imagenet (torchvision, needs
        # _with_replaced_head), timm's own create_model(..., pretrained=True,
        # num_classes=N) already replaces the classifier head correctly for
        # N != 1000 -- no separate head-replacement branch needed. Same
        # lazy-import precedent as convnext_tiny_imagenet below (timm is
        # already a pinned dependency, not previously imported for a
        # *pretrained* architecture from src/ard).
        import timm

        return timm.create_model(
            "mobilenetv4_conv_small.e1200_r224_in1k", pretrained=pretrained, num_classes=num_classes
        )
    if architecture == "convnextv2_atto_imagenet":
        # Plan 0103: ConvNeXt V2-Atto (Woo et al./Meta-KAIST, arXiv:2301.00808,
        # 2023), 3.71M params -- a LayerNorm-based lightweight architecture,
        # a genuine miniaturization of the same ConvNeXt family
        # Singh/Croce/Hein 2023 validated at ConvNeXt-T scale (~28M), added
        # to test whether their architecture-family findings scale down.
        # timm ships only the FCMAE-pretrained-then-ImageNet-1k-finetuned
        # checkpoint at this scale (no pure supervised-from-scratch
        # checkpoint exists in timm for Atto) -- disclosed in plan 0103,
        # judged acceptable since this project uses the checkpoint as an
        # off-the-shelf non-robust classifier init, not as a training-recipe
        # claim. Same lazy-import, no-head-replacement pattern as
        # mobilenetv4_conv_small_imagenet above.
        import timm

        return timm.create_model("convnextv2_atto.fcmae_ft_in1k", pretrained=pretrained, num_classes=num_classes)
    if architecture == "xcit_nano_imagenet":
        # Plan 0103: XCiT-Nano-12/p16 (El-Nouby et al./Meta, arXiv:2106.09681,
        # NeurIPS 2021), 3.05M params -- a LayerNorm-based linear
        # (cross-covariance) attention architecture, still widely cited as a
        # baseline in 2023-2025 hybrid-attention papers, and the same
        # architecture family Debenedetti et al. 2022 used for their own
        # small-ViT ImageNet-100 ablations (this project targets full
        # ImageNet-1k instead). Uses the non-distilled checkpoint
        # (`.fb_in1k`, not `.fb_dist_in1k`) to avoid a knowledge-distillation
        # confound in the pretrained init. Same lazy-import,
        # no-head-replacement pattern as mobilenetv4_conv_small_imagenet.
        import timm

        return timm.create_model("xcit_nano_12_p16_224.fb_in1k", pretrained=pretrained, num_classes=num_classes)
    if architecture == "ghostnetv2_imagenet":
        # Plan 0103: GhostNetV2 1.0x (Tang et al./Huawei Noah's Ark Lab,
        # arXiv:2211.12905, NeurIPS 2022), 6.16M params -- a BatchNorm-based
        # architecture whose "cheap operation" (Ghost module) design is
        # architecturally distinct from the depthwise-separable-conv family
        # (MobileNet/EfficientNet lineage) already represented in this
        # project's cohort, diversifying the BatchNorm-CNN comparison point
        # rather than duplicating it. Same lazy-import, no-head-replacement
        # pattern as mobilenetv4_conv_small_imagenet.
        import timm

        return timm.create_model("ghostnetv2_100.in1k", pretrained=pretrained, num_classes=num_classes)
    if architecture == "convnext_tiny_imagenet":
        # NOT torchvision.models.convnext_tiny. Registered to validate this
        # project's AutoAttack evaluation pipeline at ImageNet scale against
        # a known published result: the Singh/Croce/Hein 2023 ("Revisiting
        # Adversarial Training for ImageNet", arXiv 2303.01870) reference
        # ConvNeXt-T eps=4/255 checkpoint. Direct state_dict key/shape
        # comparison confirmed that checkpoint is built on timm's ConvNeXt
        # (`stem`/`stages`/`blocks`/`conv_dw`/`gamma`/`head.norm`/`head.fc`
        # naming), not torchvision's (`features`/`block`/`layer_scale`/
        # `classifier` naming) -- the two are architecturally equivalent
        # (plain "original" PatchStem ConvNeXt-Tiny, not their ConvStem
        # "CvSt" variant) but not state_dict-compatible. timm is already a
        # pinned project dependency (requirements/environment.lock, used by
        # .external/robustbench) but not previously imported directly from
        # src/ard; imported lazily here rather than at module level since
        # every other architecture in this file only needs torch/torchvision.
        import timm

        return timm.create_model("convnext_tiny", pretrained=False, num_classes=num_classes)
    raise ValueError(f"unknown architecture: {architecture}")


def build_student(config: ModelConfig, *, tier: str = "dev") -> PixelModel:
    if config.architecture == "fixture_cnn" and tier not in {"dev", "smoke"}:
        raise ValueError("fixture_cnn is restricted to dev/smoke tiers")
    model = build_architecture(config.architecture, config.num_classes, pretrained=config.pretrained)
    return PixelModel(model, config.normalization)
