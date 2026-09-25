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
        if images.numel():
            # Same condition and tolerance as before, checked on every forward,
            # but as a tensor-side assertion instead of a Python branch on
            # ``.amin()/.amax()``: the old ``bool(tensor)`` forced a GPU->CPU
            # sync per forward and broke torch.compile graphs. ``_assert_async``
            # is traceable (it stays inside the compiled graph for every
            # backend) and fails closed: on CPU it raises ``RuntimeError`` with
            # this message immediately; on CUDA it triggers a device-side
            # assert that surfaces as a ``RuntimeError`` at the next
            # synchronization and leaves the CUDA context unusable, i.e. the
            # process dies. NaN pixels pass, exactly as before (both
            # comparisons are false for NaN).
            detached = images.detach()
            out_of_range = (detached.amin() < -1e-6) | (detached.amax() > 1 + 1e-6)
            torch._assert_async(torch.logical_not(out_of_range), "model adapter expects pixels in [0, 1]")
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
        "efficientnet_b0_imagenet",
        "mobilenetv4_conv_medium_imagenet",
        "convnext_atto_imagenet",
        "deit_tiny_imagenet",
        "mobilevit_s_imagenet",
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
    if architecture == "efficientnet_b0_imagenet":
        # Plan 0103 architecture survey: EfficientNet-B0 (Tan & Le, ICML 2019), 5.29M params / 0.39
        # GMACs; BatchNorm, depthwise-separable, SE blocks, SiLU. `ra_in1k` checkpoint (77.7% @224)
        # chosen over `ra4_e3600` (78.6%) because it uses the standard ImageNet mean/std this
        # project already uses; `ra4` needs mean/std 0.5.
        import timm

        return timm.create_model("efficientnet_b0.ra_in1k", pretrained=pretrained, num_classes=num_classes)
    if architecture == "mobilenetv4_conv_medium_imagenet":
        # Plan 0103 architecture survey: MobileNetV4-Conv-Medium (arXiv:2404.10518), 9.72M params /
        # 0.83 GMACs; same family as mobilenetv4_conv_small_imagenet, so the pair isolates capacity
        # within one architecture family (79.1% @224).
        import timm

        return timm.create_model("mobilenetv4_conv_medium.e500_r224_in1k", pretrained=pretrained, num_classes=num_classes)
    if architecture == "convnext_atto_imagenet":
        # Plan 0103 architecture survey: ConvNeXt (V1) Atto, timm variant of Liu et al. CVPR 2022
        # (arXiv:2201.03545), 3.70M params / 0.55 GMACs; LayerNorm, patchify 4x4 stem, GELU. Purely
        # supervised ImageNet-1k checkpoint (75.7% @224), chosen over ConvNeXt V2-Atto to avoid GRN
        # + FCMAE-pretraining confounds.
        import timm

        return timm.create_model("convnext_atto.d2_in1k", pretrained=pretrained, num_classes=num_classes)
    if architecture == "deit_tiny_imagenet":
        # Plan 0103 architecture survey: DeiT-Tiny / ViT-Ti/16 (Touvron et al. 2021), 5.72M params /
        # 1.07 GMACs; plain ViT, LayerNorm, patch-16 stem, GELU. Non-distilled checkpoint (72.2%
        # @224). The direct miniaturization of the ViT family Singh/Croce/Hein 2023 studied at ViT-S
        # scale.
        import timm

        return timm.create_model("deit_tiny_patch16_224.fb_in1k", pretrained=pretrained, num_classes=num_classes)
    if architecture == "mobilevit_s_imagenet":
        # Plan 0103 architecture survey: MobileViT-S (Mehta & Rastegari, ICLR 2022), 5.58M params /
        # 1.42 GMACs at 224; mobile conv+transformer hybrid (BatchNorm in conv, LayerNorm in
        # transformer blocks). Checkpoint trained at 256px with no mean/std normalization (78.3%
        # @256); used here at 224 for a uniform input size. Same model as Heuillet et al. 2025
        # (arXiv:2508.14079).
        import timm

        return timm.create_model("mobilevit_s.cvnets_in1k", pretrained=pretrained, num_classes=num_classes)
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
