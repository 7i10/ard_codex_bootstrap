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


def build_architecture(architecture: str, num_classes: int) -> nn.Module:
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
    raise ValueError(f"unknown architecture: {architecture}")


def build_student(config: ModelConfig, *, tier: str = "dev") -> PixelModel:
    if config.architecture == "fixture_cnn" and tier not in {"dev", "smoke"}:
        raise ValueError("fixture_cnn is restricted to dev/smoke tiers")
    return PixelModel(build_architecture(config.architecture, config.num_classes), config.normalization)
