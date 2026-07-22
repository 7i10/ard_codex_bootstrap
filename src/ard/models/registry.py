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


def build_architecture(architecture: str, num_classes: int) -> nn.Module:
    if architecture == "resnet18_cifar":
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
