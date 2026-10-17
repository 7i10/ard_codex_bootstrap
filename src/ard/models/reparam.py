"""Structural reparameterization of MobileNetV4-Conv-S (plan 0107 method candidate 3, human-approved 2026-10-18).

Training-time students carry extra *linear* branches (RepVGG arXiv:2101.03697, MobileOne arXiv:2206.04040)
that are folded away after training, so the deployed network is exactly the plain architecture. Design and
rationale: ``docs/plans/0107-train-time-capacity-deploy-plain.md``, "Design: structural reparameterization".

A *position* is a conv + BatchNorm whose BatchNorm is followed by a ReLU (in timm the ReLU lives inside
``BatchNormAct2d.act``). At each position :class:`ReparamConvBnAct` replaces the conv, the BatchNormAct2d
becomes ``nn.Identity``, and the module computes::

    act( BN0(conv0 x) + sum_i BN_i(conv_i x) + BN_s(conv1x1 x) + BN_id(x) )

In eval mode every BN is a fixed affine map, so the sum is one conv with bias (:func:`fold_student`), which
is written into the plain model's conv weight and ``BatchNormAct2d`` (weight 1, the folded bias, running mean 0,
running variance ``1 - eps``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from torch import nn

from ard.config.schema import ModelConfig

if TYPE_CHECKING:
    from timm.layers import BatchNormAct2d

FOLD_CONTRACT = "ard.reparam.fold.v1"


@dataclass(frozen=True)
class ReparamSpec:
    deployed_architecture: str
    extra_branches: int


# Training-time student id -> the plain architecture its folded weights load into. One id per
# over-parameterization factor, each with its own config (the project's one-id-one-config convention).
REPARAM_ARCHITECTURES: dict[str, ReparamSpec] = {
    "mobilenetv4_conv_small_reparam_imagenet": ReparamSpec("mobilenetv4_conv_small_imagenet", extra_branches=1),
}


class ConvBN(nn.Module):
    """One branch: bias-free conv followed by a plain BatchNorm (no activation)."""

    def __init__(self, conv: nn.Conv2d, bn: nn.BatchNorm2d) -> None:
        super().__init__()
        self.conv = conv
        self.bn = bn

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(inputs))


def _check_foldable(conv: nn.Conv2d) -> None:
    kernel_h, kernel_w = conv.kernel_size
    problems = []
    if type(conv) is not nn.Conv2d:
        problems.append(f"type {type(conv).__name__} (only plain nn.Conv2d is folded)")
    if kernel_h != kernel_w or kernel_h % 2 != 1:
        problems.append(f"kernel {conv.kernel_size} (odd square kernels only)")
    if conv.padding != (kernel_h // 2, kernel_w // 2):
        problems.append(f"padding {conv.padding} (symmetric k // 2 only)")
    if conv.dilation != (1, 1) or conv.padding_mode != "zeros" or conv.bias is not None:
        problems.append("dilation / padding mode / bias")
    if problems:
        raise ValueError("position cannot be folded exactly: " + "; ".join(problems))


def _new_conv(template: nn.Conv2d, kernel_size: int) -> nn.Conv2d:
    from timm.models._efficientnet_builder import _init_weight_goog

    conv = nn.Conv2d(
        template.in_channels,
        template.out_channels,
        kernel_size,
        stride=cast(tuple[int, int], template.stride),
        padding=kernel_size // 2,
        groups=template.groups,
        bias=False,
    )
    _init_weight_goog(conv)
    return conv


def _new_bn(template: nn.BatchNorm2d, channels: int) -> nn.BatchNorm2d:
    # nn.BatchNorm2d's own init is weight 1, bias 0, running mean 0, running var 1 (= timm's goog init).
    return nn.BatchNorm2d(channels, eps=template.eps, momentum=template.momentum)


class ReparamConvBnAct(nn.Module):
    """Training-time multi-branch replacement of one conv + ``BatchNormAct2d`` (ReLU) position.

    ``conv``/``bn`` are the original branch (the plain model's conv and a plain-BN copy of its
    ``BatchNormAct2d`` state); ``branches`` are the extra k x k conv+BN branches; ``scale`` is the 1x1
    conv+BN branch (k > 1 only); ``identity`` is the BN-only branch (in = out, stride 1 only). ``drop`` and
    ``act`` are the original ``BatchNormAct2d``'s modules, applied once after the sum.
    """

    def __init__(
        self, conv: nn.Conv2d, norm_act: BatchNormAct2d, *, extra_branches: int, deploy_norm_name: str
    ) -> None:
        super().__init__()
        from timm.layers import BatchNormAct2d

        if extra_branches < 0:
            raise ValueError("extra_branches must be >= 0")
        if type(norm_act) is not BatchNormAct2d or type(norm_act.act) is not nn.ReLU:
            raise ValueError("a reparameterization position must be conv + BatchNormAct2d with a ReLU activation")
        _check_foldable(conv)
        kernel = conv.kernel_size[0]
        self.deploy_norm_name = deploy_norm_name
        self.conv = conv
        self.bn = nn.BatchNorm2d(norm_act.num_features, eps=norm_act.eps, momentum=norm_act.momentum)
        self.bn.load_state_dict(norm_act.state_dict(), strict=True)
        self.branches: nn.ModuleList = nn.ModuleList(
            ConvBN(_new_conv(conv, kernel), _new_bn(norm_act, conv.out_channels)) for _ in range(extra_branches)
        )
        self.scale: ConvBN | None = (
            ConvBN(_new_conv(conv, 1), _new_bn(norm_act, conv.out_channels)) if kernel > 1 else None
        )
        has_identity = conv.in_channels == conv.out_channels and conv.stride == (1, 1)
        self.identity: nn.BatchNorm2d | None = _new_bn(norm_act, conv.out_channels) if has_identity else None
        self.drop: nn.Module = norm_act.drop
        self.act: nn.Module = norm_act.act

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.bn(self.conv(inputs))
        for branch in self.branches:
            outputs = outputs + branch(inputs)
        if self.scale is not None:
            outputs = outputs + self.scale(inputs)
        if self.identity is not None:
            outputs = outputs + self.identity(inputs)
        return self.act(self.drop(outputs))

    def _identity_kernel(self) -> torch.Tensor:
        conv = self.conv
        kernel = conv.kernel_size[0]
        per_group = conv.in_channels // conv.groups
        weight = torch.zeros(conv.out_channels, per_group, kernel, kernel, dtype=torch.float64)
        for channel in range(conv.out_channels):
            weight[channel, channel % per_group, kernel // 2, kernel // 2] = 1.0
        return weight

    @torch.no_grad()
    def folded_kernel_and_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The single conv kernel and bias equal to this module's eval-mode pre-activation (float64)."""
        kernel = self.conv.kernel_size[0]
        pad = kernel // 2

        def fuse(weight: torch.Tensor, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
            assert bn.running_mean is not None and bn.running_var is not None
            scale = bn.weight.double() / torch.sqrt(bn.running_var.double() + bn.eps)
            return weight.double() * scale.view(-1, 1, 1, 1), bn.bias.double() - bn.running_mean.double() * scale

        total_weight, total_bias = fuse(self.conv.weight, self.bn)
        parts = [fuse(branch.conv.weight, branch.bn) for branch in cast(list[ConvBN], list(self.branches))]
        if self.scale is not None:
            weight, bias = fuse(self.scale.conv.weight, self.scale.bn)
            parts.append((F.pad(weight, [pad, pad, pad, pad]), bias))
        if self.identity is not None:
            parts.append(fuse(self._identity_kernel(), self.identity))
        for weight, bias in parts:
            total_weight = total_weight + weight
            total_bias = total_bias + bias
        return total_weight, total_bias


def _mobilenetv4_positions(model: nn.Module) -> list[tuple[str, str, str]]:
    """``(parent path, conv attribute, norm attribute)`` for every conv + ReLU-BatchNormAct2d position."""
    from timm.layers import BatchNormAct2d

    positions = []
    for parent_name, parent in model.named_modules():
        for conv_name, norm_name in (("conv_stem", "bn1"), ("conv_head", "norm_head"), ("conv", "bn1"), ("conv", "bn")):
            conv, norm = getattr(parent, conv_name, None), getattr(parent, norm_name, None)
            if isinstance(conv, nn.Conv2d) and isinstance(norm, BatchNormAct2d) and type(norm.act) is nn.ReLU:
                positions.append((parent_name, conv_name, norm_name))
    return positions


def reparameterize_(model: nn.Module, *, extra_branches: int) -> int:
    """Replace every conv + ReLU-BatchNormAct2d position in place; return the number of positions."""
    from timm.layers import BatchNormAct2d

    positions = _mobilenetv4_positions(model)
    for parent_name, conv_name, norm_name in positions:
        parent = model.get_submodule(parent_name) if parent_name else model
        block = ReparamConvBnAct(
            getattr(parent, conv_name),
            getattr(parent, norm_name),
            extra_branches=extra_branches,
            deploy_norm_name=norm_name,
        )
        setattr(parent, conv_name, block)
        setattr(parent, norm_name, nn.Identity())
    # Fail closed: no ReLU-gated BatchNormAct2d may be left unconverted (a timm layout change would show here).
    leftover = [
        name
        for name, module in model.named_modules()
        if isinstance(module, BatchNormAct2d) and type(module.act) is not nn.Identity
    ]
    if leftover:
        raise ValueError(f"activation-gated BatchNormAct2d positions were not reparameterized: {leftover}")
    return len(positions)


def build_mobilenetv4_conv_small_reparam(*, num_classes: int, extra_branches: int = 1) -> nn.Module:
    """The plain ``mobilenetv4_conv_small_imagenet`` network (built first, so the original branch has the
    plain model's seeded init), then reparameterized in place (extra-branch init draws come after)."""
    from ard.models.registry import build_architecture

    model = build_architecture("mobilenetv4_conv_small_imagenet", num_classes)
    reparameterize_(model, extra_branches=extra_branches)
    return model


def build_reparam_architecture(architecture: str, num_classes: int) -> nn.Module:
    spec = REPARAM_ARCHITECTURES[architecture]
    if spec.deployed_architecture != "mobilenetv4_conv_small_imagenet":  # pragma: no cover - one family so far
        raise ValueError(f"no reparameterization builder for {spec.deployed_architecture}")
    return build_mobilenetv4_conv_small_reparam(num_classes=num_classes, extra_branches=spec.extra_branches)


def folded_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """State dict of the plain network that computes ``model``'s eval-mode function.

    Works on any module containing :class:`ReparamConvBnAct` blocks (the bare timm network or the
    ``PixelModel`` student wrapper). Keys outside the reparameterized blocks are copied unchanged.
    """
    state = model.state_dict()
    folded: dict[str, torch.Tensor] = {}
    blocks = {name: module for name, module in model.named_modules() if isinstance(module, ReparamConvBnAct)}
    if not blocks:
        raise ValueError("model has no reparameterized positions to fold")
    for key, value in state.items():
        if not any(key.startswith(name + ".") for name in blocks):
            folded[key] = value.detach().clone()
    for name, block in blocks.items():
        parent = name.rsplit(".", 1)[0] + "." if "." in name else ""
        norm = f"{parent}{block.deploy_norm_name}."
        weight, bias = block.folded_kernel_and_bias()
        dtype = block.conv.weight.dtype
        folded[f"{name}.weight"] = weight.to(dtype)
        channels = bias.numel()
        eps = block.bn.eps
        folded[f"{norm}weight"] = torch.ones(channels, dtype=dtype)
        folded[f"{norm}bias"] = bias.to(dtype)
        folded[f"{norm}running_mean"] = torch.zeros(channels, dtype=dtype)
        folded[f"{norm}running_var"] = torch.full((channels,), 1.0 - eps, dtype=dtype)
        tracked = block.bn.num_batches_tracked
        assert tracked is not None
        folded[f"{norm}num_batches_tracked"] = tracked.detach().clone()
    return folded


def deployed_student_config(student_config: ModelConfig) -> ModelConfig:
    """The plain-architecture ``ModelConfig`` a reparameterized student config deploys as."""
    spec = REPARAM_ARCHITECTURES[student_config.architecture]
    return ModelConfig.model_validate({**student_config.model_dump(), "architecture": spec.deployed_architecture})


def fold_student(student: nn.Module, student_config: ModelConfig) -> nn.Module:
    """A new plain-architecture student (``PixelModel``) holding the folded weights of ``student``."""
    from ard.models.registry import build_student

    with torch.random.fork_rng(devices=[]):  # building the plain model must not move the caller's RNG
        plain = build_student(deployed_student_config(student_config), tier="dev")
    plain.load_state_dict(folded_state_dict(student), strict=True)
    return plain.to(next(student.parameters()).device)


def fold_student_state_dict(
    state: Mapping[str, torch.Tensor], student_config: ModelConfig, *, tier: str = "dev"
) -> dict[str, torch.Tensor]:
    """Fold a saved training-time student ``state_dict`` into the plain architecture's ``state_dict``."""
    from ard.models.registry import build_student

    with torch.random.fork_rng(devices=[]):
        training = build_student(student_config, tier=tier)
    training.load_state_dict(state, strict=True)
    return fold_student(training, student_config).state_dict()


def is_reparam_architecture(architecture: str) -> bool:
    return architecture in REPARAM_ARCHITECTURES


__all__ = [
    "FOLD_CONTRACT",
    "REPARAM_ARCHITECTURES",
    "ConvBN",
    "ReparamConvBnAct",
    "ReparamSpec",
    "build_mobilenetv4_conv_small_reparam",
    "build_reparam_architecture",
    "deployed_student_config",
    "fold_student",
    "fold_student_state_dict",
    "folded_state_dict",
    "is_reparam_architecture",
    "reparameterize_",
]
