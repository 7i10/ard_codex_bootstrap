"""Hook-based parameter and multiply-accumulate (MAC) counter for student architectures.

One forward pass of a single ``3 x image_size x image_size`` image in eval mode. Every term counts
whole hooked tensors (not per-sample slices), so models that fold spatial patches into the batch
dimension (MobileViT's unfold -> transformer -> fold) are counted in full. Two numbers are reported:

- ``conv_linear``: ``nn.Conv2d`` (``output elements x in_channels / groups x kernel area``) plus
  ``nn.Linear`` (``output elements x in_features``, i.e. every token). This is the convention of plan
  0103's Phase 1 architecture table (DeiT-Tiny 1.075 G, MobileViT-S 1.417 G at 224).
- ``attention``: the two attention matmuls of timm ``vision_transformer.Attention``, ``q @ k^T`` and
  ``attn @ v``: ``2 x N^2 x C`` per attention sequence for ``N`` tokens of width ``C``, times the
  number of sequences (the hooked input's leading dimension). Its ``qkv``/``proj`` linears are
  already in ``conv_linear``.

``macs = conv_linear + attention``. Normalization, activations, pooling, residual adds and biases are
not counted. A module class that performs attention but is not hooked fails closed rather than being
silently under-counted.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class Complexity:
    params: int
    conv_linear_macs: int
    attention_macs: int

    @property
    def macs(self) -> int:
        return self.conv_linear_macs + self.attention_macs


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def count_mac_terms(model: nn.Module, *, image_size: int = 224, in_channels: int = 3) -> tuple[int, int]:
    """``(conv_linear, attention)`` MACs of one forward pass of a single image (eval mode)."""
    from timm.models.vision_transformer import Attention as TimmAttention

    conv_linear = 0
    attention = 0

    def conv_hook(module: nn.Conv2d, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal conv_linear
        kernel_area = module.kernel_size[0] * module.kernel_size[1]
        conv_linear += output.numel() * (module.in_channels // module.groups) * kernel_area

    def linear_hook(module: nn.Linear, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal conv_linear
        conv_linear += output.numel() * module.in_features

    def attention_hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal attention
        sequences, tokens, width = inputs[0].shape
        attention += sequences * 2 * tokens * tokens * width

    handles = []
    try:
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                handles.append(module.register_forward_hook(conv_hook))
            elif isinstance(module, nn.Linear):
                handles.append(module.register_forward_hook(linear_hook))
            elif isinstance(module, TimmAttention):
                handles.append(module.register_forward_hook(attention_hook))
            elif "attention" in type(module).__name__.lower():
                raise NotImplementedError(f"MAC counting is not defined for attention module {type(module).__name__}")
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                model(torch.zeros(1, in_channels, image_size, image_size))
        finally:
            model.train(was_training)
    finally:
        for handle in handles:
            handle.remove()
    return conv_linear, attention


def count_macs(model: nn.Module, *, image_size: int = 224, in_channels: int = 3) -> int:
    """Total MACs (conv + linear + attention matmuls)."""
    return sum(count_mac_terms(model, image_size=image_size, in_channels=in_channels))


def measure(model: nn.Module, *, image_size: int = 224) -> Complexity:
    conv_linear, attention = count_mac_terms(model, image_size=image_size)
    return Complexity(params=count_parameters(model), conv_linear_macs=conv_linear, attention_macs=attention)
