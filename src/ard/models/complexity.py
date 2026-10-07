"""Hook-based parameter and multiply-accumulate (MAC) counter for student architectures.

Counting convention (the one timm/fvcore-style "GMACs" tables use for these models):

- ``nn.Conv2d``: ``output elements x (in_channels / groups) x kernel area`` per sample;
- ``nn.Linear``: ``output elements x in_features`` per sample (token-wise linears count every token);
- timm ``vision_transformer.Attention``: the two attention matmuls, ``q @ k^T`` and ``attn @ v``,
  ``2 x N^2 x C`` per sample for ``N`` tokens of width ``C``. Its ``qkv``/``proj`` linears are counted
  by the ``nn.Linear`` hook.

Normalization, activations, pooling, residual adds and biases are not counted. A module class that
performs attention but is not hooked fails closed rather than being silently under-counted.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class Complexity:
    params: int
    macs: int


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def count_macs(model: nn.Module, *, image_size: int = 224, in_channels: int = 3) -> int:
    """MACs of one forward pass of a single ``in_channels x image_size x image_size`` image (eval mode)."""
    from timm.models.vision_transformer import Attention as TimmAttention

    total = 0

    def conv_hook(module: nn.Conv2d, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal total
        kernel_area = module.kernel_size[0] * module.kernel_size[1]
        total += output[0].numel() * (module.in_channels // module.groups) * kernel_area

    def linear_hook(module: nn.Linear, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal total
        total += output[0].numel() * module.in_features

    def attention_hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal total
        tokens, width = inputs[0].shape[1], inputs[0].shape[2]
        total += 2 * tokens * tokens * width

    handles = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, TimmAttention):
            handles.append(module.register_forward_hook(attention_hook))
        elif "attention" in type(module).__name__.lower():
            for handle in handles:
                handle.remove()
            raise NotImplementedError(f"MAC counting is not defined for attention module {type(module).__name__}")
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(torch.zeros(1, in_channels, image_size, image_size))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return total


def measure(model: nn.Module, *, image_size: int = 224) -> Complexity:
    return Complexity(params=count_parameters(model), macs=count_macs(model, image_size=image_size))
