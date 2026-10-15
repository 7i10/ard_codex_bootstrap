"""ReLU forward with a smooth (parametric-softplus) backward, for the training updates only.

Xie, Tan, Gong, Yuille, Le 2020, "Smooth Adversarial Training" (arXiv:2006.14536), Sec. 3 ("improving
gradient quality"): keep ReLU in every forward pass, but in selected backward passes replace ReLU's
derivative ``1[x > 0]`` by the derivative of their parametric softplus (Eqn. 2)

    f(alpha, x) = (1 / alpha) * log(1 + exp(alpha * x)),   d/dx f(alpha, x) = 1 / (1 + exp(-alpha x))  (Eqn. 3)
                                                                          = sigmoid(alpha * x),

with alpha = 10 in the paper. ``beta`` below is that alpha (the name PyTorch's ``nn.Softplus`` uses for the
same parameter). Human-approved 2026-10-15 (plan 0103 Phase 2); config: ``training.relu_surrogate_gradient``.

Mechanism. :func:`install_relu_surrogate` replaces every ``nn.ReLU`` module of the student with a
:class:`SurrogateGradientReLU` bound to one :class:`ReluSurrogateGradient` switch. The forward is
``torch.relu`` in every case, so outputs are bitwise ReLU's. The switch is read at BACKWARD time: while it
is off, the backward is ``threshold_backward(grad, x, 0)`` -- exactly the formula PyTorch's own ReLU
backward uses (``x <= 0`` iff ``relu(x) <= 0``, NaN included), so every true gradient is bitwise
nn.ReLU's; while it is on (inside :meth:`ReluSurrogateGradient.active`), the upstream gradient is
multiplied by ``sigmoid(beta * x)`` at the ReLU input ``x``. The Trainer turns the switch on only around
the training loss's ``backward()`` and, for ``scope=attack_and_optimizer``, around the training-attack
call; every other gradient (selection / validation / probe / held-out attacks, AWP's weight-ascent
direction) is computed with the switch off. It is never tied to ``train()`` / ``eval()``: the training
attack runs the student in eval mode.

ReLU has no parameters or buffers, so the replacement leaves ``state_dict`` unchanged and checkpoints load
into the plain architecture; ``ard.cli.evaluate`` builds the plain ReLU model and never sees this module.
In-place ReLUs become out-of-place when gradients are recorded (the input is kept for the backward), so
activation memory grows by one tensor per ReLU; values are identical.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class ReluSurrogateGradient:
    """The on/off switch shared by every installed :class:`SurrogateGradientReLU` of one student.

    A plain attribute, not thread-local: the CUDA autograd engine runs backward functions on its own
    device threads, which must see the state the calling thread set.
    """

    def __init__(self, beta: float) -> None:
        beta = float(beta)
        if not (beta > 0 and torch.isfinite(torch.tensor(beta))):
            raise ValueError("relu surrogate gradient beta must be finite and positive")
        self.beta = beta
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    @contextmanager
    def active(self) -> Iterator[None]:
        """Use the softplus derivative in every backward run inside this block (restores the previous state)."""
        previous = self._active
        self._active = True
        try:
            yield
        finally:
            self._active = previous


class _SurrogateReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, switch: ReluSurrogateGradient) -> torch.Tensor:
        ctx.save_for_backward(inputs)
        ctx.switch = switch
        return torch.relu(inputs)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (inputs,) = ctx.saved_tensors
        switch: ReluSurrogateGradient = ctx.switch
        if switch.is_active:
            # SAT Eqn. 3: d/dx (1/beta) log(1 + exp(beta x)) = sigmoid(beta x).
            return grad_output * torch.sigmoid(switch.beta * inputs), None
        # nn.ReLU's own derivative formula (PyTorch derivatives.yaml: threshold_backward(grad, result, 0)).
        return torch.ops.aten.threshold_backward(grad_output, inputs, 0), None


class SurrogateGradientReLU(nn.Module):
    """Drop-in for ``nn.ReLU``: ReLU forward; true or softplus-derivative backward per its switch."""

    def __init__(self, switch: ReluSurrogateGradient, *, inplace: bool = False) -> None:
        super().__init__()
        self.switch = switch
        self.inplace = inplace

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled() and inputs.requires_grad:
            return _SurrogateReLUFunction.apply(inputs, self.switch)
        # Nothing is recorded, so no backward can follow: exactly nn.ReLU.
        return F.relu(inputs, inplace=self.inplace)

    def extra_repr(self) -> str:
        return f"beta={self.switch.beta}" + (", inplace=True" if self.inplace else "")


def install_relu_surrogate(model: nn.Module, switch: ReluSurrogateGradient) -> int:
    """Replace every ``nn.ReLU`` child module (exact type) of ``model`` in place; return how many.

    Refuses a model with no ``nn.ReLU``: the option would then silently do nothing.
    """
    replaced = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if type(child) is nn.ReLU:
                setattr(parent, name, SurrogateGradientReLU(switch, inplace=child.inplace))
                replaced += 1
    if replaced == 0:
        raise ValueError(
            f"training.relu_surrogate_gradient requires a student with nn.ReLU modules; {type(model).__name__} has none"
        )
    return replaced
