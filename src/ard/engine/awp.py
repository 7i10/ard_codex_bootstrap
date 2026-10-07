"""Adversarial Weight Perturbation (Wu, Xia & Wang 2020, arXiv:2004.05884) for ``pgd_at``.

A transcription of the official ``csdongxian/AWP`` ``AT_AWP/utils_awp.py``
(``AdvWeightPerturb``, ``diff_in_weights``, ``add_into_weights``) as used by
``AT_AWP/train_cifar10.py``:

1. ``proxy.load_state_dict(model.state_dict())``; ``proxy.train()``.
2. ``loss = -L(proxy(x_adv), y)``; one step of ``SGD(proxy.parameters(), lr=0.01)``
   (no momentum, no weight decay: stateless).
3. For every state entry with ``ndim > 1`` whose name contains ``weight``:
   ``d = ||w|| / (||w_proxy - w|| + 1e-20) * (w_proxy - w)``.
4. ``w += gamma * d`` on the matching named parameters; the training step runs
   at the perturbed weights; after the optimizer step ``w -= gamma * d``.

``L`` is the run's own per-sample PGD-AT objective averaged over the valid
examples -- the official code's ``F.cross_entropy`` mean when
``method.label_smoothing`` is 0 (the label-smoothed CE otherwise). The proxy
forward is in train mode, as upstream: it updates only the proxy's (discarded)
BatchNorm statistics and draws from the default CUDA RNG if the model has
dropout.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from collections.abc import Callable

import torch
from torch import nn

__all__ = ["AWP_EPS", "AWP_PROXY_LEARNING_RATE", "AdversarialWeightPerturbation"]

# Official code constants (AT_AWP/utils_awp.py EPS; train_cifar10.py proxy_opt lr).
AWP_EPS = 1e-20
AWP_PROXY_LEARNING_RATE = 0.01


class AdversarialWeightPerturbation:
    def __init__(self, model: nn.Module, *, gamma: float) -> None:
        if not gamma > 0:
            raise ValueError("AWP gamma must be positive")
        self.model = model
        self.gamma = float(gamma)
        self.proxy = copy.deepcopy(model)
        for parameter in self.proxy.parameters():
            parameter.requires_grad_(True)
            parameter.grad = None
        self.proxy_optimizer = torch.optim.SGD(self.proxy.parameters(), lr=AWP_PROXY_LEARNING_RATE)

    def compute(
        self, inputs: torch.Tensor, loss_fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> OrderedDict[str, torch.Tensor]:
        """Return the rescaled per-layer ascent direction ``d`` (``calc_awp``)."""
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()
        loss = -loss_fn(self.proxy(inputs))
        self.proxy_optimizer.zero_grad()
        loss.backward()
        self.proxy_optimizer.step()
        diff: OrderedDict[str, torch.Tensor] = OrderedDict()
        with torch.no_grad():
            model_state = self.model.state_dict()
            proxy_state = self.proxy.state_dict()
            for (old_key, old_value), (new_key, new_value) in zip(
                model_state.items(), proxy_state.items(), strict=True
            ):
                if old_key != new_key:
                    raise RuntimeError("AWP proxy state does not align with the model state")
                if old_value.ndim <= 1:
                    continue
                if "weight" in old_key:
                    difference = new_value - old_value
                    diff[old_key] = old_value.norm() / (difference.norm() + AWP_EPS) * difference
        # Drop the proxy's gradients: they are not reused and would hold memory.
        self.proxy_optimizer.zero_grad(set_to_none=True)
        return diff

    def _add(self, diff: OrderedDict[str, torch.Tensor], coefficient: float) -> None:
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if name in diff:
                    parameter.add_(coefficient * diff[name])

    def perturb(self, diff: OrderedDict[str, torch.Tensor]) -> None:
        self._add(diff, 1.0 * self.gamma)

    def restore(self, diff: OrderedDict[str, torch.Tensor]) -> None:
        self._add(diff, -1.0 * self.gamma)
