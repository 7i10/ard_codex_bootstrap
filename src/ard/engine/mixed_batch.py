"""Mixed clean + adversarial batches and split (auxiliary) BatchNorm (plan 0103 Phase 2 batch A).

See ``ard.config.schema.MixedBatchConfig`` for the scientific definition. This
module owns the pure pieces the Trainer composes:

* :func:`adversarial_count` -- ``k = floor(fraction * m)``: the first ``k``
  positions of each per-rank batch are attacked.
* :func:`example_weights` -- lambda for the attacked positions, 1 otherwise;
  the loss is ``sum(w_i * L_i * valid_i) / sum(w_i * valid_i)`` (summed over
  ranks), which is Kurakin et al.'s ``(sum_clean L + lambda sum_adv L) /
  ((m - k) + lambda k)`` on a padding-free batch.
* :class:`AuxiliaryBatchNorm` -- the clean-branch BatchNorm state for
  ``split_batchnorm``. The model's own BatchNorm layers are the MAIN
  (adversarial) BN and stay the only BN the saved ``model`` weights,
  validation, selection and evaluation ever see. This module holds a second
  copy of every BatchNorm layer's affine parameters and running statistics
  (initialized from the main BN at construction) and runs the clean sub-batch
  through the model with those tensors substituted
  (``torch.func.functional_call``), so the clean forward's batch statistics,
  running-statistic updates and affine gradients go to the auxiliary copy only.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import torch
from torch import nn
from torch.func import functional_call
from torch.nn.modules.batchnorm import _BatchNorm

__all__ = ["AuxiliaryBatchNorm", "adversarial_count", "check_split_batchnorm_batch", "example_weights"]


def adversarial_count(batch_size: int, fraction: float) -> int:
    """Number of attacked examples ``k = floor(fraction * batch_size)`` (``0 < fraction < 1``)."""
    if batch_size < 0:
        raise ValueError("batch_size must be non-negative")
    if not 0.0 < fraction < 1.0:
        raise ValueError("adversarial_fraction must lie in (0, 1)")
    return int(math.floor(fraction * batch_size))


def check_split_batchnorm_batch(batch_size: int, fraction: float, *, where: str) -> None:
    """Refuse a batch whose split-BN sub-batches would hold exactly one example.

    BatchNorm in train mode on one example is undefined wherever the layer has
    a single value per channel (e.g. MobileNetV4's head BN after global
    pooling) and degenerate elsewhere. An empty adversarial sub-batch is fine
    (no adversarial forward). The batch is refused, never trimmed: dropping
    examples would silently change the data each epoch sees.
    """
    adversarial = adversarial_count(batch_size, fraction)
    clean = batch_size - adversarial
    if adversarial == 1 or clean == 1:
        raise ValueError(
            f"method.mixed_batch.split_batchnorm: {where} of {batch_size} examples splits into {adversarial} "
            f"adversarial / {clean} clean; a 1-example BatchNorm sub-batch is refused (choose a per_rank_batch_size "
            "or adversarial_fraction that avoids it)"
        )


def example_weights(
    batch_size: int, adversarial: int, weight: float, *, device: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Per-position loss weight: ``weight`` (lambda) for the first ``adversarial`` positions, else 1."""
    if not 0 <= adversarial <= batch_size:
        raise ValueError("adversarial count must lie in [0, batch_size]")
    weights = torch.ones(batch_size, device=device, dtype=dtype)
    weights[:adversarial] = weight
    return weights


def _key(name: str) -> str:
    # Module paths contain '.', which ParameterDict/buffer names may not.
    return name.replace(".", "__")


class AuxiliaryBatchNorm(nn.Module):
    """Clean-branch copy of every BatchNorm layer of ``model`` (split BN).

    Construct it from the (unwrapped) student BEFORE the optimizer, and give
    its parameters to the optimizer together with the student's. Its
    ``state_dict`` is checkpointed separately (``auxiliary_batchnorm``) so a
    resume restores it exactly; the student's own ``state_dict`` is unchanged.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        layers = [(name, module) for name, module in model.named_modules() if isinstance(module, _BatchNorm)]
        if not layers:
            raise ValueError(
                "method.mixed_batch.split_batchnorm requires a student with BatchNorm layers; this student has "
                "none (LayerNorm/GroupNorm-only models hold no batch statistics to split)"
            )
        self.layer_names: tuple[str, ...] = tuple(name for name, _ in layers)
        self.affine_parameters = nn.ParameterDict()
        for name, module in layers:
            if module.affine:
                self.affine_parameters[_key(name) + "__weight"] = nn.Parameter(module.weight.detach().clone())
                self.affine_parameters[_key(name) + "__bias"] = nn.Parameter(module.bias.detach().clone())
            if module.track_running_stats:
                assert module.running_mean is not None and module.running_var is not None
                assert module.num_batches_tracked is not None
                self.register_buffer(_key(name) + "__running_mean", module.running_mean.detach().clone())
                self.register_buffer(_key(name) + "__running_var", module.running_var.detach().clone())
                self.register_buffer(_key(name) + "__num_batches_tracked", module.num_batches_tracked.detach().clone())
        self._layer_flags = {name: (module.affine, module.track_running_stats) for name, module in layers}

    def overrides(self) -> dict[str, torch.Tensor]:
        """``{model state name: auxiliary tensor}`` for ``functional_call``."""
        mapping: dict[str, torch.Tensor] = {}
        buffers = dict(self.named_buffers())
        for name in self.layer_names:
            affine, tracked = self._layer_flags[name]
            if affine:
                mapping[f"{name}.weight"] = self.affine_parameters[_key(name) + "__weight"]
                mapping[f"{name}.bias"] = self.affine_parameters[_key(name) + "__bias"]
            if tracked:
                for suffix in ("running_mean", "running_var", "num_batches_tracked"):
                    mapping[f"{name}.{suffix}"] = buffers[_key(name) + "__" + suffix]
        return mapping

    def forward_clean(self, model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        """Run ``model`` on ``inputs`` with every BatchNorm swapped for its auxiliary copy."""
        return functional_call(model, self.overrides(), (inputs,), strict=False)

    def matches(self, model: nn.Module) -> bool:
        """True iff ``model`` still has exactly the BatchNorm layers this copy was built from."""
        names = tuple(name for name, module in model.named_modules() if isinstance(module, _BatchNorm))
        return names == self.layer_names

    def iter_parameters(self) -> Iterator[nn.Parameter]:
        return iter(self.affine_parameters.values())
