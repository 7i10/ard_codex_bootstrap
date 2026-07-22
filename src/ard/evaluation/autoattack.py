"""Explicit AutoAttack adapter; full AutoAttack is never an automated test."""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn


class AutoAttackUnavailable(RuntimeError):
    pass


def run_autoattack(
    *,
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    norm: str,
    epsilon: float,
    output_path: Path,
    seed: int,
    batch_size: int = 128,
    autoattack_cls: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run only when the standalone evaluation CLI explicitly requests it.

    This function is deliberately isolated from training and ordinary PGD
    evaluation.  Callers must have already loaded a saved checkpoint.
    """
    if norm != "linf":
        raise ValueError("AutoAttack adapter supports the validated Linf threat model only")
    if autoattack_cls is None:
        try:
            import autoattack

            version = getattr(autoattack, "__version__", "unknown")
            adapter_class: Callable[..., Any] = autoattack.AutoAttack
        except ImportError as exc:
            raise AutoAttackUnavailable("AutoAttack is optional; install it for a separate evaluation process") from exc
    else:
        version = "injected"
        adapter_class = autoattack_cls
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    was_training = model.training
    model.eval()
    try:
        adversary: Any = adapter_class(model, norm="Linf", eps=epsilon, version="standard", device=images.device)
        adversary.seed = seed
        adversarial = adversary.run_standard_evaluation(images, labels, bs=batch_size)
        with torch.no_grad():
            accuracy = model(adversarial).argmax(1).eq(labels).float().mean().item()
    finally:
        model.train(was_training)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "autoattack_accuracy": float(accuracy),
        "seed": seed,
        "norm": "Linf",
        "epsilon": epsilon,
        "version": version,
        "batch_size": batch_size,
    }
    import json

    output_path.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    return result
