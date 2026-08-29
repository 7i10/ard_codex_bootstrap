"""One-batch eager/compiled functional parity check for the runtime audit."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import torch
from ert_rslad_runtime_benchmark import _load_student_checkpoint, _loader
from torch.optim import SGD

from ard.attacks import AttackRequest, LinfPGD
from ard.config import load_config
from ard.models import build_student, build_teacher
from ard.objectives import RSLADObjective


def _seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run(
    model: torch.nn.Module,
    teacher: torch.nn.Module,
    config: Any,
    batch: Any,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    images = batch.images.to(device)
    labels = batch.labels.to(device)
    attack = LinfPGD(config.method.attack)
    objective = RSLADObjective(
        temperature=config.method.temperature,
        temperature_squared=config.method.temperature_squared,
    )
    optimizer = SGD(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        momentum=config.optimizer.momentum,
        weight_decay=config.optimizer.weight_decay,
        nesterov=config.optimizer.nesterov,
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        teacher_clean = teacher(images.float()).detach().float()
    attack_result = attack.generate(
        AttackRequest(
            inputs=images,
            labels=labels,
            student=model,
            teacher=teacher,
            target_logits=teacher_clean,
            generator=generator,
        )
    )
    adversarial_logits = model(attack_result.adversarial)
    clean_logits = model(images)
    terms = objective(
        student_logits=adversarial_logits,
        labels=labels,
        teacher_logits=teacher_clean,
        clean_student_logits=clean_logits,
    )
    loss = terms.total.mean()
    loss.backward()
    optimizer.step()
    return {
        "teacher_clean": teacher_clean.detach(),
        "adversarial": attack_result.adversarial.detach(),
        "adversarial_logits": adversarial_logits.detach(),
        "clean_logits": clean_logits.detach(),
        "loss": loss.detach(),
        "parameters": [parameter.detach().clone() for parameter in model.parameters()],
        "gradients": [
            None if parameter.grad is None else parameter.grad.detach().clone() for parameter in model.parameters()
        ],
    }


def _max_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().detach().cpu())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--candidate",
        choices=("default", "reduce-overhead", "max-autotune-no-cudagraphs"),
        default="default",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    config = load_config(args.config)
    if config.training.deterministic:
        torch.use_deterministic_algorithms(True)
    _seed(args.seed)
    loader = _loader(config, workers=0, pin_memory=False, persistent_workers=False, prefetch_factor=None)
    batch = next(iter(loader))
    eager = build_student(config.student, tier="production").to(device)
    checkpoint_sha256 = _load_student_checkpoint(eager, args.checkpoint)
    compiled = copy.deepcopy(eager)
    teacher = build_teacher(config.teacher, tier="production").to(device)
    teacher.eval()
    compiled = torch.compile(compiled, mode=args.candidate, fullgraph=False)
    eager_result = _run(eager, teacher, config, batch, device, args.seed + 1)
    compiled_result = _run(compiled, teacher, config, batch, device, args.seed + 1)
    checks = {
        "teacher_clean_max_abs_diff": _max_diff(eager_result["teacher_clean"], compiled_result["teacher_clean"]),
        "adversarial_max_abs_diff": _max_diff(eager_result["adversarial"], compiled_result["adversarial"]),
        "adversarial_logits_max_abs_diff": _max_diff(
            eager_result["adversarial_logits"], compiled_result["adversarial_logits"]
        ),
        "clean_logits_max_abs_diff": _max_diff(eager_result["clean_logits"], compiled_result["clean_logits"]),
        "loss_abs_diff": _max_diff(eager_result["loss"], compiled_result["loss"]),
        "parameter_max_abs_diff": max(
            _max_diff(a, b) for a, b in zip(eager_result["parameters"], compiled_result["parameters"], strict=True)
        ),
        "gradient_max_abs_diff": max(
            _max_diff(a, b)
            for a, b in zip(eager_result["gradients"], compiled_result["gradients"], strict=True)
            if a is not None and b is not None
        ),
    }
    result = {
        "schema_version": 1,
        "kind": "ert_rslad_runtime_parity_v2",
        "config": str(args.config),
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "candidate": args.candidate,
        "device": str(device),
        "seed": args.seed,
        "checks": checks,
        "predeclared_tolerances": {
            "teacher_clean_max_abs_diff": 1e-6,
            "adversarial_max_abs_diff": 1e-5,
            "adversarial_logits_max_abs_diff": 1e-4,
            "clean_logits_max_abs_diff": 1e-4,
            "loss_abs_diff": 1e-5,
            "parameter_max_abs_diff": 1e-4,
            "gradient_max_abs_diff": 1e-3,
        },
        "torch_version": torch.__version__,
    }
    result["pass"] = all(checks[key] <= value for key, value in result["predeclared_tolerances"].items())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
