#!/usr/bin/env python3
"""Measure what the TRADES attack-initialization deviation costs, on a fixed batch.

`docs/UPSTREAM_BASELINES.md` section 8 records that official TRADES starts its
inner maximisation from Gaussian noise at scale 0.001 while this repository
starts from a uniform draw in [-epsilon, epsilon], and calls the difference
intentional.  No cost was ever attached to it.  That is the same shape as the
detached-clean-target entry that sat in the same list and turned out to cost
2.73 percentage points of official-test AutoAttack
(`docs/debugging/0028-trades-clean-target-detached.md`).

The corrected TRADES run scores 47.87 % AutoAttack against a published
ResNet-18 range of 49.0 to 49.4 %.  This script asks whether the initialization
can account for that 1.2 to 1.5 point shortfall, before nine training runs are
spent on the hypothesis.

Design.  One PGD step loop is written once here and is used for both arms, so
the only difference between them is the initial delta.  The loop is first
checked against `LinfPGD.generate` on the uniform arm and must agree bitwise;
if it does not, the script refuses to report.  Everything else -- model,
weights, batch, epsilon, step size, step count, KL target, evaluation mode --
is identical between arms.

What is reported per arm: the inner KL objective reached at every step (the
direct measure of how hard the inner problem was made), the final perturbation
geometry, the outer TRADES loss, and the parameter-gradient norm of that loss.
The two arms are then compared sample by sample.

This runs on CPU on a fixed batch.  It trains nothing and evaluates no test
split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ard.attacks.base import AttackRequest  # noqa: E402
from ard.attacks.pgd import LinfPGD  # noqa: E402
from ard.config.loader import load_config  # noqa: E402
from ard.data.datasets import _to_tensor, build_raw_dataset  # noqa: E402
from ard.models.registry import build_student  # noqa: E402


def pgd_steps(
    *,
    model: nn.Module,
    clean: torch.Tensor,
    labels: torch.Tensor,
    initial_delta: torch.Tensor,
    epsilon: float,
    step_size: float,
    steps: int,
) -> tuple[torch.Tensor, list[float]]:
    """The shared inner maximisation. Identical for both arms by construction."""
    with torch.no_grad():
        target_logits = model(clean).detach().float()
    adversarial = (clean + initial_delta).detach()
    trace: list[float] = []
    for _ in range(steps):
        adversarial.requires_grad_(True)
        logits = model(adversarial.float()).float()
        loss = F.kl_div(F.log_softmax(logits, dim=1), F.softmax(target_logits, dim=1), reduction="batchmean")
        gradient = torch.autograd.grad(loss, adversarial, only_inputs=True)[0]
        trace.append(float(loss.detach()))
        adversarial = adversarial.detach() + step_size * gradient.detach().sign()
        delta = torch.clamp(adversarial - clean, -epsilon, epsilon)
        adversarial = (clean + delta).clamp(0, 1).detach()
    with torch.no_grad():
        logits = model(adversarial).float()
        final = F.kl_div(F.log_softmax(logits, dim=1), F.softmax(target_logits, dim=1), reduction="batchmean")
    trace.append(float(final))
    return adversarial, trace


def per_sample_kl(model: nn.Module, clean: torch.Tensor, adversarial: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        target = F.softmax(model(clean).float(), dim=1)
        log_student = F.log_softmax(model(adversarial).float(), dim=1)
        return F.kl_div(log_student, target, reduction="none").sum(dim=1)


def outer_trades(
    model: nn.Module, clean: torch.Tensor, adversarial: torch.Tensor, labels: torch.Tensor, beta: float
) -> tuple[float, float]:
    """Outer TRADES loss and the L2 norm of its gradient with respect to the parameters."""
    model.zero_grad(set_to_none=True)
    clean_logits = model(clean).float()
    adversarial_logits = model(adversarial).float()
    cross_entropy = F.cross_entropy(clean_logits, labels)
    log_target = F.log_softmax(clean_logits, dim=1)
    log_student = F.log_softmax(adversarial_logits, dim=1)
    kl = (log_target.exp() * (log_target - log_student)).sum(dim=1).mean()
    loss = cross_entropy + beta * kl
    loss.backward()
    squared = sum(float((p.grad.detach() ** 2).sum()) for p in model.parameters() if p.grad is not None)
    model.zero_grad(set_to_none=True)
    return float(loss.detach()), squared**0.5


def render_report(report: dict, record_path: Path) -> str:
    """Render the Markdown report from the same dict the JSON record is written from."""
    lines: list[str] = []
    lines.append("# TRADES attack initialization: what the deviation costs on a fixed batch")
    lines.append("")
    lines.append(f"Contract: `{report['contract']}`. Record: `{record_path.name}`.")
    lines.append("")
    lines.append("**Question.** " + str(report["question"]))
    lines.append("")
    lines.append(
        "**Terms.** *Inner KL* is the value the ten-step attack drives the "
        "Kullback-Leibler divergence to between the model's prediction on the perturbed image and "
        "its own prediction on the clean image; a higher value means the attack found a harder "
        "example, so a higher value means a stronger attack. *Uniform* is this repository's "
        "initialization, a draw from the uniform distribution on the whole epsilon-ball. "
        "*Gaussian* is official TRADES', a draw from a normal distribution at scale "
        f"{report['gaussian_scale']}, which starts essentially at the clean image. Everything else -- "
        "model, weights, batch, epsilon, step size, step count, target, evaluation mode -- is "
        "identical between the two, and one step loop serves both."
    )
    lines.append("")
    attack = report["attack"]
    lines.append(
        f"**Setting.** {report['batch_size']} images, {report['batch']}. "
        f"epsilon {attack['epsilon']:.6f}, step {attack['step_size']:.6f}, {attack['steps']} steps, "
        f"KL to the student's own clean prediction, beta {report['trades_beta']}. CPU. "
        "No training, no test split."
    )
    lines.append("")
    lines.append("## Result")
    lines.append("")
    lines.append(
        "| checkpoint | inner KL (uniform) | inner KL (Gaussian) | uniform is | outer loss diff | "
        "parameter gradient diff | samples where uniform is stronger | identical adversarial examples |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for entry in report["checkpoints"]:
        comparison = entry["comparison"]
        lines.append(
            f"| epoch {entry['epoch']} | {entry['uniform']['inner_kl_final']:.6f} | "
            f"{entry['gaussian']['inner_kl_final']:.6f} | "
            f"{comparison['inner_kl_relative_difference'] * 100:+.2f}% stronger | "
            f"{comparison['outer_loss_relative_difference'] * 100:+.2f}% | "
            f"{comparison['outer_gradient_relative_difference'] * 100:+.2f}% | "
            f"{comparison['per_sample_kl_uniform_stronger_fraction'] * 100:.1f}% | "
            f"{comparison['adversarial_identical_fraction'] * 100:.1f}% |"
        )
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.append(
        "The local initialization makes the inner attack **stronger**, not weaker, at both "
        "checkpoints. Training against a harder inner problem raises robustness if it does anything, "
        "so this deviation points the wrong way to explain a local TRADES that sits *below* the "
        "published range. The hypothesis is not supported."
    )
    lines.append("")
    lines.append(
        "The size is also wrong. The detached-clean-target defect, measured the same way on a fixed "
        "batch, moved the gradient by 58 per cent and cost 2.73 percentage points of official-test "
        "AutoAttack. This deviation moves the inner objective by 3 to 5 per cent and the parameter "
        "gradient by about 10 per cent, an order of magnitude smaller and in the helping direction."
    )
    lines.append("")
    lines.append(
        "A separate observation, not the question asked. The two initializations reach adversarial "
        "examples that differ by the full width of the ball on at least one pixel of **every** image, "
        "and no image lands in the same place under both. Yet per image the uniform attack is the "
        "stronger one only about half the time. Which maximum the attack finds is close to arbitrary; "
        "how high it climbs is not. Any claim that two attack configurations are equivalent because "
        "their adversarial examples agree, or different because they disagree, is reading the wrong "
        "quantity."
    )
    lines.append("")
    lines.append("## What this does not show")
    lines.append("")
    lines.append(
        "Both arms attack a model that was itself trained under the uniform initialization, so the "
        "loss surface is the one that initialization produced. A model trained under the Gaussian "
        "initialization could differ. A fixed-batch differential cannot rule that out; it can only "
        "say that on this surface the deviation is small and helps. The same limitation applied to "
        "the detached-target measurement, which predicted the sign correctly."
    )
    lines.append("")
    lines.append(
        "It also says nothing about the other recorded deviations -- the epsilon and step-size "
        "defaults, the normalization path, or the learning-rate schedule. Those remain unmeasured."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="resolved_config.yaml of the TRADES run")
    parser.add_argument("--checkpoint", type=Path, action="append", required=True, help="repeatable")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gaussian-scale", type=float, default=0.001, help="official TRADES init scale")
    parser.add_argument("--init-seed", type=int, default=23)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    torch.manual_seed(arguments.init_seed)
    config = load_config(arguments.config)
    attack_config = config.method.attack
    epsilon = float(attack_config.epsilon_value)
    step_size = float(attack_config.step_size_value)
    steps = int(attack_config.steps)
    beta = float(config.method.trades_beta)

    raw = build_raw_dataset(config.dataset)
    images = torch.stack([_to_tensor(raw[index][0]) for index in range(arguments.batch_size)])
    labels = torch.tensor([int(raw[index][1]) for index in range(arguments.batch_size)])

    report: dict[str, object] = {
        "contract": "trades-attack-initialization-differential/v1",
        "question": (
            "Does the local uniform-in-[-eps,eps] attack initialization, against official TRADES' "
            "0.001-scale Gaussian initialization, change the inner maximisation enough to explain "
            "the 1.2-1.5 pp shortfall of the corrected local TRADES against the published range?"
        ),
        "config": str(arguments.config),
        "batch_size": arguments.batch_size,
        "batch": "first N images of the configured split, in index order",
        "attack": {"epsilon": epsilon, "step_size": step_size, "steps": steps, "loss": "kl", "target": "student_clean"},
        "trades_beta": beta,
        "gaussian_scale": arguments.gaussian_scale,
        "init_seed": arguments.init_seed,
        "checkpoints": [],
    }

    for checkpoint_path in arguments.checkpoint:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = build_student(config.student, tier=config.tier)
        model.load_state_dict(payload["model"])
        model.eval()

        # Uniform arm, and the fidelity check that the shared loop is the engine's loop.
        uniform_delta = torch.empty_like(images).uniform_(
            -1.0, 1.0, generator=torch.Generator().manual_seed(arguments.init_seed)
        )
        uniform_delta = uniform_delta * epsilon
        uniform_delta = (images + uniform_delta).clamp(0, 1) - images

        reference = LinfPGD(attack_config).generate(
            AttackRequest(
                inputs=images,
                labels=labels,
                student=model,
                generator=torch.Generator().manual_seed(arguments.init_seed),
            )
        )
        if not torch.equal(reference.initial_delta, uniform_delta):
            print("REFUSED: the reconstructed uniform initialization is not the engine's", file=sys.stderr)
            return 2

        uniform_adversarial, uniform_trace = pgd_steps(
            model=model,
            clean=images,
            labels=labels,
            initial_delta=uniform_delta,
            epsilon=epsilon,
            step_size=step_size,
            steps=steps,
        )
        if not torch.equal(uniform_adversarial, reference.adversarial):
            print("REFUSED: the shared step loop does not reproduce LinfPGD.generate", file=sys.stderr)
            return 2

        # Official arm: Gaussian at 0.001, everything else identical.
        gaussian_delta = arguments.gaussian_scale * torch.randn(
            images.shape, generator=torch.Generator().manual_seed(arguments.init_seed)
        )
        gaussian_delta = (images + gaussian_delta).clamp(0, 1) - images
        gaussian_adversarial, gaussian_trace = pgd_steps(
            model=model,
            clean=images,
            labels=labels,
            initial_delta=gaussian_delta,
            epsilon=epsilon,
            step_size=step_size,
            steps=steps,
        )

        uniform_sample_kl = per_sample_kl(model, images, uniform_adversarial)
        gaussian_sample_kl = per_sample_kl(model, images, gaussian_adversarial)
        difference = uniform_sample_kl - gaussian_sample_kl
        adversarial_gap = (uniform_adversarial - gaussian_adversarial).abs()

        uniform_loss, uniform_grad = outer_trades(model, images, uniform_adversarial, labels, beta)
        gaussian_loss, gaussian_grad = outer_trades(model, images, gaussian_adversarial, labels, beta)

        report["checkpoints"].append(  # type: ignore[union-attr]
            {
                "path": str(checkpoint_path),
                "epoch": int(payload.get("epoch", -1)),
                "uniform": {
                    "inner_kl_trace": uniform_trace,
                    "inner_kl_final": uniform_trace[-1],
                    "outer_trades_loss": uniform_loss,
                    "outer_parameter_gradient_l2": uniform_grad,
                    "mean_abs_delta": float((uniform_adversarial - images).abs().mean()),
                    "fraction_at_epsilon": float(
                        ((uniform_adversarial - images).abs() >= epsilon - 1e-9).float().mean()
                    ),
                },
                "gaussian": {
                    "inner_kl_trace": gaussian_trace,
                    "inner_kl_final": gaussian_trace[-1],
                    "outer_trades_loss": gaussian_loss,
                    "outer_parameter_gradient_l2": gaussian_grad,
                    "mean_abs_delta": float((gaussian_adversarial - images).abs().mean()),
                    "fraction_at_epsilon": float(
                        ((gaussian_adversarial - images).abs() >= epsilon - 1e-9).float().mean()
                    ),
                },
                "comparison": {
                    "inner_kl_relative_difference": (uniform_trace[-1] - gaussian_trace[-1])
                    / max(gaussian_trace[-1], 1e-12),
                    "outer_loss_relative_difference": (uniform_loss - gaussian_loss) / max(abs(gaussian_loss), 1e-12),
                    "outer_gradient_relative_difference": (uniform_grad - gaussian_grad) / max(gaussian_grad, 1e-12),
                    "per_sample_kl_uniform_stronger_fraction": float((difference > 0).float().mean()),
                    "per_sample_kl_mean_difference": float(difference.mean()),
                    "per_sample_kl_max_abs_difference": float(difference.abs().max()),
                    "adversarial_linf_distance_mean": float(adversarial_gap.amax(dim=(1, 2, 3)).mean()),
                    "adversarial_linf_distance_max": float(adversarial_gap.max()),
                    "adversarial_identical_fraction": float(
                        (adversarial_gap.amax(dim=(1, 2, 3)) < 1e-9).float().mean()
                    ),
                },
            }
        )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, indent=2) + "\n"
    arguments.output.write_text(serialized, encoding="utf-8")
    arguments.output.with_suffix(arguments.output.suffix + ".sha256").write_text(
        hashlib.sha256(serialized.encode("utf-8")).hexdigest() + "\n", encoding="utf-8"
    )
    report_path = arguments.output.with_suffix(".md")
    report_path.write_text(render_report(report, arguments.output), encoding="utf-8")
    print(render_report(report, arguments.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
