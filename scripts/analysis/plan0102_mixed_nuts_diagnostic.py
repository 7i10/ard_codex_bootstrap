"""Plan 0102 Workstream A: post-hoc logit-mixing diagnostic (throwaway, not a
scientific record -- same status as plan 0101's own Stage A pretrained-
checkpoint check).

Question: how much of the clean-accuracy loss from adversarial fine-tuning
is recoverable purely at inference time, by mixing the softmax outputs of
the already-completed PGD-AT-tuned checkpoint (robust) and the original
pretrained backbone (clean), the way MixedNUTS (arXiv:2402.02263) mixes a
robust and a standard classifier? No retraining; ~1 GPU-hour per the
literature review's own estimate (plan 0102 Progress log, 2026-09-15).

This is a WHITE-BOX attack against the actual mixture (not a reused PGD
trace from the robust model alone): the attack backpropagates through both
branches of ``mixed_log_probs`` below, matching how the mixture would
really be attacked if deployed. It is a self-contained manual PGD loop
(not ``ard.attacks.LinfPGD``, which has no two-model-mixture request
shape) using this project's own eps=4/255, step=8/765, 10-step, random-
start convention (the resolved training config's own selection attack).

Not a scientific record: no run-bundle, no checkpoint, no tracker. Prints
a small table to stdout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ard.config.loader import load_resolved_config_for_evaluation  # noqa: E402
from ard.data import collate_indexed  # noqa: E402
from ard.data.datasets import build_dataset  # noqa: E402
from ard.evaluation import load_saved_student_checkpoint  # noqa: E402
from ard.models import build_student  # noqa: E402

EPS = 4.0 / 255.0
STEP = 8.0 / 765.0
STEPS = 10
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def mixed_log_probs(robust: torch.nn.Module, clean: torch.nn.Module, images: torch.Tensor, lam: float) -> torch.Tensor:
    p_robust = F.softmax(robust(images), dim=1)
    p_clean = F.softmax(clean(images), dim=1)
    mixed = lam * p_robust + (1.0 - lam) * p_clean
    return torch.log(mixed.clamp_min(1e-12))


def pgd_against_mixture(
    robust: torch.nn.Module, clean: torch.nn.Module, images: torch.Tensor, labels: torch.Tensor, lam: float
) -> torch.Tensor:
    images = images.detach()
    delta = torch.empty_like(images).uniform_(-EPS, EPS)
    delta = (torch.clamp(images + delta, 0.0, 1.0) - images).detach().requires_grad_(True)
    for _ in range(STEPS):
        log_p = mixed_log_probs(robust, clean, images + delta, lam)
        loss = F.nll_loss(log_p, labels)
        (grad,) = torch.autograd.grad(loss, delta)
        delta = (delta.detach() + STEP * grad.sign()).clamp(-EPS, EPS)
        delta = (torch.clamp(images + delta, 0.0, 1.0) - images).detach().requires_grad_(True)
    return (images + delta).detach()


def run_one(*, train_dir: Path, sample_count: int, device: torch.device) -> None:
    resolved = load_resolved_config_for_evaluation(train_dir / "resolved_config.yaml")
    config = resolved.config
    robust = build_student(config.student, tier=config.tier).to(device).eval()
    load_saved_student_checkpoint(train_dir / "best.pt", robust)
    for p in robust.parameters():
        p.requires_grad_(False)
    clean = build_student(config.student, tier=config.tier).to(device).eval()
    for p in clean.parameters():
        p.requires_grad_(False)

    dataset = build_dataset(config.evaluation.dataset)
    generator = torch.Generator().manual_seed(0)
    indices = torch.randperm(len(dataset), generator=generator)[:sample_count].tolist()
    loader = DataLoader(Subset(dataset, indices), batch_size=128, shuffle=False, collate_fn=collate_indexed)

    print(f"\n=== {config.student.architecture} ({train_dir}) ===")
    print(f"n={sample_count} images, eps={EPS:.4f}, step={STEP:.4f}, steps={STEPS}")
    print(f"{'lambda':>8} {'clean_acc':>10} {'robust_acc':>11}")
    for lam in LAMBDAS:
        clean_correct = 0
        robust_correct = 0
        total = 0
        for batch in loader:
            images = batch.images.to(device)
            labels = batch.labels.to(device)
            with torch.no_grad():
                log_p = mixed_log_probs(robust, clean, images, lam)
                clean_correct += (log_p.argmax(dim=1) == labels).sum().item()
            adv_images = pgd_against_mixture(robust, clean, images, labels, lam)
            with torch.no_grad():
                log_p_adv = mixed_log_probs(robust, clean, adv_images, lam)
                robust_correct += (log_p_adv.argmax(dim=1) == labels).sum().item()
            total += labels.numel()
        print(f"{lam:8.2f} {clean_correct / total:10.4f} {robust_correct / total:11.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, action="append", required=True, help="Repeatable.")
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    for train_dir in args.train_dir:
        run_one(train_dir=train_dir, sample_count=args.sample_count, device=device)


if __name__ == "__main__":
    main()
