"""Pure, no-update primitives for the I100 Secant BDD forensic audit."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F


def dynamic_pair_margin(
    logits: torch.Tensor, labels: torch.Tensor, rival: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Runtime-identical Student-selected non-true rival and signed margin."""
    if rival is None:
        masked = logits.detach().clone()
        masked.scatter_(1, labels[:, None], float("-inf"))
        rival = masked.argmax(dim=1)
    else:
        rival = rival.detach()
    margin = logits.float().gather(1, labels[:, None]).squeeze(1) - logits.float().gather(1, rival[:, None]).squeeze(1)
    return margin, rival


def secant_components(
    *,
    student_adv_margin: torch.Tensor,
    student_clean_margin: torch.Tensor,
    teacher_adv_margin: torch.Tensor,
    teacher_clean_margin: torch.Tensor,
    rho: torch.Tensor,
    selected: torch.Tensor,
    epsilon: float,
) -> dict[str, torch.Tensor]:
    """Return exact v2 per-sample S-BDD quantities without a reduction.

    ``q_student`` remains graph-bearing by construction.  Teacher quantities
    and the hard gate are detached.  The caller owns the full-batch mean and
    any frozen coefficient; this avoids silently selected-count normalizing
    the diagnostic.
    """
    if epsilon <= 0:
        raise ValueError("secant epsilon must be positive")
    rho = rho.detach().float()
    teacher_adv = teacher_adv_margin.detach().float()
    teacher_clean = teacher_clean_margin.detach().float()
    selected = selected.detach().to(dtype=student_adv_margin.dtype)
    q_student = (student_adv_margin - student_clean_margin).abs() / (rho + epsilon)
    q_teacher = (teacher_adv - teacher_clean).abs() / (rho + epsilon)
    d_student = student_adv_margin / (q_student + epsilon)
    d_teacher = teacher_adv / (q_teacher.detach() + epsilon)
    teacher_pair_gate = (teacher_adv > 0).to(dtype=student_adv_margin.dtype)
    nonzero_rho = (rho > epsilon).to(dtype=student_adv_margin.dtype)
    hinge_gap = d_teacher - d_student
    hinge_positive = (hinge_gap > 0).to(dtype=student_adv_margin.dtype)
    active = selected * teacher_pair_gate * nonzero_rho
    raw_loss = 0.5 * F.relu(hinge_gap).square() * active
    return {
        "student_margin_delta_sign": torch.sign(student_adv_margin - student_clean_margin).detach(),
        "q_student": q_student,
        "q_teacher": q_teacher.detach(),
        "d_student": d_student,
        "d_teacher": d_teacher.detach(),
        "teacher_pair_gate": teacher_pair_gate.detach(),
        "nonzero_rho": nonzero_rho.detach(),
        "hinge_gap": hinge_gap,
        "hinge_positive": hinge_positive.detach(),
        "active": active.detach(),
        "raw_loss": raw_loss,
    }


def scalar_secant_loss(
    student_adv_margin: torch.Tensor,
    student_clean_margin: torch.Tensor,
    *,
    rho: torch.Tensor,
    d_teacher: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Scalar-varying v2 loss for autograd/central-difference validation.

    The teacher distance, radius, active mask, rival, and hinge region must be
    frozen by the caller.  This intentionally tests only the Student formula.
    """
    q_student = (student_adv_margin - student_clean_margin).abs() / (rho + epsilon)
    d_student = student_adv_margin / (q_student + epsilon)
    return 0.5 * F.relu(d_teacher.detach() - d_student).square()


def central_difference(function, value: torch.Tensor, *, step: float) -> torch.Tensor:
    """Central derivative for a scalar-valued function at a frozen smooth point."""
    if step <= 0:
        raise ValueError("finite-difference step must be positive")
    return (function(value + step) - function(value - step)) / (2.0 * step)


def quantile_summary(values: torch.Tensor) -> dict[str, float]:
    """Finite-only numerical summary required by the forensic report."""
    flattened = values.detach().double().flatten()
    finite = flattened[torch.isfinite(flattened)]
    if finite.numel() == 0:
        keys = ("count", "finite_count", "min", "p1", "p5", "p25", "median", "p75", "p95", "p99", "max")
        return {key: float("nan") for key in keys}
    return {
        "count": float(flattened.numel()),
        "finite_count": float(finite.numel()),
        "min": float(finite.min()),
        "p1": float(torch.quantile(finite, 0.01)),
        "p5": float(torch.quantile(finite, 0.05)),
        "p25": float(torch.quantile(finite, 0.25)),
        "median": float(torch.quantile(finite, 0.50)),
        "p75": float(torch.quantile(finite, 0.75)),
        "p95": float(torch.quantile(finite, 0.95)),
        "p99": float(torch.quantile(finite, 0.99)),
        "max": float(finite.max()),
    }


def rank_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    """Small dependency-free Spearman correlation for finite paired tensors."""
    a = left.detach().double().flatten()
    b = right.detach().double().flatten()
    valid = torch.isfinite(a) & torch.isfinite(b)
    a, b = a[valid], b[valid]
    if a.numel() < 2:
        return float("nan")
    # The requested audit is descriptive.  Average tied ranks are not needed
    # for continuous geometry values; stable argsort ranks keep this local.
    ra = torch.empty_like(a)
    rb = torch.empty_like(b)
    ra[torch.argsort(a, stable=True)] = torch.arange(a.numel(), dtype=a.dtype, device=a.device)
    rb[torch.argsort(b, stable=True)] = torch.arange(b.numel(), dtype=b.dtype, device=b.device)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = torch.sqrt(ra.square().sum() * rb.square().sum())
    return float((ra * rb).sum() / denom) if float(denom) else float("nan")


def state_tensor_hash(state: Mapping[str, torch.Tensor]) -> str:
    """Canonical hash helper for no-update parameter/buffer restoration checks."""
    import hashlib

    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()
