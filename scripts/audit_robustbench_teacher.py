#!/usr/bin/env python3
"""Audit one already-acquired RobustBench teacher without permitting downloads."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acquire_robustbench_teachers import (  # noqa: E402
    ALLOWED_REGISTRY_IDS,
    DEFAULT_MODEL_DIR,
    AcquisitionError,
    LoadModel,
    LoadModelResolver,
    _pinned_load_model,
    expected_checkpoint_path,
)

from ard.attacks import AttackRequest, LinfPGD  # noqa: E402
from ard.config.schema import AttackConfig, TeacherConfig  # noqa: E402
from ard.models.teacher import TeacherAdapter, build_teacher  # noqa: E402
from ard.models.teacher_registry import TeacherRegistry, TeacherRegistryError, sha256_file  # noqa: E402

PGD_STEP_SIZE = 2.0 / 255.0
PARITY_ATOL = 1e-7

RegistryLoader = Callable[[Path], TeacherRegistry]
ArdLoader = Callable[[TeacherConfig], TeacherAdapter]


class AuditError(RuntimeError):
    """An existing teacher does not meet a local loader or gradient contract."""


@dataclass(frozen=True)
class AuditReport:
    registry_id: str
    upstream_model_id: str
    device: str
    checkpoint_sha256: str
    preprocessing_owner: str
    normalization_profile: str
    parity_atol: float
    max_abs_diff: float
    logits_dtype: str
    backend_flags: dict[str, bool]
    input_gradient_l1: float
    pgd_linf: float


@dataclass(frozen=True)
class BackendFlags:
    deterministic_algorithms: bool
    cudnn_benchmark: bool
    cudnn_deterministic: bool
    cuda_matmul_allow_tf32: bool
    cudnn_allow_tf32: bool

    def report(self) -> dict[str, bool]:
        return {
            "deterministic_algorithms": self.deterministic_algorithms,
            "cudnn_benchmark": self.cudnn_benchmark,
            "cudnn_deterministic": self.cudnn_deterministic,
            "cuda_matmul_allow_tf32": self.cuda_matmul_allow_tf32,
            "cudnn_allow_tf32": self.cudnn_allow_tf32,
        }


def audit(
    *,
    root: Path,
    registry_id: str,
    model_dir: Path,
    device: torch.device,
    registry_loader: RegistryLoader = TeacherRegistry.load,
    load_model_resolver: LoadModelResolver = _pinned_load_model,
    ard_loader: ArdLoader | None = None,
) -> AuditReport:
    """Run one fresh-process, bounded audit over local external/cache checkpoint bytes."""
    if registry_id not in ALLOWED_REGISTRY_IDS:
        raise AuditError(f"registry ID is not audit-allowlisted: {registry_id!r}")
    selected_root = root.resolve()
    registry = registry_loader(selected_root)
    try:
        spec = registry.spec(registry_id)
        registry.validate_external()
        cache_checkpoint = registry.checkpoint_path(spec)
        registry.validate_local_checkpoint(spec, cache_checkpoint)
    except TeacherRegistryError as exc:
        raise AuditError(str(exc)) from exc
    source_checkpoint = expected_checkpoint_path(model_dir.resolve(), spec)
    if not source_checkpoint.is_file():
        raise AuditError(
            f"external RobustBench checkpoint is missing; local-only audit will not download: {source_checkpoint}"
        )
    source_sha = sha256_file(source_checkpoint)
    if source_sha != spec.checkpoint_sha256:
        raise AuditError("external RobustBench checkpoint SHA does not match the registered ARD teacher SHA")

    with _deterministic_backend() as backend_flags:
        loader = load_model_resolver(selected_root, True)
        pinned_model = _load_pinned_local(loader, spec.upstream_model_id, model_dir.resolve(), device)
        fixed_pixels = _fixed_pixels(device)
        with torch.no_grad():
            pinned_logits = pinned_model(fixed_pixels).detach().float().cpu()
        del pinned_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        config = TeacherConfig.model_validate(
            {
                "source": "robustbench",
                "registry_id": spec.registry_id,
                "architecture": spec.architecture,
                "num_classes": 10,
                "normalization": spec.preprocessing.normalization(),
                "preprocessing_owner": spec.preprocessing.owner,
                "checkpoint": cache_checkpoint,
                "checkpoint_sha256": spec.checkpoint_sha256,
                "threat_norm": spec.threat.norm,
                "threat_epsilon": spec.threat.epsilon,
            }
        )
        loader_ard = ard_loader or (lambda value: build_teacher(value, tier="production"))
        teacher = loader_ard(config).to(device)
        teacher.eval()
        _validate_teacher_contract(teacher, spec.preprocessing.owner)
        with torch.no_grad():
            ard_logits = teacher.logits(fixed_pixels, require_input_grad=False).detach().float().cpu()
        max_abs_diff = float((ard_logits - pinned_logits).abs().amax())
        try:
            torch.testing.assert_close(ard_logits, pinned_logits, rtol=0.0, atol=PARITY_ATOL)
        except AssertionError as exc:
            raise AuditError(f"strict ARD/pinned RobustBench FP32 logit parity failed (atol={PARITY_ATOL})") from exc
        gradient_l1 = _validate_input_gradient(teacher, fixed_pixels)
        pgd_linf = _validate_one_step_pgd(teacher, fixed_pixels)
        if any(parameter.grad is not None for parameter in teacher.parameters()):
            raise AuditError("frozen teacher parameter gradients must remain None")
    return AuditReport(
        registry_id=spec.registry_id,
        upstream_model_id=spec.upstream_model_id,
        device=str(device),
        checkpoint_sha256=source_sha,
        preprocessing_owner=teacher.metadata.preprocessing_owner,
        normalization_profile=teacher.metadata.normalization.profile,
        parity_atol=PARITY_ATOL,
        max_abs_diff=max_abs_diff,
        logits_dtype=str(ard_logits.dtype),
        backend_flags=backend_flags.report(),
        input_gradient_l1=gradient_l1,
        pgd_linf=pgd_linf,
    )


def _load_pinned_local(loader: LoadModel, upstream_model_id: str, model_dir: Path, device: torch.device) -> nn.Module:
    model = loader(model_name=upstream_model_id, model_dir=model_dir, dataset="cifar10", threat_model="Linf")
    if not isinstance(model, nn.Module):
        raise AuditError("pinned RobustBench load_model did not return a torch.nn.Module")
    model = model.to(device)
    model.eval()
    return model


def _fixed_pixels(device: torch.device) -> torch.Tensor:
    return torch.linspace(0.0, 1.0, steps=3 * 32 * 32, dtype=torch.float32, device=device).reshape(1, 3, 32, 32)


def _validate_teacher_contract(teacher: TeacherAdapter, expected_owner: str) -> None:
    if teacher.training or teacher.model.training:
        raise AuditError("strict ARD teacher must be in eval mode")
    if teacher.metadata.preprocessing_owner != expected_owner:
        raise AuditError("strict ARD teacher preprocessing owner does not match the registered teacher")
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise AuditError("strict ARD teacher parameters must be frozen")


def _validate_input_gradient(teacher: TeacherAdapter, pixels: torch.Tensor) -> float:
    inputs = pixels.detach().clone().requires_grad_(True)
    logits = teacher.logits(inputs, require_input_grad=True)
    if tuple(logits.shape) != (1, 10) or not torch.isfinite(logits).all():
        raise AuditError("strict ARD teacher produced invalid logits for input-gradient audit")
    gradient = torch.autograd.grad(logits.square().sum(), inputs, only_inputs=True)[0]
    if not torch.isfinite(gradient).all():
        raise AuditError("teacher input gradient is non-finite")
    magnitude = float(gradient.detach().abs().sum().cpu())
    if magnitude <= 0.0:
        raise AuditError("teacher input gradient must be nonzero")
    return magnitude


def _validate_one_step_pgd(teacher: TeacherAdapter, pixels: torch.Tensor) -> float:
    labels = torch.zeros((1,), dtype=torch.long, device=pixels.device)
    attack = LinfPGD(
        AttackConfig(
            epsilon="8/255",
            step_size="2/255",
            steps=1,
            random_start=False,
            loss="ce",
            student_mode="eval",
            teacher_mode="eval",
        )
    )
    prior_mode = teacher.training
    prior_model_mode = teacher.model.training
    result = attack.generate(AttackRequest(inputs=pixels, labels=labels, student=teacher))
    adversarial = result.adversarial
    if teacher.training != prior_mode or teacher.model.training != prior_model_mode:
        raise AuditError("one-step CE PGD did not restore frozen teacher eval mode")
    delta = (adversarial - pixels).detach().abs().amax()
    bound = float(delta.cpu())
    if abs(bound - result.max_abs_delta) > 1e-7:
        raise AuditError("one-step CE PGD result.max_abs_delta does not match the independently measured perturbation")
    if bound > PGD_STEP_SIZE + 1e-7:
        raise AuditError(f"one-step CE PGD violates canonical 2/255 step bound: {bound}")
    if bound <= 0.0:
        raise AuditError("one-step CE PGD must make a nonzero update on the fixed audit batch")
    if not torch.isfinite(adversarial).all() or adversarial.amin() < 0 or adversarial.amax() > 1:
        raise AuditError("one-step CE PGD violates pixel clamp [0, 1]")
    return bound


@contextmanager
def _deterministic_backend() -> Iterator[BackendFlags]:
    previous = BackendFlags(
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
    )
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    enforced = BackendFlags(
        deterministic_algorithms=True,
        cudnn_benchmark=False,
        cudnn_deterministic=True,
        cuda_matmul_allow_tf32=False,
        cudnn_allow_tf32=False,
    )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield enforced
    finally:
        torch.use_deterministic_algorithms(previous.deterministic_algorithms, warn_only=previous_warn_only)
        torch.backends.cudnn.benchmark = previous.cudnn_benchmark
        torch.backends.cudnn.deterministic = previous.cudnn_deterministic
        torch.backends.cuda.matmul.allow_tf32 = previous.cuda_matmul_allow_tf32
        torch.backends.cudnn.allow_tf32 = previous.cudnn_allow_tf32


def _default_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--registry-id", required=True, choices=sorted(ALLOWED_REGISTRY_IDS))
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", default=None, help="one local device, default cuda:0 when available else cpu")
    args = parser.parse_args()
    try:
        report = audit(
            root=args.root,
            registry_id=args.registry_id,
            model_dir=args.model_dir,
            device=torch.device(args.device) if args.device is not None else _default_device(),
        )
    except (AcquisitionError, AuditError) as exc:
        parser.error(str(exc))
    print(json.dumps(asdict(report), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
