"""FP32 pixel-space L-infinity PGD with explicit loss and model-mode contracts."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final

import torch
import torch.nn.functional as F
from torch import nn

from ard.config.schema import AttackConfig

from .base import AttackGenerator, AttackRequest, AttackResult

_U64_MASK: Final[int] = (1 << 64) - 1
_STREAM_OFFSET: Final[int] = 0x9E3779B97F4A7C15


def _splitmix64(value: int) -> int:
    """Stable scalar mixer used only to derive a PyTorch Generator seed."""
    value = (value + _STREAM_OFFSET) & _U64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _U64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _U64_MASK
    return (value ^ (value >> 31)) & _U64_MASK


def _stream_hash(stream_tag: str) -> int:
    # FNV-1a is stable across processes and Python versions; unlike hash(), it
    # is not salted per process.  The resulting value is only a seed component.
    value = 0xCBF29CE484222325
    for byte in stream_tag.encode("utf-8"):
        value = ((value ^ byte) * 0x100000001B3) & _U64_MASK
    return value


def _sample_seed(*, attack_seed: int, epoch: int, source_id: int, stream_tag: str, restart_index: int) -> int:
    if attack_seed < 0 or epoch < 0 or source_id < 0 or restart_index < 0:
        raise ValueError("sample-keyed random-start key fields must be non-negative")
    value = _splitmix64(attack_seed & _U64_MASK)
    for component in (epoch, source_id, _stream_hash(stream_tag), restart_index):
        value = _splitmix64(value ^ (component & _U64_MASK))
    # torch.Generator.manual_seed accepts signed 64-bit seeds.
    return value & ((1 << 63) - 1)


def sample_keyed_random_start(
    clean: torch.Tensor,
    source_ids: torch.Tensor,
    *,
    attack_seed: int,
    epoch: int,
    stream_tag: str = "train_pgd",
    restart_index: int = 0,
) -> torch.Tensor:
    """Return source-keyed uniform random starts with batch-local memory.

    Each source gets its own generator seeded only by the frozen key fields.
    The reference implementation intentionally uses PyTorch's native
    generator rather than a custom tensor PRNG; vectorization can be added
    later only if it preserves this contract and passes the same tests.
    """
    if source_ids.ndim != 1 or source_ids.shape[0] != clean.shape[0]:
        raise ValueError("source_ids must be a one-dimensional vector matching the batch")
    if source_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("source_ids must contain integer stable IDs")
    # CPU is the canonical draw device: the key must produce the same values
    # when a source moves between ranks/devices, and only this batch-local
    # tensor is transferred to the attack device.
    output_cpu = torch.empty(clean.shape, dtype=clean.dtype, device="cpu")
    for position, raw_source_id in enumerate(source_ids.detach().to(device="cpu", dtype=torch.long).tolist()):
        generator = torch.Generator(device="cpu").manual_seed(
            _sample_seed(
                attack_seed=attack_seed,
                epoch=epoch,
                source_id=int(raw_source_id),
                stream_tag=stream_tag,
                restart_index=restart_index,
            )
        )
        output_cpu[position].uniform_(-1.0, 1.0, generator=generator)
    return output_cpu if clean.device.type == "cpu" else output_cpu.to(device=clean.device)


@contextmanager
def _temporary_modes(
    student: nn.Module, teacher: nn.Module | None, *, student_train: bool, teacher_train: bool
) -> Iterator[None]:
    student_mode = student.training
    teacher_mode = None if teacher is None else teacher.training
    student.train(student_train)
    if teacher is not None:
        teacher.train(teacher_train)
    try:
        yield
    finally:
        student.train(student_mode)
        if teacher is not None and teacher_mode is not None:
            teacher.train(teacher_mode)


def _validate_pixels(inputs: torch.Tensor) -> None:
    if not inputs.is_floating_point():
        raise TypeError("attack inputs must be floating-point pixels")
    if inputs.numel() and (inputs.detach().amin() < 0 or inputs.detach().amax() > 1):
        raise ValueError("attack inputs must lie in pixel domain [0, 1]")


def _budget(value: torch.Tensor | None, scalar: float, *, batch: int, device: torch.device, name: str) -> torch.Tensor:
    if value is None:
        return torch.full((batch, 1, 1, 1), scalar, device=device, dtype=torch.float32)
    if value.ndim == 1 and value.shape[0] == batch:
        value = value.reshape(batch, 1, 1, 1)
    if value.shape != (batch, 1, 1, 1):
        raise ValueError(f"{name} override must have shape [batch] or [batch,1,1,1]")
    value = value.detach().to(device=device, dtype=torch.float32)
    if not torch.isfinite(value).all() or bool((value < 0).any()):
        raise ValueError(f"{name} override must be finite and non-negative")
    return value


@dataclass(frozen=True)
class PGDTrajectory:
    """Device-side output of ``LinfPGD.perturb`` (no host-derived fields)."""

    adversarial: torch.Tensor
    initial_delta: torch.Tensor
    step_losses: tuple[torch.Tensor, ...]
    captured_adversarial: torch.Tensor | None


class LinfPGD(AttackGenerator):
    def __init__(self, config: AttackConfig) -> None:
        if config.norm != "linf" or config.input_domain != "pixel_0_1":
            raise ValueError("LinfPGD supports only linf attacks in pixel [0,1]")
        self.config = config

    @property
    def requires_teacher_clean_target(self) -> bool:
        return self.config.loss == "kl" and self.config.kl_target == "teacher_clean"

    def _resolve_target(
        self, request: AttackRequest, clean: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return ``(target_logits, target_probabilities)``; at most one is non-None.

        Resolved once per call, before the step loop, so the target stays
        fixed across every PGD step regardless of which source produced it.
        """
        if self.config.loss == "ce":
            return None, None
        if self.config.kl_target == "rectified":
            if request.target_logits is not None:
                raise ValueError("kl_target='rectified' takes target_probabilities, not target_logits")
            if request.target_probabilities is None:
                raise ValueError("rectified KL PGD requires target_probabilities on the request")
            return None, request.target_probabilities.detach().float()
        if request.target_probabilities is not None:
            raise ValueError("target_probabilities is only accepted when kl_target='rectified'")
        if request.target_logits is not None:
            return request.target_logits.detach().float(), None
        if self.config.kl_target == "student_clean":
            with torch.no_grad(), torch.autocast(device_type=clean.device.type, enabled=False):
                return request.student(clean).detach().float(), None
        if self.config.kl_target == "teacher_clean":
            if request.teacher is None:
                raise ValueError("teacher_clean KL PGD requires a teacher")
            with torch.no_grad(), torch.autocast(device_type=clean.device.type, enabled=False):
                return request.teacher(clean).detach().float(), None
        raise RuntimeError("validated KL attack has no target source")

    def _loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        target_logits: torch.Tensor | None,
        target_probabilities: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.config.loss == "ce":
            return F.cross_entropy(logits, labels)
        temperature = self.config.temperature
        if target_probabilities is not None:
            # Already a resolved distribution (e.g. an EMA-blended rectified
            # label): no further temperature/softmax resolution on the
            # target side.  The student side is still divided by
            # temperature for consistency with the other KL targets; callers
            # that want a plain, untempered cross-entropy (as ADR does) set
            # attack.temperature=1.0 explicitly rather than special-casing it
            # here.
            target = target_probabilities
        else:
            assert target_logits is not None
            target = F.softmax(target_logits / temperature, dim=1)
        loss = F.kl_div(F.log_softmax(logits / temperature, dim=1), target, reduction="batchmean")
        return loss * (temperature * temperature) if self.config.temperature_squared else loss

    def budgets(self, request: AttackRequest) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-sample ``[batch,1,1,1]`` FP32 epsilon and step-size tensors.

        Without overrides this is two fill kernels and no host synchronization;
        an override is validated on the host (one synchronization each).
        """
        batch, device = request.inputs.shape[0], request.inputs.device
        epsilon = self.config.epsilon_value
        step_size = self.config.step_size_value
        assert epsilon is not None and step_size is not None  # resolved by AttackConfig validation
        epsilon_tensor = _budget(request.epsilon_override, epsilon, batch=batch, device=device, name="epsilon")
        step_tensor = _budget(request.step_size_override, step_size, batch=batch, device=device, name="step_size")
        return epsilon_tensor, step_tensor

    def perturb(
        self,
        request: AttackRequest,
        *,
        epsilon: torch.Tensor,
        step_size: torch.Tensor,
        unit_noise: torch.Tensor | None,
    ) -> PGDTrajectory:
        """Synchronization-free PGD core shared by ``generate`` and the CUDA-graph step.

        ``unit_noise`` is the raw ``U(-1, 1)`` random-start draw (``None``: no
        random start). It is scaled by ``epsilon`` and clamped to the pixel box
        here, exactly as ``generate`` always did. No host validation happens
        here and no tensor is read back to the host (except the debug-only
        ``trace_step_losses``, which returns device tensors for the caller to
        read), so the call can be captured in a CUDA graph. ``generate`` is the
        validating wrapper; call this directly only after the same checks.
        """
        clean = request.inputs.detach().float()
        if unit_noise is None:
            delta = torch.zeros_like(clean)
        else:
            delta = unit_noise * epsilon
            delta = (clean + delta).clamp(0, 1) - clean
        initial_delta = delta.detach().clone()
        adversarial = (clean + delta).detach()
        losses: list[torch.Tensor] | None = [] if self.config.trace_step_losses else None
        captured = None
        with _temporary_modes(
            request.student,
            request.teacher,
            student_train=self.config.student_mode == "train",
            teacher_train=self.config.teacher_mode == "train",
        ):
            target_logits, target_probabilities = self._resolve_target(request, clean)
            for step in range(1, self.config.steps + 1):
                adversarial.requires_grad_(True)
                with torch.autocast(device_type=adversarial.device.type, enabled=False):
                    logits = request.student(adversarial.float())
                    loss = self._loss(logits.float(), request.labels, target_logits, target_probabilities)
                gradient = torch.autograd.grad(loss, adversarial, only_inputs=True)[0]
                if losses is not None:
                    losses.append(loss.detach())
                adversarial = adversarial.detach() + step_size * gradient.detach().sign()
                delta = torch.maximum(torch.minimum(adversarial - clean, epsilon), -epsilon)
                adversarial = (clean + delta).clamp(0, 1).detach()
                if step == request.capture_step:
                    captured = adversarial.clone()
        return PGDTrajectory(
            adversarial=adversarial,
            initial_delta=initial_delta,
            step_losses=() if losses is None else tuple(losses),
            captured_adversarial=captured,
        )

    def generate(self, request: AttackRequest) -> AttackResult:
        _validate_pixels(request.inputs)
        clean = request.inputs.detach().float()
        epsilon_tensor, step_tensor = self.budgets(request)
        if bool((step_tensor > epsilon_tensor).any()):
            raise ValueError("per-sample PGD step size cannot exceed epsilon")
        if request.capture_step is not None and not 1 <= request.capture_step < self.config.steps:
            raise ValueError("captured PGD step must be a strict positive prefix of the configured trajectory")
        unit_noise = None
        if self.config.random_start and bool((epsilon_tensor > 0).any()):
            if self.config.random_start_keying == "sample_keyed_v1":
                if request.source_ids is None or request.epoch is None or request.attack_seed is None:
                    raise ValueError("sample-keyed random starts require source_ids, epoch, and attack_seed")
                unit_noise = sample_keyed_random_start(
                    clean,
                    request.source_ids,
                    attack_seed=request.attack_seed,
                    epoch=request.epoch,
                    stream_tag=request.stream_tag,
                    restart_index=request.restart_index,
                )
            else:
                unit_noise = torch.zeros_like(clean)
                unit_noise.uniform_(-1.0, 1.0, generator=request.generator)
        trajectory = self.perturb(request, epsilon=epsilon_tensor, step_size=step_tensor, unit_noise=unit_noise)
        adversarial = trajectory.adversarial
        final_delta = adversarial - clean
        return AttackResult(
            adversarial=adversarial,
            initial_delta=trajectory.initial_delta,
            step_losses=tuple(float(loss.cpu()) for loss in trajectory.step_losses),
            max_abs_delta=float(final_delta.detach().abs().amax().cpu()),
            captured_adversarial=trajectory.captured_adversarial,
        )


def teacher_input_gradient(teacher: nn.Module, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Differentiate teacher output w.r.t. pixels while keeping all teacher parameters frozen."""
    _validate_pixels(inputs)
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise ValueError("teacher parameters must be frozen before requesting input gradients")
    original_mode = teacher.training
    teacher.eval()
    pixels = inputs.detach().float().requires_grad_(True)
    try:
        with torch.autocast(device_type=pixels.device.type, enabled=False):
            loss = F.cross_entropy(teacher(pixels), labels)
        return torch.autograd.grad(loss, pixels, only_inputs=True)[0].detach()
    finally:
        teacher.train(original_mode)
