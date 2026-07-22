"""FP32 pixel-space L-infinity PGD with explicit loss and model-mode contracts."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn

from ard.config.schema import AttackConfig

from .base import AttackGenerator, AttackRequest, AttackResult


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


class LinfPGD(AttackGenerator):
    def __init__(self, config: AttackConfig) -> None:
        if config.norm != "linf" or config.input_domain != "pixel_0_1":
            raise ValueError("LinfPGD supports only linf attacks in pixel [0,1]")
        self.config = config

    def _target_logits(self, request: AttackRequest, clean: torch.Tensor) -> torch.Tensor | None:
        if self.config.loss == "ce":
            return None
        if request.target_logits is not None:
            return request.target_logits.detach().float()
        if self.config.kl_target == "student_clean":
            with torch.no_grad(), torch.autocast(device_type=clean.device.type, enabled=False):
                return request.student(clean).detach().float()
        if self.config.kl_target == "teacher_clean":
            if request.teacher is None:
                raise ValueError("teacher_clean KL PGD requires a teacher")
            with torch.no_grad(), torch.autocast(device_type=clean.device.type, enabled=False):
                return request.teacher(clean).detach().float()
        raise RuntimeError("validated KL attack has no target source")

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor, target_logits: torch.Tensor | None) -> torch.Tensor:
        if self.config.loss == "ce":
            return F.cross_entropy(logits, labels)
        assert target_logits is not None
        temperature = self.config.temperature
        loss = F.kl_div(
            F.log_softmax(logits / temperature, dim=1),
            F.softmax(target_logits / temperature, dim=1),
            reduction="batchmean",
        )
        return loss * (temperature * temperature) if self.config.temperature_squared else loss

    def generate(self, request: AttackRequest) -> AttackResult:
        _validate_pixels(request.inputs)
        clean = request.inputs.detach().float()
        epsilon = self.config.epsilon_value
        step_size = self.config.step_size_value
        assert epsilon is not None and step_size is not None  # resolved by AttackConfig validation
        delta = torch.zeros_like(clean)
        if self.config.random_start and epsilon > 0:
            delta.uniform_(-epsilon, epsilon, generator=request.generator)
            delta = (clean + delta).clamp(0, 1) - clean
        initial_delta = delta.detach().clone()
        adversarial = (clean + delta).detach()
        losses: list[float] = []
        with _temporary_modes(
            request.student,
            request.teacher,
            student_train=self.config.student_mode == "train",
            teacher_train=self.config.teacher_mode == "train",
        ):
            target_logits = self._target_logits(request, clean)
            for _ in range(self.config.steps):
                adversarial.requires_grad_(True)
                with torch.autocast(device_type=adversarial.device.type, enabled=False):
                    logits = request.student(adversarial.float())
                    loss = self._loss(logits.float(), request.labels, target_logits)
                gradient = torch.autograd.grad(loss, adversarial, only_inputs=True)[0]
                losses.append(float(loss.detach().cpu()))
                adversarial = adversarial.detach() + step_size * gradient.detach().sign()
                delta = (adversarial - clean).clamp(-epsilon, epsilon)
                adversarial = (clean + delta).clamp(0, 1).detach()
        final_delta = adversarial - clean
        return AttackResult(
            adversarial=adversarial,
            initial_delta=initial_delta,
            step_losses=tuple(losses),
            max_abs_delta=float(final_delta.detach().abs().amax().cpu()),
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
