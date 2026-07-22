"""CPU clean-room evidence for documented local-versus-official TRADES contracts.

The mandatory tests intentionally do not import the legacy upstream package.
They encode its published fixed-batch equations independently, so they remain
valid on CPU-only current PyTorch environments.  The source-only evidence is
explicitly opt-in and never executes the legacy CUDA-dependent runtime.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml
from torch import nn

from ard.attacks import LinfPGD
from ard.attacks.base import AttackRequest
from ard.config.schema import AttackConfig, MethodConfig
from ard.objectives import TRADESObjective

pytestmark = [pytest.mark.t2, pytest.mark.regression]

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_TRADES_SHA = "6e8e11b7c281371c2f027ffadfbaea80361f09de"


def _official_trades_loss(
    *, clean_logits: torch.Tensor, adversarial_logits: torch.Tensor, labels: torch.Tensor, beta: float
) -> torch.Tensor:
    """Clean-room reference for the pinned official outer objective.

    Its clean softmax target is deliberately not detached.  ``reduction='sum'
    / batch`` is the legacy ``KLDivLoss(size_average=False)`` contract.
    """
    clean = F.cross_entropy(clean_logits, labels)
    robust = (
        F.kl_div(F.log_softmax(adversarial_logits, dim=1), F.softmax(clean_logits, dim=1), reduction="sum")
        / clean_logits.shape[0]
    )
    return clean + beta * robust


def _local_trades_loss(
    *, clean_logits: torch.Tensor, adversarial_logits: torch.Tensor, labels: torch.Tensor, beta: float
) -> torch.Tensor:
    return TRADESObjective(beta=beta)(
        student_logits=adversarial_logits, clean_student_logits=clean_logits, labels=labels
    ).total.mean()


def test_fixed_batch_outer_loss_beta_reduction_and_clean_target_gradient_difference() -> None:
    labels = torch.tensor([0, 2])
    clean_base = torch.tensor([[0.4, -0.2, 0.7], [-0.5, 0.6, 0.1]], dtype=torch.float64)
    adversarial_base = torch.tensor([[0.1, 0.5, -0.4], [0.8, -0.3, 0.2]], dtype=torch.float64)
    beta = 6.0

    local_clean = clean_base.clone().requires_grad_()
    local_adversarial = adversarial_base.clone().requires_grad_()
    local_loss = _local_trades_loss(
        clean_logits=local_clean, adversarial_logits=local_adversarial, labels=labels, beta=beta
    )
    local_clean_gradient, local_adversarial_gradient = torch.autograd.grad(local_loss, (local_clean, local_adversarial))

    official_clean = clean_base.clone().requires_grad_()
    official_adversarial = adversarial_base.clone().requires_grad_()
    official_loss = _official_trades_loss(
        clean_logits=official_clean, adversarial_logits=official_adversarial, labels=labels, beta=beta
    )
    official_clean_gradient, official_adversarial_gradient = torch.autograd.grad(
        official_loss, (official_clean, official_adversarial)
    )

    assert torch.allclose(local_loss, official_loss, rtol=0, atol=1e-14)
    assert torch.allclose(local_adversarial_gradient, official_adversarial_gradient, rtol=0, atol=1e-14)
    assert not torch.allclose(local_clean_gradient, official_clean_gradient, rtol=0, atol=1e-14)
    # The local clean branch receives CE only; the official branch additionally
    # receives the non-detached target-side KL gradient.
    clean_for_ce = clean_base.clone().requires_grad_()
    expected_local_clean = torch.autograd.grad(F.cross_entropy(clean_for_ce, labels), clean_for_ce)[0]
    assert torch.allclose(local_clean_gradient, expected_local_clean, rtol=0, atol=1e-14)


def test_one_sgd_delta_records_the_documented_nondetached_clean_branch_difference() -> None:
    inputs = torch.tensor([[0.2, -0.1], [0.7, 0.3]], dtype=torch.float64)
    adversarial_inputs = torch.tensor([[0.4, -0.2], [0.5, 0.9]], dtype=torch.float64)
    labels = torch.tensor([0, 1])
    initial_weight = torch.tensor([[0.2, -0.4], [0.3, 0.1]], dtype=torch.float64)

    def one_step(local: bool) -> tuple[torch.Tensor, torch.Tensor]:
        model = nn.Linear(2, 2, bias=False, dtype=torch.float64)
        with torch.no_grad():
            model.weight.copy_(initial_weight)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        clean_logits, adversarial_logits = model(inputs), model(adversarial_inputs)
        loss = (
            _local_trades_loss(
                clean_logits=clean_logits, adversarial_logits=adversarial_logits, labels=labels, beta=6.0
            )
            if local
            else _official_trades_loss(
                clean_logits=clean_logits, adversarial_logits=adversarial_logits, labels=labels, beta=6.0
            )
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.detach(), model.weight.detach()

    local_loss, local_weight = one_step(local=True)
    official_loss, official_weight = one_step(local=False)
    assert torch.allclose(local_loss, official_loss, rtol=0, atol=1e-14)
    assert not torch.allclose(local_weight, official_weight, rtol=0, atol=1e-14)


class _ModeRecordingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[0.3, -0.2, 0.1, 0.4], [-0.1, 0.5, 0.2, -0.3]]))
        self.observed_modes: list[bool] = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.observed_modes.append(self.training)
        return self.linear(inputs.flatten(1))


def test_clean_room_attack_initialization_projection_clamp_and_mode_contracts() -> None:
    inputs = torch.tensor([[[[0.0, 1.0], [0.03, 0.97]]]])
    labels = torch.tensor([0])
    epsilon = "1/5"
    attack = LinfPGD(
        AttackConfig(
            epsilon=epsilon,
            step_size="1/10",
            steps=1,
            random_start=True,
            loss="kl",
            kl_target="student_clean",
            student_mode="eval",
        )
    )
    model = _ModeRecordingModel()
    model.train()
    expected_generator = torch.Generator(device="cpu").manual_seed(23)
    expected_delta = torch.empty_like(inputs).uniform_(-0.2, 0.2, generator=expected_generator)
    expected_delta = (inputs + expected_delta).clamp(0, 1) - inputs
    result = attack.generate(
        AttackRequest(
            inputs=inputs,
            labels=labels,
            student=model,
            generator=torch.Generator(device="cpu").manual_seed(23),
        )
    )

    assert torch.equal(result.initial_delta, expected_delta)
    assert float(result.adversarial.amin()) >= 0.0 and float(result.adversarial.amax()) <= 1.0
    assert result.max_abs_delta <= 0.2 + 1e-7
    assert model.observed_modes == [False, False]
    assert model.training

    # Official TRADES begins at 0.001 Gaussian noise; its first construction
    # is not the local uniform-in-epsilon, immediately-clamped initialization.
    upstream_initial = inputs + 0.001 * torch.randn(inputs.shape, generator=torch.Generator().manual_seed(23))
    assert not torch.equal(result.initial_delta, upstream_initial - inputs)
    assert float(upstream_initial.amin()) < 0.0 or float(upstream_initial.amax()) > 1.0
    upstream_gradient = torch.tensor([[[[-1.0, 1.0], [1.0, -1.0]]]])
    upstream_stepped = upstream_initial + 0.1 * upstream_gradient.sign()
    upstream_projected = torch.minimum(torch.maximum(upstream_stepped, inputs - 0.2), inputs + 0.2).clamp(0, 1)
    assert float((upstream_projected - inputs).abs().amax()) <= 0.2 + 1e-7
    assert float(upstream_projected.amin()) >= 0.0 and float(upstream_projected.amax()) <= 1.0

    local_defaults = AttackConfig()
    assert (local_defaults.epsilon, local_defaults.step_size, local_defaults.steps) == ("8/255", "2/255", 10)
    official_defaults = (0.031, 0.007, 10, 6.0)
    assert (local_defaults.epsilon, local_defaults.step_size, local_defaults.steps) != (".031", ".007", 10)
    trades_method = MethodConfig(
        id="trades",
        version=1,
        attack=AttackConfig(loss="kl", kl_target="student_clean"),
    )
    assert trades_method.trades_beta == official_defaults[3]


@pytest.mark.upstream
def test_opt_in_pinned_trades_source_evidence_without_legacy_import() -> None:
    if os.environ.get("ARD_TRADES_SOURCE_EVIDENCE") != "1":
        pytest.skip("set ARD_TRADES_SOURCE_EVIDENCE=1 after bootstrapping the pinned TRADES checkout")
    checkout = ROOT / ".external" / "trades"
    lock = yaml.safe_load((ROOT / "external.lock.yaml").read_text(encoding="utf-8"))
    assert lock["repositories"]["trades"]["commit"] == UPSTREAM_TRADES_SHA
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip() == UPSTREAM_TRADES_SHA
    assert not subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=checkout, text=True
    ).strip()
    source = (checkout / "trades.py").read_text(encoding="utf-8")
    for contract in (
        "model.eval()",
        "0.001 * torch.randn",
        "x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())",
        "x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)",
        "x_adv = torch.clamp(x_adv, 0.0, 1.0)",
        "model.train()",
        "loss_natural = F.cross_entropy",
        "F.softmax(model(x_natural), dim=1)",
        "loss = loss_natural + beta * loss_robust",
    ):
        assert contract in source
    runner = (checkout / "train_trades_cifar10.py").read_text(encoding="utf-8")
    assert "transforms.ToTensor()" in runner
    assert "transforms.Normalize" not in runner
    assert "model = WideResNet().to(device)" in runner
