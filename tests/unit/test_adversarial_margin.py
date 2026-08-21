from __future__ import annotations

import pytest
import torch

from ard.objectives import ObjectiveTerms


def _terms(batch: int = 2) -> ObjectiveTerms:
    zeros = torch.zeros(batch, requires_grad=False)
    return ObjectiveTerms(
        hard=zeros.clone(),
        kd=zeros.clone(),
        regularization=zeros.clone(),
        adversarial_kd=zeros.clone(),
        clean_kd=zeros.clone(),
    )


def test_probability_margin_hinge_has_expected_values_and_gradient() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    labels = torch.tensor([0, 0])
    result = _terms().add_adversarial_margin(
        logits,
        labels,
        torch.tensor([0.8, 0.8]),
        torch.ones(2),
        coefficient=1.0,
    )
    margin = torch.tensor([torch.tanh(torch.tensor(1.0)), -torch.tanh(torch.tensor(1.0))])
    expected = torch.relu(torch.tensor(0.8) - margin).sum()
    assert torch.allclose(result.hard.sum(), expected, atol=1e-6)
    result.hard.sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_margin_target_is_detached_and_multiplier_is_full_batch() -> None:
    logits = torch.tensor([[0.0, 1.0], [1.0, 0.0]], requires_grad=True)
    labels = torch.tensor([0, 0])
    target = torch.tensor([0.5, 0.5], requires_grad=True)
    result = _terms().add_adversarial_margin(logits, labels, target, torch.tensor([1.0, 0.0]), coefficient=2.0)
    result.hard.sum().backward()
    assert target.grad is None
    assert torch.allclose(result.hard[1], torch.tensor(0.0))


def test_margin_rejects_nonfinite_or_negative_multiplier() -> None:
    logits = torch.zeros((2, 3))
    labels = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="finite"):
        _terms().add_adversarial_margin(
            logits, labels, torch.tensor([0.1, float("nan")]), torch.ones(2), coefficient=1.0
        )
    with pytest.raises(ValueError, match="non-negative"):
        _terms().add_adversarial_margin(logits, labels, torch.ones(2), torch.tensor([1.0, -1.0]), coefficient=1.0)
