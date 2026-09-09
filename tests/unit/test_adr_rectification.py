"""ADR's rectified label (paper Eq. 3-5) and the two ADR objectives.

Wu, Wang & Chen, "Annealing Self-Distillation Rectification Improves
Adversarial Training", ICLR 2024, arXiv:2305.12118.  The mechanism is a label
replacement, not a KD term added to hard CE, so what has to be tested is (a)
the rectified target matches the paper's formula on a hand-computed example,
(b) it never receives gradient -- it is an EMA-of-student output and must be
as inert as a frozen teacher's, and (c) both objectives (PGD-AT base and
TRADES base) route the rectified target correctly.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from ard.objectives.adr import ADRObjective, ADRTRADESObjective, rectify_label


def test_rectify_label_matches_hand_computed_formula_when_ema_is_correct() -> None:
    # EMA's top class matches the true label (class 0): no discount to lambda_floor.
    ema_logits = torch.tensor([[3.0, 0.0, 0.0]])
    labels = torch.tensor([0])
    temperature = 2.0
    lambda_floor = 0.7

    rectified = rectify_label(
        ema_clean_logits=ema_logits, labels=labels, temperature=temperature, lambda_floor=lambda_floor
    )

    teacher_probabilities = F.softmax(ema_logits / temperature, dim=1)
    top = teacher_probabilities.amax(dim=1)
    true = teacher_probabilities.gather(1, labels[:, None]).squeeze(1)
    assert torch.equal(top, true)  # confirms this fixture actually exercises the "no discount" branch
    expected_lambda = lambda_floor  # clip(0.7 - 0, 0, 1)
    one_hot = torch.tensor([[1.0, 0.0, 0.0]])
    expected = expected_lambda * teacher_probabilities + (1 - expected_lambda) * one_hot
    assert torch.allclose(rectified, expected, atol=1e-6)
    assert torch.allclose(rectified.sum(dim=1), torch.ones(1), atol=1e-6)


def test_rectify_label_discounts_lambda_when_ema_disagrees_with_the_true_label() -> None:
    # EMA's top class (1) disagrees with the true label (0): lambda_i must drop
    # by exactly the size of the EMA's mistake, per Eq. 4.
    ema_logits = torch.tensor([[0.0, 3.0, 0.0]])
    labels = torch.tensor([0])
    temperature = 1.0
    lambda_floor = 0.9

    rectified = rectify_label(
        ema_clean_logits=ema_logits, labels=labels, temperature=temperature, lambda_floor=lambda_floor
    )

    teacher_probabilities = F.softmax(ema_logits, dim=1)
    top = float(teacher_probabilities.amax(dim=1))
    true = float(teacher_probabilities[0, 0])
    expected_lambda = max(0.0, min(1.0, lambda_floor - (top - true)))
    assert expected_lambda < lambda_floor  # the discount actually fired
    one_hot = torch.tensor([[1.0, 0.0, 0.0]])
    expected = expected_lambda * teacher_probabilities + (1 - expected_lambda) * one_hot
    assert torch.allclose(rectified, expected, atol=1e-6)


def test_rectify_label_clips_lambda_to_the_unit_interval() -> None:
    # A large mistake must clip lambda_i to exactly 0 (pure one-hot), not go negative.
    ema_logits = torch.tensor([[-10.0, 10.0]])
    labels = torch.tensor([0])
    rectified = rectify_label(ema_clean_logits=ema_logits, labels=labels, temperature=1.0, lambda_floor=0.5)
    assert torch.allclose(rectified, torch.tensor([[1.0, 0.0]]), atol=1e-6)


def test_rectify_label_rejects_a_target_that_still_requires_grad() -> None:
    ema_logits = torch.tensor([[1.0, 0.0]], requires_grad=True)
    with pytest.raises(ValueError, match="requires a detached EMA forward"):
        rectify_label(ema_clean_logits=ema_logits, labels=torch.tensor([0]), temperature=1.0, lambda_floor=0.7)


def test_rectified_target_never_receives_gradient_through_either_objective() -> None:
    """The rectified target is an EMA output computed under no_grad and passed in
    as a plain tensor; even so, confirm neither objective differentiates through it
    if a caller ever passed one that still tracked gradient by mistake."""
    torch.manual_seed(0)
    ema_logits = (torch.randn(4, 5) * 2).requires_grad_(True)
    labels = torch.randint(0, 5, (4,))
    rectified = rectify_label(
        ema_clean_logits=ema_logits.detach(), labels=labels, temperature=2.0, lambda_floor=0.8
    ).requires_grad_(True)
    student_logits = torch.randn(4, 5, requires_grad=True)

    terms = ADRObjective()(student_logits=student_logits, labels=labels, rectified_target_probabilities=rectified)
    (grad,) = torch.autograd.grad(terms.total.mean(), rectified, allow_unused=True, materialize_grads=True)
    assert torch.equal(grad, torch.zeros_like(grad))


def test_adr_objective_is_unreduced_and_matches_hand_computed_kl() -> None:
    student_logits = torch.tensor([[2.0, 0.0, -1.0], [0.5, 1.5, 0.0]], requires_grad=True)
    labels = torch.tensor([0, 1])
    rectified = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])

    terms = ADRObjective()(student_logits=student_logits, labels=labels, rectified_target_probabilities=rectified)
    # ``probabilities_to_student_kl`` returns KL(target || student), which is
    # plain cross-entropy plus the target's own (student-independent) negative
    # entropy -- not literally the cross-entropy value.  The two share a
    # gradient with respect to the student, which is the property this
    # module's design relies on and which the second assertion below checks.
    log_student = F.log_softmax(student_logits, dim=1)
    expected_kl = F.kl_div(log_student, rectified, reduction="none").sum(dim=1)

    assert terms.hard.shape == (2,)
    assert torch.equal(terms.kd, torch.zeros(2))
    assert torch.allclose(terms.hard, expected_kl, atol=1e-6)
    terms.total.mean().backward()
    assert student_logits.grad is not None

    kl_grad = student_logits.grad.clone()
    student_logits.grad = None
    plain_ce = -(rectified * F.log_softmax(student_logits, dim=1)).sum(dim=1)
    plain_ce.mean().backward()
    assert torch.allclose(kl_grad, student_logits.grad, atol=1e-6), (
        "KL-against-a-fixed-target and plain cross-entropy against it must produce "
        "identical student gradients (they differ only by the target's constant entropy)"
    )


def test_adr_objective_requires_a_rectified_target() -> None:
    with pytest.raises(ValueError, match="requires a rectified target"):
        ADRObjective()(student_logits=torch.zeros(2, 3), labels=torch.tensor([0, 1]))


def test_adr_trades_objective_replaces_only_the_natural_ce_label_and_keeps_the_kl_term_intact() -> None:
    """The KL robustness term must be byte-identical to plain TRADESObjective's
    (same clean/adversarial pair, same beta, same non-detached gradient routing);
    only the natural-CE branch's label source changes."""
    from ard.objectives.trades import TRADESObjective

    torch.manual_seed(1)
    student_adv = torch.randn(3, 4, requires_grad=True)
    student_clean = torch.randn(3, 4, requires_grad=True)
    labels = torch.tensor([0, 1, 2])
    rectified = torch.softmax(torch.randn(3, 4), dim=1)
    beta = 6.0

    adr_terms = ADRTRADESObjective(beta=beta)(
        student_logits=student_adv,
        labels=labels,
        clean_student_logits=student_clean,
        rectified_target_probabilities=rectified,
    )
    plain_trades_kd = TRADESObjective(beta=beta)(
        student_logits=student_adv, labels=labels, clean_student_logits=student_clean
    ).kd
    assert torch.allclose(adr_terms.kd, plain_trades_kd, atol=1e-6)

    expected_hard = F.kl_div(F.log_softmax(student_clean, dim=1), rectified, reduction="none").sum(dim=1)
    assert torch.allclose(adr_terms.hard, expected_hard, atol=1e-6)

    # The clean branch must still receive gradient through the KL term (same
    # contract as plain TRADES, docs/debugging/0028), independent of the
    # rectified-label CE term also touching it.
    (grad,) = torch.autograd.grad(adr_terms.kd.mean(), student_clean, retain_graph=True)
    assert torch.linalg.vector_norm(grad) > 0


def test_adr_trades_objective_requires_clean_logits_and_a_rectified_target() -> None:
    labels = torch.tensor([0, 1])
    rectified = torch.softmax(torch.randn(2, 3), dim=1)
    with pytest.raises(ValueError, match="requires student logits on the clean input"):
        ADRTRADESObjective(beta=6.0)(
            student_logits=torch.zeros(2, 3), labels=labels, rectified_target_probabilities=rectified
        )
    with pytest.raises(ValueError, match="requires a rectified target"):
        ADRTRADESObjective(beta=6.0)(
            student_logits=torch.zeros(2, 3), labels=labels, clean_student_logits=torch.zeros(2, 3)
        )


def test_module_level_flags_declare_the_right_extra_forwards() -> None:
    assert ADRObjective.requires_rectified_target_probabilities is True
    assert ADRObjective.requires_clean_student_logits is False
    assert ADRTRADESObjective.requires_rectified_target_probabilities is True
    assert ADRTRADESObjective.requires_clean_student_logits is True


def test_cosine_schedule_reference_values_match_the_papers_cifar10_endpoints() -> None:
    """Sanity check the schedule this module's caller is expected to use
    (ard.schedules.cosine_value.cosine_anneal) against the paper's stated
    CIFAR-10 endpoints, so a future change to either side is caught here too."""
    from ard.schedules.cosine_value import cosine_anneal

    total = 200 * 391  # 200 epochs, CIFAR-10 batch 128 -> 391 iterations/epoch
    first = cosine_anneal(start=2.5, end=2.0, iteration=0, total_iterations=total)
    last = cosine_anneal(start=2.5, end=2.0, iteration=total - 1, total_iterations=total)
    assert first == pytest.approx(2.5)
    assert last == pytest.approx(2.0, abs=1e-4)
    assert first > last  # temperature anneals downward

    lambda_first = cosine_anneal(start=0.7, end=0.95, iteration=0, total_iterations=total)
    lambda_last = cosine_anneal(start=0.7, end=0.95, iteration=total - 1, total_iterations=total)
    assert lambda_first == pytest.approx(0.7)
    assert lambda_last == pytest.approx(0.95, abs=1e-4)
    assert lambda_last > lambda_first  # lambda anneals upward
