"""Distillation proxies contract v2 (plan 0103 Phase 2, human-approved 2026-10-09), CPU fixtures.

Hand-checkable cases for metrics A (exposed vulnerability fraction), B
(input-gradient alignment) and C (correct-and-reactive rates), the RSLAD
attack identity, the frozen same-recipe teacher, and v1 reproducibility.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ard.config.schema import AttackConfig, NormalizationConfig
from ard.distillation.proxies import (
    FrozenEvalModel,
    compute_pair_proxies,
    compute_pair_proxies_v2,
    exposed_vulnerability_fraction,
    gradient_cosine,
    input_gradients,
    reactive_rates,
    rslad_attack_from_config,
)
from ard.models.registry import FixtureCNN, PixelModel

ROOT = Path(__file__).resolve().parents[2]
RSLAD_CONFIG = ROOT / "configs" / "scientific" / "imagenet_mobilenetv4_rslad_phase2_salman_r50_online.yaml"
CE = AttackConfig(
    loss="ce", epsilon="4/255", step_size="8/765", steps=2, random_start=True, student_mode="eval", teacher_mode="eval"
)
RSLAD = AttackConfig(loss="kl", kl_target="teacher_clean", epsilon="4/255", step_size="8/765", steps=2)
CPU = torch.device("cpu")

# compute_pair_proxies (contract v1) on the v1 test fixture, recorded from the source
# *before* the v2 refactor (5b6b443): the default CLI path must keep these numbers.
V1_GOLDEN = {
    "a_teacher_on_student_adv": 0.0,
    "count": 12,
    "kl_teacher_student_adv_vs_clean": 2.3749424144625664e-05,
    "kl_teacher_student_adv_vs_teacher_adv": 4.588634086151918e-05,
    "student_clean_accuracy": 0.0,
    "student_pgd10_accuracy": 0.0,
    "tas_ratio": 0.16666666666666666,
    "teacher_accuracy_on_teacher_adv": 0.0,
    "teacher_clean_accuracy_on_crops": 0.0,
    "teacher_entropy_clean": 5.671570738156636,
    "teacher_entropy_student_adv": 5.671246846516927,
    "teacher_top100_mass_coverage": 0.435646357635657,
    "teacher_top10_mass_coverage": 0.05538024318714937,
    "teacher_top1_mass_coverage": 0.006022689631208777,
    "teacher_top200_mass_coverage": 0.7527230530977249,
    "teacher_top20_mass_coverage": 0.1042280209561189,
    "teacher_top50_mass_coverage": 0.23908743262290955,
    "teacher_top5_mass_coverage": 0.028962435976912577,
    "teacher_true_class_in_top1": 0.0,
    "teacher_true_class_in_top10": 0.0,
    "teacher_true_class_in_top100": 0.16666666666666666,
    "teacher_true_class_in_top20": 0.08333333333333333,
    "teacher_true_class_in_top200": 0.5,
    "teacher_true_class_in_top5": 0.0,
    "teacher_true_class_in_top50": 0.16666666666666666,
    "teacher_true_class_mass_on_crops": 0.0031653875679088137,
}


def _model(seed: int, classes: int = 300) -> PixelModel:
    torch.manual_seed(seed)
    return PixelModel(FixtureCNN(classes), NormalizationConfig(profile="imagenet_standard"))


def _loader(classes: int = 300) -> DataLoader:
    torch.manual_seed(0)
    images = torch.rand(12, 3, 8, 8)
    labels = torch.randint(0, classes, (12,))
    return DataLoader(TensorDataset(images, labels, torch.arange(12)), batch_size=5)


# ------------------------------------------------------------------ v1 reproducibility


def test_v1_numbers_are_unchanged() -> None:
    metrics = compute_pair_proxies(_model(1), _model(2), _loader(), attack_config=CE, device=CPU)
    assert metrics == V1_GOLDEN


def test_v2_ce_reproduces_every_v1_metric() -> None:
    v1 = compute_pair_proxies(_model(1), _model(2), _loader(), attack_config=CE, device=CPU)
    v2 = compute_pair_proxies_v2(
        _model(1), _model(2), _loader(), selection_attack=CE, student_attacks={"ce": CE}, device=CPU
    )["ce"]
    assert {key: v2[key] for key in v1} == v1
    assert v2["student_accuracy_on_student_adv"] == v1["student_pgd10_accuracy"]


# ------------------------------------------------------------------ metric A


def test_exposed_vulnerability_fraction_hand_cases() -> None:
    # Teacher correct on 80 clean, 60 under the student's dS, 40 under its own dT: dS exposes half the drop.
    assert exposed_vulnerability_fraction(80, 60, 40) == 0.5
    assert exposed_vulnerability_fraction(80, 40, 40) == 1.0
    assert exposed_vulnerability_fraction(80, 30, 40) == 1.25  # dS hurts the teacher more than its own PGD-10
    assert exposed_vulnerability_fraction(80, 90, 40) == -0.25  # dS raises teacher accuracy
    assert exposed_vulnerability_fraction(80, 60, 80) is None  # guarded: own attack removes nothing
    assert exposed_vulnerability_fraction(0, 0, 0) is None


# ------------------------------------------------------------------ metric B


def test_gradient_cosine_hand_cases() -> None:
    first = torch.tensor([[1.0, 0.0], [1.0, 1.0], [3.0, 4.0], [0.0, 0.0], [2.0, 0.0]])
    second = torch.tensor([[0.0, 2.0], [-1.0, -1.0], [6.0, 8.0], [1.0, 0.0], [1.0, 1.0]])
    cosine, valid = gradient_cosine(first, second)
    assert valid.tolist() == [True, True, True, False, True]
    assert torch.allclose(cosine, torch.tensor([0.0, -1.0, 1.0, 0.0, 1 / math.sqrt(2)], dtype=torch.float64))
    # Flattening: an image-shaped gradient behaves like its flattened vector.
    cosine_4d, _ = gradient_cosine(first.reshape(5, 1, 1, 2), second.reshape(5, 2, 1, 1))
    assert torch.equal(cosine_4d, cosine)


class _Linear(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.flatten(1) @ self.weight.T


def test_input_gradients_are_per_sample_ce_gradients() -> None:
    # logits = W x with W = [[1, 0], [0, 1]], x = 0 -> softmax uniform, so grad_x CE = W^T (p - onehot(y)).
    model = _Linear(torch.eye(2))
    inputs = torch.zeros(2, 1, 1, 2)
    labels = torch.tensor([0, 1])
    gradient = input_gradients(model, inputs, labels)
    assert torch.allclose(gradient.flatten(1), torch.tensor([[-0.5, 0.5], [0.5, -0.5]]))
    # Sum reduction: each row is that sample's own gradient, independent of the batch.
    assert torch.allclose(gradient[1:], input_gradients(model, inputs[1:], labels[1:]))
    assert model.weight.grad is None
    # A model and an exact copy are perfectly aligned; the negated model is anti-aligned.
    cosine, _ = gradient_cosine(gradient, input_gradients(_Linear(torch.eye(2)), inputs, labels))
    assert torch.allclose(cosine, torch.ones(2, dtype=torch.float64))


def test_self_pair_has_unit_clean_gradient_alignment() -> None:
    student = _model(1, classes=3)
    teacher = FrozenEvalModel(copy.deepcopy(student))
    metrics = compute_pair_proxies_v2(
        student, teacher, _loader(3), selection_attack=CE, student_attacks={"ce": CE}, device=CPU
    )["ce"]
    assert metrics["input_gradient_cosine_clean"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["input_gradient_cosine_student_adv"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["input_gradient_cosine_clean_excluded_zero_gradient_count"] == 0


# ------------------------------------------------------------------ metric C


def test_reactive_rates_hand_case() -> None:
    kl = torch.tensor([0.1, 0.2, 0.3, 0.4])
    correct = torch.tensor([True, True, False, True])
    decreased = torch.tensor([False, True, True, True])
    rates = reactive_rates(kl, correct, decreased)
    assert rates["correct_and_reactive_kl_threshold"] == pytest.approx(0.25)  # mean of the two middle values
    assert rates["correct_and_reactive_rate"] == 0.25  # only sample 3 is correct and above the median
    assert rates["correct_and_reactive_rate_among_teacher_correct"] == pytest.approx(1 / 3)
    assert rates["correct_and_true_mass_decreased_rate"] == 0.5  # samples 1 and 3 (sample 2 is wrong)
    assert rates["correct_and_true_mass_decreased_rate_among_teacher_correct"] == pytest.approx(2 / 3)
    # Strictly above the median: an odd count's middle sample is never reactive.
    all_correct, none_decreased = torch.ones(3, dtype=torch.bool), torch.zeros(3, dtype=torch.bool)
    odd = reactive_rates(torch.tensor([1.0, 2.0, 3.0]), all_correct, none_decreased)
    assert odd["correct_and_reactive_kl_threshold"] == 2.0 and odd["correct_and_reactive_rate"] == pytest.approx(1 / 3)
    none_correct = reactive_rates(kl, torch.zeros(4, dtype=torch.bool), decreased)
    assert none_correct["correct_and_reactive_rate"] == 0.0
    assert none_correct["correct_and_reactive_rate_among_teacher_correct"] is None


# ------------------------------------------------------------------ the full v2 pass


def test_v2_metrics_are_consistent_across_attacks() -> None:
    def run() -> dict:
        return compute_pair_proxies_v2(
            _model(1, 3),
            _model(2, 3),
            _loader(3),
            selection_attack=CE,
            student_attacks={"ce": CE, "rslad": RSLAD},
            device=CPU,
        )

    results = run()
    assert set(results) == {"ce", "rslad"} and results == run()  # deterministic for a fixed attack seed
    ce, rslad = results["ce"], results["rslad"]
    assert "student_pgd10_accuracy" in ce and "student_pgd10_accuracy" not in rslad
    shared = ("teacher_clean_accuracy_on_crops", "teacher_accuracy_on_teacher_adv", "input_gradient_cosine_clean")
    assert all(ce[key] == rslad[key] for key in shared)
    for metrics in results.values():
        count = metrics["count"]
        clean = round(metrics["teacher_clean_accuracy_on_crops"] * count)
        on_student = round(metrics["a_teacher_on_student_adv"] * count)
        on_teacher = round(metrics["teacher_accuracy_on_teacher_adv"] * count)
        assert metrics["exposed_vulnerability_numerator_count"] == clean - on_student
        assert metrics["exposed_vulnerability_denominator_count"] == clean - on_teacher
        expected = exposed_vulnerability_fraction(clean, on_student, on_teacher)
        assert metrics["exposed_vulnerability_fraction"] == expected
        assert metrics["correct_and_reactive_rate"] <= metrics["a_teacher_on_student_adv"] + 1e-12
        assert metrics["correct_and_true_mass_decreased_rate"] <= metrics["a_teacher_on_student_adv"] + 1e-12
        assert -1.0 <= metrics["input_gradient_cosine_clean"] <= 1.0


def test_v2_guards() -> None:
    def run(**overrides: object) -> None:
        kwargs = {"selection_attack": CE, "student_attacks": {"ce": CE}, "device": CPU} | overrides
        compute_pair_proxies_v2(_model(1, 3), _model(2, 3), _loader(3), **kwargs)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="selection attack"):
        run(student_attacks={"ce": CE.model_copy(update={"steps": 3})})
    with pytest.raises(ValueError, match="threat budget"):
        run(student_attacks={"rslad": AttackConfig(loss="kl", kl_target="teacher_clean", epsilon="8/255", steps=2)})
    with pytest.raises(ValueError, match="KL-to-teacher-clean"):
        wrong_target = AttackConfig(loss="kl", kl_target="student_clean", epsilon="4/255", step_size="1/255")
        run(student_attacks={"rslad": wrong_target})
    with pytest.raises(ValueError, match="subset"):
        run(student_attacks={"trades": CE})


# ------------------------------------------------------------------ RSLAD identity and frozen teacher


def test_rslad_attack_comes_verbatim_from_the_phase2_config() -> None:
    attack, provenance = rslad_attack_from_config(RSLAD_CONFIG)
    assert attack.identity() == {
        "norm": "linf",
        "input_domain": "pixel_0_1",
        "epsilon": "4/255",
        "epsilon_value": 4 / 255,
        "step_size": "8/765",
        "step_size_value": 8 / 765,
        "steps": 3,
        "random_start": True,
        "loss": "kl",
        "kl_target": "teacher_clean",
        "temperature": 1.0,
        "temperature_squared": True,
        "student_mode": "eval",
        "teacher_mode": "eval",
    }
    assert provenance["field"] == "method.attack" and len(provenance["config_sha256"]) == 64


def test_rslad_attack_config_refuses_a_non_rslad_config(tmp_path: Path) -> None:
    path = tmp_path / "pgd.yaml"
    path.write_text("method: {id: pgd_at, version: 1, attack: {loss: ce}}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="RSLAD-family"):
        rslad_attack_from_config(path)


def test_frozen_same_recipe_teacher_never_moves() -> None:
    torch.manual_seed(3)
    inner = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.Flatten(), nn.LazyLinear(3))
    inner(torch.rand(2, 3, 8, 8))
    teacher = FrozenEvalModel(inner)
    teacher.train()
    assert not teacher.training and not inner.training and not inner[1].training
    assert not any(parameter.requires_grad for parameter in teacher.parameters())
    before = copy.deepcopy(inner.state_dict())
    compute_pair_proxies_v2(
        _model(1, 3), teacher, _loader(3), selection_attack=CE, student_attacks={"ce": CE, "rslad": RSLAD}, device=CPU
    )
    after = inner.state_dict()
    assert all(torch.equal(before[key], after[key]) for key in before)
    assert all(parameter.grad is None for parameter in teacher.parameters())
