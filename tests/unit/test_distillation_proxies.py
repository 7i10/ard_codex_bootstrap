"""Cheap distillation proxies and teacher sanity helpers (plan 0103 Phase 2 batch D), CPU fixtures."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from ard.config.schema import AttackConfig, NormalizationConfig
from ard.distillation.proxies import TOP_K_GRID, compute_pair_proxies
from ard.distillation.soft_label_bank import kl_rows
from ard.distillation.teacher_sanity import SANITY_ATTACK, accuracy_under_pgd, forward_throughput, sanity_subset_ids
from ard.models.registry import FixtureCNN, PixelModel

ATTACK = AttackConfig(
    loss="ce", epsilon="4/255", step_size="8/765", steps=2, random_start=True, student_mode="eval", teacher_mode="eval"
)


def _model(seed: int, classes: int = 300) -> PixelModel:
    torch.manual_seed(seed)
    return PixelModel(FixtureCNN(classes), NormalizationConfig(profile="imagenet_standard"))


def _loader(classes: int = 300) -> DataLoader:
    torch.manual_seed(0)
    images = torch.rand(12, 3, 8, 8)
    labels = torch.randint(0, classes, (12,))
    return DataLoader(TensorDataset(images, labels, torch.arange(12)), batch_size=5)


def test_pair_proxies_are_bounded_and_self_consistent() -> None:
    metrics = compute_pair_proxies(_model(1), _model(2), _loader(), attack_config=ATTACK, device=torch.device("cpu"))
    assert metrics["count"] == 12
    for name in ("a_teacher_on_student_adv", "tas_ratio", "teacher_clean_accuracy_on_crops", "student_pgd10_accuracy"):
        assert 0.0 <= metrics[name] <= 1.0
    coverage = [metrics[f"teacher_top{k}_mass_coverage"] for k in TOP_K_GRID]
    assert coverage == sorted(coverage) and coverage[-1] <= 1.0 + 1e-6
    hits = [metrics[f"teacher_true_class_in_top{k}"] for k in TOP_K_GRID]
    assert hits == sorted(hits)
    assert metrics["teacher_entropy_clean"] > 0 and metrics["kl_teacher_student_adv_vs_clean"] >= 0
    # Deterministic for a fixed attack seed.
    again = compute_pair_proxies(_model(1), _model(2), _loader(), attack_config=ATTACK, device=torch.device("cpu"))
    assert again == metrics


def test_tas_definition_direction() -> None:
    p_clean = torch.tensor([[0.9, 0.1]])
    p_on_student_adv = torch.tensor([[0.2, 0.8]])
    p_on_teacher_adv = torch.tensor([[0.1, 0.9]])
    # Teacher's response to dS is closer to its own adversarial response than to clean: transferable.
    assert bool(kl_rows(p_on_student_adv, p_clean) >= kl_rows(p_on_student_adv, p_on_teacher_adv))
    assert torch.allclose(kl_rows(p_clean, p_clean), torch.zeros(1))
    expected = (p_on_student_adv * (p_on_student_adv / p_clean).log()).sum()
    assert torch.allclose(kl_rows(p_on_student_adv, p_clean), expected[None])


def test_proxy_attack_must_be_eval_ce() -> None:
    kl = AttackConfig(loss="kl", kl_target="teacher_clean", epsilon="4/255", step_size="8/765", steps=1)
    with pytest.raises(ValueError, match="selection attack"):
        compute_pair_proxies(_model(1), _model(2), _loader(), attack_config=kl, device=torch.device("cpu"))


def test_teacher_sanity_helpers() -> None:
    assert SANITY_ATTACK.steps == 10 and SANITY_ATTACK.epsilon == "4/255" and SANITY_ATTACK.step_size == "8/765"
    result = accuracy_under_pgd(_model(3), _loader(), device=torch.device("cpu"), attack_config=ATTACK)
    assert result["count"] == 12 and result["pgd10_accuracy"] <= 1.0
    throughput = forward_throughput(
        _model(3), device=torch.device("cpu"), batch_size=2, image_size=8, seconds=0.05, warmup_batches=1
    )
    assert throughput["images_per_second"] > 0
    ids = sanity_subset_ids([index % 10 for index in range(200)], size=50)
    assert len(ids) == 50 and ids == sorted(ids) and len(set(ids)) == 50
