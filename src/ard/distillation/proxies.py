"""Cheap (student, teacher) compatibility proxies (plan 0103 Phase 2, discussion register E).

On a fixed 10,000-image class-stratified subset of the *training partition*,
viewed through the student's own training crops (``EpochImageNetTransform`` at
one epoch, the config's augmentation seed), for a student's Phase 1 PGD-AT
checkpoint and one frozen teacher:

* ``x + dS``: the student's PGD-10 adversarial example (its own selection attack:
  CE, Linf 4/255, step 8/765, random start, eval mode);
* ``x + dT``: the teacher's own PGD-10 adversarial example (same identity, on the teacher).

Reported (all means over the subset):

* ``a_teacher_on_student_adv`` -- A_{T<-S}: teacher top-1 on ``x + dS``;
* ``tas_ratio`` -- SAAD (Lee & Chung, arXiv:2512.10275, Eq. 7) transferable-sample
  fraction: ``KL(T(x+dS) || T(x)) >= KL(T(x+dS) || T(x+dT))`` (SAAD uses PGD-20 at
  CIFAR; here PGD-10 at the ImageNet budget), plus both KL means;
* teacher entropy on ``x`` and on ``x + dS``;
* FKD-style label quality on the training crops: teacher top-1, mean true-class
  probability, true class in top-K, and top-K probability-mass coverage for
  K in ``TOP_K_GRID`` (this is the measurement that chooses the bank's K);
* context: student clean / PGD-10 accuracy and teacher accuracy on ``x + dT``.

These are training-time proxies, never official results.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ard.attacks import AttackRequest, LinfPGD
from ard.config.schema import AttackConfig
from ard.data.datasets import SourceIndexedSubset, train_probe_ids

from .soft_label_bank import entropy, kl_rows

PROXY_SUBSET_SEED = 20261009
TOP_K_GRID = (1, 5, 10, 20, 50, 100, 200)


class FixedEpochSubset(Dataset[tuple[torch.Tensor, int, int]]):
    """Selected training-partition source IDs through the training transform at one epoch."""

    def __init__(self, train_view: SourceIndexedSubset, ids: Sequence[int], *, epoch: int) -> None:
        self.base = train_view.dataset
        self.ids = list(ids)
        self.base.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, int, int]:
        image, label, source_id = self.base[self.ids[position]]
        return image, label, source_id


def proxy_subset_ids(train_view: SourceIndexedSubset, *, size: int = 10000, seed: int = PROXY_SUBSET_SEED) -> list[int]:
    targets = getattr(train_view.dataset.dataset, "targets", None)
    if not isinstance(targets, (list, tuple)):
        raise TypeError("proxy subset requires a label-only targets sequence")
    return train_probe_ids(train_view.indices, targets, size=size, seed=seed)


def _pgd(model: nn.Module, attack: LinfPGD, images: torch.Tensor, labels: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=images.device).manual_seed(seed)
    return attack.generate(
        AttackRequest(inputs=images, labels=labels, student=model, generator=generator, stream_tag="proxy_pgd")
    ).adversarial


def compute_pair_proxies(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader[Any],
    *,
    attack_config: AttackConfig,
    device: torch.device,
    attack_seed: int = 0,
) -> dict[str, Any]:
    if attack_config.loss != "ce" or attack_config.student_mode != "eval":
        raise ValueError("proxy attacks are the eval-mode CE selection attack")
    attack = LinfPGD(attack_config)
    student.eval()
    teacher.eval()
    sums: dict[str, float] = {}
    count = 0

    def add(name: str, values: torch.Tensor) -> None:
        sums[name] = sums.get(name, 0.0) + float(values.double().sum().item())

    for batch_index, (images, labels, _) in enumerate(loader):
        images, labels = images.to(device).float(), labels.to(device)
        seed = attack_seed + 1_000_003 * batch_index
        student_adv = _pgd(student, attack, images, labels, seed)
        teacher_adv = _pgd(teacher, attack, images, labels, seed + 17)
        with torch.no_grad():
            s_clean, s_adv = student(images).float(), student(student_adv).float()
            p_clean = F.softmax(teacher(images).float(), dim=1)
            p_on_s = F.softmax(teacher(student_adv).float(), dim=1)
            p_on_t = F.softmax(teacher(teacher_adv).float(), dim=1)
        add("student_clean_accuracy", s_clean.argmax(1) == labels)
        add("student_pgd10_accuracy", s_adv.argmax(1) == labels)
        add("a_teacher_on_student_adv", p_on_s.argmax(1) == labels)
        add("teacher_accuracy_on_teacher_adv", p_on_t.argmax(1) == labels)
        kl_to_clean = kl_rows(p_on_s, p_clean)
        kl_to_teacher_adv = kl_rows(p_on_s, p_on_t)
        add("tas_ratio", kl_to_clean >= kl_to_teacher_adv)
        add("kl_teacher_student_adv_vs_clean", kl_to_clean)
        add("kl_teacher_student_adv_vs_teacher_adv", kl_to_teacher_adv)
        add("teacher_entropy_clean", entropy(p_clean))
        add("teacher_entropy_student_adv", entropy(p_on_s))
        add("teacher_clean_accuracy_on_crops", p_clean.argmax(1) == labels)
        true_mass = p_clean.gather(1, labels[:, None]).squeeze(1)
        add("teacher_true_class_mass_on_crops", true_mass)
        ordered = p_clean.sort(dim=1, descending=True)
        cumulative = ordered.values.cumsum(dim=1)
        rank = (ordered.indices == labels[:, None]).float().argmax(dim=1)
        for k in TOP_K_GRID:
            if k <= p_clean.shape[1]:
                add(f"teacher_top{k}_mass_coverage", cumulative[:, k - 1])
                add(f"teacher_true_class_in_top{k}", rank < k)
        count += labels.shape[0]
    if count == 0:
        raise ValueError("proxy loader is empty")
    return {name: value / count for name, value in sorted(sums.items())} | {"count": count}


def ids_digest(ids: Sequence[int]) -> str:
    return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()
