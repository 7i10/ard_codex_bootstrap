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

Contract v2 (``ard-distillation-proxies-v2``, :func:`compute_pair_proxies_v2`)
keeps every v1 metric (same names; the ones that depend on ``x + dS`` are
measured under the selected student attack, and ``student_pgd10_accuracy`` is
emitted only for the ``ce`` attack, next to the always-present
``student_accuracy_on_student_adv``) and adds:

* a choice of student attack for ``x + dS``: ``ce`` (the v1 selection attack
  above, same random-start seeds, so v2/ce reproduces v1's numbers) or ``rslad``
  (RSLAD's own inner maximisation as RSLAD training runs it: ``KL(softmax(T(x))
  || softmax(S(x')))`` maximised over ``x'`` with the teacher's clean logits as a
  fixed target, identity taken verbatim from a Phase 2 RSLAD config's
  ``method.attack`` -- PGD-3, Linf 4/255, step 8/765, random start, eval mode).
  The target is the full online teacher softmax (the ``online_teacher`` target
  source), never a soft-label-bank truncation.  Because the RSLAD attack needs the
  teacher's target, it is per (student, teacher) pair, like everything else here;
* metric A, ``exposed_vulnerability_fraction``: ``(n_T(x) - n_T(x+dS)) /
  (n_T(x) - n_T(x+dT))`` with ``n_T(.)`` the teacher's correct-prediction *count*
  on the subset (a ratio of counts, not a mean of per-sample ratios); ``None``
  when the denominator is not positive.  It is the share of the teacher's own
  white-box PGD-10 accuracy drop that the student's perturbation already
  exposes; it can exceed 1 or be negative;
* metric B, ``input_gradient_cosine_clean`` (and ``..._student_adv`` at ``x + dS``):
  per-sample cosine between the flattened pixel-space gradients of
  ``CE(S(.), y)`` and ``CE(T(.), y)``, averaged over samples where both gradients
  are non-zero (the excluded count is reported);
* metric C, ``correct_and_reactive_rate``: fraction of subset samples with
  ``argmax T(x+dS) == y`` AND ``KL(T(x+dS) || T(x))`` strictly above that
  (pair, attack)'s median KL over the subset (``correct_and_reactive_kl_threshold``;
  the usual median, the mean of the two middle values for an even count); plus
  ``correct_and_true_mass_decreased_rate``: ``argmax T(x+dS) == y`` AND
  ``p_T(y | x+dS) < p_T(y | x)``.  Both divide by the whole subset; the
  ``..._among_teacher_correct`` versions divide by the number of samples the
  teacher still classifies correctly at ``x + dS`` (``None`` if zero).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ard.attacks import AttackRequest, LinfPGD
from ard.config.schema import AttackConfig
from ard.data.datasets import SourceIndexedSubset, train_probe_ids

from .soft_label_bank import entropy, kl_rows

PROXY_SUBSET_SEED = 20261009
TOP_K_GRID = (1, 5, 10, 20, 50, 100, 200)
PROXIES_V1_CONTRACT = "ard-distillation-proxies-v1"
PROXIES_V2_CONTRACT = "ard-distillation-proxies-v2"
STUDENT_ATTACKS = ("ce", "rslad")
# Per-batch random-start seed offsets (batch seed = attack_seed + 1_000_003 * batch_index).
# ``ce`` and the teacher's own attack keep the v1 offsets, so v2/ce reproduces v1's numbers.
SEED_OFFSETS = {"ce": 0, "teacher_own": 17, "rslad": 29}

Accumulate = Callable[[str, torch.Tensor], None]


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


def _add_teacher_crop_quality(add: Accumulate, p_clean: torch.Tensor, labels: torch.Tensor) -> None:
    """FKD-style label quality of the teacher's clean softmax on the training crops."""
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


def compute_pair_proxies(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader[Any],
    *,
    attack_config: AttackConfig,
    device: torch.device,
    attack_seed: int = 0,
) -> dict[str, Any]:
    """Contract v1 (unchanged numbers; kept as the CLI default)."""
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
        _add_teacher_crop_quality(add, p_clean, labels)
        count += labels.shape[0]
    if count == 0:
        raise ValueError("proxy loader is empty")
    return {name: value / count for name, value in sorted(sums.items())} | {"count": count}


def ids_digest(ids: Sequence[int]) -> str:
    return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()


# --------------------------------------------------------------------------- v2


def validate_rslad_attack(attack: AttackConfig, selection_attack: AttackConfig | None = None) -> None:
    if attack.loss != "kl" or attack.kl_target != "teacher_clean":
        raise ValueError("the rslad proxy attack is the KL-to-teacher-clean inner maximisation")
    if attack.student_mode != "eval" or attack.teacher_mode != "eval" or attack.random_start_keying != "batch":
        raise ValueError("the rslad proxy attack runs both models in eval mode with batch-keyed random starts")
    if selection_attack is not None and (
        attack.norm != selection_attack.norm
        or attack.input_domain != selection_attack.input_domain
        or attack.epsilon_value != selection_attack.epsilon_value
    ):
        raise ValueError("the rslad attack and the student's selection attack must share the threat budget")


def rslad_attack_from_config(path: Path) -> tuple[AttackConfig, dict[str, str]]:
    """RSLAD's training attack, verbatim from a Phase 2 RSLAD config's ``method.attack``.

    Only the ``method`` block is read (it needs no environment interpolation and
    must contain none); the file's SHA-256 is returned as provenance.
    """
    data = path.read_bytes()
    method = (yaml.safe_load(data) or {}).get("method") or {}
    if method.get("id") not in {"rslad", "rslad_advt"} or not isinstance(method.get("attack"), dict):
        raise ValueError(f"{path} is not an RSLAD-family config with a method.attack block")
    if "${" in repr(method["attack"]):
        raise ValueError("method.attack must not contain environment placeholders")
    attack = AttackConfig.model_validate(method["attack"])
    validate_rslad_attack(attack)
    return attack, {"config": str(path), "config_sha256": hashlib.sha256(data).hexdigest(), "field": "method.attack"}


class FrozenEvalModel(nn.Module):
    """A Phase 1 checkpoint used as a proxy teacher: parameters frozen, always in eval mode.

    Mirrors :class:`ard.models.teacher.TeacherAdapter` (no BatchNorm statistic
    can move even if an attack asks for ``train()``) and keeps the pixel adapter
    the checkpoint was trained with (inputs stay pixels in ``[0, 1]``).
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self.train(False)

    def train(self, mode: bool = True) -> FrozenEvalModel:
        del mode
        super().train(False)
        self.model.eval()
        return self

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.model(pixels)


def input_gradients(model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-sample pixel gradients of ``CE(model(x), y)``.

    Sum reduction, so row ``i`` is exactly ``grad_x CE(model(x_i), y_i)`` for an
    eval-mode model (no cross-sample coupling).  Only the input is
    differentiated; no parameter ``.grad`` is touched.
    """
    pixels = inputs.detach().float().requires_grad_(True)
    with torch.enable_grad(), torch.autocast(device_type=pixels.device.type, enabled=False):
        loss = F.cross_entropy(model(pixels).float(), labels, reduction="sum")
    return torch.autograd.grad(loss, pixels, only_inputs=True)[0].detach()


def gradient_cosine(first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample cosine of flattened gradients (FP64) and the mask of rows where both are non-zero."""
    a, b = first.flatten(1).double(), second.flatten(1).double()
    norm_a, norm_b = a.norm(dim=1), b.norm(dim=1)
    valid = (norm_a > 0) & (norm_b > 0)
    cosine = (a * b).sum(dim=1) / (norm_a * norm_b).clamp_min(torch.finfo(torch.float64).tiny)
    return torch.where(valid, cosine, torch.zeros_like(cosine)), valid


def exposed_vulnerability_fraction(
    teacher_clean_correct: int, teacher_correct_on_student_adv: int, teacher_correct_on_teacher_adv: int
) -> float | None:
    """Metric A as a ratio of counts; ``None`` when the teacher's own attack removes no correct prediction."""
    denominator = teacher_clean_correct - teacher_correct_on_teacher_adv
    if denominator <= 0:
        return None
    return (teacher_clean_correct - teacher_correct_on_student_adv) / denominator


def reactive_rates(
    kl_to_clean: torch.Tensor, teacher_correct: torch.Tensor, true_mass_decreased: torch.Tensor
) -> dict[str, float | None]:
    """Metric C over the whole subset of one (pair, attack); one entry per sample in each vector."""
    if kl_to_clean.ndim != 1 or kl_to_clean.numel() == 0:
        raise ValueError("reactive rates need a non-empty per-sample KL vector")
    if teacher_correct.shape != kl_to_clean.shape or true_mass_decreased.shape != kl_to_clean.shape:
        raise ValueError("per-sample vectors must have the same length")
    kl = kl_to_clean.detach().double().cpu()
    threshold = float(np.median(kl.numpy()))
    correct = teacher_correct.detach().bool().cpu()
    reactive = int((correct & (kl > threshold)).sum())
    decreased = int((correct & true_mass_decreased.detach().bool().cpu()).sum())
    total, n_correct = kl.numel(), int(correct.sum())
    return {
        "correct_and_reactive_kl_threshold": threshold,
        "correct_and_reactive_rate": reactive / total,
        "correct_and_reactive_rate_among_teacher_correct": None if n_correct == 0 else reactive / n_correct,
        "correct_and_true_mass_decreased_rate": decreased / total,
        "correct_and_true_mass_decreased_rate_among_teacher_correct": (
            None if n_correct == 0 else decreased / n_correct
        ),
    }


def _student_adversarial(
    name: str,
    attack: LinfPGD,
    student: nn.Module,
    teacher: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    teacher_clean_logits: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    if name == "ce":
        return _pgd(student, attack, images, labels, seed)
    # As in RSLAD training (ard.engine.trainer): the teacher's clean logits are the
    # fixed KL target and the student crafts x' (both models in eval mode).
    generator = torch.Generator(device=images.device).manual_seed(seed)
    request = AttackRequest(
        inputs=images,
        labels=labels,
        student=student,
        teacher=teacher,
        target_logits=teacher_clean_logits,
        generator=generator,
        stream_tag="proxy_rslad",
    )
    return attack.generate(request).adversarial


def compute_pair_proxies_v2(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader[Any],
    *,
    selection_attack: AttackConfig,
    student_attacks: Mapping[str, AttackConfig],
    device: torch.device,
    attack_seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """All v2 metrics for one (student, teacher) pair: one metrics dict per student attack.

    Attack-independent quantities (teacher-clean label quality, the teacher's own
    PGD-10 ``x + dT``, clean input-gradient alignment) are computed once per batch
    and repeated in every attack's dict.
    """
    if selection_attack.loss != "ce" or selection_attack.student_mode != "eval":
        raise ValueError("proxy attacks are the eval-mode CE selection attack")
    if not student_attacks or set(student_attacks) - set(STUDENT_ATTACKS):
        raise ValueError(f"student_attacks must be a non-empty subset of {STUDENT_ATTACKS}")
    if "ce" in student_attacks and student_attacks["ce"].identity() != selection_attack.identity():
        raise ValueError("the ce student attack is the student's own selection attack")
    if "rslad" in student_attacks:
        validate_rslad_attack(student_attacks["rslad"], selection_attack)
    attacks = {name: LinfPGD(student_attacks[name]) for name in STUDENT_ATTACKS if name in student_attacks}
    teacher_attack = LinfPGD(selection_attack)
    student.eval()
    teacher.eval()
    shared: dict[str, float] = {}
    per_attack: dict[str, dict[str, float]] = {name: {} for name in attacks}
    samples: dict[str, dict[str, list[torch.Tensor]]] = {
        name: {"kl": [], "correct": [], "decreased": []} for name in attacks
    }
    count = 0

    def adder(target: dict[str, float]) -> Accumulate:
        def add(name: str, values: torch.Tensor) -> None:
            target[name] = target.get(name, 0.0) + float(values.double().sum().item())

        return add

    add_shared = adder(shared)
    for batch_index, (images, labels, _) in enumerate(loader):
        images, labels = images.to(device).float(), labels.to(device)
        seed = attack_seed + 1_000_003 * batch_index
        teacher_adv = _pgd(teacher, teacher_attack, images, labels, seed + SEED_OFFSETS["teacher_own"])
        with torch.no_grad():
            s_clean = student(images).float()
            t_clean_logits = teacher(images).float()
            p_clean = F.softmax(t_clean_logits, dim=1)
            p_on_t = F.softmax(teacher(teacher_adv).float(), dim=1)
        add_shared("student_clean_accuracy", s_clean.argmax(1) == labels)
        add_shared("teacher_accuracy_on_teacher_adv", p_on_t.argmax(1) == labels)
        add_shared("teacher_entropy_clean", entropy(p_clean))
        _add_teacher_crop_quality(add_shared, p_clean, labels)
        cosine, valid = gradient_cosine(
            input_gradients(student, images, labels), input_gradients(teacher, images, labels)
        )
        add_shared("input_gradient_cosine_clean_sum", cosine)
        add_shared("input_gradient_cosine_clean_valid", valid)
        true_mass_clean = p_clean.gather(1, labels[:, None]).squeeze(1)
        for name, attack in attacks.items():
            add = adder(per_attack[name])
            student_adv = _student_adversarial(
                name, attack, student, teacher, images, labels, t_clean_logits, seed + SEED_OFFSETS[name]
            )
            with torch.no_grad():
                s_adv = student(student_adv).float()
                p_on_s = F.softmax(teacher(student_adv).float(), dim=1)
            student_correct = s_adv.argmax(1) == labels
            add("student_accuracy_on_student_adv", student_correct)
            if name == "ce":
                add("student_pgd10_accuracy", student_correct)
            teacher_correct = p_on_s.argmax(1) == labels
            add("a_teacher_on_student_adv", teacher_correct)
            kl_to_clean = kl_rows(p_on_s, p_clean)
            kl_to_teacher_adv = kl_rows(p_on_s, p_on_t)
            add("tas_ratio", kl_to_clean >= kl_to_teacher_adv)
            add("kl_teacher_student_adv_vs_clean", kl_to_clean)
            add("kl_teacher_student_adv_vs_teacher_adv", kl_to_teacher_adv)
            add("teacher_entropy_student_adv", entropy(p_on_s))
            samples[name]["kl"].append(kl_to_clean.detach().cpu())
            samples[name]["correct"].append(teacher_correct.cpu())
            samples[name]["decreased"].append((p_on_s.gather(1, labels[:, None]).squeeze(1) < true_mass_clean).cpu())
            cosine, valid = gradient_cosine(
                input_gradients(student, student_adv, labels), input_gradients(teacher, student_adv, labels)
            )
            add("input_gradient_cosine_student_adv_sum", cosine)
            add("input_gradient_cosine_student_adv_valid", valid)
        count += labels.shape[0]
    if count == 0:
        raise ValueError("proxy loader is empty")
    results: dict[str, dict[str, Any]] = {}
    for name in attacks:
        sums = shared | per_attack[name]
        metrics: dict[str, Any] = {
            key: value / count for key, value in sums.items() if not key.startswith("input_gradient_cosine_")
        }
        for where in ("clean", "student_adv"):
            valid_count = round(sums[f"input_gradient_cosine_{where}_valid"])
            metrics[f"input_gradient_cosine_{where}"] = (
                None if valid_count == 0 else sums[f"input_gradient_cosine_{where}_sum"] / valid_count
            )
            metrics[f"input_gradient_cosine_{where}_excluded_zero_gradient_count"] = count - valid_count
        clean_correct = round(sums["teacher_clean_accuracy_on_crops"])
        on_student = round(sums["a_teacher_on_student_adv"])
        on_teacher = round(sums["teacher_accuracy_on_teacher_adv"])
        metrics["exposed_vulnerability_fraction"] = exposed_vulnerability_fraction(
            clean_correct, on_student, on_teacher
        )
        metrics["exposed_vulnerability_numerator_count"] = clean_correct - on_student
        metrics["exposed_vulnerability_denominator_count"] = clean_correct - on_teacher
        metrics |= reactive_rates(
            torch.cat(samples[name]["kl"]), torch.cat(samples[name]["correct"]), torch.cat(samples[name]["decreased"])
        )
        results[name] = dict(sorted(metrics.items())) | {"count": count}
    return results


METRIC_DEFINITIONS_V2 = {
    "x_plus_dS": "the student's adversarial example under the recorded student attack",
    "x_plus_dT": "the teacher's own white-box PGD-10 (the student's CE selection attack identity, run on the teacher)",
    "exposed_vulnerability_fraction": (
        "(n correct T(x) - n correct T(x+dS)) / (n correct T(x) - n correct T(x+dT)); ratio of counts over the "
        "subset; null when the denominator <= 0"
    ),
    "input_gradient_cosine_clean": (
        "mean over samples of cos(grad_x CE(S(x),y), grad_x CE(T(x),y)), each gradient flattened per sample; "
        "samples where either gradient is exactly zero are excluded and counted"
    ),
    "input_gradient_cosine_student_adv": "as input_gradient_cosine_clean, evaluated at x+dS",
    "correct_and_reactive_rate": (
        "fraction of subset samples with argmax T(x+dS) == y and KL(T(x+dS)||T(x)) > the median of that KL over "
        "the subset for this (pair, attack) (correct_and_reactive_kl_threshold)"
    ),
    "correct_and_true_mass_decreased_rate": (
        "fraction of subset samples with argmax T(x+dS) == y and p_T(y|x+dS) < p_T(y|x)"
    ),
    "among_teacher_correct": "the *_among_teacher_correct variants divide by the count with argmax T(x+dS) == y",
}
