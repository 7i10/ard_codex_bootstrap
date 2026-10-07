"""Distillation-target configuration for ImageNet RSLAD (plan 0103 Phase 2, batch D).

Kept apart from ``schema.py`` so the Phase 2 distillation contract is reviewable
in one place.  This module must not import ``ard.config.schema`` (schema imports
it); cross-field checks receive the validated ``ExperimentConfig`` duck-typed.

Two target sources produce the *same* RSLAD target when the soft-label bank
keeps every class:

* ``online_teacher``: the frozen teacher runs on each training batch's clean
  crops, exactly as CIFAR RSLAD does.
* ``soft_label_bank``: the teacher's softmax on the exact deterministic training
  crops (``EpochImageNetTransform`` keyed by augmentation seed, epoch and source
  ID) was precomputed (``ard.cli.build_soft_label_bank``) and stored as top-K
  probabilities plus residual mass (FKD-style marginal smoothing).  Training
  reads the bank instead of running the teacher and refuses any batch whose
  crop keys differ from the bank's.

Only the teacher-clean target changes source.  The attack (KL to the
teacher-clean target, student-crafted), the 5/6 adversarial + 1/6 clean KD
objective, temperature and ``T^2`` are RSLAD's own and are not touched.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

IMAGENET_TEACHER_REGISTRY_IDS: frozenset[str] = frozenset(
    {
        "salman2020_resnet50_linf_eps4",
        "singh2023_convnext_t_convstem",
        "singh2023_vit_s_convstem",
        "singh2023_convnext_b_convstem",
        "ard_mobilenetv4_conv_medium_phase1_random",
    }
)

# Methods whose objective and attack consume only the teacher-clean logits, so
# a bank of teacher-clean soft labels is a complete substitute for the teacher.
#   ``rslad_advt`` additionally needs the teacher on the adversarial example;
#   in bank mode that forward runs online and its softmax is truncated exactly
#   like the bank (same K, same marginal smoothing).
DISTILLATION_METHODS: frozenset[str] = frozenset({"rslad", "rslad_advt"})
ADVERSARIAL_TEACHER_TARGET_METHODS: frozenset[str] = frozenset({"rslad_advt"})


# Format + storage of ard.distillation.soft_label_bank (kept here so the
# config layer does not import the runtime module).
BANK_STORAGE_IDENTITY = "ard-soft-label-bank-v1/top_k_marginal_smoothing_v1/fp16"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


class SoftLabelBankConfig(_StrictModel):
    """One precomputed bank: the path is provenance, the manifest digest is the identity."""

    path: Path
    manifest_sha256: str
    top_k: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_digest(self) -> SoftLabelBankConfig:
        if not _is_sha256(self.manifest_sha256):
            raise ValueError("distillation.bank.manifest_sha256 must be a lowercase 64-character SHA-256 hex digest")
        return self


class DistillationConfig(_StrictModel):
    target_source: Literal["online_teacher", "soft_label_bank"]
    bank: SoftLabelBankConfig | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def validate_bank(self) -> DistillationConfig:
        if (self.target_source == "soft_label_bank") != (self.bank is not None):
            raise ValueError("distillation.bank is required for, and only valid with, target_source=soft_label_bank")
        return self

    def protocol_identity(self, teacher: Any) -> dict[str, object]:
        """The pooling identity: target source, teacher profile and digest, bank storage and K.

        The bank manifest digest is deliberately absent: a bank is built per
        augmentation seed, so seeds of one arm have different digests and must
        still pool.  The digest is per-run lineage (:meth:`run_lineage`).
        """
        return {
            "target_source": self.target_source,
            "teacher_registry_id": None if teacher is None else teacher.registry_id,
            "teacher_checkpoint_sha256": None if teacher is None else teacher.checkpoint_sha256,
            "bank_storage": None if self.bank is None else BANK_STORAGE_IDENTITY,
            "bank_top_k": None if self.bank is None else self.bank.top_k,
        }

    def run_lineage(self) -> dict[str, object] | None:
        """Per-run (per-seed) bank lineage, recorded next to the result, never pooled on."""
        if self.bank is None:
            return None
        return {"bank_manifest_sha256": self.bank.manifest_sha256, "bank_path": str(self.bank.path)}


def validate_distillation_cross_fields(config: Any) -> None:
    """Fail closed for every combination the Phase 2 distillation paths do not implement."""
    distillation: DistillationConfig | None = config.distillation
    teacher = config.teacher
    imagenet_teacher = teacher is not None and teacher.source == "imagenet_registry"
    if imagenet_teacher and distillation is None:
        raise ValueError("an imagenet_registry teacher requires an explicit distillation block (target source)")
    if config.method.id in ADVERSARIAL_TEACHER_TARGET_METHODS and distillation is None:
        raise ValueError(f"method {config.method.id} is defined only with an explicit distillation block")
    if distillation is None:
        return
    errors: list[str] = []
    if not imagenet_teacher:
        errors.append("teacher.source=imagenet_registry")
    if config.dataset.name != "imagenet":
        errors.append("dataset.name=imagenet")
    if config.method.id not in DISTILLATION_METHODS:
        errors.append("method.id in " + ", ".join(sorted(DISTILLATION_METHODS)))
    if config.method.target_policy is not None:
        errors.append("no method.target_policy")
    if config.intervention is not None or config.prescriptive_v3 is not None:
        errors.append("no intervention/prescriptive_v3")
    if config.observation.profile != "off":
        # teacher_response observation runs the teacher on adversarial inputs,
        # which the bank cannot answer; keep both modes on the same footing.
        errors.append("observation.profile=off")
    if distillation.target_source == "soft_label_bank":
        if config.dataset.imagenet_heavy_augmentation:
            errors.append(
                "dataset.imagenet_heavy_augmentation=false (RandAugment/RandomErasing use the global RNG, so the "
                "training crop is not a function of the crop key)"
            )
        attack = config.method.attack
        # A truncated bank target is defined at temperature 1; any other
        # temperature would re-sharpen the smoothed tail, an unreviewed semantics.
        if not math.isclose(config.method.temperature, 1.0, rel_tol=0, abs_tol=0) or (
            attack is not None and not math.isclose(attack.temperature, 1.0, rel_tol=0, abs_tol=0)
        ):
            errors.append("method.temperature=1 and method.attack.temperature=1")
    if errors:
        raise ValueError("distillation requires " + "; ".join(errors))
