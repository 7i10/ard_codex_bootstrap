"""Typed scientific configuration with explicit units and strict keys."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator


def parse_rational(value: str) -> float:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("quantity must be a non-empty string such as '8/255'")
    text = value.strip()
    try:
        if "/" in text:
            numerator, denominator = text.split("/", maxsplit=1)
            result = float(numerator) / float(denominator)
        else:
            result = float(text)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"invalid rational quantity: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError("quantity must be finite")
    return result


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ProtocolConfig(StrictModel):
    """Versioned experiment protocol identity; M1 owns its concrete registry."""

    id: Literal[
        "saad_paper_reproduction_v1",
        "saad_code_295121c_audit_v1",
        "controlled_cifar10_r18_v1",
        "controlled_cifar10_r18_cropshift_v1",
        "controlled_cifar10_r18_cropshift_prefix_v1",
        "controlled_cifar10_r18_crop_re_v1",
        "controlled_cifar10_r18_idbh_weak_v1",
        "controlled_cifar10_r18_stagewise_augmentation_v1",
        "controlled_cifar10_r18_delayed_multistep_v1",
        "controlled_cifar10_r18_prescriptive_v3_v1",
        "controlled_cifar10_r18_pilot_v1",
        "controlled_cifar10_r18_pilot_1ep_v1",
        "controlled_cifar10_r18_pilot_3ep_v1",
        "synthetic_smoke_v2",
    ]


class SeedsConfig(StrictModel):
    """Independent deterministic seeds, resolved without a legacy scalar alias."""

    split: int
    model_init: int
    data_order: int
    augmentation: int
    train_attack: int
    evaluation_attack: int
    qualitative_panel: int


class OptimizerConfig(StrictModel):
    """Optimizer identity frozen for the M1 protocol implementation."""

    id: Literal["sgd"]
    learning_rate: float = Field(gt=0)
    momentum: float = Field(ge=0, lt=1)
    weight_decay: float = Field(ge=0)
    nesterov: bool

    @model_validator(mode="after")
    def validate_nesterov(self) -> OptimizerConfig:
        if self.nesterov and self.momentum <= 0:
            raise ValueError("nesterov SGD requires positive momentum")
        return self


class SchedulerConfig(StrictModel):
    """Scheduler identity frozen before M1 supplies concrete schedules."""

    id: Literal["identity", "multistep"]
    milestones: tuple[int, ...]
    gamma: float = Field(gt=0)
    step_at: Literal["epoch_end"]

    @model_validator(mode="after")
    def validate_schedule(self) -> SchedulerConfig:
        if self.id == "identity":
            if self.milestones or self.gamma != 1.0:
                raise ValueError("identity scheduler requires milestones=[] and gamma=1.0")
        elif not self.milestones or tuple(sorted(set(self.milestones))) != self.milestones or self.milestones[0] < 0:
            raise ValueError("multistep scheduler requires strictly increasing non-negative milestones")
        return self


class TargetPolicyConfig(StrictModel):
    """Identity for adversarial-student teacher-target calibration only."""

    id: Literal["teacher_target_uniform_mix"]
    version: Literal[1]
    risk_transform: Literal["identity"]
    mixing: Literal["uniform"]
    apply_to: Literal["adversarial_student_kd"]
    rho_max: float = Field(default=0.5, ge=0, le=1)


class NormalizationConfig(StrictModel):
    """A named, pixel-space normalization contract owned by one model adapter."""

    input_domain: Literal["pixel_0_1"] = "pixel_0_1"
    profile: Literal[
        "fixture_unit",
        "cifar10_raw_identity",
        "cifar10_standard",
        "robustbench_cifar10_bartoldson_embedded",
        "cifar100_standard",
        "tiny_imagenet_standard",
        "custom",
    ] = "fixture_unit"
    mean: tuple[float, float, float] | None = None
    std: tuple[float, float, float] | None = None
    provenance: str | None = None

    @model_validator(mode="after")
    def validate_std(self) -> NormalizationConfig:
        profiles = {
            "fixture_unit": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), "ARD fixture identity profile"),
            "cifar10_raw_identity": (
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
                "CIFAR-10 raw-pixel identity profile for clean-room SAAD student",
            ),
            "cifar10_standard": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), "CIFAR-10 repository profile"),
            "robustbench_cifar10_bartoldson_embedded": (
                (0.4914, 0.4822, 0.4465),
                (0.2471, 0.2435, 0.2616),
                "RobustBench dm_wide_resnet.CIFAR10_MEAN/CIFAR10_STD at 78fcc9e48a07a861268f295a777b975f25155964",
            ),
            "cifar100_standard": (
                (0.5071, 0.4865, 0.4409),
                (0.2673, 0.2564, 0.2762),
                "CIFAR-100 repository profile; not claimed upstream-exact",
            ),
            "tiny_imagenet_standard": (
                (0.4802, 0.4481, 0.3975),
                (0.2302, 0.2265, 0.2262),
                "Tiny-ImageNet repository profile",
            ),
        }
        if self.profile == "custom":
            if self.mean is None or self.std is None or not self.provenance:
                raise ValueError("custom normalization requires mean, std, and provenance")
        else:
            expected_mean, expected_std, expected_provenance = profiles[self.profile]
            if self.mean is not None and self.mean != expected_mean:
                raise ValueError(f"normalization mean does not match named profile {self.profile}")
            if self.std is not None and self.std != expected_std:
                raise ValueError(f"normalization std does not match named profile {self.profile}")
            if self.provenance is not None and self.provenance != expected_provenance:
                raise ValueError(f"normalization provenance does not match named profile {self.profile}")
            object.__setattr__(self, "mean", expected_mean)
            object.__setattr__(self, "std", expected_std)
            object.__setattr__(self, "provenance", expected_provenance)
        assert self.mean is not None and self.std is not None
        if any(not math.isfinite(value) for value in self.mean):
            raise ValueError("normalization mean values must be finite")
        if any(value <= 0 or not math.isfinite(value) for value in self.std):
            raise ValueError("normalization std values must be finite and positive")
        return self


class AttackConfig(StrictModel):
    norm: Literal["linf"] = "linf"
    input_domain: Literal["pixel_0_1"] = "pixel_0_1"
    epsilon: str = "8/255"
    epsilon_value: float | None = None
    step_size: str = "2/255"
    step_size_value: float | None = None
    steps: int = Field(default=10, ge=1)
    random_start: bool = True
    # ``batch`` preserves the historical stream.  ``sample_keyed_v1`` is a
    # separately versioned contract for interventions that must not let
    # DataLoader order decide which fixed source ID receives a random start.
    random_start_keying: Literal["batch", "sample_keyed_v1"] = "batch"
    loss: Literal["ce", "kl"] = "ce"
    kl_target: Literal["student_clean", "teacher_clean"] | None = None
    temperature: float = Field(default=1.0, gt=0)
    temperature_squared: bool = True
    student_mode: Literal["train", "eval"] = "eval"
    teacher_mode: Literal["train", "eval"] = "eval"
    # Trace collection is debugging-only and deliberately excluded from the
    # complete 14-field scientific attack identity below.
    trace_step_losses: bool = False

    def identity(self) -> dict[str, object]:
        """JSON-safe complete attack identity; never omit a scientific field."""
        identity = {
            "norm": self.norm,
            "input_domain": self.input_domain,
            "epsilon": self.epsilon,
            "epsilon_value": self.epsilon_value,
            "step_size": self.step_size,
            "step_size_value": self.step_size_value,
            "steps": self.steps,
            "random_start": self.random_start,
            "loss": self.loss,
            "kl_target": self.kl_target,
            "temperature": self.temperature,
            "temperature_squared": self.temperature_squared,
            "student_mode": self.student_mode,
            "teacher_mode": self.teacher_mode,
        }
        # Keep the historical 14-field identity byte-compatible for all
        # existing batch-keyed runs.  The new algorithm is intentionally a
        # distinct identity and is therefore explicit only for that mode.
        if self.random_start_keying != "batch":
            identity["random_start_keying"] = self.random_start_keying
        return identity

    def identity_json(self) -> str:
        return json.dumps(self.identity(), sort_keys=True, separators=(",", ":"))

    def identity_sha256(self) -> str:
        return hashlib.sha256(self.identity_json().encode()).hexdigest()

    @model_validator(mode="after")
    def resolve_quantities(self) -> AttackConfig:
        epsilon = parse_rational(self.epsilon)
        step_size = parse_rational(self.step_size)
        if epsilon < 0 or step_size <= 0:
            raise ValueError("epsilon must be non-negative and step_size must be positive")
        if epsilon > 1 or step_size > 1:
            raise ValueError("pixel-domain epsilon and step_size must not exceed 1")
        if self.epsilon_value is not None and not math.isclose(self.epsilon_value, epsilon, rel_tol=0, abs_tol=1e-15):
            raise ValueError("epsilon_value does not match epsilon")
        if self.step_size_value is not None and not math.isclose(
            self.step_size_value, step_size, rel_tol=0, abs_tol=1e-15
        ):
            raise ValueError("step_size_value does not match step_size")
        object.__setattr__(self, "epsilon_value", epsilon)
        object.__setattr__(self, "step_size_value", step_size)
        if self.loss == "ce" and self.kl_target is not None:
            raise ValueError("kl_target is valid only for KL attacks")
        if self.loss == "kl" and self.kl_target is None:
            raise ValueError("KL attacks require an explicit kl_target")
        if self.loss == "kl" and self.kl_target == "teacher_clean" and self.teacher_mode != "eval":
            raise ValueError("teacher_clean KL attacks require teacher_mode=eval")
        return self


class DatasetConfig(StrictModel):
    name: Literal["synthetic_cifar", "cifar10", "cifar100", "tiny_imagenet"] = "synthetic_cifar"
    root: Path | None = None
    split: Literal["train", "val", "test"] = "train"
    download: bool = False
    num_samples: int = Field(default=16, ge=1)
    num_classes: int = Field(default=10, ge=2)
    image_size: int = Field(default=32, ge=1)
    seed: int = 0
    content_sha256: str | None = None
    augmentation_policy: Literal["canonical", "cropshift", "crop_re", "idbh_weak", "stagewise"] = "canonical"
    augmentation_crop_shift_high: int = Field(default=11, ge=1)
    stagewise_switch_epoch: int | None = Field(default=None, ge=1, le=199)
    stagewise_late_policy: Literal["crop_re", "idbh_weak"] | None = None

    @model_validator(mode="after")
    def validate_dataset(self) -> DatasetConfig:
        if self.name in {"cifar10", "cifar100"} and self.split == "val":
            raise ValueError("CIFAR has no validation split alias; use official train or test")
        expected = {"cifar10": 10, "cifar100": 100}.get(self.name)
        if expected is not None and self.num_classes != expected:
            raise ValueError(f"{self.name} requires num_classes={expected}")
        if self.name == "tiny_imagenet" and self.root is None:
            raise ValueError("tiny_imagenet requires an explicit root")
        if self.augmentation_policy != "canonical" and self.name not in {"cifar10", "cifar100"}:
            raise ValueError("non-canonical augmentation policies are currently defined only for CIFAR datasets")
        if self.augmentation_policy == "stagewise":
            if self.stagewise_switch_epoch is None or self.stagewise_late_policy is None:
                raise ValueError("stagewise augmentation requires a switch epoch and late policy")
        elif self.stagewise_switch_epoch is not None or self.stagewise_late_policy is not None:
            raise ValueError("stagewise fields are valid only with augmentation_policy=stagewise")
        if self.content_sha256 is not None and (
            len(self.content_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.content_sha256)
        ):
            raise ValueError("dataset content_sha256 must be a lowercase 64-character SHA-256 hex digest")
        return self


class ModelConfig(StrictModel):
    architecture: Literal[
        "saad_resnet18_cifar_v1",
        "torchvision_resnet18_cifar_norm_v1",
        "resnet18_cifar",
        "mobilenet_v2_cifar",
        "fixture_cnn",
    ] = "fixture_cnn"
    num_classes: int = Field(default=10, ge=2)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    preprocessing_owner: Literal["student_adapter"] = "student_adapter"


class TeacherConfig(StrictModel):
    source: Literal["checkpoint", "fixture", "robustbench"] = "fixture"
    architecture: Literal[
        "saad_resnet18_cifar_v1",
        "torchvision_resnet18_cifar_norm_v1",
        "resnet18_cifar",
        "mobilenet_v2_cifar",
        "fixture_cnn",
        "robustbench_wide_resnet",
        "robustbench_dm_wide_resnet",
    ] = "fixture_cnn"
    num_classes: int = Field(default=10, ge=2)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    preprocessing_owner: Literal["teacher_adapter", "model_embedded"] = "teacher_adapter"
    checkpoint: Path | None = None
    checkpoint_sha256: str | None = None
    registry_id: Literal[
        "chen2021_ltd_wrn34_10",
        "chen2021_ltd_wrn34_20",
        "bartoldson2024_adversarial_wrn94_16",
    ] | None = None
    threat_norm: Literal["linf"] = "linf"
    threat_epsilon: str = "8/255"
    fixture_seed: int = 1729

    @model_validator(mode="after")
    def validate_source(self) -> TeacherConfig:
        if self.source in {"checkpoint", "robustbench"} and (self.checkpoint is None or self.checkpoint_sha256 is None):
            raise ValueError(f"{self.source} teachers require checkpoint and checkpoint_sha256")
        if self.source == "robustbench" and self.registry_id is None:
            raise ValueError("robustbench teachers require registry_id")
        if self.source != "robustbench" and self.registry_id is not None:
            raise ValueError("registry_id is only valid for robustbench teachers")
        if (
            self.source == "robustbench"
            and self.registry_id == "chen2021_ltd_wrn34_10"
            and self.preprocessing_owner != "teacher_adapter"
        ):
            raise ValueError("Chen RobustBench teacher requires teacher_adapter preprocessing")
        if self.preprocessing_owner == "model_embedded":
            if self.source != "robustbench" or self.registry_id != "bartoldson2024_adversarial_wrn94_16":
                raise ValueError("model_embedded preprocessing is restricted to the Bartoldson RobustBench teacher")
        if self.threat_epsilon != "8/255":
            raise ValueError("teacher threat_epsilon must remain the explicit canonical value 8/255")
        if self.checkpoint_sha256 is not None and (
            len(self.checkpoint_sha256) != 64 or any(char not in "0123456789abcdef" for char in self.checkpoint_sha256)
        ):
            raise ValueError("checkpoint_sha256 must be a lowercase 64-character digest")
        return self


class MethodConfig(StrictModel):
    id: Literal[
        "pgd_at",
        "trades",
        "rslad",
        "rslad_entropy",
        "rslad_student",
        "rslad_joint",
        "rslad_joint_downweight",
        "rslad_hard_fallback",
        "rslad_frozen_oracle_softening",
    ]
    version: Literal[1]
    attack: AttackConfig = Field(default_factory=AttackConfig)
    selection_attack: AttackConfig | None = None
    temperature: float = Field(default=1.0, gt=0)
    temperature_squared: bool = True
    trades_beta: float = Field(default=6.0, ge=0)
    entropy_gamma: float = Field(default=1.0, gt=0)
    student_ema_decay: float = Field(default=0.9, ge=0, lt=1)
    student_policy_warmup_epochs: int = Field(default=1, ge=1)
    target_policy: TargetPolicyConfig | None = None
    oracle_mask: bool = False
    frozen_oracle_manifest: Path | None = None
    frozen_oracle_manifest_sha256: str | None = None

    @property
    def name(self) -> str:
        """Runtime compatibility only; resolved configuration serializes ``id``."""
        return self.id

    @model_validator(mode="after")
    def resolve_selection_attack(self) -> MethodConfig:
        expected_loss = "ce" if self.id == "pgd_at" else "kl"
        expected_target = {
            "trades": "student_clean",
            "rslad": "teacher_clean",
            "rslad_entropy": "teacher_clean",
            "rslad_student": "teacher_clean",
            "rslad_joint": "teacher_clean",
            "rslad_joint_downweight": "teacher_clean",
            "rslad_hard_fallback": "teacher_clean",
            "rslad_frozen_oracle_softening": "teacher_clean",
        }.get(self.id)
        if self.attack.loss != expected_loss:
            raise ValueError(f"{self.id} requires attack.loss={expected_loss}")
        if self.attack.kl_target != expected_target:
            raise ValueError(f"{self.id} requires attack.kl_target={expected_target!r}")
        selection = self.selection_attack
        if selection is None:
            selection = self.attack.model_copy(
                update={
                    "loss": "ce",
                    "kl_target": None,
                    "student_mode": "eval",
                    "teacher_mode": "eval",
                    "random_start_keying": "batch",
                }
            )
            object.__setattr__(self, "selection_attack", selection)
        if selection.loss != "ce":
            raise ValueError("checkpoint selection attack must use hard-label CE")
        if selection.student_mode != "eval" or selection.teacher_mode != "eval":
            raise ValueError("checkpoint selection attack must keep student and teacher in eval mode")
        mismatched = []
        # Selection uses CE, whereas training may use KL.  Temperature and
        # KL-only fields do not define CE threat parity, and controlled
        # selection deliberately uses a stronger 20-step attack than the
        # 10-step training inner maximization.
        for field in ("norm", "input_domain", "random_start"):
            if getattr(selection, field) != getattr(self.attack, field):
                mismatched.append(field)
        assert selection.epsilon_value is not None and self.attack.epsilon_value is not None
        assert selection.step_size_value is not None and self.attack.step_size_value is not None
        if not math.isclose(selection.epsilon_value, self.attack.epsilon_value, rel_tol=0, abs_tol=1e-15):
            mismatched.append("epsilon")
        if not math.isclose(selection.step_size_value, self.attack.step_size_value, rel_tol=0, abs_tol=1e-15):
            mismatched.append("step_size")
        if mismatched:
            raise ValueError(
                "checkpoint selection attack must match the training threat model: " + ", ".join(mismatched)
            )
        if self.id == "rslad_entropy" and self.entropy_gamma != 1.0:
            raise ValueError("rslad_entropy currently implements Shannon entropy only (entropy_gamma=1)")
        risk_methods = {
            "rslad_student",
            "rslad_joint",
            "rslad_joint_downweight",
            "rslad_hard_fallback",
        }
        if self.id in risk_methods:
            if self.student_ema_decay != 0.9:
                raise ValueError(
                    f"{self.id} is the canonical EMA=0.9 method; use a separate method ID for other decays"
                )
            if self.student_policy_warmup_epochs != 1:
                raise ValueError(
                    f"{self.id} is the canonical one-epoch-warmup method; use a separate method ID for variants"
                )
        target_methods = {"rslad_student", "rslad_joint", "rslad_frozen_oracle_softening"}
        if self.id in target_methods:
            if self.target_policy is None:
                raise ValueError(f"{self.id} requires an explicit target_policy")
        elif self.target_policy is not None:
            raise ValueError(f"target_policy is only defined for {sorted(target_methods)}")
        if self.oracle_mask and self.id != "rslad_hard_fallback":
            raise ValueError("oracle_mask is only defined for rslad_hard_fallback")
        frozen_fields = (self.frozen_oracle_manifest, self.frozen_oracle_manifest_sha256)
        if self.id == "rslad_frozen_oracle_softening":
            if any(value is None for value in frozen_fields):
                raise ValueError("rslad_frozen_oracle_softening requires frozen_oracle_manifest and exact SHA-256")
            assert self.target_policy is not None
            if self.target_policy.rho_max != 0.5:
                raise ValueError("rslad_frozen_oracle_softening fixes target_policy.rho_max=0.5")
            if self.oracle_mask:
                raise ValueError("rslad_frozen_oracle_softening uses only its frozen external manifest")
        elif any(value is not None for value in frozen_fields):
            raise ValueError("frozen_oracle_manifest fields are only defined for rslad_frozen_oracle_softening")
        if self.frozen_oracle_manifest_sha256 is not None and (
            len(self.frozen_oracle_manifest_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.frozen_oracle_manifest_sha256)
        ):
            raise ValueError("frozen_oracle_manifest_sha256 must be a lowercase 64-character digest")
        return self


class TrainingConfig(StrictModel):
    epochs: int = Field(default=1, ge=1)
    checkpoint_epochs: tuple[int, ...] = (49, 99, 149, 199)
    per_rank_batch_size: int = Field(ge=1)
    global_batch_size: int = Field(ge=1)
    num_workers: int = Field(default=0, ge=0)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    deterministic: bool = True
    validation_fraction: float = Field(default=0.25, gt=0, lt=1)
    # This is a protocol identity, not a performance option. Ordinary DDP
    # computes BatchNorm statistics independently on each rank.
    batchnorm_mode: Literal["local_per_rank"] = "local_per_rank"

    @model_validator(mode="after")
    def validate_batch_identity(self) -> TrainingConfig:
        if self.global_batch_size < self.per_rank_batch_size:
            raise ValueError("global_batch_size must be at least per_rank_batch_size")
        ordered = tuple(sorted(set(self.checkpoint_epochs)))
        if any(epoch < 1 for epoch in self.checkpoint_epochs) or ordered != self.checkpoint_epochs:
            raise ValueError("checkpoint_epochs must be strictly increasing positive epoch numbers")
        return self


def validate_global_batch_size(*, per_rank_batch_size: int, global_batch_size: int, world_size: int) -> int:
    """Validate and return the effective global batch for an initialized job."""
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    effective_global_batch_size = per_rank_batch_size * world_size
    if global_batch_size != effective_global_batch_size:
        raise ValueError(
            "global_batch_size must equal per_rank_batch_size * world_size "
            f"({global_batch_size} != {per_rank_batch_size} * {world_size})"
        )
    return effective_global_batch_size


def training_execution_identity(*, training: TrainingConfig, world_size: int) -> dict[str, int | str]:
    """Return the immutable hardware-sensitive training identity."""
    effective_global_batch_size = validate_global_batch_size(
        per_rank_batch_size=training.per_rank_batch_size,
        global_batch_size=training.global_batch_size,
        world_size=world_size,
    )
    return {
        "world_size": world_size,
        "per_rank_batch_size": training.per_rank_batch_size,
        "global_batch_size": training.global_batch_size,
        "effective_global_batch_size": effective_global_batch_size,
        "batchnorm_mode": training.batchnorm_mode,
    }


class TrackingConfig(StrictModel):
    """Tracking is explicit so production cannot silently become untracked."""

    mode: Literal["disabled", "offline", "offline_sync", "online"] = "disabled"
    project: str | None = None
    entity: str | None = None
    run_id: str | None = None
    name: str | None = None
    group: str | None = None
    log_every_steps: int | None = None
    diagnostics_mode: Literal["off", "summary", "panel"] = "panel"
    panel_size: int = Field(default=24, ge=0)
    panel_interval_epochs: int = Field(default=5, ge=1)
    artifact_interval_epochs: int = Field(default=5, ge=1)
    # Checkpoints and the complete run bundle remain authoritative on the
    # local output filesystem.  W&B receives lightweight metrics/lineage by
    # default; explicit promotion is required for heavyweight artifacts.
    artifact_retention: Literal["metrics_only", "best_last", "full"] = "metrics_only"

    @model_validator(mode="after")
    def validate_wandb_identity(self) -> TrackingConfig:
        if self.mode in {"offline", "offline_sync", "online"} and not self.project:
            raise ValueError("tracked runs require tracking.project")
        if self.run_id is not None and not self.run_id.strip():
            raise ValueError("tracking.run_id must not be empty")
        if self.log_every_steps is not None:
            raise ValueError("bootstrap tracking is epoch-only; tracking.log_every_steps must be null")
        if self.diagnostics_mode == "panel" and self.panel_size == 0:
            raise ValueError("panel diagnostics require tracking.panel_size > 0")
        return self


class ObservationConfig(StrictModel):
    """Detached, method-independent training observations.

    Profiles are deliberately an explicit cost/lineage axis.  They store raw
    primitives only; a proposed risk, threshold, or loss intervention must be
    configured separately through an existing policy/method.
    """

    profile: Literal["off", "student_history", "teacher_response"] = "off"

    @property
    def records_student_history(self) -> bool:
        return self.profile in {"student_history", "teacher_response"}

    @property
    def records_teacher_response(self) -> bool:
        return self.profile == "teacher_response"


class InterventionParentConfig(StrictModel):
    """Immutable provenance required for a common-state intervention arm."""

    checkpoint_sha256: str
    raw_config_sha256: str
    git_sha: str
    epoch: Literal[39, 79, 99]
    world_size: Literal[1]
    teacher_checkpoint_sha256: str
    sample_state_records: Literal[45000]
    sample_state_sha256: str
    train_partition_manifest: Path
    train_partition_manifest_sha256: str
    train_partition_ids_labels_sha256: str
    artifact_attestation: Path
    artifact_attestation_sha256: str
    artifact_inventory: Path
    artifact_inventory_sha256: str

    @model_validator(mode="after")
    def validate_hashes(self) -> InterventionParentConfig:
        for name, value, length in (
            ("checkpoint_sha256", self.checkpoint_sha256, 64),
            ("raw_config_sha256", self.raw_config_sha256, 64),
            ("git_sha", self.git_sha, 40),
            ("teacher_checkpoint_sha256", self.teacher_checkpoint_sha256, 64),
            ("sample_state_sha256", self.sample_state_sha256, 64),
            ("train_partition_manifest_sha256", self.train_partition_manifest_sha256, 64),
            ("train_partition_ids_labels_sha256", self.train_partition_ids_labels_sha256, 64),
            ("artifact_attestation_sha256", self.artifact_attestation_sha256, 64),
            ("artifact_inventory_sha256", self.artifact_inventory_sha256, 64),
        ):
            if len(value) != length or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"intervention parent {name} must be a lowercase {length}-character digest")
        return self


class InterventionMaskProvenanceConfig(StrictModel):
    source: Literal[
        "seed0_bartoldson_frozen_predictor",
        "class_matched_random",
        "online_history_epoch39_v2",
        "class_state_count_matched_random_epoch39_v2",
        "prescriptive_v3_online_history",
        "prescriptive_v3_matched_random",
        "ffnr_route_a_strong_ce_pgd20",
        "ffnr_route_a_matched_random",
        "ffnr_route_b_strong_ce_pgd20",
        "ffnr_route_b_matched_random",
    ]
    approved_selector_spec_sha256: str | None = None
    selector_spec_path: Path | None = None
    parent_checkpoint_sha256: str
    parent_sample_state_sha256: str
    random_seed: int | None = None
    generator: str | None = None
    generator_version: str | None = None
    reference_history_mask_sha256: str | None = None
    reference_selected_count: int | None = None
    reference_selected_class_counts: dict[str, int] | None = None
    reference_history_selector_spec_sha256: str | None = None
    route: Literal["peak_failure", "non_recovery"] | None = None
    anchor_robust_correct: bool | None = None

    def exact_payload(self) -> dict[str, object]:
        """Return the byte-level payload expected by the corresponding mask.

        H3 masks intentionally serialized their explicit ``null`` fields;
        epoch-39 v2 masks omit unavailable fields.  Keep that distinction so
        extending the schema cannot invalidate the already-registered H3
        manifests.
        """
        payload = self.model_dump(mode="json")
        if self.source in {
            "online_history_epoch39_v2",
            "class_state_count_matched_random_epoch39_v2",
            "prescriptive_v3_online_history",
            "prescriptive_v3_matched_random",
        }:
            return {key: value for key, value in payload.items() if value is not None}
        payload.pop("route", None)
        payload.pop("anchor_robust_correct", None)
        return payload

    @model_validator(mode="after")
    def validate_provenance(self) -> InterventionMaskProvenanceConfig:
        for name, candidate in (
            ("parent_checkpoint_sha256", self.parent_checkpoint_sha256),
            ("parent_sample_state_sha256", self.parent_sample_state_sha256),
        ):
            if len(candidate) != 64 or any(character not in "0123456789abcdef" for character in candidate):
                raise ValueError(f"intervention mask provenance {name} must be a lowercase 64-character digest")
        if self.source == "seed0_bartoldson_frozen_predictor":
            if self.approved_selector_spec_sha256 is None or self.selector_spec_path is None:
                raise ValueError("history mask provenance requires an approved selector specification SHA-256")
            if any(
                value is not None
                for value in (
                    self.random_seed,
                    self.generator,
                    self.generator_version,
                    self.reference_history_mask_sha256,
                    self.reference_selected_count,
                    self.reference_selected_class_counts,
                    self.reference_history_selector_spec_sha256,
                )
            ):
                raise ValueError("history mask provenance cannot carry random-mask fields")
        elif self.source == "class_matched_random":
            if (
                self.approved_selector_spec_sha256 is not None
                or self.selector_spec_path is not None
                or any(
                    value is None
                    for value in (
                        self.random_seed,
                        self.generator,
                        self.generator_version,
                        self.reference_history_mask_sha256,
                        self.reference_selected_count,
                        self.reference_selected_class_counts,
                        self.reference_history_selector_spec_sha256,
                    )
                )
            ):
                raise ValueError("random mask provenance requires fixed generator and reference history budget")
        elif self.source in {"online_history_epoch39_v2", "prescriptive_v3_online_history"}:
            if self.approved_selector_spec_sha256 is None or self.selector_spec_path is None:
                raise ValueError("online-history mask provenance requires an approved selector specification SHA-256")
            if self.route is None or self.anchor_robust_correct is None:
                raise ValueError("online-history mask provenance requires route and anchor correctness")
            if any(
                value is not None
                for value in (
                    self.random_seed,
                    self.generator,
                    self.generator_version,
                    self.reference_history_mask_sha256,
                    self.reference_selected_count,
                    self.reference_selected_class_counts,
                    self.reference_history_selector_spec_sha256,
                )
            ):
                raise ValueError("online-history mask provenance cannot carry random-mask fields")
        elif self.source in {
            "ffnr_route_a_strong_ce_pgd20",
            "ffnr_route_a_matched_random",
            "ffnr_route_b_strong_ce_pgd20",
            "ffnr_route_b_matched_random",
        }:
            if any(
                value is not None
                for value in (
                    self.approved_selector_spec_sha256,
                    self.selector_spec_path,
                    self.route,
                    self.anchor_robust_correct,
                    self.reference_history_mask_sha256,
                    self.reference_selected_count,
                    self.reference_selected_class_counts,
                    self.reference_history_selector_spec_sha256,
                )
            ):
                raise ValueError("FFNR causal mask provenance must not carry legacy selector fields")
            if self.source.endswith("matched_random"):
                if self.random_seed is None or self.generator is None or self.generator_version is None:
                    raise ValueError("FFNR matched-random provenance requires generator metadata")
            elif any(value is not None for value in (self.random_seed, self.generator, self.generator_version)):
                raise ValueError("FFNR selected provenance must not carry random generator metadata")
        else:
            if (
                self.approved_selector_spec_sha256 is not None
                or self.selector_spec_path is not None
                or self.route is None
                or self.anchor_robust_correct is None
                or any(
                    value is None
                    for value in (
                        self.random_seed,
                        self.generator,
                        self.generator_version,
                        self.reference_history_mask_sha256,
                        self.reference_selected_count,
                        self.reference_selected_class_counts,
                        self.reference_history_selector_spec_sha256,
                    )
                )
            ):
                raise ValueError("v2 random mask provenance requires route, generator, and reference history budget")
        if self.approved_selector_spec_sha256 is not None and (
            len(self.approved_selector_spec_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.approved_selector_spec_sha256)
        ):
            raise ValueError(
                "intervention mask provenance approved_selector_spec_sha256 must be a lowercase 64-character digest"
            )
        if self.reference_history_mask_sha256 is not None and (
            len(self.reference_history_mask_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.reference_history_mask_sha256)
        ):
            raise ValueError(
                "intervention mask provenance reference_history_mask_sha256 must be a lowercase 64-character digest"
            )
        if self.reference_history_selector_spec_sha256 is not None and (
            len(self.reference_history_selector_spec_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.reference_history_selector_spec_sha256)
        ):
            raise ValueError(
                "intervention mask provenance reference selector spec SHA must be a lowercase 64-character digest"
            )
        return self


class InterventionMaskConfig(StrictModel):
    """Hash-bound train-only selected IDs for a fixed intervention arm."""

    path: Path
    sha256: str
    selected_ids_sha256: str
    selected_count: int = Field(gt=0)
    selected_class_counts: dict[str, int]
    provenance: InterventionMaskProvenanceConfig

    @model_validator(mode="after")
    def validate_mask(self) -> InterventionMaskConfig:
        for name, value in (("sha256", self.sha256), ("selected_ids_sha256", self.selected_ids_sha256)):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"intervention mask {name} must be a lowercase 64-character digest")
        counts: dict[int, int] = {}
        for raw_class, count in self.selected_class_counts.items():
            try:
                class_id = int(raw_class)
            except ValueError as exc:
                raise ValueError("intervention mask class-count keys must be integer strings") from exc
            if str(class_id) != raw_class or class_id < 0 or count < 0:
                raise ValueError("intervention mask class counts must be non-negative canonical integer mappings")
            counts[class_id] = count
        if sum(counts.values()) != self.selected_count:
            raise ValueError("intervention mask selected_count must equal its selected_class_counts total")
        return self


class InterventionConfig(StrictModel):
    """Registered immutable intervention arms; legacy H3 and epoch-39 v2 are disjoint."""

    arm: Literal["C", "HS", "RS", "HD", "RD", "PF_TA", "PF_R", "NR_TA", "NR_R", "C79", "RA", "RAR", "RB", "RBR"]
    selector: Literal[
        "none", "student_history", "class_matched_random", "online_history", "class_state_count_matched_random",
        "route_a_strong", "route_a_matched_random", "route_b_strong", "route_b_matched_random"
    ]
    kind: Literal[
        "ordinary_rslad",
        "uniform_target_softening",
        "adversarial_kd_downweight",
        "teacher_target_true_label_mix",
        "route_a_ce_anchor",
        "route_b_ce_anchor",
    ]
    parent: InterventionParentConfig
    mask: InterventionMaskConfig | None = None
    selector_bundle_path: Path | None = None
    selector_bundle_sha256: str | None = None
    uniform_target_softening_rho: float = Field(default=0.5, ge=0, le=1)
    adversarial_kd_multiplier: float = Field(default=0.5, ge=0, le=1)
    adversarial_ce_coefficient: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def validate_registered_arm(self) -> InterventionConfig:
        legacy = {
            "C": ("none", "ordinary_rslad", False),
            "HS": ("student_history", "uniform_target_softening", True),
            "RS": ("class_matched_random", "uniform_target_softening", True),
            "HD": ("student_history", "adversarial_kd_downweight", True),
            "RD": ("class_matched_random", "adversarial_kd_downweight", True),
        }
        v2 = {
            "PF_TA": ("online_history", "teacher_target_true_label_mix", True, "peak_failure", True),
            "PF_R": ("class_state_count_matched_random", "teacher_target_true_label_mix", True, "peak_failure", True),
            "NR_TA": ("online_history", "teacher_target_true_label_mix", True, "non_recovery", False),
            "NR_R": ("class_state_count_matched_random", "teacher_target_true_label_mix", True, "non_recovery", False),
        }
        causal = {
            "C79": ("none", "ordinary_rslad", False, None, 0.5, 0.0),
            "RA": ("route_a_strong", "route_a_ce_anchor", True, "ffnr_route_a_strong_ce_pgd20", 0.5, 0.25),
            "RAR": ("route_a_matched_random", "route_a_ce_anchor", True, "ffnr_route_a_matched_random", 0.5, 0.25),
            "RB": ("route_b_strong", "route_b_ce_anchor", True, "ffnr_route_b_strong_ce_pgd20", 1.0, 0.25),
            "RBR": ("route_b_matched_random", "route_b_ce_anchor", True, "ffnr_route_b_matched_random", 1.0, 0.25),
        }
        if self.arm in legacy:
            expected = legacy[self.arm]
            if self.parent.epoch != 99:
                raise ValueError("legacy intervention arms require the epoch-99 parent contract")
        elif self.arm in causal:
            selector, kind, has_mask, source, kd_multiplier, ce_coefficient = causal[self.arm]
            expected = (selector, kind, has_mask)
            if self.parent.epoch != 79:
                raise ValueError("FFNR causal arms require the epoch-79 parent contract")
            if self.adversarial_kd_multiplier != kd_multiplier or self.adversarial_ce_coefficient != ce_coefficient:
                raise ValueError("FFNR causal arm treatment coefficients are frozen by the preregistered pilot")
            if self.mask is not None and self.mask.provenance.source != source:
                raise ValueError("FFNR causal mask provenance source does not match its registered arm")
        else:
            selector, kind, has_mask, route, anchor_correct = v2[self.arm]
            expected = (selector, kind, has_mask)
            if self.parent.epoch != 39:
                raise ValueError("history-routing v2 arms require the epoch-39 parent contract")
            if self.mask is not None and (
                self.mask.provenance.route != route or self.mask.provenance.anchor_robust_correct is not anchor_correct
            ):
                raise ValueError("history-routing v2 mask provenance route/state does not match its registered arm")
            if self.selector_bundle_path is None or self.selector_bundle_sha256 is None:
                raise ValueError("history-routing v2 arms require an immutable selector bundle path and SHA-256")
            if len(self.selector_bundle_sha256) != 64 or any(
                character not in "0123456789abcdef" for character in self.selector_bundle_sha256
            ):
                raise ValueError("history-routing v2 selector bundle SHA-256 must be a lowercase 64-character digest")
        if (self.selector, self.kind, self.mask is not None) != expected:
            raise ValueError("intervention arm must use its registered selector, treatment, and mask presence")
        if self.arm in legacy and (self.uniform_target_softening_rho != 0.5 or self.adversarial_kd_multiplier != 0.5):
            raise ValueError("the registered intervention screen fixes both treatment strengths at 0.5")
        if self.arm in legacy and self.adversarial_ce_coefficient != 0.0:
            raise ValueError("legacy intervention arms must not carry adversarial CE")
        if self.arm in legacy and (self.selector_bundle_path is not None or self.selector_bundle_sha256 is not None):
            raise ValueError("legacy intervention arms must not carry history-routing v2 selector bundles")
        if self.mask is not None:
            parent = self.parent
            provenance = self.mask.provenance
            if (
                provenance.parent_checkpoint_sha256 != parent.checkpoint_sha256
                or provenance.parent_sample_state_sha256 != parent.sample_state_sha256
            ):
                raise ValueError("intervention mask provenance must bind the exact parent checkpoint and sample state")
            if self.selector == "student_history" and provenance.source != "seed0_bartoldson_frozen_predictor":
                raise ValueError("history-selected arms require frozen predictor provenance")
            if self.selector == "class_matched_random" and provenance.source != "class_matched_random":
                raise ValueError("random-selected arms require class-matched random provenance")
            if self.selector == "online_history" and provenance.source != "online_history_epoch39_v2":
                raise ValueError("online history-selected arms require epoch-39 online provenance")
            if (
                self.selector == "class_state_count_matched_random"
                and provenance.source != "class_state_count_matched_random_epoch39_v2"
            ):
                raise ValueError("v2 random-selected arms require class/state/count-matched provenance")
            if self.arm in causal and provenance.source != causal[self.arm][3]:
                raise ValueError("FFNR causal arms require the registered route provenance")
        return self


class PrescriptiveV3Config(StrictModel):
    """Separate epoch-79 retention/prefix treatment contract; never v2."""

    arm: Literal["PF_RET_H", "PF_RET_R", "NR_PFX_H", "NR_PFX_R"]
    parent: InterventionParentConfig
    mask: InterventionMaskConfig
    selector_bundle_path: Path
    selector_bundle_sha256: str
    anchor_checkpoint: Path
    anchor_checkpoint_sha256: str
    pf_teacher_mix: float = 0.75
    pf_anchor_mix: float = 0.25
    pf_start_epoch: int = 80
    pf_end_epoch: int = 129
    nr_prefix_step: int = 5
    nr_full_steps: int = 10
    nr_start_epoch: int = 80
    nr_end_epoch: int = 99

    @model_validator(mode="after")
    def validate_contract(self) -> PrescriptiveV3Config:
        if self.parent.epoch != 79:
            raise ValueError("prescriptive v3 arms require the epoch-79 parent")
        expected_source = (
            "prescriptive_v3_online_history" if self.arm.endswith("_H") else "prescriptive_v3_matched_random"
        )
        route = "peak_failure" if self.arm.startswith("PF_") else "non_recovery"
        if self.mask.provenance.source != expected_source or self.mask.provenance.route != route:
            raise ValueError("prescriptive v3 arm/mask provenance route drifted")
        if (
            self.mask.provenance.parent_checkpoint_sha256 != self.parent.checkpoint_sha256
            or self.mask.provenance.parent_sample_state_sha256 != self.parent.sample_state_sha256
        ):
            raise ValueError("prescriptive v3 mask provenance must bind the exact parent checkpoint and state")
        expected_anchor_state = self.arm.startswith("PF_")
        if self.mask.provenance.anchor_robust_correct is not expected_anchor_state:
            raise ValueError("prescriptive v3 mask anchor state does not match its PF/NR route")
        for name, value in (
            ("selector bundle", self.selector_bundle_sha256),
            ("anchor checkpoint", self.anchor_checkpoint_sha256),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"prescriptive v3 {name} SHA-256 must be lowercase")
        if self.anchor_checkpoint_sha256 != self.parent.checkpoint_sha256:
            raise ValueError("prescriptive v3 anchor must be the exact epoch-79 parent")
        if (
            self.pf_teacher_mix != 0.75
            or self.pf_anchor_mix != 0.25
            or self.pf_start_epoch != 80
            or self.pf_end_epoch != 129
            or self.nr_prefix_step != 5
            or self.nr_full_steps != 10
            or self.nr_start_epoch != 80
            or self.nr_end_epoch != 99
        ):
            raise ValueError("prescriptive v3 treatment values are frozen by the registered contract")
        return self


class EvaluationConfig(StrictModel):
    """Saved-checkpoint evaluation contract; no training-time signal is exposed."""

    checkpoints: Literal["best", "last", "both"] = "both"
    seed: int = 0
    attack: AttackConfig | None = None
    dataset: DatasetConfig | None = None
    autoattack: bool = False
    write_sample_stats: bool = False
    panel_size: int = Field(default=24, ge=0)
    autoattack_batch_size: int = Field(default=128, ge=1)

    @model_validator(mode="after")
    def validate_attack(self) -> EvaluationConfig:
        if self.attack is not None:
            if self.attack.loss != "ce" or self.attack.kl_target is not None:
                raise ValueError("evaluation PGD must use explicit hard-label CE")
            if self.attack.student_mode != "eval" or self.attack.teacher_mode != "eval":
                raise ValueError("evaluation PGD must keep models in eval mode")
        if self.dataset is not None and self.dataset.split not in {"val", "test"}:
            raise ValueError("evaluation.dataset must name the official val or test split")
        return self


class ExperimentConfig(StrictModel):
    schema_version: Literal[2]
    protocol: ProtocolConfig
    tier: Literal["dev", "smoke", "repro", "pilot", "production"] = "dev"
    seeds: SeedsConfig
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    student: ModelConfig = Field(default_factory=ModelConfig)
    teacher: TeacherConfig | None = None
    method: MethodConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    training: TrainingConfig
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    observation: ObservationConfig = Field(default_factory=ObservationConfig)
    intervention: InterventionConfig | None = None
    prescriptive_v3: PrescriptiveV3Config | None = None
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output_dir: Path = Path("outputs/dev")
    # Compatibility with M1 checkpoints/configs.  New paths use tracking.run_id.
    tracker_run_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_pre_v2_schema(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        version = value.get("schema_version")
        if version != 2:
            if version is None:
                raise ValueError("schema_version is required and must be exactly 2; v1/missing configs are unsupported")
            raise ValueError(f"schema_version must be exactly 2; received {version!r}")
        return value

    @property
    def seed(self) -> int:
        """Runtime compatibility only; resolved configuration serializes ``seeds.model_init``."""
        return self.seeds.model_init

    @model_validator(mode="after")
    def validate_cross_fields(self) -> ExperimentConfig:
        if (
            self.tracker_run_id is not None
            and self.tracking.run_id is not None
            and self.tracker_run_id != self.tracking.run_id
        ):
            raise ValueError("tracker_run_id and tracking.run_id must match when both are set")
        if self.tier == "production":
            if self.tracking.mode == "disabled":
                raise ValueError("production requires non-disabled tracking")
            if not self.tracking.project or not self.tracking.entity:
                raise ValueError("production requires tracking.project and tracking.entity")
            if not self.tracking.group:
                raise ValueError("production requires tracking.group")
            if self.tracking.diagnostics_mode != "panel" or self.tracking.panel_size == 0:
                raise ValueError("production requires panel diagnostics with tracking.panel_size > 0")
        if self.tier == "smoke" and self.tracking.mode not in {"disabled", "offline"}:
            raise ValueError("smoke permits only disabled or offline tracking")
        if self.tier in {"repro", "pilot", "production"} and self.tracking.mode not in {"online", "offline_sync"}:
            raise ValueError("repro/pilot/production require online or offline_sync tracking")
        if self.tier == "pilot" and (not self.tracking.project or not self.tracking.entity or not self.tracking.group):
            raise ValueError("pilot requires tracking.project, tracking.entity, and tracking.group")
        if self.tier in {"repro", "pilot", "production"} and (
            self.dataset.name == "synthetic_cifar" or self.student.architecture == "fixture_cnn"
        ):
            raise ValueError("repro/pilot/production forbid synthetic datasets and fixture students")
        if (
            self.tier in {"repro", "pilot", "production"}
            and self.dataset.name == "tiny_imagenet"
            and not self.dataset.content_sha256
        ):
            raise ValueError("repro/pilot/production Tiny-ImageNet requires dataset.content_sha256")
        if self.student.num_classes != self.dataset.num_classes:
            raise ValueError("student and dataset num_classes must match")
        if self.teacher is not None and self.teacher.num_classes != self.dataset.num_classes:
            raise ValueError("teacher and dataset num_classes must match")
        rslad_methods = {
            "rslad",
            "rslad_entropy",
            "rslad_student",
            "rslad_joint",
            "rslad_joint_downweight",
            "rslad_hard_fallback",
            "rslad_frozen_oracle_softening",
        }
        if self.method.id in rslad_methods and self.teacher is None:
            raise ValueError(f"{self.method.id} requires a frozen teacher")
        if self.method.oracle_mask and self.tier != "dev":
            raise ValueError("oracle_mask is scientific/dev-only and is forbidden for smoke, repro, and production")
        if self.method.id == "rslad_frozen_oracle_softening" and self.tier not in {"dev", "production"}:
            raise ValueError("rslad_frozen_oracle_softening permits only dev tests or guarded production runs")
        if self.intervention is not None:
            if self.method.id != "rslad" or self.method.target_policy is not None:
                raise ValueError("intervention arms require baseline method=rslad without a method target policy")
            if self.observation.profile != "teacher_response":
                raise ValueError("intervention arms require teacher_response parent-compatible observations")
            if (
                self.teacher is None
                or self.teacher.checkpoint_sha256 != self.intervention.parent.teacher_checkpoint_sha256
            ):
                raise ValueError("intervention parent teacher SHA must exactly match the arm teacher")
        if self.prescriptive_v3 is not None:
            if (
                self.intervention is not None
                or self.method.id != "rslad"
                or self.observation.profile != "teacher_response"
                or self.protocol.id != "controlled_cifar10_r18_prescriptive_v3_v1"
            ):
                raise ValueError("prescriptive v3 requires its standalone observed RSLAD protocol identity")
            if (
                self.teacher is None
                or self.teacher.checkpoint_sha256 != self.prescriptive_v3.parent.teacher_checkpoint_sha256
            ):
                raise ValueError("prescriptive v3 parent teacher SHA must exactly match the arm teacher")
        if self.observation.records_teacher_response and self.teacher is None:
            raise ValueError("observation.profile=teacher_response requires a frozen teacher")
        if self.teacher is not None and self.teacher.source == "fixture" and self.tier not in {"dev", "smoke"}:
            raise ValueError("fixture teachers are restricted to dev/smoke tiers")
        if self.tier == "pilot" and self.teacher is not None and self.teacher.source != "robustbench":
            raise ValueError("pilot teacher must be a registered RobustBench teacher")
        expected_profile = {
            "synthetic_cifar": "fixture_unit",
            "cifar10": "cifar10_standard",
            "cifar100": "cifar100_standard",
            "tiny_imagenet": "tiny_imagenet_standard",
        }[self.dataset.name]
        if self.student.architecture == "saad_resnet18_cifar_v1" and self.dataset.name == "cifar10":
            expected_profile = "cifar10_raw_identity"
        if self.student.normalization.profile != expected_profile:
            raise ValueError(f"dataset {self.dataset.name} requires student normalization profile {expected_profile}")
        self._validate_protocol_contract()
        return self

    def _validate_protocol_contract(self) -> None:
        """Fail closed for the runnable, versioned protocol identities."""
        from ard.protocols import get_protocol

        spec = get_protocol(self.protocol.id)
        if not spec.runnable_locally:
            return
        pilot_protocols = {
            "controlled_cifar10_r18_pilot_v1",
            "controlled_cifar10_r18_pilot_1ep_v1",
            "controlled_cifar10_r18_pilot_3ep_v1",
        }
        if self.protocol.id in pilot_protocols and self.tier != "pilot":
            raise ValueError(f"{self.protocol.id} requires tier=pilot")
        if self.tier == "pilot" and self.protocol.id not in pilot_protocols:
            raise ValueError("tier=pilot requires a controlled CIFAR-10 pilot protocol")
        if self.protocol.id == "synthetic_smoke_v2":
            smoke_errors: list[str] = []
            if self.tier not in {"dev", "smoke"}:
                smoke_errors.append("tier must be dev or smoke")
            if self.dataset.name != "synthetic_cifar":
                smoke_errors.append("dataset.name must be synthetic_cifar")
            if self.student.architecture != "fixture_cnn":
                smoke_errors.append("student.architecture must be fixture_cnn")
            if smoke_errors:
                raise ValueError("synthetic_smoke_v2 contract violation: " + "; ".join(smoke_errors))
            return
        if self.protocol.id not in {
            "controlled_cifar10_r18_v1",
            "controlled_cifar10_r18_cropshift_v1",
            "controlled_cifar10_r18_cropshift_prefix_v1",
            "controlled_cifar10_r18_delayed_multistep_v1",
            "controlled_cifar10_r18_prescriptive_v3_v1",
            *pilot_protocols,
        }:
            return
        metadata = spec.metadata
        errors: list[str] = []
        dataset = cast(Mapping[str, object], metadata["dataset"])
        student = cast(Mapping[str, object], metadata["student"])
        training = cast(Mapping[str, object], metadata["training"])
        seeds = cast(Mapping[str, object], metadata["seeds"])
        evaluation = cast(Mapping[str, object], metadata["evaluation"])
        assert all(isinstance(item, Mapping) for item in (dataset, student, training, seeds, evaluation))
        for field, expected in dataset.items():
            if getattr(self.dataset, field) != expected:
                errors.append(f"dataset.{field} must be {expected!r}")
        for field, expected in student.items():
            actual = (
                self.student.normalization.profile if field == "normalization_profile" else getattr(self.student, field)
            )
            if actual != expected:
                errors.append(f"student.{field} must be {expected!r}")
        for field, expected in training.items():
            actual = getattr(self.training, field)
            causal_short_horizon = (
                self.intervention is not None
                and self.intervention.arm in {"C79", "RA", "RAR", "RB", "RBR"}
                and field == "epochs"
                and actual in {84, 89, 94}
                and expected == 200
            )
            if causal_short_horizon:
                continue
            if isinstance(expected, float):
                matches = math.isclose(actual, expected, rel_tol=0, abs_tol=1e-15)
            else:
                matches = actual == expected
            if not matches:
                errors.append(f"training.{field} must be {expected!r}")
        for field, expected in seeds.items():
            if getattr(self.seeds, field) != expected:
                errors.append(f"seeds.{field} must be {expected!r}")
        for field, expected in evaluation.items():
            if getattr(self.evaluation, field) != expected:
                errors.append(f"evaluation.{field} must be {expected!r}")
        optimizer = metadata["optimizer"]
        assert isinstance(optimizer, Mapping)
        for field, expected in optimizer.items():
            if getattr(self.optimizer, field) != expected:
                errors.append(f"optimizer.{field} must be {expected!r}")
        schedule = metadata["scheduler"]
        assert isinstance(schedule, Mapping)
        for field, expected in schedule.items():
            if getattr(self.scheduler, field) != expected:
                errors.append(f"scheduler.{field} must be {expected!r}")
        attack_family = self.method.id if self.method.id in {"pgd_at", "trades"} else "rslad"
        train_attacks = metadata["train_attacks"]
        assert isinstance(train_attacks, Mapping)
        attack = train_attacks[attack_family]
        selection = metadata["selection_attack"]
        assert isinstance(attack, Mapping) and isinstance(selection, Mapping)
        actual_selection = self.method.selection_attack
        assert actual_selection is not None
        configured_attacks = (
            (self.method.attack, attack, "method.attack"),
            (actual_selection, selection, "method.selection_attack"),
        )
        for configured, expected, name in configured_attacks:
            for field, value in expected.items():
                if getattr(configured, field) != value:
                    errors.append(f"{name}.{field} must be {value!r}")
            if configured.norm != "linf" or configured.input_domain != "pixel_0_1":
                errors.append(f"{name} must be Linf in raw pixel [0,1] domain")
            if configured.student_mode != "eval" or configured.teacher_mode != "eval":
                errors.append(f"{name} must preserve eval attack modes")
        if errors:
            raise ValueError(f"{self.protocol.id} contract violation: " + "; ".join(errors))
