"""Typed scientific configuration with explicit units and strict keys."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .distillation import IMAGENET_TEACHER_REGISTRY_IDS, DistillationConfig, validate_distillation_cross_fields


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
        "controlled_cifar10_r18_adr_v1",
        "controlled_cifar10_mobilenetv2_adr_v1",
        "controlled_cifar10_r18_trades_49k_validation_v1",
        "synthetic_smoke_v2",
        "imagenet_stage0_dev_v1",
        "controlled_imagenet_stage01_r18_mobilenetv3_adr_v1",
        # Plan 0101: mobile-scale clean-accuracy-floor recipe (modern
        # checkpoint + optional epsilon warmup), deliberately a separate
        # protocol identity from plan 0100's -- not a modification of that
        # plan's own frozen contract.
        "controlled_imagenet_stage01_mobilenetv4_pgd_at_v1",
        # Plan 0102 Workstream A: TRADES-beta probe on the same
        # architecture, a separate protocol identity from plan 0101's own
        # PGD-AT contract.
        "controlled_imagenet_stage01_mobilenetv4_trades_v1",
        # Plan 0102 Workstream B: adr on MobileNetV4-Conv-Small, testing
        # plan 0100's own root-cause hypothesis (temperature mistransferred
        # from a 200-class setting) for adr's ImageNet failure.
        "controlled_imagenet_stage01_mobilenetv4_adr_v1",
        # Plan 0102 Workstream A: Singh/Croce/Hein 2023's own pretrained-init
        # ImageNet recipe (AdamW, cosine decay, label smoothing, heavy
        # augmentation, weight EMA), transplanted onto MobileNetV4-Conv-Small
        # -- a separate protocol identity from this plan's own pgd_at/trades
        # identities.
        "controlled_imagenet_stage01_mobilenetv4_revisiting_at_recipe_v1",
        # Decision 0017, option E: the identical revisiting_at recipe
        # (same attack identity, optimizer, scheduler, augmentation,
        # weight-EMA) on resnet18_imagenet instead of MobileNetV4-Conv-Small
        # -- an architecture-swap control, human's own proposal (chat,
        # 2026-09-22), testing whether this recipe's catastrophic-collapse
        # failure mode is specific to the extra-lightweight architecture. A
        # separate protocol identity from the MobileNetV4 arm's, matching
        # this project's own precedent of one id per architecture pairing.
        "controlled_imagenet_stage01_r18_revisiting_at_recipe_v1",
        # Plan 0103: lightweight-architecture survey, all candidates trained
        # with Arm A's own pinned, working, plain-SGD PGD-AT recipe (never
        # the collapsed AdamW recipe) -- one scientific contract shared
        # across every candidate architecture, since architecture is the
        # single studied variable.
        "controlled_imagenet_stage02_lightweight_architecture_survey_v1",
        # Plan 0103 Phase 0: training-budget x initialization 2x2 on
        # MobileNetV4-Conv-Small (100-epoch cells; schedule differs from Arm A).
        "controlled_imagenet_stage02_budget_init_v1",
        # Plan 0104: fine-tuning-shaped schedule on MobileNetV4-Conv-Small --
        # Arm A with peak learning rate 0.05 -> 0.005 (decision 0019).
        "controlled_imagenet_stage02_finetune_lr_v1",
        # Plan 0104 stage 2: per-init learning-rate comparison on
        # MobileNetV4-Conv-Small (pretrained 0.015; random init 0.025 / 0.1).
        "controlled_imagenet_stage02_init_lr_grid_v1",
        # Plan 0104 stage 3: ImageNet-100 proxy validation -- the six
        # MobileNetV4-Conv-Small stage-1/2 cells rerun on ImageNet-100
        # (``imagenet`` adapter, 100-class root, pinned manifest hashes, a
        # 100-way head) with everything else identical, to test whether the
        # proxy's learning-rate ranking matches ImageNet-1k's.
        "controlled_imagenet100_proxy_lr_v1",
        # Plan 0103 option A (human-approved 2026-09-27): clean-training
        # control for the random-init budget cells -- Arm A's recipe with
        # method ``standard`` (no training attack) at 50 and 100 epochs:
        # is PGD-AT's 50-epoch saturation specific to the robust objective
        # or a property of the recipe?
        "controlled_imagenet_stage02_clean_budget_v1",
        # Plan 0103 (human-approved 2026-09-30): AdvXL-inspired two-stage
        # adversarial training of MobileNetV4-Conv-Small from random init,
        # FLOPs-matched to the single-stage 50-epoch run -- stage 1 at a
        # 112px training resolution with PGD-1, stage 2 a 224px PGD-3
        # fine-tune initialized from stage 1's last.pt. Each config sets
        # exactly one of training.train_image_size (stage 1) or
        # training.init_checkpoint (stage 2).
        "controlled_imagenet_stage02_two_stage_lowres_v1",
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
    """Optimizer identity. ``sgd`` is the original M1 protocol implementation.

    Plan 0102 Workstream A: ``adamw`` reproduces Singh/Croce/Hein 2023's own
    pretrained-init ImageNet recipe (arXiv:2303.01870, Appendix A.1; fetched
    from the paper this session, not paraphrased) -- AdamW, beta1/beta2
    configurable (paper uses 0.9/0.95, not PyTorch's 0.9/0.999 default, so
    both are explicit config fields rather than library defaults). The two
    optimizer identities use disjoint field sets (sgd: momentum/nesterov;
    adamw: beta1/beta2) enforced below, rather than one flat "works for
    either" set someone could half-fill in.
    """

    id: Literal["sgd", "adamw"]
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    momentum: float | None = Field(default=None, ge=0, lt=1)
    nesterov: bool | None = None
    beta1: float | None = Field(default=None, gt=0, lt=1)
    beta2: float | None = Field(default=None, gt=0, lt=1)
    # Plan 0103 Phase 2 batch A (human-approved 2026-10-08), sgd only: put
    # every parameter with ndim <= 1 (normalization affine weights/biases and
    # every bias) in a weight_decay=0 parameter group -- the same split AdamW
    # always uses (ard.cli.train._weight_decay_parameter_groups). AdamW refuses
    # the flag (it is already on there). Serialized only when true, so every
    # existing config keeps a byte-identical resolved config and hash; when
    # true it is in the optimizer dump, hence in the evaluation record's
    # training_protocol_identity.
    exclude_norm_bias_from_weight_decay: bool = Field(default=False, exclude_if=lambda value: value is False)

    @model_validator(mode="after")
    def validate_by_id(self) -> OptimizerConfig:
        if self.id == "sgd":
            if self.momentum is None or self.nesterov is None:
                raise ValueError("sgd requires momentum and nesterov")
            if self.beta1 is not None or self.beta2 is not None:
                raise ValueError("sgd does not use beta1/beta2")
            if self.nesterov and self.momentum <= 0:
                raise ValueError("nesterov SGD requires positive momentum")
        else:
            if self.beta1 is None or self.beta2 is None:
                raise ValueError("adamw requires beta1 and beta2")
            if self.momentum is not None or self.nesterov is not None:
                raise ValueError("adamw does not use momentum/nesterov")
            if self.exclude_norm_bias_from_weight_decay:
                raise ValueError(
                    "optimizer.exclude_norm_bias_from_weight_decay is only for sgd; adamw always excludes "
                    "normalization affine parameters and biases from weight decay"
                )
        return self


class SchedulerConfig(StrictModel):
    """Scheduler identity frozen before M1 supplies concrete schedules.

    Plan 0102 Workstream A: ``warmup_cosine`` reproduces Singh/Croce/Hein
    2023's own pretrained-init ImageNet recipe (arXiv:2303.01870, Appendix
    A.1) -- linear warmup (same shape ``warmup_multistep`` already has, and
    for the same reason: "peak LR attained at epoch 10" of a 50-epoch run)
    followed by cosine decay to 0 at the final epoch, instead of multistep
    decay. ``milestones``/``gamma`` are meaningless for a cosine decay (there
    is no discrete drop), so this id requires the same no-op values
    "identity" does, rather than silently ignoring whatever is supplied.
    """

    id: Literal["identity", "multistep", "warmup_multistep", "warmup_cosine"]
    milestones: tuple[int, ...]
    gamma: float = Field(gt=0)
    step_at: Literal["epoch_end"]
    # Plan 0100 (scientific review finding P1-5): a linear LR warmup over the
    # first warmup_epochs epochs, reaching the base LR exactly at epoch
    # warmup_epochs, before multistep decay -- only defined for
    # "warmup_multistep". Singh/Croce/Hein 2023's own pretrained-init recipe
    # (this project's own literature source for the 50-epoch horizon) uses
    # exactly this shape: linear warmup for the first ~20% of a 50-epoch
    # run, then decay -- see ard.schedules.warmup_multistep_multiplier.
    warmup_epochs: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_schedule(self) -> SchedulerConfig:
        if self.id == "identity":
            if self.milestones or self.gamma != 1.0:
                raise ValueError("identity scheduler requires milestones=[] and gamma=1.0")
            if self.warmup_epochs is not None:
                raise ValueError("warmup_epochs is only defined for warmup_multistep/warmup_cosine")
        elif self.id == "multistep":
            if not self.milestones or tuple(sorted(set(self.milestones))) != self.milestones or self.milestones[0] < 0:
                raise ValueError("multistep scheduler requires strictly increasing non-negative milestones")
            if self.warmup_epochs is not None:
                raise ValueError("warmup_epochs is only defined for warmup_multistep/warmup_cosine")
        elif self.id == "warmup_multistep":
            if not self.milestones or tuple(sorted(set(self.milestones))) != self.milestones or self.milestones[0] < 0:
                raise ValueError("warmup_multistep scheduler requires strictly increasing non-negative milestones")
            if self.warmup_epochs is None:
                raise ValueError("warmup_multistep requires warmup_epochs")
            if self.milestones[0] < self.warmup_epochs:
                raise ValueError("warmup_multistep's first decay milestone must not occur during warmup")
        else:
            if self.milestones or self.gamma != 1.0:
                raise ValueError("warmup_cosine scheduler requires milestones=[] and gamma=1.0 (no discrete decay)")
            if self.warmup_epochs is None:
                raise ValueError("warmup_cosine requires warmup_epochs")
        return self


class TargetPolicyConfig(StrictModel):
    """Identity for adversarial-student teacher-target calibration only."""

    id: Literal["teacher_target_uniform_mix"]
    version: Literal[1]
    risk_transform: Literal["identity"]
    mixing: Literal["uniform"]
    apply_to: Literal["adversarial_student_kd"]
    rho_max: float = Field(default=0.5, ge=0, le=1)


class AdrConfig(StrictModel):
    """EMA-teacher and rectification hyperparameters for ``adr``/``adr_trades``.

    ADR: Wu, Wang & Chen, "Annealing Self-Distillation Rectification Improves
    Adversarial Training", ICLR 2024, arXiv:2305.12118.  Defaults match the
    paper's CIFAR-10 configuration exactly (its §5.1 and the official code's
    per-dataset gin configs, github.com/yuyuwu5/ADR); this is a fixed decay
    updated once per training iteration (not per epoch).  Temperature always
    anneals by the per-iteration cosine schedule
    (``ard.schedules.cosine_value.cosine_anneal``), covering the full
    training run.  Lambda does too **only** when ``lambda_source="cosine"``
    (the default -- every existing config validates unchanged); when a
    config explicitly opts into ``lambda_source="gap_adaptive"`` (plan 0098),
    lambda instead comes from ``ard.schedules.gap_adaptive``, recomputed once
    per epoch boundary from that epoch's own train/val robust-accuracy gap
    and held fixed through the next epoch's iterations -- see
    ``docs/plans/0098-gap-adaptive-adr-cifar10.md`` and
    ``docs/SCIENTIFIC_INVARIANTS.md``'s ADR section for what that gap
    actually measures (a live scientific-review open question, not settled
    by this schema).  The EMA-decay is a genuinely separate piece of state from
    ``MethodConfig.student_ema_decay`` (a per-sample robust-margin EMA used
    by an unrelated risk-weighting mechanism) and must not be confused with
    it.
    """

    ema_decay: float = Field(default=0.995, ge=0, lt=1)
    temperature_high: float = Field(default=2.5, gt=0)
    temperature_low: float = Field(default=2.0, gt=0)
    lambda_low: float = Field(default=0.7, ge=0, le=1)
    lambda_high: float = Field(default=0.95, ge=0, le=1)
    # Plan 0098: replaces lambda's *source* only. "cosine" (default) is
    # exactly plan 0097's validated behavior -- a pure function of epoch
    # index, unchanged. "gap_adaptive" drives lambda instead from a live,
    # self-normalizing measurement of train/val robust-accuracy gap
    # (ard.schedules.gap_adaptive), recomputed once per epoch boundary and
    # held fixed through that epoch's iterations. Temperature's own cosine
    # anneal is untouched either way.
    lambda_source: Literal["cosine", "gap_adaptive"] = "cosine"
    gap_smoothing_beta: float = Field(default=0.9, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_bounds(self) -> AdrConfig:
        if self.temperature_low > self.temperature_high:
            raise ValueError("temperature_low must not exceed temperature_high (temperature anneals downward)")
        if self.lambda_low > self.lambda_high:
            raise ValueError("lambda_low must not exceed lambda_high (lambda anneals upward)")
        if self.lambda_source == "cosine" and self.gap_smoothing_beta != 0.9:
            raise ValueError("gap_smoothing_beta is only meaningful when lambda_source=gap_adaptive")
        return self


class MixedBatchConfig(StrictModel):
    """Plan 0103 Phase 2 batch A: mixed clean + adversarial batches for ``pgd_at``.

    Kurakin, Goodfellow & Bengio 2017 (arXiv:1611.01236, Sec. 3.1): of every
    minibatch of ``m`` examples, ``k`` are replaced by adversarial examples and
    the loss is ``(sum_clean L + lambda * sum_adv L) / ((m - k) + lambda * k)``
    (paper: m=32, k=16, lambda=0.3). Here ``k = floor(adversarial_fraction * m)``
    for each per-rank batch of ``m`` examples (the last partial batch included)
    and the adversarial examples are the FIRST ``k`` positions of the batch:
    fixed by position, no extra random draw. The sampler already shuffles every
    epoch, so the attacked subset is a seeded, uniformly random subset of the
    data that changes every epoch. ``adversarial_weight`` is lambda. Only the
    ``k`` selected examples are attacked (the configured training attack, its
    random start drawn for those ``k`` only); the other ``m - k`` enter the
    training forward clean. The normalizer counts valid examples. World size 1
    only (the schema and the Trainer refuse DDP).

    ``split_batchnorm`` (Xie & Yuille 2019, arXiv:1906.03787, "MBN"; the same
    mechanism as AdvProp's auxiliary BN, arXiv:1911.09665): the adversarial
    sub-batch goes through the model's own BatchNorm layers (the MAIN BN) and
    the clean sub-batch through an auxiliary copy of every BatchNorm layer's
    affine parameters and running statistics. The main BN is the adversarial
    BN: the training attack, validation, checkpoint selection, saved ``model``
    weights and evaluation all use it; the auxiliary (clean) BN exists only
    during training and is checkpointed separately for resume. BatchNorm
    models only (refused when the student has no BatchNorm layer, e.g.
    LayerNorm-only ViT/ConvNeXt); LayerNorm/GroupNorm layers, which hold no
    batch statistics, stay shared. Refused unless every sub-batch has at least
    2 examples: the schema checks the full per-rank batch, ``ard.cli.train`` the
    last partial batch of the epoch (examples are never dropped); refused with
    ``training.compile`` and ``training.init_checkpoint``. The train-mode
    clean-position accuracy is measured through the auxiliary BN.
    """

    adversarial_fraction: float = Field(gt=0, lt=1)
    adversarial_weight: float = Field(gt=0)
    split_batchnorm: bool = False

    @model_validator(mode="after")
    def validate_weight(self) -> MixedBatchConfig:
        if not math.isfinite(self.adversarial_weight):
            raise ValueError("mixed_batch.adversarial_weight must be finite")
        return self


class AwpConfig(StrictModel):
    """Plan 0103 Phase 2 batch A: Adversarial Weight Perturbation on top of ``pgd_at``.

    Wu, Xia & Wang 2020 (arXiv:2004.05884), official code csdongxian/AWP
    ``AT_AWP/`` (``utils_awp.py``, ``train_cifar10.py``; pinned in
    external.lock.yaml as ``awp`` at a7acf5d8, vendored under ``.external/awp``
    and imported by the parity test). This is AWP on this project's PGD-AT,
    not an AT-AWP reproduction: upstream crafts its PGD examples with the model
    in train mode, ours with the configured ``attack.student_mode`` (default
    eval), plus our own data, schedule and architecture. After the PGD attack, a
    proxy copy of the student (train mode) takes one SGD step (lr 0.01) that
    ascends the adversarial loss; the per-layer difference ``d`` is rescaled to
    ``||w_l|| / (||d_l|| + 1e-20) * d_l`` for every state entry with ndim > 1
    whose name contains ``weight``; the student is perturbed by ``gamma * d``,
    takes its ordinary training step at the perturbed weights, and the
    perturbation is subtracted again after the optimizer step. Defaults are the
    AT-AWP code's ``--awp-gamma 0.01`` and ``--awp-warmup 0`` (AWP active from
    epoch ``warmup_epochs`` on). Extra compute: one proxy forward+backward per
    step plus a state-dict copy (PGD-K step: K+1 -> K+2 forward/backward passes)
    and one more model copy in memory. Single process only. The step's pre-update
    observability (``train_robust_accuracy``, per-sample diagnostic rows) is
    measured at the perturbed weights the training forward used; post-step
    metrics, validation, EMA and checkpoints see the restored weights.
    """

    gamma: float = Field(default=0.01, gt=0)
    warmup_epochs: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_gamma(self) -> AwpConfig:
        if not math.isfinite(self.gamma):
            raise ValueError("awp.gamma must be finite")
        return self


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
        "imagenet_standard",
        "imagenet_raw_identity",
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
            "imagenet_standard": (
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                "Standard ILSVRC-2012 mean/std convention (torchvision.models default preprocessing)",
            ),
            "imagenet_raw_identity": (
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
                "ImageNet checkpoints trained on raw [0,1] pixels without mean/std (e.g. timm mobilevit_*.cvnets_in1k)",
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
    # "rectified" means the caller resolves the target itself and supplies it
    # as AttackRequest.target_probabilities (already a probability
    # distribution, not logits) -- used by ard.objectives.adr's EMA-blended
    # label, which must be identical across the attack and the training loss.
    kl_target: Literal["student_clean", "teacher_clean", "rectified"] | None = None
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


def _is_sha256_hex(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


class DerivedDatasetConfig(StrictModel):
    """A dataset root that is a declared, deterministic derivative of another.

    Produced by ``scripts/build_resized_imagenet.py``: the source's ``train/``
    tree with identical relative paths (so class indices, source IDs and the
    seeded validation split are unchanged), every image whose shorter side
    exceeds ``short_side`` resized to that shorter side (``resample`` filter)
    and re-encoded as JPEG at ``jpeg_quality``; smaller images copied byte for
    byte. ``content_sha256`` is the *source* dataset's manifest digest; the
    derived root's own digest is the enclosing ``dataset.content_sha256``.
    ``manifest_sha256`` pins the build manifest at the derived root, which is
    checked against these fields whenever the dataset is loaded.
    """

    content_sha256: str
    transform: Literal["resize_short_side"]
    short_side: int = Field(ge=1)
    jpeg_quality: int = Field(ge=1, le=100)
    resample: Literal["lanczos"]
    manifest_sha256: str

    @model_validator(mode="after")
    def validate_digests(self) -> DerivedDatasetConfig:
        for name in ("content_sha256", "manifest_sha256"):
            if not _is_sha256_hex(getattr(self, name)):
                raise ValueError(f"dataset.derived_from.{name} must be a lowercase 64-character SHA-256 hex digest")
        return self


class DatasetConfig(StrictModel):
    name: Literal["synthetic_cifar", "cifar10", "cifar100", "tiny_imagenet", "imagenet"] = "synthetic_cifar"
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
    # Identity only, never a path: the same mask lives at different paths on
    # different hosts, and a path in a hashed config makes two identical runs
    # look different.  The verified mask itself is supplied at run time and is
    # checked against these three fields before any image is transformed.
    stagewise_late_mask_selected_ids_sha256: str | None = None
    stagewise_late_mask_selected_count: int | None = Field(default=None, ge=1)
    # Plan 0102 Workstream A: Singh/Croce/Hein 2023's own pretrained-init
    # ImageNet recipe (arXiv:2303.01870, Appendix A.2) uses RandAugment(2
    # layers, magnitude 9) + RandomErasing(p=0.25) on top of the existing
    # RandomResizedCrop+flip. Deliberately NOT built against this project's
    # own local-``torch.Generator`` determinism discipline (human decision,
    # chat, 2026-09-21): torchvision's own ``RandAugment``/``RandomErasing``
    # draw from the *global* RNG, so unlike ``EpochImageNetTransform``'s
    # crop+flip, a resumed epoch is not guaranteed to reproduce the exact
    # same augmented view per source ID when this is enabled. Accepted for
    # this exploratory recipe-matching experiment; revisit for full
    # determinism if this becomes a lasting part of the recipe rather than
    # a one-off comparison. Default False reproduces today's exact
    # behavior for every existing config.
    imagenet_heavy_augmentation: bool = False
    # Plan 0103 Phase 2 batch C (human-approved 2026-10-08): IDBH (Li &
    # Spratling, ICLR 2023, "Data augmentation alone can improve adversarial
    # training") appended to the ImageNet RandomResizedCrop + flip -- see
    # ard.data.datasets.EpochImageNetTransform. Unlike
    # imagenet_heavy_augmentation, every IDBH draw comes from a generator
    # keyed by (augmentation seed, epoch, source id), so a resumed epoch
    # reproduces the same view per sample. "idbh_weak_nocropshift" (the
    # human-chosen Phase 2 arm): ColorShape('color') + Random Erasing p=0.5,
    # no CropShift on top of RandomResizedCrop. "idbh_weak" / "idbh_strong":
    # additionally a fraction-preserving CropShift adaptation, erasing p=0.5 /
    # 1.0. Serialized only when not "standard", so every existing config keeps
    # a byte-identical resolved config and config hash.
    imagenet_augmentation: Literal["standard", "idbh_weak_nocropshift", "idbh_weak", "idbh_strong"] = Field(
        default="standard", exclude_if=lambda value: value == "standard"
    )
    # Plan 0103 loader speedup C (human-approved 2026-09-30): the root is a
    # pre-resized derivative of the dataset named by
    # derived_from.content_sha256; see DerivedDatasetConfig. The training
    # pixels differ, so it is part of the dataset identity (and the
    # enclosing content_sha256 is the derived root's own). ImageNet train
    # split only. Serialized only when set, so every existing config keeps a
    # byte-identical resolved config and config hash.
    derived_from: DerivedDatasetConfig | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def validate_dataset(self) -> DatasetConfig:
        if self.name in {"cifar10", "cifar100"} and self.split == "val":
            raise ValueError("CIFAR has no validation split alias; use official train or test")
        expected = {"cifar10": 10, "cifar100": 100}.get(self.name)
        if expected is not None and self.num_classes != expected:
            raise ValueError(f"{self.name} requires num_classes={expected}")
        mask_fields = (self.stagewise_late_mask_selected_ids_sha256, self.stagewise_late_mask_selected_count)
        if any(field is not None for field in mask_fields):
            if any(field is None for field in mask_fields):
                raise ValueError("a stagewise late mask needs both its ID digest and its count, or neither")
            if self.augmentation_policy != "stagewise":
                raise ValueError("a stagewise late mask only applies to the stagewise augmentation policy")
        if self.name == "tiny_imagenet" and self.root is None:
            raise ValueError("tiny_imagenet requires an explicit root")
        if self.name == "imagenet" and self.root is None:
            raise ValueError("imagenet requires an explicit root")
        if self.augmentation_policy != "canonical" and self.name not in {"cifar10", "cifar100"}:
            raise ValueError("non-canonical augmentation policies are currently defined only for CIFAR datasets")
        if self.imagenet_heavy_augmentation and self.name != "imagenet":
            raise ValueError("imagenet_heavy_augmentation is only defined for the imagenet dataset")
        if self.imagenet_augmentation != "standard":
            if self.name != "imagenet" or self.split != "train":
                raise ValueError("imagenet_augmentation is only defined for the imagenet train split")
            if self.imagenet_heavy_augmentation:
                raise ValueError("imagenet_augmentation and imagenet_heavy_augmentation are mutually exclusive")
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
        if self.derived_from is not None:
            if self.name != "imagenet" or self.split != "train":
                raise ValueError("dataset.derived_from is only defined for the imagenet train split")
            if self.content_sha256 is None:
                raise ValueError("dataset.derived_from requires the derived root's own dataset.content_sha256")
            if self.content_sha256 == self.derived_from.content_sha256:
                raise ValueError("dataset.content_sha256 must be the derived root's digest, not its source's")
        return self


class ModelConfig(StrictModel):
    architecture: Literal[
        "saad_resnet18_cifar_v1",
        "torchvision_resnet18_cifar_norm_v1",
        "resnet18_cifar",
        "mobilenet_v2_cifar",
        "fixture_cnn",
        # ImageNet-scale, native-resolution (plan 0099 prep; unpatched
        # torchvision definitions -- no 32px conv1/maxpool patch). Available
        # for a future config to select; this does not itself pick Stage 1's
        # architecture, see docs/plans/0099-imagenet-stage0-prep.md.
        "resnet18_imagenet",
        "resnet50_imagenet",
        "mobilenet_v2_imagenet",
        "mobilenet_v3_small_imagenet",
        # Plan 0101: modern (2024) mobile-scale replacement for
        # mobilenet_v3_small_imagenet -- see registry.py's build_architecture
        # comment.
        "mobilenetv4_conv_small_imagenet",
        # timm's ConvNeXt-Tiny (not torchvision's -- see registry.py's
        # build_architecture comment). Registered to evaluate the
        # Singh/Croce/Hein 2023 (arXiv 2303.01870) published eps=4/255
        # ConvNeXt-T checkpoint through this project's own AutoAttack
        # pipeline as a pipeline-correctness check, not to select an
        # architecture for any ImageNet training campaign.
        "convnext_tiny_imagenet",
        # Plan 0103: lightweight-architecture-survey candidates, all
        # trained with Arm A's own pinned plain-SGD PGD-AT recipe -- see
        # registry.py's build_architecture and
        # docs/plans/0103-lightweight-imagenet-architecture-survey.md.
        "efficientnet_b0_imagenet",
        "mobilenetv4_conv_medium_imagenet",
        "convnext_atto_imagenet",
        "deit_tiny_imagenet",
        "mobilevit_s_imagenet",
        # Plan 0103 Phase 2 architecture variants (human-approved 2026-10-08), random init only
        # (absent from the pretrained allowlist below) -- see ard.models.variants.
        "mobilenetv4_conv_small_silu_imagenet",
        "mobilenetv4_conv_small_gelu_imagenet",
        "mobilenetv4_conv_small_se_imagenet",
        "mobilenetv4_conv_small_silu_se_imagenet",
        "deit_tiny_convstem_imagenet",
        "convnext_atto_deep_narrow_imagenet",
        "convnext_atto_ols_imagenet",
        "convnext_atto_convstem_imagenet",
        "mobilenetv4_conv_small_se_fullhead_imagenet",
    ] = "fixture_cnn"
    num_classes: int = Field(default=10, ge=2)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    preprocessing_owner: Literal["student_adapter"] = "student_adapter"
    # Plan 0100: initialize from standard (non-robust) ImageNet-1k pretrained
    # weights before adversarial fine-tuning, per Singh/Croce/Hein 2023's own
    # recipe (torchvision's public weights are their explicitly-sanctioned
    # in-kind substitute for "standard models from the timm library or the
    # original papers"). Default False reproduces today's exact behavior
    # (weights=None) for every existing config that never mentions this
    # field. Only defined for the two architectures plan 0100 actually uses;
    # see registry.py's build_architecture for the enforced allowlist.
    pretrained: bool = False

    @model_validator(mode="after")
    def validate_pretrained(self) -> ModelConfig:
        if self.pretrained and self.architecture not in {
            "resnet18_imagenet",
            "mobilenet_v3_small_imagenet",
            "mobilenetv4_conv_small_imagenet",
            "efficientnet_b0_imagenet",
            "mobilenetv4_conv_medium_imagenet",
            "convnext_atto_imagenet",
            "deit_tiny_imagenet",
            "mobilevit_s_imagenet",
        }:
            raise ValueError(f"pretrained=True is not supported for architecture: {self.architecture}")
        return self


class TeacherConfig(StrictModel):
    # Plan 0103 Phase 2 batch D: ``imagenet_registry`` teachers are the pinned
    # ImageNet-1k Linf 4/255 profiles in ard.models.imagenet_teacher_registry
    # (checkpoint SHA-256, architecture, normalization and threat restated
    # here and checked against the profile at build time).
    source: Literal["checkpoint", "fixture", "robustbench", "imagenet_registry"] = "fixture"
    architecture: Literal[
        "saad_resnet18_cifar_v1",
        "torchvision_resnet18_cifar_norm_v1",
        "resnet18_cifar",
        "mobilenet_v2_cifar",
        "fixture_cnn",
        "robustbench_wide_resnet",
        "robustbench_dm_wide_resnet",
        "resnet50_imagenet",
        "convnext_tiny_convstem_imagenet",
        "vit_s_convstem_imagenet",
        "convnext_base_convstem_imagenet",
        "mobilenetv4_conv_medium_imagenet",
    ] = "fixture_cnn"
    num_classes: int = Field(default=10, ge=2)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    preprocessing_owner: Literal["teacher_adapter", "model_embedded"] = "teacher_adapter"
    checkpoint: Path | None = None
    checkpoint_sha256: str | None = None
    registry_id: (
        Literal[
            "chen2021_ltd_wrn34_10",
            "chen2021_ltd_wrn34_20",
            "bartoldson2024_adversarial_wrn94_16",
            "salman2020_resnet50_linf_eps4",
            "singh2023_convnext_t_convstem",
            "singh2023_vit_s_convstem",
            "singh2023_convnext_b_convstem",
            "ard_mobilenetv4_conv_medium_phase1_random",
        ]
        | None
    ) = None
    threat_norm: Literal["linf"] = "linf"
    threat_epsilon: str = "8/255"
    fixture_seed: int = 1729

    @model_validator(mode="after")
    def validate_source(self) -> TeacherConfig:
        imagenet_ids = IMAGENET_TEACHER_REGISTRY_IDS
        if self.source in {"checkpoint", "robustbench", "imagenet_registry"} and (
            self.checkpoint is None or self.checkpoint_sha256 is None
        ):
            raise ValueError(f"{self.source} teachers require checkpoint and checkpoint_sha256")
        if self.source in {"robustbench", "imagenet_registry"} and self.registry_id is None:
            raise ValueError(f"{self.source} teachers require registry_id")
        if self.source not in {"robustbench", "imagenet_registry"} and self.registry_id is not None:
            raise ValueError("registry_id is only valid for robustbench and imagenet_registry teachers")
        if self.source == "robustbench" and self.registry_id in imagenet_ids:
            raise ValueError("ImageNet registry IDs require teacher.source=imagenet_registry")
        if self.source == "imagenet_registry":
            if self.registry_id not in imagenet_ids:
                raise ValueError("teacher.source=imagenet_registry requires an ImageNet registry ID")
            if self.threat_epsilon != "4/255" or self.preprocessing_owner != "teacher_adapter":
                raise ValueError("ImageNet registry teachers are Linf 4/255 with teacher_adapter-owned preprocessing")
            # The full profile (architecture, normalization, digest) is checked
            # against ard.models.imagenet_teacher_registry when the teacher is built.
            return self._validate_digest()
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
        return self._validate_digest()

    def _validate_digest(self) -> TeacherConfig:
        if self.checkpoint_sha256 is not None and (
            len(self.checkpoint_sha256) != 64 or any(char not in "0123456789abcdef" for char in self.checkpoint_sha256)
        ):
            raise ValueError("checkpoint_sha256 must be a lowercase 64-character digest")
        return self


class MethodConfig(StrictModel):
    id: Literal[
        # Plan 0103 option A: standard (non-adversarial) training -- the same
        # hard-label CE as pgd_at, on the clean training batch, with no
        # training attack at all. method.selection_attack is still required,
        # so validation keeps reporting clean AND PGD accuracy.
        "standard",
        "pgd_at",
        "trades",
        "rslad",
        # Plan 0103 Phase 2 batch D (human-approved 2026-10-08): RSLAD whose
        # 5/6 adversarial KL targets the teacher on the student's adversarial
        # example, softmax(T(x')/tau), instead of T(x); attack, clean term,
        # coefficients and temperature are RSLAD's. AdaAD's outer loss (Huang
        # et al. CVPR 2023, Eq. 11) with RSLAD's attack. ImageNet distillation
        # only (see ard.config.distillation).
        "rslad_advt",
        "rslad_entropy",
        "rslad_student",
        "rslad_joint",
        "rslad_joint_downweight",
        "rslad_hard_fallback",
        "rslad_frozen_oracle_softening",
        "adr",
        "adr_trades",
    ]
    version: Literal[1]
    # None only for ``standard`` (no training attack); every other method
    # requires one -- see ``resolve_training_attack``.
    attack: AttackConfig | None = Field(default_factory=AttackConfig)
    selection_attack: AttackConfig | None = None
    temperature: float = Field(default=1.0, gt=0)
    temperature_squared: bool = True
    trades_beta: float = Field(default=6.0, ge=0)
    entropy_gamma: float = Field(default=1.0, gt=0)
    # Plan 0102 Workstream A: pgd_at's own label smoothing (Singh/Croce/Hein
    # 2023, arXiv:2303.01870, Appendix A.1: "label smoothing coefficient of
    # 0.1"). Default 0.0 reproduces today's exact plain hard-label CE for
    # every existing pgd_at config. Wired for pgd_at and standard (the same
    # CE objective) -- not a statement that other methods couldn't use it,
    # just not yet asked for.
    label_smoothing: float = Field(default=0.0, ge=0, lt=1)
    student_ema_decay: float = Field(default=0.9, ge=0, lt=1)
    student_policy_warmup_epochs: int = Field(default=1, ge=1)
    target_policy: TargetPolicyConfig | None = None
    oracle_mask: bool = False
    frozen_oracle_manifest: Path | None = None
    frozen_oracle_manifest_sha256: str | None = None
    adr: AdrConfig | None = None
    # Plan 0103 two-stage run, stage 1 only (human-approved 2026-09-30):
    # the training attack is PGD-1 with step 4/255 (= epsilon) while
    # checkpoint selection -- and therefore the default evaluation attack --
    # stays the reference's PGD-10 with step 8/765, so stage-1 checkpoints
    # are measured under the same threat identity as every other ImageNet
    # arm. True exempts *only* the step-size equality below; norm, input
    # domain, epsilon and random start must still match, selection must be
    # given explicitly and equal the reference PGD-10 identity exactly
    # (docs/SCIENTIFIC_INVARIANTS.md records this exception, human-approved
    # 2026-09-30), and ExperimentConfig restricts the flag to
    # controlled_imagenet_stage02_two_stage_lowres_v1 with
    # training.train_image_size set. Serialized only when true, so every
    # existing config keeps a byte-identical resolved config.
    selection_step_size_independent: bool = Field(default=False, exclude_if=lambda value: value is False)
    # Plan 0103 Phase 2 batch A (human-approved 2026-10-08), pgd_at only,
    # mutually exclusive; see MixedBatchConfig / AwpConfig. Serialized only
    # when set, so every existing config keeps a byte-identical resolved
    # config and hash; when set they are in method_identity (and so the
    # evaluation record's identity) through the method dump.
    mixed_batch: MixedBatchConfig | None = Field(default=None, exclude_if=lambda value: value is None)
    awp: AwpConfig | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def validate_phase2_batch_a_options(self) -> MethodConfig:
        for name in ("mixed_batch", "awp"):
            if getattr(self, name) is not None and self.id != "pgd_at":
                raise ValueError(f"method.{name} is defined only for method pgd_at")
        if self.mixed_batch is not None and self.awp is not None:
            raise ValueError("method.mixed_batch and method.awp are not specified together")
        return self

    @property
    def name(self) -> str:
        """Runtime compatibility only; resolved configuration serializes ``id``."""
        return self.id

    def require_training_attack(self) -> AttackConfig:
        """Return ``attack`` itself (the same object), or refuse ``standard``.

        For callers that only make sense for a method with a training attack;
        typed non-optional so they need no per-site ``None`` handling.
        """
        if self.attack is None:
            raise ValueError(f"method {self.id} has no training attack")
        return self.attack

    @model_validator(mode="before")
    @classmethod
    def resolve_training_attack(cls, value: Any) -> Any:
        """``standard`` has no training attack; every other method has one.

        A ``standard`` config may omit ``attack`` or give it as ``null`` (its
        resolved config serializes ``null``); an attack mapping there would be
        silently ignored, so it is refused. Any other method given
        ``attack: null`` is refused rather than silently defaulted.
        """
        if not isinstance(value, dict):
            return value
        if value.get("id") == "standard":
            if value.get("attack") is not None:
                raise ValueError(
                    "method standard trains on clean inputs and has no training attack; remove method.attack "
                    "(checkpoint selection uses method.selection_attack)"
                )
            return {**value, "attack": None}
        if "attack" in value and value["attack"] is None:
            raise ValueError(f"method {value.get('id')!r} requires a training attack; method.attack must not be null")
        return value

    @model_validator(mode="after")
    def resolve_selection_attack(self) -> MethodConfig:
        if self.id == "standard":
            return self._resolve_standard()
        if self.attack is None:
            raise ValueError(f"method {self.id} requires a training attack")
        expected_loss = "ce" if self.id == "pgd_at" else "kl"
        expected_target = {
            "trades": "student_clean",
            "rslad": "teacher_clean",
            "rslad_advt": "teacher_clean",
            "rslad_entropy": "teacher_clean",
            "rslad_student": "teacher_clean",
            "rslad_joint": "teacher_clean",
            "rslad_joint_downweight": "teacher_clean",
            "rslad_hard_fallback": "teacher_clean",
            "rslad_frozen_oracle_softening": "teacher_clean",
            # Plain ADR's attack ascends the EMA-blended rectified label the
            # caller supplies via AttackRequest.target_probabilities. The
            # TRADES+ADR variant does NOT: the official ADR code's inner PGD
            # attack ignores the rectified label and keeps TRADES' original
            # inner-max (student clean vs perturbed) -- confirmed against
            # .external/adr/src/util/trades_attack.py. Only the outer
            # natural-CE term is rectified for adr_trades.
            "adr": "rectified",
            "adr_trades": "student_clean",
        }.get(self.id)
        if self.attack.loss != expected_loss:
            raise ValueError(f"{self.id} requires attack.loss={expected_loss}")
        if self.attack.kl_target != expected_target:
            raise ValueError(f"{self.id} requires attack.kl_target={expected_target!r}")
        selection = self.selection_attack
        if selection is None and self.selection_step_size_independent:
            raise ValueError("method.selection_step_size_independent requires an explicit method.selection_attack")
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
        if not self.selection_step_size_independent and not math.isclose(
            selection.step_size_value, self.attack.step_size_value, rel_tol=0, abs_tol=1e-15
        ):
            mismatched.append("step_size")
        if mismatched:
            raise ValueError(
                "checkpoint selection attack must match the training threat model: " + ", ".join(mismatched)
            )
        if self.selection_step_size_independent:
            # The exemption is only safe because selection (and so the
            # default evaluation attack) is pinned to the reference identity.
            pinned = AttackConfig(
                loss="ce",
                epsilon="4/255",
                step_size="8/765",
                steps=10,
                random_start=True,
                student_mode="eval",
                teacher_mode="eval",
            )
            if selection.identity() != pinned.identity():
                raise ValueError(
                    "method.selection_step_size_independent requires the reference selection attack exactly "
                    "(CE, Linf eps 4/255, step 8/765, 10 steps, random start, eval/eval)"
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
        adr_methods = {"adr", "adr_trades"}
        if self.id in adr_methods:
            if self.adr is None:
                raise ValueError(f"{self.id} requires an explicit adr configuration")
            if self.attack.temperature != 1.0:
                raise ValueError(
                    f"{self.id} attack must use temperature=1.0; the EMA-teacher temperature is annealed via "
                    "the adr configuration block, not attack.temperature, and the rectified target is already "
                    "a resolved distribution the attack must not re-scale"
                )
        elif self.adr is not None:
            raise ValueError(f"adr configuration is only defined for {sorted(adr_methods)}")
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

    def _resolve_standard(self) -> MethodConfig:
        """Validate ``standard``: clean CE training plus an explicit selection attack.

        There is no training threat model to derive the selection attack from,
        so it must be given explicitly, under the same rules as every other
        method (hard-label CE, eval modes). Fields no ``standard`` code path
        reads must keep their defaults; a non-default value would be silently
        ignored. ``label_smoothing`` is honored (the same CE as pgd_at).
        """
        if self.attack is not None:
            raise ValueError("method standard has no training attack")
        selection = self.selection_attack
        if selection is None:
            raise ValueError(
                "method standard requires an explicit method.selection_attack (there is no training attack "
                "to derive it from)"
            )
        if selection.loss != "ce":
            raise ValueError("checkpoint selection attack must use hard-label CE")
        if selection.student_mode != "eval" or selection.teacher_mode != "eval":
            raise ValueError("checkpoint selection attack must keep student and teacher in eval mode")
        ignored = [
            name
            for name in (
                "temperature",
                "temperature_squared",
                "trades_beta",
                "entropy_gamma",
                "student_ema_decay",
                "student_policy_warmup_epochs",
            )
            if getattr(self, name) != type(self).model_fields[name].default
        ]
        if ignored:
            raise ValueError("method standard does not use " + ", ".join(f"method.{name}" for name in ignored))
        if self.target_policy is not None:
            raise ValueError("target_policy is not defined for method standard")
        if self.adr is not None:
            raise ValueError("adr configuration is not defined for method standard")
        if self.oracle_mask:
            raise ValueError("oracle_mask is only defined for rslad_hard_fallback")
        if self.frozen_oracle_manifest is not None or self.frozen_oracle_manifest_sha256 is not None:
            raise ValueError("frozen_oracle_manifest fields are only defined for rslad_frozen_oracle_softening")
        return self


class InitCheckpointConfig(StrictModel):
    """Student initialization from one of this project's own saved checkpoints.

    Plan 0103's two-stage run: stage 2 starts from stage 1's final student
    weights. Only the ``model`` entry of a ``last.pt`` payload is loaded
    (``strict=True``) after the file's SHA-256 matches ``sha256``; optimizer,
    scheduler, scaler, RNG, sampler, sample state, epoch and best-selection
    state all start fresh. This is *not* resume, which continues one run's
    own trajectory (see ``ard.engine.checkpoint.load_init_student_weights``).
    The path is provenance, the digest is the identity; the file itself is
    read only by ``ard.cli.train`` (never at config validation or evaluation
    time, since the evaluation host need not hold it).
    """

    path: Path
    sha256: str

    @model_validator(mode="after")
    def validate_digest(self) -> InitCheckpointConfig:
        if len(self.sha256) != 64 or any(character not in "0123456789abcdef" for character in self.sha256):
            raise ValueError("training.init_checkpoint.sha256 must be a lowercase 64-character SHA-256 hex digest")
        return self


class TrainingConfig(StrictModel):
    epochs: int = Field(default=1, ge=1)
    checkpoint_epochs: tuple[int, ...] = (49, 99, 149, 199)
    per_rank_batch_size: int = Field(ge=1)
    global_batch_size: int = Field(ge=1)
    num_workers: int = Field(default=0, ge=0)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    # Default false reproduces today's exact behavior (scaler=None,
    # unconditionally, regardless of device) for every existing config that
    # never mentions this field -- see docs/plans/0099-imagenet-stage0-prep.md
    # checklist item 4 and docs/decisions/0011-post-cifar-adr-next-step.md.
    # Only meaningful on a CUDA device; enabling it on CPU constructs a
    # disabled (no-op) GradScaler rather than erroring, matching the
    # existing "auto" device-resolution pattern elsewhere in this config.
    amp: bool = False
    deterministic: bool = True
    # Throughput options (human-approved 2026-09-25). Both default false,
    # which reproduces today's exact behavior for every existing config that
    # never mentions them: cuDNN autotuning off (PyTorch's own default) and
    # the student run eagerly. cudnn_benchmark lets cuDNN time and pick
    # convolution algorithms per input shape; that choice is not
    # reproducible, so it requires deterministic=false. compile also
    # requires deterministic=false: bitwise reproducibility of inductor's
    # generated kernels is unproven here. compile wraps only
    # the student's training-time forward in torch.compile (default mode);
    # checkpoints, EMA copies and state_dict I/O always use the original
    # module, so saved keys are unchanged. Both are recorded in the
    # evaluation record's training_protocol_identity so arms that differ in
    # them are never pooled silently.
    cudnn_benchmark: bool = False
    compile: bool = False
    # Throughput option (human-approved 2026-09-26). True (default) keeps
    # today's two post-optimizer-step eval-mode/no-grad student forwards
    # (clean batch and the already-crafted adversarial batch) that exist only
    # for logged diagnostics: train_clean_accuracy,
    # train_robust_accuracy_eval_mode, train_robust_overtakes_clean and
    # ADR's train_ema_student_agreement. False skips them (~17% of step
    # compute); those metrics are then absent from the epoch rows. No
    # training decision reads them (gap-adaptive ADR reads the train-mode
    # train_robust_accuracy; online S2 reads boundary_* only), so the model,
    # optimizer, EMA, RNG streams and checkpoints are unchanged. Serialized
    # only when false, so every existing config keeps a byte-identical
    # resolved config and config hash. Deliberately NOT part of the
    # evaluation training_protocol_identity (training-neutral, like
    # train_probe_size), so True and False seeds of one arm pool.
    # pydantic>=2.12 is required for Field(exclude_if=...).
    step_diagnostics: bool = Field(default=True, exclude_if=lambda value: value is True)
    validation_fraction: float = Field(default=0.25, gt=0, lt=1)
    # This is a protocol identity, not a performance option. Ordinary DDP
    # computes BatchNorm statistics independently on each rank.
    batchnorm_mode: Literal["local_per_rank"] = "local_per_rank"
    # Plan 0101: linear, epoch-indexed ramp of the *training* attack's
    # epsilon from 0 up to the method's configured target over the first
    # ``epsilon_warmup_epochs`` epochs (Debenedetti, Sehwag, Mittal,
    # arXiv:2209.07399; see ard.schedules.epsilon_warmup for the exact
    # verified mechanism and this project's one deliberate deviation from
    # it). Default None reproduces today's exact behavior -- no warmup, the
    # full configured epsilon from epoch 0 -- for every existing config that
    # never mentions this field. Never applied to the selection or
    # evaluation attack, only the training attack (CLAUDE.md rule 6: the
    # evaluated threat model never changes).
    epsilon_warmup_epochs: int | None = Field(default=None, ge=0)
    # Plan 0102 Workstream A: plain weight-space EMA ("Adversarial Training
    # with Stochastic Weight Average", arXiv:2009.10526, surfaced by this
    # session's literature review, 2026-09-15), decayed every training
    # iteration exactly like ADR's own EMA (this reuses
    # ``Trainer._update_ema`` unchanged -- see there for the exact update
    # rule and why it is method-agnostic already). Independent of, and
    # mutually exclusive with, ``method.adr`` (adr already tracks its own
    # EMA as a distillation target; this field is for a plain pgd_at/trades
    # run that wants a weight-averaged shadow model purely for its own
    # sake, not as a training-time target). Default None reproduces today's
    # exact behavior -- no EMA model constructed -- for every existing
    # config that never mentions this field. 0.0 is a permitted but
    # degenerate value (the EMA becomes an exact copy of the live model
    # every step, so best-ema.pt is just a second, differently-selected
    # copy of the student, not a real average) -- not rejected, since it
    # is harmless rather than incorrect, but never a value to actually
    # launch with.
    weight_ema_decay: float | None = Field(default=None, ge=0, lt=1)
    # Observability only (plan 0103): after each epoch, measure clean and
    # selection-attack accuracy on this many fixed training-partition images
    # under the validation transform and eval mode -- identical conditions to
    # the validation slice, so seen-vs-unseen is a real generalization gap
    # and seen-set robust accuracy is the training error. Logged train_*
    # metrics cannot serve (augmented crops, train-mode BatchNorm, 3-step
    # attack). Never used for checkpoint selection. None (default) disables it.
    train_probe_size: int | None = Field(default=None, ge=1)
    # Plan 0103 Phase 2 (human decision 2026-10-08): lighter per-epoch
    # validation. When set, the per-epoch selection metric (so best.pt /
    # best-ema.pt and the catastrophic-overfitting check) is measured on a
    # FIXED, class-stratified subset of this many images of the held-out
    # validation split, seeded by seeds.split (ard.data.selection_subset_ids;
    # at least dataset.num_classes). Those numbers are named val_subset_* in
    # the epoch rows and best_subset_* / last_subset_* in the run summary,
    # never val_* / best_* / last_*, so they cannot be read as full-split
    # numbers. The train / held-out partition itself is unchanged
    # (validation_fraction keeps its meaning; the training data are
    # identical). Nothing that drives training reads the selection metric
    # (refused with ADR's gap-adaptive lambda, which does), so the trained
    # weights and last.pt are identical with or without it. At the final
    # epoch the last weights AND best.pt (and best-ema.pt) are evaluated on
    # the FULL held-out split and on its complement (held-out minus the
    # subset) in one pass each, with shared random starts: val_full_* /
    # best_val_full_* and val_complement_* / best_val_complement_*. It
    # changes which checkpoint is "best", so it is part of the config hash
    # and of the evaluation training_protocol_identity. Only ard.cli.train
    # implements it. None (default) is today's exact behavior; serialized
    # only when set, so every existing config keeps a byte-identical
    # resolved config and config hash.
    selection_subset_size: int | None = Field(default=None, ge=1, exclude_if=lambda value: value is None)
    # Plan 0103 two-stage run (stage 1). The square output size of the
    # *training-partition* views only: the RandomResizedCrop training crop
    # and the in-training validation / train-probe views (ImageNetEvalTransform
    # at this size: shorter side to round(size*256/224), centre crop), so
    # per-epoch selection is measured at the resolution the model trains at.
    # dataset.image_size and evaluation.dataset (the official evaluation)
    # are untouched. Resizing is PIL bilinear on PIL images, which is
    # anti-aliased on downscale. ImageNet only. Serialized only when set, so
    # every existing config keeps a byte-identical resolved config and
    # config hash.
    train_image_size: int | None = Field(default=None, ge=1, exclude_if=lambda value: value is None)
    # Plan 0103 two-stage run (stage 2); see InitCheckpointConfig. Serialized
    # only when set, like train_image_size.
    init_checkpoint: InitCheckpointConfig | None = Field(default=None, exclude_if=lambda value: value is None)
    # Plan 0103 loader speedup B (human-approved 2026-09-30). The ImageNet
    # RandomResizedCrop training view decodes each JPEG at the largest DCT
    # reduction (1/2, 1/4, 1/8, via PIL Image.draft) that keeps the sampled
    # crop >= the output size in both dimensions, so the final resize is
    # still an anti-aliased downscale. Crop boxes, flips and RNG consumption
    # are identical to the default path; the pixels differ slightly (libjpeg
    # DCT-domain downscaling), so it is part of the run identity (config
    # hash and training_protocol_identity). Validation / probe / evaluation
    # views are unchanged. ImageNet only. Serialized only when true.
    jpeg_draft_decode: bool = Field(default=False, exclude_if=lambda value: value is False)
    # Plan 0105 throughput option (human-approved 2026-09-30). Captures the
    # full PGD-AT training step (attack, forward, loss, backward, SGD update,
    # epoch accumulators) in one CUDA graph and replays it per full batch.
    # Equal to the eager step by contract -- bitwise in deterministic mode;
    # with deterministic=false exact RNG streams and a one-step result within
    # 4x max(eager spread, FP32 rounding) of the nearest eager outcome, per
    # tensor group -- so it is allowed only where that contract is tested
    # (cudnn_benchmark=true is not): one CUDA device, FP32,
    # step_diagnostics=false (tracking diagnostics are supported), eager
    # student, method pgd_at (optionally with method.mixed_batch without split
    # BN, or method.awp) or ImageNet rslad / rslad_advt distillation (bank or
    # online target; an in-step teacher must be allowlisted), a batch-keyed
    # random start and a fixed budget, SGD or (since 2026-10-09) AdamW built with capturable=True,
    # no policy treatment/intervention
    # (see ExperimentConfig._validate_cuda_graph); a plain weight EMA is admitted. The
    # first full batch of every epoch and the last partial batch run
    # eagerly; the graph is re-captured every epoch (the scheduler changes
    # the learning rate only at epoch ends) and a guard refuses to replay
    # if any optimizer hyperparameter or state tensor changed since
    # capture. Serialized only when true, so every existing config keeps a
    # byte-identical resolved config and config hash.
    cuda_graph: bool = Field(default=False, exclude_if=lambda value: value is False)

    @model_validator(mode="after")
    def validate_batch_identity(self) -> TrainingConfig:
        if self.global_batch_size < self.per_rank_batch_size:
            raise ValueError("global_batch_size must be at least per_rank_batch_size")
        ordered = tuple(sorted(set(self.checkpoint_epochs)))
        if any(epoch < 1 for epoch in self.checkpoint_epochs) or ordered != self.checkpoint_epochs:
            raise ValueError("checkpoint_epochs must be strictly increasing positive epoch numbers")
        if self.epsilon_warmup_epochs is not None and self.epsilon_warmup_epochs > self.epochs:
            raise ValueError("epsilon_warmup_epochs must not exceed the total number of training epochs")
        if self.cudnn_benchmark and self.deterministic:
            raise ValueError(
                "training.cudnn_benchmark=true requires training.deterministic=false: cuDNN autotuning "
                "selects convolution algorithms nondeterministically"
            )
        if self.compile and self.deterministic:
            raise ValueError(
                "training.compile=true requires training.deterministic=false: bitwise reproducibility of "
                "torch.compile/inductor kernels is not established"
            )
        if self.cuda_graph:
            # (condition that must hold, why) -- every entry fails closed.
            requirements = (
                (self.device == "cuda", "training.device=cuda (graphs exist only on CUDA; 'auto' is not enough)"),
                # deterministic=false is admitted (human decision 2026-10-03):
                # bitwise equal to eager in deterministic mode; otherwise every RNG
                # stream stays exact and one step lands within eager noise / FP32 rounding
                # (tests/integration/test_cuda_graph_training_step.py).
                (not self.amp, "training.amp=false (a GradScaler step syncs and is not captured)"),
                (not self.compile, "training.compile=false"),
                (
                    not self.cudnn_benchmark,
                    "training.cudnn_benchmark=false (under benchmark a captured step was seen to use a "
                    "different cuDNN algorithm than the eager step of the same process; plan 0105)",
                ),
                (not self.step_diagnostics, "training.step_diagnostics=false"),
                (
                    self.global_batch_size == self.per_rank_batch_size,
                    "global_batch_size == per_rank_batch_size (world size 1; DDP is not captured)",
                ),
                (self.epsilon_warmup_epochs is None, "training.epsilon_warmup_epochs unset (per-epoch attack budget)"),
                # training.weight_ema_decay is admitted (plan 0103 Phase 2 batch A,
                # 2026-10-08): the EMA update runs inside the captured step, right
                # after the SGD update, with the eager step's exact kernels (parity-tested).
            )
            missing = [reason for holds, reason in requirements if not holds]
            if missing:
                raise ValueError("training.cuda_graph=true requires " + "; ".join(missing))
        return self


# training.cuda_graph (plan 0105): student architectures whose captured step is
# shown equal to the eager step by
# tests/integration/test_cuda_graph_training_step.py (bitwise when
# deterministic, one step within eager noise / FP32 rounding when not; eval-mode attack,
# torch 2.11 on an RTX 4090) -- the SGD-trained BatchNorm CNNs this
# project uses. Anything else is refused until that test covers it; rerun it
# after a torch or driver upgrade.
CUDA_GRAPH_ARCHITECTURES: frozenset[str] = frozenset(
    {
        "mobilenetv4_conv_small_imagenet",
        "mobilenetv4_conv_medium_imagenet",
        "efficientnet_b0_imagenet",
        # 2026-10-09 (human-approved; LayerNorm / GELU / depthwise 7x7 / SDPA attention, SGD and AdamW):
        # the plan 0103 ConvNeXt-Atto and DeiT-Tiny students and their batch-B variants.
        "convnext_atto_imagenet",
        "convnext_atto_deep_narrow_imagenet",
        "convnext_atto_ols_imagenet",
        "convnext_atto_convstem_imagenet",
        "deit_tiny_imagenet",
        "deit_tiny_convstem_imagenet",
    }
)
# training.cuda_graph with a teacher that runs INSIDE the captured step (RSLAD with
# distillation.target_source=online_teacher, and rslad_advt's teacher forward on x'):
# frozen eval-mode ImageNet teacher architectures whose captured forward is
# parity-tested bitwise (tests/integration/test_cuda_graph_training_step.py, human
# decision 2026-10-08). RSLAD from a soft-label bank never runs the teacher.
CUDA_GRAPH_TEACHER_ARCHITECTURES: frozenset[str] = frozenset(
    {
        "resnet50_imagenet",
        "convnext_tiny_convstem_imagenet",
        "convnext_base_convstem_imagenet",
        "vit_s_convstem_imagenet",
        "mobilenetv4_conv_medium_imagenet",
    }
)
# Methods whose step the graph transcribes (plan 0105; RSLAD / RSLAD-advT since 2026-10-08).
CUDA_GRAPH_METHODS: frozenset[str] = frozenset({"pgd_at", "rslad", "rslad_advt"})


def reject_throughput_options(training: TrainingConfig, *, runtime: str) -> None:
    """Refuse a config whose throughput options a runtime does not implement.

    Only ``ard.cli.train`` applies ``training.compile``,
    ``training.cudnn_benchmark``, ``training.cuda_graph`` and
    ``training.step_diagnostics=false``.
    Every other Trainer builder would silently run eagerly without
    autotuning, with the diagnostic forwards, while the config says
    otherwise, so it must call this first.
    """
    # (field, value this runtime would ignore, value it must be run with)
    unsupported = [(name, "true", "false") for name in ("compile", "cudnn_benchmark") if getattr(training, name)]
    if not training.step_diagnostics:
        unsupported.append(("step_diagnostics", "false", "true"))
    # Plan 0103 two-stage fields: equally applied only by ard.cli.train.
    if training.train_image_size is not None:
        unsupported.append(("train_image_size", str(training.train_image_size), "null"))
    if training.init_checkpoint is not None:
        unsupported.append(("init_checkpoint", "<set>", "null"))
    if training.jpeg_draft_decode:
        unsupported.append(("jpeg_draft_decode", "true", "false"))
    if training.cuda_graph:
        unsupported.append(("cuda_graph", "true", "false"))
    if training.selection_subset_size is not None:
        unsupported.append(("selection_subset_size", str(training.selection_subset_size), "null"))
    if unsupported:
        requested = ", ".join(f"training.{name}={value}" for name, value, _ in unsupported)
        remediation = ", ".join(f"training.{name}={value}" for name, _, value in unsupported)
        raise ValueError(
            f"{runtime} does not implement {requested}; run it with {remediation} "
            "(only ard.cli.train applies these options)"
        )


def reject_phase2_batch_a_options(config: ExperimentConfig, *, runtime: str) -> None:
    """Refuse plan 0103 Phase 2 batch A options a Trainer builder does not implement.

    Only ``ard.cli.train`` wires ``method.mixed_batch``, ``method.awp`` and
    ``optimizer.exclude_norm_bias_from_weight_decay`` into the Trainer and its
    optimizer; every other builder would silently train without them.
    """
    requested = []
    if config.method.mixed_batch is not None:
        requested.append("method.mixed_batch")
    if config.method.awp is not None:
        requested.append("method.awp")
    if config.optimizer.exclude_norm_bias_from_weight_decay:
        requested.append("optimizer.exclude_norm_bias_from_weight_decay")
    if requested:
        raise ValueError(
            f"{runtime} does not implement {', '.join(requested)}; only ard.cli.train applies these options"
        )


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
        "none",
        "student_history",
        "class_matched_random",
        "online_history",
        "class_state_count_matched_random",
        "route_a_strong",
        "route_a_matched_random",
        "route_b_strong",
        "route_b_matched_random",
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
    # Plan 0100 (scientific review finding P1-4): AutoAttack materializes its
    # full evaluation set on the GPU as one tensor before attacking --
    # ~123 MB for CIFAR-10's 10k test images (never a problem), ~30 GB for
    # ImageNet's 50k val images (guaranteed CUDA OOM on any single GPU this
    # project has). None (default) preserves today's exact behavior --
    # attack every image the loader yields, unchanged for every existing
    # CIFAR config. When set, only a fixed-seed uniform-random subset of
    # this size (drawn from the full evaluation.dataset, keyed on
    # evaluation.seed) is attacked -- not a prefix of the dataset, since
    # ImageNetDataset's own sample ordering is grouped by class and a
    # prefix would concentrate on only the first few classes.
    autoattack_sample_count: int | None = Field(default=None, ge=1)

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
    # Plan 0103 Phase 2 batch D (see ard.config.distillation). Serialized only
    # when set, so every existing config keeps a byte-identical resolved
    # config and config hash.
    distillation: DistillationConfig | None = Field(default=None, exclude_if=lambda value: value is None)
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
        if (
            self.tier in {"repro", "pilot", "production"}
            and self.dataset.name == "imagenet"
            and not self.dataset.content_sha256
        ):
            raise ValueError("repro/pilot/production ImageNet requires dataset.content_sha256")
        if self.training.weight_ema_decay is not None and self.method.adr is not None:
            raise ValueError(
                "training.weight_ema_decay cannot be combined with method.adr: adr already tracks its own "
                "EMA as a distillation target; a second, independent weight EMA would need a second shadow model"
            )
        if self.training.train_image_size is not None:
            if self.dataset.name != "imagenet":
                raise ValueError("training.train_image_size is only defined for the imagenet dataset")
            if self.training.train_image_size == self.dataset.image_size:
                raise ValueError("training.train_image_size equals dataset.image_size; omit it")
        subset_size = self.training.selection_subset_size
        if subset_size is not None:
            if self.method.adr is not None and self.method.adr.lambda_source == "gap_adaptive":
                raise ValueError(
                    "training.selection_subset_size cannot be combined with method.adr.lambda_source=gap_adaptive: "
                    "the gap-adaptive lambda reads the per-epoch validation PGD accuracy, so the subset would "
                    "change the training trajectory, not only checkpoint selection"
                )
            if subset_size < self.dataset.num_classes:
                raise ValueError(
                    f"training.selection_subset_size ({subset_size}) must be at least dataset.num_classes "
                    f"({self.dataset.num_classes}) so the class-stratified subset holds every class"
                )
        if self.training.jpeg_draft_decode and self.dataset.name != "imagenet":
            raise ValueError("training.jpeg_draft_decode is only defined for the imagenet dataset")
        if self.training.init_checkpoint is not None and self.student.pretrained:
            raise ValueError(
                "training.init_checkpoint cannot be combined with student.pretrained=true: the student would "
                "have two initializations"
            )
        if (
            self.training.init_checkpoint is not None
            and self.protocol.id != "controlled_imagenet_stage02_two_stage_lowres_v1"
        ):
            raise ValueError(
                "training.init_checkpoint is defined only for controlled_imagenet_stage02_two_stage_lowres_v1"
            )
        if self.method.selection_step_size_independent and (
            self.protocol.id != "controlled_imagenet_stage02_two_stage_lowres_v1"
            or self.training.train_image_size is None
        ):
            raise ValueError(
                "method.selection_step_size_independent is defined only for the reduced-resolution stage 1 of "
                "controlled_imagenet_stage02_two_stage_lowres_v1"
            )
        if self.protocol.id == "controlled_imagenet_stage02_two_stage_lowres_v1" and (
            (self.training.train_image_size is None) == (self.training.init_checkpoint is None)
        ):
            raise ValueError(
                "controlled_imagenet_stage02_two_stage_lowres_v1 requires exactly one of "
                "training.train_image_size (stage 1) or training.init_checkpoint (stage 2)"
            )
        if self.student.num_classes != self.dataset.num_classes:
            raise ValueError("student and dataset num_classes must match")
        if self.teacher is not None and self.teacher.num_classes != self.dataset.num_classes:
            raise ValueError("teacher and dataset num_classes must match")
        rslad_methods = {
            "rslad",
            "rslad_advt",
            "rslad_entropy",
            "rslad_student",
            "rslad_joint",
            "rslad_joint_downweight",
            "rslad_hard_fallback",
            "rslad_frozen_oracle_softening",
        }
        if self.method.id in rslad_methods and self.teacher is None:
            raise ValueError(f"{self.method.id} requires a frozen teacher")
        if self.method.id == "standard":
            # Plan 0103 option A: nothing in standard training reads a
            # teacher or a training-attack budget, so either would be
            # silently ignored.
            if self.teacher is not None:
                raise ValueError("method standard trains without a teacher; remove teacher")
            if self.training.epsilon_warmup_epochs is not None:
                raise ValueError("method standard has no training attack; training.epsilon_warmup_epochs is undefined")
            # Mirrors the Trainer's attack-free refusals that a config can
            # express, so they fail here, before any tracker run exists.
            # (intervention/prescriptive_v3 already require method rslad;
            # target_policy/adr/oracle_mask/frozen oracle are refused on
            # MethodConfig.)
            if self.observation.profile != "off":
                raise ValueError(
                    "method standard records no adversarial per-sample state; observation.profile must be off"
                )
            if self.protocol.id != "controlled_imagenet_stage02_clean_budget_v1":
                raise ValueError("method standard is defined only under controlled_imagenet_stage02_clean_budget_v1")
        if self.protocol.id == "controlled_imagenet_stage02_clean_budget_v1" and self.method.id != "standard":
            raise ValueError("controlled_imagenet_stage02_clean_budget_v1 requires method standard")
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
            "imagenet": "imagenet_standard",
        }[self.dataset.name]
        if self.student.architecture == "saad_resnet18_cifar_v1" and self.dataset.name == "cifar10":
            expected_profile = "cifar10_raw_identity"
        # Plan 0103: timm's mobilevit_s.cvnets_in1k checkpoint was trained on
        # raw [0,1] pixels; pin it to that profile (and no other architecture).
        if self.student.architecture == "mobilevit_s_imagenet" and self.dataset.name == "imagenet":
            expected_profile = "imagenet_raw_identity"
        if self.student.normalization.profile != expected_profile:
            raise ValueError(f"dataset {self.dataset.name} requires student normalization profile {expected_profile}")
        validate_distillation_cross_fields(self)
        self._validate_phase2_batch_a()
        self._validate_cuda_graph()
        self._validate_protocol_contract()
        return self

    def _validate_phase2_batch_a(self) -> None:
        """Plan 0103 Phase 2 batch A: runtime limits of mixed batch, split BN and AWP (single process)."""
        mixed_batch, awp = self.method.mixed_batch, self.method.awp
        if mixed_batch is None and awp is None:
            return
        feature = "method.mixed_batch" if mixed_batch is not None else "method.awp"
        single_process = self.training.global_batch_size == self.training.per_rank_batch_size
        if not single_process:
            raise ValueError(
                f"{feature} requires world size 1 (global_batch_size == per_rank_batch_size); the mixed-batch "
                "normalizer, split BN and the AWP perturbation are tested only in a single process"
            )
        # Mirrors the Trainer's scope check so a config fails before any tracker run exists.
        if self.observation.profile != "off":
            raise ValueError(f"{feature} requires observation.profile=off")
        if self.teacher is not None or self.distillation is not None:
            raise ValueError(f"{feature} is defined for plain pgd_at without a teacher or distillation block")
        if mixed_batch is not None:
            per_rank = self.training.per_rank_batch_size
            adversarial = math.floor(mixed_batch.adversarial_fraction * per_rank)
            if mixed_batch.split_batchnorm:
                if adversarial < 2 or per_rank - adversarial < 2:
                    raise ValueError(
                        "method.mixed_batch.split_batchnorm needs at least 2 adversarial and 2 clean examples per "
                        f"batch (BatchNorm on a 1-example sub-batch, e.g. a head BN after pooling, is undefined); "
                        f"per_rank_batch_size {per_rank} with adversarial_fraction "
                        f"{mixed_batch.adversarial_fraction} gives {adversarial} / {per_rank - adversarial}"
                    )
                if self.training.compile:
                    raise ValueError("method.mixed_batch.split_batchnorm cannot be combined with training.compile")
                if self.training.init_checkpoint is not None:
                    raise ValueError(
                        "method.mixed_batch.split_batchnorm cannot be combined with training.init_checkpoint (the "
                        "auxiliary BN state is not carried between stages)"
                    )
        if awp is not None:
            if self.training.amp:
                raise ValueError("method.awp cannot be combined with training.amp")
            if awp.warmup_epochs >= self.training.epochs:
                raise ValueError("method.awp.warmup_epochs must be smaller than training.epochs (AWP never active)")

    def _validate_cuda_graph(self) -> None:
        """``training.cuda_graph`` scope: only the step whose eager parity is proven (plan 0105)."""
        if not self.training.cuda_graph:
            return
        attack = self.method.attack
        method_id = self.method.id
        distillation = method_id in {"rslad", "rslad_advt"}
        # The teacher runs inside the captured step for an online target or rslad_advt's forward on x'.
        teacher_in_step = distillation and (
            method_id == "rslad_advt"
            or (self.distillation is not None and self.distillation.target_source == "online_teacher")
        )
        mixed_batch = self.method.mixed_batch
        requirements = (
            (
                self.student.architecture in CUDA_GRAPH_ARCHITECTURES,
                "a parity-tested student.architecture (" + ", ".join(sorted(CUDA_GRAPH_ARCHITECTURES)) + ")",
            ),
            (
                attack is not None and attack.student_mode == "eval",
                "method.attack.student_mode=eval (train-mode attacks are not parity-tested)",
            ),
            (method_id in CUDA_GRAPH_METHODS, "method.id in " + ", ".join(sorted(CUDA_GRAPH_METHODS))),
            (method_id != "pgd_at" or self.teacher is None, "no teacher (pgd_at)"),
            (
                not distillation or (self.teacher is not None and self.distillation is not None),
                "rslad/rslad_advt with a teacher and a distillation block (ImageNet distillation)",
            ),
            # Admitted scope == parity-tested scope (review of d2e82b2, P2-1): advT only from a bank, and
            # distillation at temperature 1 only (the tested value; the bank already requires it).
            (
                method_id != "rslad_advt"
                or (self.distillation is not None and self.distillation.target_source == "soft_label_bank"),
                "rslad_advt with distillation.target_source=soft_label_bank (online advT is not parity-tested)",
            ),
            (
                not distillation
                or (self.method.temperature == 1.0 and attack is not None and attack.temperature == 1.0),
                "method.temperature=1 and method.attack.temperature=1 for rslad/rslad_advt (the parity-tested value)",
            ),
            (
                not teacher_in_step
                or (self.teacher is not None and self.teacher.architecture in CUDA_GRAPH_TEACHER_ARCHITECTURES),
                "a parity-tested teacher.architecture when the teacher runs inside the step ("
                + ", ".join(sorted(CUDA_GRAPH_TEACHER_ARCHITECTURES))
                + ")",
            ),
            (self.method.adr is None, "no method.adr"),
            (self.method.target_policy is None, "no method.target_policy"),
            (not self.method.oracle_mask, "method.oracle_mask=false"),
            (self.intervention is None, "no intervention"),
            (self.prescriptive_v3 is None, "no prescriptive_v3"),
            (self.observation.profile == "off", "observation.profile=off"),
            (
                mixed_batch is None or not mixed_batch.split_batchnorm,
                "no method.mixed_batch.split_batchnorm (not parity-tested)",
            ),
            (
                self.method.awp is None or self.training.deterministic,
                "method.awp only with training.deterministic=true (the nondeterministic one-step check could not "
                "resolve an AWP step: its eager outcomes were too spread; plan 0105)",
            ),
            # AdamW since 2026-10-09 (human-approved): ard.cli.train then builds it with capturable=True
            # (device step counter, device bias corrections), for the eager and the captured steps alike.
            (self.optimizer.id in {"sgd", "adamw"}, "optimizer.id=sgd or adamw"),
            (
                attack is not None
                and (
                    attack.loss == "ce"
                    if method_id == "pgd_at"
                    else attack.loss == "kl" and attack.kl_target == "teacher_clean"
                ),
                "a CE training attack (pgd_at) or a KL-to-teacher-clean training attack (rslad/rslad_advt)",
            ),
            (
                attack is not None and attack.random_start_keying == "batch",
                "method.attack.random_start_keying=batch (sample-keyed starts are drawn on the host)",
            ),
            (attack is not None and not attack.trace_step_losses, "method.attack.trace_step_losses=false"),
            (
                attack is not None
                and attack.epsilon_value is not None
                and attack.step_size_value is not None
                and (not attack.random_start or attack.epsilon_value > 0)
                and attack.step_size_value <= attack.epsilon_value,
                "a fixed training budget with 0 < epsilon (under a random start) and step_size <= epsilon",
            ),
        )
        missing = [reason for holds, reason in requirements if not holds]
        if missing:
            raise ValueError("training.cuda_graph=true requires " + "; ".join(missing))

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
            "controlled_cifar10_r18_adr_v1",
            "controlled_cifar10_mobilenetv2_adr_v1",
            "controlled_cifar10_r18_trades_49k_validation_v1",
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
        if self.method.attack is None:
            raise ValueError(f"{self.protocol.id} contract defines no method without a training attack")
        attack_family = self.method.id if self.method.id in {"pgd_at", "trades", "adr", "adr_trades"} else "rslad"
        train_attacks = metadata["train_attacks"]
        assert isinstance(train_attacks, Mapping)
        if attack_family not in train_attacks:
            raise ValueError(
                f"{self.protocol.id} contract has no train_attacks entry for method family {attack_family!r}"
            )
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
