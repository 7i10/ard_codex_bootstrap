"""Single training loop composed from attack and unreduced outer objective interfaces."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ard.attacks import AttackGenerator, AttackRequest
from ard.data import IndexedBatch
from ard.objectives import DistillationObjective, ObjectiveTerms
from ard.policies import (
    FixedInterventionMask,
    PolicyContext,
    PolicyWeights,
    WeightPolicy,
    student_risk_from_margin,
    teacher_risk_from_entropy,
)
from ard.signals import (
    RobustMarginSignal,
    TeacherConfidenceBatch,
    shannon_entropy,
    teacher_confidence_primitives,
)
from ard.state import SampleStateStore
from ard.targets import AnchoredTeacherTargetPolicy, TeacherTargetPolicy
from ard.tracking.diagnostics import TrainingDiagnostics

from .checkpoint import TrainingState, load_checkpoint, save_checkpoint
from .distributed import (
    gather_objects,
    get_rank,
    get_world_size,
    reduce_max,
    reduce_min,
    reduce_sums,
    suspend_ddp_buffer_broadcasts,
)

if TYPE_CHECKING:
    from ard.analysis.frozen_oracle import FrozenRiskLookup


@contextmanager
def _evaluation_mode(model: nn.Module) -> Iterator[None]:
    mode = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(mode)


def _jensen_shannon_response(clean_logits: torch.Tensor, adversarial_logits: torch.Tensor) -> torch.Tensor:
    """Return stable per-sample JS divergence for detached FP32 teacher logits."""
    if clean_logits.ndim != 2 or clean_logits.shape != adversarial_logits.shape:
        raise ValueError("teacher JS response requires aligned [batch, class] logits")
    clean_log_probabilities = F.log_softmax(clean_logits.detach().float(), dim=1)
    adversarial_log_probabilities = F.log_softmax(adversarial_logits.detach().float(), dim=1)
    mixture_log_probabilities = torch.logaddexp(clean_log_probabilities, adversarial_log_probabilities) - math.log(2.0)
    response = 0.5 * (
        F.kl_div(
            mixture_log_probabilities,
            clean_log_probabilities,
            reduction="none",
            log_target=True,
        ).sum(dim=1)
        + F.kl_div(
            mixture_log_probabilities,
            adversarial_log_probabilities,
            reduction="none",
            log_target=True,
        ).sum(dim=1)
    )
    response = response.detach().float()
    if not bool(torch.isfinite(response).all()):
        raise FloatingPointError("teacher JS response is non-finite")
    # JS is non-negative analytically, but the two FP32 KL terms can cancel to
    # a small negative value for nearly identical distributions.
    if bool((response < -1e-6).any()):
        raise FloatingPointError("teacher JS response is below the FP32 rounding floor")
    return response.clamp_min(0.0)


def _reduce_epoch_observability(
    local_totals: torch.Tensor,
    *,
    local_seconds: float,
    local_cuda_peak_allocated_bytes: int,
    local_cuda_peak_reserved_bytes: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply the epoch SUM/MAX contract and derive globally valid throughput."""
    if local_totals.shape != (6,):
        raise ValueError("epoch totals must contain six scalar accumulators")
    global_totals = reduce_sums(local_totals)
    rank_max = reduce_max(
        torch.tensor(
            [local_seconds, float(local_cuda_peak_allocated_bytes), float(local_cuda_peak_reserved_bytes)],
            dtype=torch.float64,
            device=local_totals.device,
        )
    )
    valid_examples = float(global_totals[3].item())
    seconds = float(rank_max[0].item())
    return global_totals, {
        "valid_examples": valid_examples,
        "seconds": seconds,
        "images_per_second": valid_examples / seconds if seconds > 0 else 0.0,
        "cuda_peak_allocated_bytes": float(rank_max[1].item()),
        "cuda_peak_reserved_bytes": float(rank_max[2].item()),
        "teacher_clean_forward_calls": float(global_totals[4].item()),
        "teacher_adversarial_forward_calls": float(global_totals[5].item()),
    }


class Trainer:
    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Any,
        scaler: Any,
        attack: AttackGenerator,
        selection_attack: AttackGenerator,
        objective: DistillationObjective,
        device: torch.device,
        output_dir: Path,
        config_hash: str,
        seed: int,
        evaluation_attack_seed: int | None = None,
        tracker_run_id: str | None = None,
        fork_lineage: Mapping[str, Any] | None = None,
        teacher: nn.Module | None = None,
        policy: WeightPolicy | None = None,
        sample_store: SampleStateStore | None = None,
        target_policy: TeacherTargetPolicy | None = None,
        intervention_mask: FixedInterventionMask | None = None,
        anchor_model: nn.Module | None = None,
        prescriptive_v3_route: str | None = None,
        adversarial_kd_multiplier: float | None = None,
        adversarial_ce_coefficient: float | None = None,
        clean_ce_coefficient: float | None = None,
        clean_wrong_mode: str | None = None,
        clean_wrong_attack_skip: bool = False,
        selected_attack_epsilon: float | None = None,
        selected_attack_step_size: float | None = None,
        extra_clean_ce_coefficient: float | None = None,
        adversarial_bce_coefficient: float | None = None,
        adaptive_advkd_gamma: float | None = None,
        margin_coefficient: float | None = None,
        margin_target_mode: str | None = None,
        margin_gamma: float | None = None,
        margin_floor: float | None = None,
        margin_cap: float | None = None,
        teacher_clean_reliability_mask: FixedInterventionMask | None = None,
        iad_inspired: bool = False,
        dynamic_s3_router: Any | None = None,
        policy_warmup_epochs: int = 0,
        oracle_mask: bool = False,
        frozen_risk_lookup: FrozenRiskLookup | None = None,
        diagnostics: TrainingDiagnostics | None = None,
        observation_profile: str = "off",
        checkpoint_epochs: tuple[int, ...] = (),
    ) -> None:
        self.model = model.to(device)
        self.teacher = None if teacher is None else teacher.to(device)
        if self.teacher is not None:
            for parameter in self.teacher.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
        self.optimizer, self.scheduler, self.scaler = optimizer, scheduler, scaler
        self.attack, self.selection_attack, self.objective = attack, selection_attack, objective
        self.device, self.output_dir, self.config_hash, self.seed = device, output_dir, config_hash, seed
        self.evaluation_attack_seed = seed if evaluation_attack_seed is None else evaluation_attack_seed
        self.tracker_run_id = tracker_run_id
        self.checkpoint_epochs = tuple(checkpoint_epochs)
        self.fork_lineage = None if fork_lineage is None else dict(fork_lineage)
        self.policy = policy
        self.sample_store = sample_store
        if observation_profile not in {"off", "student_history", "teacher_response"}:
            raise ValueError("observation_profile must be off, student_history, or teacher_response")
        self.observation_profile = observation_profile
        self._records_student_history = observation_profile in {"student_history", "teacher_response"}
        self._records_teacher_response = observation_profile == "teacher_response"
        if self._records_student_history and self.sample_store is None:
            raise ValueError("student-history observation requires sample state")
        if self._records_teacher_response and self.teacher is None:
            raise ValueError("teacher-response observation requires a frozen teacher")
        self.target_policy = target_policy
        has_treatment = any(
            value is not None
            for value in (
                target_policy,
                adversarial_kd_multiplier,
                adversarial_ce_coefficient,
                clean_wrong_mode,
                selected_attack_epsilon,
                selected_attack_step_size,
                extra_clean_ce_coefficient,
                adversarial_bce_coefficient,
                adaptive_advkd_gamma,
                margin_coefficient,
                margin_target_mode,
                margin_gamma,
                margin_floor,
                margin_cap,
                teacher_clean_reliability_mask,
                iad_inspired,
            )
        )
        # A teacher target policy can be selected by an ordinary WeightPolicy
        # (for example rslad_student) and is not itself a fixed-ID treatment.
        # Only these explicit per-ID outer-loss branches need a static mask or
        # the same-step dynamic router.
        requires_explicit_mask = any(
            value is not None
            for value in (
                adversarial_kd_multiplier,
                adversarial_ce_coefficient,
                clean_wrong_mode,
                margin_coefficient,
            )
        )
        if intervention_mask is not None and prescriptive_v3_route != "nr_prefix" and not has_treatment:
            raise ValueError("an intervention mask requires at least one registered treatment")
        if intervention_mask is None and requires_explicit_mask and dynamic_s3_router is None:
            raise ValueError("a registered treatment requires a fixed intervention mask")
        if dynamic_s3_router is not None and intervention_mask is not None:
            raise ValueError("dynamic S3 routing cannot be combined with a fixed intervention mask")
        if adversarial_kd_multiplier is not None and (
            not torch.isfinite(torch.as_tensor(adversarial_kd_multiplier)) or adversarial_kd_multiplier < 0
        ):
            raise ValueError("adversarial KD intervention multiplier must be finite and non-negative")
        if adversarial_ce_coefficient is not None and not torch.isfinite(torch.as_tensor(adversarial_ce_coefficient)):
            raise ValueError("adversarial CE intervention coefficient must be finite")
        if adversarial_ce_coefficient is not None and adversarial_ce_coefficient < 0:
            raise ValueError("adversarial CE intervention coefficient must be non-negative")
        if clean_ce_coefficient is not None and (
            not torch.isfinite(torch.as_tensor(clean_ce_coefficient)) or clean_ce_coefficient < 0
        ):
            raise ValueError("clean CE coefficient must be finite and non-negative")
        if clean_wrong_mode not in {None, "clean_ce_only", "teacher_clean_gate", "clean_kd"}:
            raise ValueError("unknown clean-wrong treatment mode")
        if clean_wrong_attack_skip and clean_wrong_mode is None:
            raise ValueError("clean-wrong attack skipping requires a clean-wrong treatment mode")
        for name, value in (
            ("selected attack epsilon", selected_attack_epsilon),
            ("selected attack step size", selected_attack_step_size),
            ("extra clean CE coefficient", extra_clean_ce_coefficient),
            ("adversarial BCE coefficient", adversarial_bce_coefficient),
            ("adaptive AdvKD gamma", adaptive_advkd_gamma),
            ("margin coefficient", margin_coefficient),
            ("margin gamma", margin_gamma),
            ("margin floor", margin_floor),
            ("margin cap", margin_cap),
        ):
            if value is not None and (not torch.isfinite(torch.as_tensor(value)) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        if (selected_attack_epsilon is None) != (selected_attack_step_size is None):
            raise ValueError("selected attack epsilon and step size must be supplied together")
        if teacher_clean_reliability_mask is not None and intervention_mask is None:
            raise ValueError("teacher reliability mask requires a selected treatment mask")
        valid_margin_modes = {None, "fixed", "teacher_zero", "teacher_floor", "teacher_abstain"}
        if margin_target_mode not in valid_margin_modes:
            raise ValueError("unknown adversarial margin target mode")
        if margin_target_mode is not None and margin_coefficient is None:
            raise ValueError("margin target mode requires a margin coefficient")
        if margin_coefficient is not None and margin_target_mode is None:
            raise ValueError("margin coefficient requires a margin target mode")
        if margin_target_mode == "fixed" and margin_gamma is None:
            raise ValueError("fixed margin target requires gamma")
        if margin_target_mode in {"teacher_zero", "teacher_floor", "teacher_abstain"} and margin_cap is None:
            raise ValueError("Teacher margin target requires a finite cap")
        if margin_floor is not None and margin_cap is not None and margin_floor > margin_cap:
            raise ValueError("margin floor cannot exceed margin cap")
        self.intervention_mask = intervention_mask
        if prescriptive_v3_route not in {None, "pf_retention", "nr_prefix"}:
            raise ValueError("prescriptive route must be pf_retention or nr_prefix")
        if (anchor_model is not None) != (prescriptive_v3_route == "pf_retention"):
            raise ValueError("only PF retention requires one frozen anchor model")
        if prescriptive_v3_route is not None and intervention_mask is None:
            raise ValueError("prescriptive treatment requires a fixed intervention mask")
        if prescriptive_v3_route == "pf_retention" and not isinstance(target_policy, AnchoredTeacherTargetPolicy):
            raise ValueError("PF retention requires the anchored teacher target")
        self.prescriptive_v3_route = prescriptive_v3_route
        self.anchor_model = None if anchor_model is None else anchor_model.to(device)
        if self.anchor_model is not None:
            for parameter in self.anchor_model.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
            # A retention anchor is a fixed epoch-79 reference, not a second
            # train-mode network.  ``_anchor_logits`` reasserts eval mode for
            # the forward, while this persistent mode prevents accidental BN
            # state updates should a future call site bypass that helper.
            self.anchor_model.eval()
        self.adversarial_kd_multiplier = adversarial_kd_multiplier
        self.adversarial_ce_coefficient = adversarial_ce_coefficient
        self.clean_ce_coefficient = clean_ce_coefficient
        self.clean_wrong_mode = clean_wrong_mode
        self.clean_wrong_attack_skip = clean_wrong_attack_skip
        self.selected_attack_epsilon = selected_attack_epsilon
        self.selected_attack_step_size = selected_attack_step_size
        self.extra_clean_ce_coefficient = extra_clean_ce_coefficient
        self.adversarial_bce_coefficient = adversarial_bce_coefficient
        self.adaptive_advkd_gamma = adaptive_advkd_gamma
        self.margin_coefficient = margin_coefficient
        self.margin_target_mode = margin_target_mode
        self.margin_gamma = margin_gamma
        self.margin_floor = margin_floor
        self.margin_cap = margin_cap
        self.teacher_clean_reliability_mask = teacher_clean_reliability_mask
        self.iad_inspired = iad_inspired
        self.dynamic_s3_router = dynamic_s3_router
        if target_policy is not None and self.teacher is None:
            raise ValueError("teacher target policy requires a frozen teacher")
        if policy_warmup_epochs < 0:
            raise ValueError("policy_warmup_epochs must be non-negative")
        if oracle_mask and sample_store is None:
            raise ValueError("oracle_mask requires student-aware sample state")
        if frozen_risk_lookup is not None and target_policy is None:
            raise ValueError("frozen oracle risk requires an explicit teacher target policy")
        self.policy_warmup_epochs, self.oracle_mask = policy_warmup_epochs, oracle_mask
        self.frozen_risk_lookup = frozen_risk_lookup
        self.current_epoch = 0
        self._robust_margin_signal = RobustMarginSignal()
        self.global_step = 0
        self.best_metric = float("-inf")
        self.selection_metadata: dict[str, Any] = {
            "metric": "val_pgd_accuracy",
            "attack": self._attack_metadata(self.selection_attack),
            "tie_break": "earliest_epoch",
            "seed_protocol": "seed+1000003*global_step+10007*rank+590017; one advancing generator per pass",
            "selected_epoch": None,
        }
        if prescriptive_v3_route is not None:
            self.selection_metadata["prescriptive_v3"] = {
                "route": prescriptive_v3_route,
                "selected_ids_sha256": intervention_mask.selected_ids_digest if intervention_mask is not None else None,
                "active_epochs": [80, 129] if prescriptive_v3_route == "pf_retention" else [80, 99],
                "selected_attack_input": "full_step10" if prescriptive_v3_route == "pf_retention" else "step5_prefix",
                "unselected_attack_input": "full_step10",
            }
        self.sample_state: dict[str, Any] = {} if sample_store is None else sample_store.state_dict()
        self.diagnostics = diagnostics
        # This detached cache is strictly intra-batch diagnostic reuse.  It is
        # cleared at every batch boundary and intentionally excluded from
        # checkpoint state.
        self._teacher_adversarial_logits: torch.Tensor | None = None
        self._teacher_adversarial_forward_calls = 0.0

    def _attack_generator(self) -> torch.Generator:
        seed = self.seed + 1_000_003 * self.global_step + 10_007 * get_rank()
        return torch.Generator(device=self.device).manual_seed(seed)

    def _selection_generator(self) -> torch.Generator:
        seed = self.evaluation_attack_seed + 1_000_003 * self.global_step + 10_007 * get_rank() + 590_017
        return torch.Generator(device=self.device).manual_seed(seed)

    @staticmethod
    def _attack_metadata(attack: AttackGenerator) -> dict[str, Any]:
        config = getattr(attack, "config", None)
        if config is None:
            return {"name": type(attack).__name__}
        return {
            "name": type(attack).__name__,
            "identity": config.identity(),
            "identity_sha256": config.identity_sha256(),
        }

    @staticmethod
    def _mask(batch: IndexedBatch) -> torch.Tensor:
        if batch.state_update_mask is None:
            return torch.ones(batch.labels.shape[0], device=batch.labels.device, dtype=torch.float32)
        return batch.state_update_mask.to(dtype=torch.float32)

    def _prescriptive_active(self) -> bool:
        return (self.prescriptive_v3_route == "pf_retention" and 80 <= self.current_epoch <= 129) or (
            self.prescriptive_v3_route == "nr_prefix" and 80 <= self.current_epoch <= 99
        )

    def _anchor_logits(self, images: torch.Tensor) -> torch.Tensor:
        if self.anchor_model is None:
            raise RuntimeError("anchored target requested without a frozen epoch-79 model")
        with (
            _evaluation_mode(self.anchor_model),
            torch.no_grad(),
            torch.autocast(device_type=self.device.type, enabled=False),
        ):
            logits = self.anchor_model(images.float()).detach().float()
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("frozen anchor logits are non-finite")
        return logits

    def _flush_sample_store(self) -> None:
        """Replicate valid sparse observations before a checkpoint is written."""
        if self.sample_store is None:
            return
        # Each rank contributes exactly its local queue.  ``merge_pending``
        # canonicalizes by stable original ID, so local records are not also
        # applied separately and a valid distributed duplicate cannot double
        # increment EMA/correctness/forgetting counters.
        pending_by_rank = gather_objects(self.sample_store.pending_state())
        self.sample_store.merge_pending(pending_by_rank)
        self.sample_state = self.sample_store.state_dict()

    def _teacher_adversarial_response(self, adversarial: torch.Tensor) -> torch.Tensor:
        """Return the one detached FP32 teacher-adv forward for this batch.

        Observation, entropy/joint policies, and qualitative diagnostics share
        this cache.  It has no graph and is cleared at the batch boundary.
        """
        if self._teacher_adversarial_logits is not None:
            return self._teacher_adversarial_logits
        if self.teacher is None:
            raise ValueError("teacher adversarial response requires a frozen teacher")
        with (
            _evaluation_mode(self.teacher),
            torch.no_grad(),
            torch.autocast(device_type=self.device.type, enabled=False),
        ):
            self._teacher_adversarial_logits = self.teacher(adversarial.float()).detach().float()
        self._teacher_adversarial_forward_calls += 1.0
        return self._teacher_adversarial_logits

    @staticmethod
    def _probability_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return ``p(y) - max_{c != y} p(c)`` in detached-safe FP32 math."""
        if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
            raise ValueError("probability margin expects [batch, classes] logits and [batch] labels")
        if labels.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8):
            raise ValueError("probability-margin labels must be integer class indices")
        if bool((labels < 0).any()) or bool((labels >= logits.shape[1]).any()):
            raise ValueError("probability-margin labels are outside the class range")
        probabilities = F.softmax(logits.float(), dim=1)
        true_probability = probabilities.gather(1, labels[:, None]).squeeze(1)
        wrong_probability = probabilities.scatter(1, labels[:, None], 0.0).amax(dim=1)
        margin = true_probability - wrong_probability
        if not bool(torch.isfinite(margin).all()):
            raise FloatingPointError("probability margin is non-finite")
        return margin

    def _adversarial_margin_target(
        self,
        *,
        teacher_adversarial_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build a frozen Teacher margin target and its active multiplier."""
        if self.margin_target_mode is None or self.margin_coefficient is None:
            raise RuntimeError("adversarial margin target requested without a configured margin treatment")
        teacher_margin = self._probability_margin(teacher_adversarial_logits, labels).detach()
        mode = self.margin_target_mode
        if mode == "fixed":
            assert self.margin_gamma is not None
            target = torch.full_like(teacher_margin, float(self.margin_gamma))
            active = torch.ones_like(teacher_margin)
        else:
            assert self.margin_cap is not None
            clipped = teacher_margin.clamp(min=0.0, max=float(self.margin_cap))
            if mode == "teacher_floor":
                if self.margin_floor is None:
                    raise RuntimeError("Teacher floor margin target requires a floor")
                target = teacher_margin.clamp(min=float(self.margin_floor), max=float(self.margin_cap))
                active = torch.ones_like(teacher_margin)
            elif mode == "teacher_abstain":
                target = clipped
                active = (teacher_margin > 0).to(dtype=teacher_margin.dtype)
            else:  # teacher_zero
                target = clipped
                active = torch.ones_like(teacher_margin)
        return target.detach(), active.detach()

    def _student_aware_signals(
        self,
        *,
        batch: IndexedBatch,
        logits: torch.Tensor,
        valid_mask: torch.Tensor,
        teacher_clean: TeacherConfidenceBatch | None = None,
        teacher_adversarial: TeacherConfidenceBatch | None = None,
        teacher_clean_to_adversarial_margin_response: torch.Tensor | None = None,
        teacher_clean_to_adversarial_js_response: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.sample_store is None:
            return {}
        margin = self._robust_margin_signal.compute(
            student_adv_logits=logits,
            labels=batch.labels,
            valid_mask=valid_mask,
        )
        robust_correct = logits.detach().argmax(dim=1).eq(batch.labels)
        self.sample_store.record_pending(
            sample_ids=batch.sample_ids,
            margins=margin.values,
            robust_correct=robust_correct,
            valid_mask=margin.valid_mask,
            update=self.global_step,
            epoch=self.current_epoch,
            rank=get_rank(),
            labels=batch.labels if self._records_teacher_response else None,
            teacher_clean=teacher_clean,
            teacher_adversarial=teacher_adversarial,
            teacher_clean_to_adversarial_margin_response=teacher_clean_to_adversarial_margin_response,
            teacher_clean_to_adversarial_js_response=teacher_clean_to_adversarial_js_response,
        )
        student_risk = student_risk_from_margin(self.sample_store.margin_ema(batch.sample_ids))
        return {"student_risk": student_risk}

    def _policy_weights(
        self,
        *,
        batch: IndexedBatch,
        adversarial: torch.Tensor,
        logits: torch.Tensor,
        valid_mask: torch.Tensor,
        student_signals: dict[str, torch.Tensor],
    ) -> PolicyWeights | None:
        frozen_risk_lookup = getattr(self, "frozen_risk_lookup", None)
        if frozen_risk_lookup is not None:
            valid = valid_mask.to(device=logits.device, dtype=logits.dtype)
            risk = frozen_risk_lookup.values(batch.sample_ids, device=logits.device, dtype=logits.dtype) * valid
            return PolicyWeights(
                hard_weight=torch.zeros_like(risk),
                kd_weight=valid,
                joint_risk=risk,
            )
        if self.policy is None:
            return None
        # Epoch zero is exactly baseline RSLAD while detached margin
        # observations are collected for the next epoch.  Missing EMA state
        # must not introduce a hard-label fallback.
        if self.sample_store is not None and self.current_epoch < self.policy_warmup_epochs:
            kd = valid_mask.to(device=logits.device, dtype=logits.dtype)
            zero = torch.zeros_like(kd)
            return PolicyWeights(hard_weight=zero, kd_weight=kd, joint_risk=zero)
        signals: dict[str, torch.Tensor] = {}
        required = self.policy.required_signals
        entropy: torch.Tensor | None = None
        if "teacher_entropy" in required or "joint_risk" in required:
            if self.teacher is None:
                raise ValueError("selected policy requires a teacher")
            entropy = shannon_entropy(self._teacher_adversarial_response(adversarial))
        if "teacher_entropy" in required:
            assert entropy is not None
            signals["teacher_entropy"] = entropy
        if "student_risk" in required:
            signals["student_risk"] = student_signals["student_risk"]
        if "joint_risk" in required:
            assert entropy is not None
            teacher_risk = teacher_risk_from_entropy(entropy, num_classes=logits.shape[1])
            signals["joint_risk"] = student_signals["student_risk"] * teacher_risk
        weights = self.policy.compute(
            signals,
            context=PolicyContext(valid_mask=valid_mask, global_min=reduce_min),
            num_classes=logits.shape[1],
        )
        if self.oracle_mask:
            # Deliberately scientific-only: current adversarial correctness is
            # an oracle for whether a hard-label fallback may be active.  It
            # is never available in smoke/repro/production/evaluation configs.
            oracle_risk = logits.detach().argmax(dim=1).ne(batch.labels).to(dtype=weights.kd_weight.dtype)
            joint_risk = weights.joint_risk
            assert joint_risk is not None
            risk = joint_risk * oracle_risk * valid_mask.to(dtype=weights.kd_weight.dtype)
            weights = PolicyWeights(
                hard_weight=risk,
                kd_weight=(1.0 - risk) * valid_mask.to(dtype=weights.kd_weight.dtype),
                joint_risk=risk,
            )
        joint_risk = weights.joint_risk
        assert joint_risk is not None
        return PolicyWeights(
            hard_weight=weights.hard_weight.to(device=logits.device, dtype=logits.dtype),
            kd_weight=weights.kd_weight.to(device=logits.device, dtype=logits.dtype),
            joint_risk=joint_risk.to(device=logits.device, dtype=logits.dtype),
        )

    def train_epoch(self, loader: DataLoader[IndexedBatch]) -> dict[str, float]:
        self.model.train()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        started_at = time.perf_counter()
        # Loss sum, clean-correct, robust-correct, valid examples, and actual
        # detached clean/adv teacher forwards.  One final SUM makes the
        # count telemetry global without adding a hot-loop collective.
        totals = torch.zeros(6, dtype=torch.float64, device=self.device)
        for batch in loader:
            if not isinstance(batch, IndexedBatch):
                raise TypeError("trainer requires IndexedBatch batches")
            batch = batch.to(self.device)
            self._teacher_adversarial_logits = None
            self._teacher_adversarial_forward_calls = 0.0
            mask = self._mask(batch)
            self.optimizer.zero_grad(set_to_none=True)
            requires_clean_student = getattr(self.objective, "requires_clean_student_logits", False)
            requires_teacher_clean = getattr(self.objective, "requires_teacher_clean_logits", False)
            attack_requires_teacher_clean = bool(getattr(self.attack, "requires_teacher_clean_target", False))
            teacher_clean_logits = None
            teacher_clean_forward_calls = 0.0
            if requires_teacher_clean or attack_requires_teacher_clean or self._records_teacher_response:
                if self.teacher is None:
                    raise ValueError("selected attack or objective requires a teacher")
                # This is the one detached FP32 target for both inner and outer
                # RSLAD-family computations.  It has no teacher parameter or
                # input graph and remains valid while the student is updated.
                with (
                    _evaluation_mode(self.teacher),
                    torch.no_grad(),
                    torch.autocast(device_type=self.device.type, enabled=False),
                ):
                    teacher_clean_logits = self.teacher(batch.images.float()).detach().float()
                teacher_clean_forward_calls = 1.0
            valid_mask = mask.to(dtype=torch.bool)
            intervention_risk = None
            if self.intervention_mask is not None:
                intervention_risk = self.intervention_mask.values(
                    batch.sample_ids,
                    device=batch.images.device,
                    dtype=batch.images.dtype,
                ) * valid_mask.to(dtype=batch.images.dtype)
            treatment_risk = intervention_risk
            if (
                treatment_risk is not None
                and self.prescriptive_v3_route is not None
                and not self._prescriptive_active()
            ):
                treatment_risk = torch.zeros_like(treatment_risk)
            epsilon_override = step_override = None
            if self.selected_attack_epsilon is not None:
                attack_config = getattr(self.attack, "config", None)
                baseline_epsilon = getattr(attack_config, "epsilon_value", None)
                baseline_step = getattr(attack_config, "step_size_value", None)
                if baseline_epsilon is None or baseline_step is None or treatment_risk is None:
                    raise ValueError("mixed selected attack budget requires a resolved PGD attack and mask")
                epsilon_override = torch.where(
                    treatment_risk > 0,
                    torch.as_tensor(self.selected_attack_epsilon, device=batch.images.device, dtype=batch.images.dtype),
                    torch.as_tensor(baseline_epsilon, device=batch.images.device, dtype=batch.images.dtype),
                )
                step_override = torch.where(
                    treatment_risk > 0,
                    torch.as_tensor(
                        self.selected_attack_step_size,
                        device=batch.images.device,
                        dtype=batch.images.dtype,
                    ),
                    torch.as_tensor(baseline_step, device=batch.images.device, dtype=batch.images.dtype),
                )
            skip_selected = (
                self.clean_wrong_attack_skip
                and treatment_risk is not None
                and bool((treatment_risk > 0).any())
            )
            if skip_selected:
                if treatment_risk is None:
                    raise RuntimeError("clean-wrong attack skip lost its intervention mask")
                attack_indices = torch.nonzero(treatment_risk <= 0, as_tuple=False).flatten()
                adversarial = batch.images.clone()
                if attack_indices.numel() > 0:
                    subset_target = (
                        None
                        if teacher_clean_logits is None
                        else teacher_clean_logits.index_select(0, attack_indices)
                    )
                    attack_result = self.attack.generate(
                        AttackRequest(
                            inputs=batch.images.index_select(0, attack_indices),
                            labels=batch.labels.index_select(0, attack_indices),
                            student=self.model,
                            teacher=self.teacher,
                        target_logits=subset_target,
                        generator=self._attack_generator(),
                        epsilon_override=(
                            None if epsilon_override is None else epsilon_override.index_select(0, attack_indices)
                        ),
                        step_size_override=(
                            None if step_override is None else step_override.index_select(0, attack_indices)
                        ),
                        )
                    )
                    adversarial.index_copy_(0, attack_indices, attack_result.adversarial)
                else:
                    attack_result = None
            else:
                attack_result = self.attack.generate(
                    AttackRequest(
                        inputs=batch.images,
                        labels=batch.labels,
                        student=self.model,
                        teacher=self.teacher,
                        target_logits=teacher_clean_logits,
                        generator=self._attack_generator(),
                        epsilon_override=epsilon_override,
                        step_size_override=step_override,
                        capture_step=5
                        if self.prescriptive_v3_route == "nr_prefix" and self._prescriptive_active()
                        else None,
                    )
                )
                adversarial = attack_result.adversarial
            if self.prescriptive_v3_route == "nr_prefix" and self._prescriptive_active():
                if (
                    attack_result is None
                    or attack_result.captured_adversarial is None
                    or self.intervention_mask is None
                ):
                    raise RuntimeError("NR prefix treatment did not capture the existing PGD-10 step-5 state")
                selected = (
                    self.intervention_mask.values(batch.sample_ids, device=adversarial.device, dtype=torch.bool)
                    & valid_mask
                )
                adversarial = torch.where(
                    selected[:, None, None, None], attack_result.captured_adversarial, adversarial
                )
            logits = self.model(adversarial)
            observed_teacher_clean = observed_teacher_adversarial = None
            teacher_clean_to_adversarial_margin_response = teacher_clean_to_adversarial_js_response = None
            if self._records_teacher_response:
                assert self.teacher is not None and teacher_clean_logits is not None
                teacher_adversarial_logits = self._teacher_adversarial_response(adversarial)
                observed_teacher_clean = teacher_confidence_primitives(
                    teacher_clean_logits,
                    batch.labels,
                    valid_mask,
                )
                observed_teacher_adversarial = teacher_confidence_primitives(
                    teacher_adversarial_logits,
                    batch.labels,
                    valid_mask,
                )
                # These are detached FP32 observation-only primitives.  They
                # intentionally reuse the existing teacher forwards and never
                # participate in the attack, objective, policy, or optimizer.
                teacher_clean_to_adversarial_js_response = _jensen_shannon_response(
                    teacher_clean_logits,
                    teacher_adversarial_logits,
                )
                teacher_clean_to_adversarial_margin_response = (
                    (
                        (
                            observed_teacher_adversarial.true_probability
                            - observed_teacher_adversarial.max_wrong_probability
                        )
                        - (observed_teacher_clean.true_probability - observed_teacher_clean.max_wrong_probability)
                    )
                    .detach()
                    .float()
                )
            student_signals = self._student_aware_signals(
                batch=batch,
                logits=logits,
                valid_mask=valid_mask,
                teacher_clean=observed_teacher_clean,
                teacher_adversarial=observed_teacher_adversarial,
                teacher_clean_to_adversarial_margin_response=teacher_clean_to_adversarial_margin_response,
                teacher_clean_to_adversarial_js_response=teacher_clean_to_adversarial_js_response,
            )
            clean_student_logits = None
            if requires_clean_student:
                with suspend_ddp_buffer_broadcasts(self.model):
                    clean_student_logits = self.model(batch.images)
            objective_inputs: dict[str, torch.Tensor] = {"student_logits": logits, "labels": batch.labels}
            if requires_teacher_clean:
                assert teacher_clean_logits is not None
                objective_inputs["teacher_logits"] = teacher_clean_logits
            if requires_clean_student:
                assert clean_student_logits is not None
                objective_inputs["clean_student_logits"] = clean_student_logits
            weights = self._policy_weights(
                batch=batch,
                adversarial=adversarial,
                logits=logits,
                valid_mask=valid_mask,
                student_signals=student_signals,
            )
            if self.target_policy is not None:
                if teacher_clean_logits is None:
                    raise ValueError("teacher target policy requires clean teacher logits")
                if weights is None or weights.joint_risk is None:
                    raise ValueError("teacher target policy requires an explicit detached risk")
                target_output = self.target_policy(
                    teacher_logits=teacher_clean_logits,
                    risk=weights.joint_risk if treatment_risk is None else treatment_risk,
                    temperature=getattr(
                        self.target_policy,
                        "target_temperature",
                        getattr(self.objective, "temperature", 1.0),
                    ),
                    labels=batch.labels if getattr(self.target_policy, "requires_labels", False) else None,
                    **(
                        {
                            "anchor_logits": (
                                self._anchor_logits(batch.images)
                                if self._prescriptive_active()
                                else teacher_clean_logits
                            ),
                        }
                        if isinstance(self.target_policy, AnchoredTeacherTargetPolicy)
                        else {}
                    ),
                )
                objective_inputs["adversarial_target_probabilities"] = target_output.probabilities
            dynamic_s3_decision = None
            if self.dynamic_s3_router is not None:
                if clean_student_logits is None or teacher_clean_logits is None or self.teacher is None:
                    raise ValueError("same-step dynamic S3 routing requires RSLAD clean Student/Teacher logits")
                dynamic_s3_decision = self.dynamic_s3_router.observe(
                    epoch=self.current_epoch,
                    sample_ids=batch.sample_ids,
                    labels=batch.labels,
                    valid_mask=valid_mask,
                    student_clean_logits=clean_student_logits,
                    student_adversarial_logits=logits,
                    teacher_clean_logits=teacher_clean_logits,
                    teacher_adversarial_logits=self._teacher_adversarial_response(adversarial),
                )
                # The router decision is a detached hard current-step mask.
                # It does not feed the PGD inner problem or any next-epoch state.
                treatment_risk = dynamic_s3_decision.action_active.to(
                    device=logits.device, dtype=logits.dtype
                )
            if self.iad_inspired:
                if clean_student_logits is None or teacher_clean_logits is None or self.teacher is None:
                    raise ValueError("IAD-inspired branch requires clean Student and Teacher logits")
                teacher_adv = self._teacher_adversarial_response(adversarial)
                with torch.no_grad():
                    alpha = F.softmax(teacher_adv.float(), dim=1).gather(1, batch.labels[:, None]).squeeze(1)
                    teacher_target = F.softmax(teacher_clean_logits.detach().float(), dim=1)
                    self_target = F.softmax(clean_student_logits.detach().float(), dim=1)
                    if treatment_risk is None:
                        selected = torch.zeros_like(alpha)
                    else:
                        selected = treatment_risk.detach().clamp(0.0, 1.0)
                    mixed = (1.0 - selected[:, None]) * teacher_target + selected[:, None] * (
                        alpha[:, None] * teacher_target + (1.0 - alpha[:, None]) * self_target
                    )
                objective_inputs["adversarial_target_probabilities"] = mixed.detach()
            terms = self.objective(**objective_inputs)
            if weights is not None:
                terms = terms.apply_policy(weights)
            if treatment_risk is not None and self.clean_wrong_mode is not None:
                if clean_student_logits is None or teacher_clean_logits is None or terms.clean_kd is None:
                    raise ValueError("clean-wrong treatment requires clean Student/Teacher logits and clean KD")
                selected = treatment_risk > 0
                clean_ce = torch.nn.functional.cross_entropy(clean_student_logits, batch.labels, reduction="none")
                if self.clean_wrong_mode == "clean_ce_only":
                    replacement_kd = torch.zeros_like(terms.clean_kd)
                elif self.clean_wrong_mode == "teacher_clean_gate":
                    replacement_kd = terms.clean_kd * teacher_clean_logits.detach().argmax(dim=1).eq(batch.labels)
                else:
                    replacement_kd = terms.clean_kd
                clean_coefficient = 1.0 if self.clean_ce_coefficient is None else self.clean_ce_coefficient
                terms = ObjectiveTerms(
                    hard=torch.where(selected, clean_coefficient * clean_ce, terms.hard),
                    kd=torch.where(selected, replacement_kd, terms.kd),
                    regularization=terms.regularization,
                    adversarial_kd=(
                        torch.where(selected, torch.zeros_like(terms.adversarial_kd), terms.adversarial_kd)
                        if terms.adversarial_kd is not None
                        else None
                    ),
                    clean_kd=torch.where(selected, replacement_kd, terms.clean_kd),
                )
            if treatment_risk is not None and self.extra_clean_ce_coefficient is not None:
                if clean_student_logits is None:
                    raise ValueError("extra CleanCE requires clean Student logits")
                clean_ce = F.cross_entropy(clean_student_logits, batch.labels, reduction="none")
                terms = ObjectiveTerms(
                    hard=terms.hard + self.extra_clean_ce_coefficient * treatment_risk * clean_ce,
                    kd=terms.kd,
                    regularization=terms.regularization,
                    adversarial_kd=terms.adversarial_kd,
                    clean_kd=terms.clean_kd,
                )
            if treatment_risk is not None and self.teacher_clean_reliability_mask is not None:
                if terms.adversarial_kd is None or terms.clean_kd is None:
                    raise ValueError("teacher reliability gate requires exposed KD branches")
                gate = self.teacher_clean_reliability_mask.values(
                    batch.sample_ids, device=logits.device, dtype=logits.dtype
                )
                multiplier = treatment_risk * (0.5 + 0.5 * gate) + (1.0 - treatment_risk)
                terms = terms.scale_adversarial_kd(
                    multiplier,
                    coefficient=float(getattr(self.objective, "ADVERSARIAL_COEFFICIENT", 1.0)),
                )
                assert terms.clean_kd is not None
                clean_multiplier = 1.0 - 0.5 * treatment_risk * (1.0 - gate)
                terms = ObjectiveTerms(
                    hard=terms.hard,
                    kd=terms.kd
                    + float(getattr(self.objective, "CLEAN_COEFFICIENT", 1.0))
                    * (clean_multiplier - 1.0)
                    * terms.clean_kd,
                    regularization=terms.regularization,
                    adversarial_kd=terms.adversarial_kd,
                    clean_kd=terms.clean_kd * clean_multiplier,
                )
            if intervention_risk is not None and self.adversarial_kd_multiplier is not None:
                multiplier = 1.0 - (1.0 - self.adversarial_kd_multiplier) * intervention_risk
                terms = terms.scale_adversarial_kd(
                    multiplier,
                    coefficient=float(getattr(self.objective, "ADVERSARIAL_COEFFICIENT", 1.0)),
                )
            if intervention_risk is not None and self.adaptive_advkd_gamma is not None:
                if terms.adversarial_kd is None:
                    raise ValueError("adaptive AdvKD requires exposed adversarial KD")
                if clean_student_logits is None:
                    raise ValueError("adaptive AdvKD requires clean Student logits")
                with torch.no_grad():
                    probability = F.softmax(clean_student_logits.detach().float(), dim=1).gather(
                        1, batch.labels[:, None]
                    ).squeeze(1)
                    adaptive = 1.0 + self.adaptive_advkd_gamma * (1.0 - probability)
                terms = terms.scale_adversarial_kd(
                    1.0 + intervention_risk * (adaptive - 1.0),
                    coefficient=float(getattr(self.objective, "ADVERSARIAL_COEFFICIENT", 1.0)),
                )
            if intervention_risk is not None and self.adversarial_ce_coefficient is not None:
                terms = terms.add_adversarial_ce(
                    batch.labels,
                    logits,
                    intervention_risk,
                    coefficient=float(self.adversarial_ce_coefficient),
                )
            if intervention_risk is not None and self.adversarial_bce_coefficient is not None:
                probabilities = F.softmax(logits.float(), dim=1)
                true_probability = probabilities.gather(1, batch.labels[:, None]).squeeze(1).clamp_min(1e-8)
                wrong = probabilities.scatter(1, batch.labels[:, None], 0.0).amax(dim=1)
                bce = -torch.log(true_probability) - torch.log((1.0 - wrong).clamp_min(1e-8))
                terms = ObjectiveTerms(
                    hard=terms.hard + self.adversarial_bce_coefficient * intervention_risk * bce,
                    kd=terms.kd,
                    regularization=terms.regularization,
                    adversarial_kd=terms.adversarial_kd,
                    clean_kd=terms.clean_kd,
                )
            if treatment_risk is not None and self.margin_target_mode is not None:
                teacher_adversarial_logits = self._teacher_adversarial_response(adversarial)
                target_margin, target_active = self._adversarial_margin_target(
                    teacher_adversarial_logits=teacher_adversarial_logits,
                    labels=batch.labels,
                )
                terms = terms.add_adversarial_margin(
                    logits,
                    batch.labels,
                    target_margin,
                    treatment_risk * target_active,
                    coefficient=float(self.margin_coefficient),
                )
            if dynamic_s3_decision is not None and self.adversarial_ce_coefficient is not None:
                if treatment_risk is None:
                    raise RuntimeError("dynamic S3 action lost its batch mask")
                terms = terms.add_adversarial_ce(
                    batch.labels,
                    logits,
                    treatment_risk,
                    coefficient=float(self.adversarial_ce_coefficient),
                )
            if self.diagnostics is not None:
                with suspend_ddp_buffer_broadcasts(self.model), _evaluation_mode(self.model), torch.no_grad():
                    diagnostic_clean = self.model(batch.images).detach()
                teacher_prediction = teacher_entropy = None
                if self.teacher is not None:
                    teacher_adversarial_logits = self._teacher_adversarial_response(adversarial)
                    teacher_prediction = teacher_adversarial_logits.argmax(1)
                    teacher_entropy = shannon_entropy(teacher_adversarial_logits)
                prior_margin = None if self.sample_store is None else self.sample_store.margin_ema(batch.sample_ids)
                sample_store = self.sample_store
                # Move scalar diagnostic fields in bounded batches.  This
                # avoids a GPU synchronization for every individual sample.
                sample_ids = batch.sample_ids.detach().cpu().tolist()
                valid = valid_mask.detach().cpu().tolist()
                labels = batch.labels.detach().cpu().tolist()
                clean_predictions = diagnostic_clean.argmax(1).detach().cpu().tolist()
                adversarial_predictions = logits.detach().argmax(1).cpu().tolist()
                teacher_predictions = None if teacher_prediction is None else teacher_prediction.detach().cpu().tolist()
                teacher_entropies = None if teacher_entropy is None else teacher_entropy.detach().cpu().tolist()
                prior_margins = None if prior_margin is None else prior_margin.detach().cpu().tolist()
                joint_risks = (
                    None
                    if weights is None or weights.joint_risk is None
                    else weights.joint_risk.detach().cpu().tolist()
                )
                kd_weights = None if weights is None else weights.kd_weight.detach().cpu().tolist()
                panel_positions = (
                    [
                        position
                        for position, sample_id in enumerate(sample_ids)
                        if sample_id in self.diagnostics.panel_ids
                    ]
                    if self.diagnostics.mode == "panel"
                    else []
                )
                panel_media: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
                if panel_positions:
                    positions = torch.tensor(panel_positions, device=self.device)
                    clean_images = batch.images.index_select(0, positions).detach().cpu()
                    adversarial_images = adversarial.index_select(0, positions).detach().cpu()
                    perturbations = (adversarial_images - clean_images).detach()
                    panel_media = {
                        position: (clean_images[index], adversarial_images[index], perturbations[index])
                        for index, position in enumerate(panel_positions)
                    }
                for position, sample_id in enumerate(sample_ids):
                    media = panel_media.get(position)
                    has_prior = (
                        prior_margins is not None and sample_store is not None and sample_id in sample_store.records
                    )
                    prior_value = None
                    unlearnability = None
                    if has_prior:
                        assert prior_margins is not None
                        prior_value = prior_margins[position]
                        unlearnability = (1 - prior_value) / 2
                    self.diagnostics.record(
                        sample_id=sample_id,
                        valid=valid[position],
                        epoch=self.current_epoch,
                        clean_image=None if media is None else media[0],
                        adversarial_image=None if media is None else media[1],
                        perturbation_visualization=None if media is None else media[2],
                        true_label=labels[position],
                        student_clean_prediction=clean_predictions[position],
                        student_adv_prediction=adversarial_predictions[position],
                        teacher_prediction=None if teacher_predictions is None else teacher_predictions[position],
                        teacher_entropy=None if teacher_entropies is None else teacher_entropies[position],
                        student_robust_margin_ema=prior_value,
                        student_unlearnability=unlearnability,
                        joint_risk=None if joint_risks is None else joint_risks[position],
                        kd_weight=0.0 if kd_weights is None else kd_weights[position],
                        clean_correct=clean_predictions[position] == labels[position],
                        robust_correct=adversarial_predictions[position] == labels[position],
                    )
                self._teacher_adversarial_logits = None
            # DDP averages gradients across ranks.  Scale each local masked
            # sum by world_size/global-effective-count so padded ranks cannot
            # dilute the update (including the size < world_size case).
            global_count = reduce_sums(mask.detach().sum().to(dtype=torch.float64)).clamp_min(1.0)
            loss = (terms.total * mask).sum() * (get_world_size() / global_count.to(dtype=terms.total.dtype))
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            if self.scaler is None:
                loss.backward()
                self.optimizer.step()
            else:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            with _evaluation_mode(self.model), torch.no_grad():
                clean_logits = self.model(batch.images)
            totals += torch.tensor(
                [
                    float((terms.total.detach() * mask).sum()),
                    float(((clean_logits.argmax(1) == batch.labels).to(mask.dtype) * mask).sum()),
                    float(((logits.detach().argmax(1) == batch.labels).to(mask.dtype) * mask).sum()),
                    float(mask.sum()),
                    teacher_clean_forward_calls,
                    self._teacher_adversarial_forward_calls,
                ],
                dtype=torch.float64,
                device=self.device,
            )
            self.global_step += 1
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak_allocated_bytes = torch.cuda.max_memory_allocated(self.device)
            peak_reserved_bytes = torch.cuda.max_memory_reserved(self.device)
        else:
            peak_allocated_bytes = 0
            peak_reserved_bytes = 0
        totals, observability = _reduce_epoch_observability(
            totals,
            local_seconds=time.perf_counter() - started_at,
            local_cuda_peak_allocated_bytes=peak_allocated_bytes,
            local_cuda_peak_reserved_bytes=peak_reserved_bytes,
        )
        count = max(observability["valid_examples"], 1.0)
        return {
            "loss": float(totals[0].item()) / count,
            "clean_accuracy": float(totals[1].item()) / count,
            "robust_accuracy": float(totals[2].item()) / count,
            **observability,
        }

    def validate_epoch(self, loader: DataLoader[IndexedBatch]) -> dict[str, float]:
        """Evaluate post-update clean and PGD accuracy without mutating model state."""
        totals = torch.zeros(3, dtype=torch.float64, device=self.device)
        generator = self._selection_generator()
        with _evaluation_mode(self.model):
            for batch in loader:
                if not isinstance(batch, IndexedBatch):
                    raise TypeError("trainer requires IndexedBatch batches")
                batch = batch.to(self.device)
                mask = self._mask(batch)
                with torch.no_grad():
                    clean_logits = self.model(batch.images)
                attack_result = self.selection_attack.generate(
                    AttackRequest(
                        inputs=batch.images,
                        labels=batch.labels,
                        student=self.model,
                        teacher=self.teacher,
                        generator=generator,
                    )
                )
                with torch.no_grad():
                    adversarial_logits = self.model(attack_result.adversarial)
                totals += torch.tensor(
                    [
                        float(((clean_logits.argmax(1) == batch.labels).to(mask.dtype) * mask).sum()),
                        float(((adversarial_logits.argmax(1) == batch.labels).to(mask.dtype) * mask).sum()),
                        float(mask.sum()),
                    ],
                    dtype=torch.float64,
                    device=self.device,
                )
        totals = reduce_sums(totals)
        count = max(float(totals[2].item()), 1.0)
        return {"clean_accuracy": float(totals[0].item()) / count, "pgd_accuracy": float(totals[1].item()) / count}

    def fit(
        self,
        loader: DataLoader[IndexedBatch],
        *,
        validation_loader: DataLoader[IndexedBatch],
        epochs: int,
        start_epoch: int = 0,
        on_epoch_end: Callable[[Mapping[str, float], bool], None] | None = None,
    ) -> list[dict[str, float]]:
        history = []
        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch
            # Record the rate that actually governs this epoch before the
            # epoch-end scheduler transition.  ``next_learning_rate`` below
            # is the checkpointed rate that a resume will use next.
            epoch_learning_rate = float(self.optimizer.param_groups[0]["lr"])
            sampler = loader.sampler
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            if hasattr(loader.dataset, "set_epoch"):
                loader.dataset.set_epoch(epoch)
            train_metrics = self.train_epoch(loader)
            self._flush_sample_store()
            if self.dynamic_s3_router is not None:
                self.dynamic_s3_router.flush_epoch(epoch)
                # The capture map is part of scientific lineage: a resumed
                # fixed arm must restore the epoch-80 decision, never rebuild
                # it from a later model state.  This assignment occurs before
                # the common checkpoint write below.
                self.fork_lineage = {
                    **({} if self.fork_lineage is None else self.fork_lineage),
                    "dynamic_s3_state": self.dynamic_s3_router.state_dict(),
                }
            if self.diagnostics is not None:
                self.diagnostics.flush()
            validation_metrics = self.validate_epoch(validation_loader)
            self.selection_metadata["last_epoch"] = epoch
            self.selection_metadata["last_clean_accuracy"] = validation_metrics["clean_accuracy"]
            self.selection_metadata["last_pgd_accuracy"] = validation_metrics["pgd_accuracy"]
            if self.scheduler is not None:
                self.scheduler.step()
            # Strictly greater deliberately keeps the earliest epoch on ties.
            improved = validation_metrics["pgd_accuracy"] > self.best_metric
            if improved:
                self.best_metric = validation_metrics["pgd_accuracy"]
                self.selection_metadata["selected_epoch"] = epoch
                self.selection_metadata["selected_clean_accuracy"] = validation_metrics["clean_accuracy"]
                self.selection_metadata["selected_pgd_accuracy"] = validation_metrics["pgd_accuracy"]
            common = dict(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                sampler=sampler,
                sample_state=self.sample_state,
                global_step=self.global_step,
                best_metric=self.best_metric,
                selection_metadata=self.selection_metadata,
                tracker_run_id=self.tracker_run_id,
                config_hash=self.config_hash,
                fork_lineage=self.fork_lineage,
            )
            save_checkpoint(self.output_dir / "last.pt", **common)
            if epoch + 1 in self.checkpoint_epochs:
                save_checkpoint(self.output_dir / f"epoch-{epoch + 1:03d}.pt", **common)
            if improved:
                save_checkpoint(self.output_dir / "best.pt", **common)
            epoch_metrics = {
                "train_loss": train_metrics["loss"],
                "train_clean_accuracy": train_metrics["clean_accuracy"],
                "train_robust_accuracy": train_metrics["robust_accuracy"],
                "train_valid_examples": train_metrics.get("valid_examples", 0.0),
                "train_seconds": train_metrics.get("seconds", 0.0),
                "train_images_per_second": train_metrics.get("images_per_second", 0.0),
                "train_cuda_peak_allocated_bytes": train_metrics.get("cuda_peak_allocated_bytes", 0.0),
                "train_cuda_peak_reserved_bytes": train_metrics.get("cuda_peak_reserved_bytes", 0.0),
                "train_teacher_clean_forward_calls": train_metrics.get("teacher_clean_forward_calls", 0.0),
                "train_teacher_adversarial_forward_calls": train_metrics.get("teacher_adversarial_forward_calls", 0.0),
                "val_clean_accuracy": validation_metrics["clean_accuracy"],
                "val_pgd_accuracy": validation_metrics["pgd_accuracy"],
                "learning_rate": epoch_learning_rate,
                "next_learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            }
            history.append(epoch_metrics)
            # The callback is deliberately after both atomic checkpoints; it
            # is observational only and cannot alter model/state selection.
            if on_epoch_end is not None:
                on_epoch_end(epoch_metrics, improved)
        return history

    def resume(self, path: Path, *, sampler: Any) -> TrainingState:
        state = load_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            sampler=sampler,
            expected_config_hash=self.config_hash,
            device=self.device,
        )
        self.global_step, self.best_metric = state.global_step, state.best_metric
        if self.tracker_run_id is not None and state.tracker_run_id != self.tracker_run_id:
            raise ValueError("checkpoint tracker run ID does not match the active tracker")
        self.tracker_run_id, self.sample_state = state.tracker_run_id, state.sample_state
        self.fork_lineage = state.fork_lineage
        if self.dynamic_s3_router is not None:
            if not isinstance(self.fork_lineage, dict) or not isinstance(
                self.fork_lineage.get("dynamic_s3_state"), dict
            ):
                raise ValueError("dynamic S3 resume checkpoint lacks immutable routing state")
            self.dynamic_s3_router.load_state_dict(self.fork_lineage["dynamic_s3_state"])
        if self.sample_store is not None:
            self.sample_store.load_state_dict(state.sample_state)
            self.sample_state = self.sample_store.state_dict()
        self.selection_metadata = state.selection_metadata
        return state
