"""Single training loop composed from attack and unreduced outer objective interfaces."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ard.attacks import AttackGenerator, AttackRequest
from ard.data import IndexedBatch
from ard.objectives import DistillationObjective
from ard.policies import (
    PolicyContext,
    PolicyWeights,
    WeightPolicy,
    student_risk_from_margin,
    teacher_risk_from_entropy,
)
from ard.signals import RobustMarginSignal, shannon_entropy
from ard.state import SampleStateStore
from ard.targets import TeacherTargetPolicy
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


@contextmanager
def _evaluation_mode(model: nn.Module) -> Iterator[None]:
    mode = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(mode)


def _reduce_epoch_observability(
    local_totals: torch.Tensor,
    *,
    local_seconds: float,
    local_cuda_peak_allocated_bytes: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply the epoch SUM/MAX contract and derive globally valid throughput."""
    if local_totals.shape != (5,):
        raise ValueError("epoch totals must contain five scalar accumulators")
    global_totals = reduce_sums(local_totals)
    rank_max = reduce_max(
        torch.tensor(
            [local_seconds, float(local_cuda_peak_allocated_bytes)],
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
        "teacher_clean_forward_calls": float(global_totals[4].item()),
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
        teacher: nn.Module | None = None,
        policy: WeightPolicy | None = None,
        sample_store: SampleStateStore | None = None,
        target_policy: TeacherTargetPolicy | None = None,
        policy_warmup_epochs: int = 0,
        oracle_mask: bool = False,
        diagnostics: TrainingDiagnostics | None = None,
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
        self.policy = policy
        self.sample_store = sample_store
        self.target_policy = target_policy
        if target_policy is not None and self.teacher is None:
            raise ValueError("teacher target policy requires a frozen teacher")
        if policy_warmup_epochs < 0:
            raise ValueError("policy_warmup_epochs must be non-negative")
        if oracle_mask and sample_store is None:
            raise ValueError("oracle_mask requires student-aware sample state")
        self.policy_warmup_epochs, self.oracle_mask = policy_warmup_epochs, oracle_mask
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
        self.sample_state: dict[str, Any] = {} if sample_store is None else sample_store.state_dict()
        self.diagnostics = diagnostics
        # This detached cache is strictly intra-batch diagnostic reuse.  It is
        # cleared at every batch boundary and intentionally excluded from
        # checkpoint state.
        self._diagnostic_teacher_adversarial_logits: torch.Tensor | None = None

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

    def _student_aware_signals(
        self,
        *,
        batch: IndexedBatch,
        logits: torch.Tensor,
        valid_mask: torch.Tensor,
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
            rank=get_rank(),
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
        teacher_adversarial_logits: torch.Tensor | None = None
        if "teacher_entropy" in required or "joint_risk" in required:
            if self.teacher is None:
                raise ValueError("selected policy requires a teacher")
            with _evaluation_mode(self.teacher), torch.no_grad():
                teacher_adversarial_logits = self.teacher(adversarial).detach().float()
                entropy = shannon_entropy(teacher_adversarial_logits)
            if self.diagnostics is not None:
                self._diagnostic_teacher_adversarial_logits = teacher_adversarial_logits
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
        # detached clean-teacher target forwards.  One final SUM makes the
        # count telemetry global without adding a hot-loop collective.
        totals = torch.zeros(5, dtype=torch.float64, device=self.device)
        for batch in loader:
            if not isinstance(batch, IndexedBatch):
                raise TypeError("trainer requires IndexedBatch batches")
            batch = batch.to(self.device)
            self._diagnostic_teacher_adversarial_logits = None
            mask = self._mask(batch)
            self.optimizer.zero_grad(set_to_none=True)
            requires_clean_student = getattr(self.objective, "requires_clean_student_logits", False)
            requires_teacher_clean = getattr(self.objective, "requires_teacher_clean_logits", False)
            attack_requires_teacher_clean = bool(getattr(self.attack, "requires_teacher_clean_target", False))
            teacher_clean_logits = None
            teacher_clean_forward_calls = 0.0
            if requires_teacher_clean or attack_requires_teacher_clean:
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
            attack_result = self.attack.generate(
                AttackRequest(
                    inputs=batch.images,
                    labels=batch.labels,
                    student=self.model,
                    teacher=self.teacher,
                    target_logits=teacher_clean_logits,
                    generator=self._attack_generator(),
                )
            )
            logits = self.model(attack_result.adversarial)
            valid_mask = mask.to(dtype=torch.bool)
            student_signals = self._student_aware_signals(batch=batch, logits=logits, valid_mask=valid_mask)
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
                adversarial=attack_result.adversarial,
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
                    risk=weights.joint_risk,
                    temperature=getattr(self.objective, "temperature", 1.0),
                )
                objective_inputs["adversarial_target_probabilities"] = target_output.probabilities
            terms = self.objective(**objective_inputs)
            if weights is not None:
                terms = terms.apply_policy(weights)
            if self.diagnostics is not None:
                with suspend_ddp_buffer_broadcasts(self.model), _evaluation_mode(self.model), torch.no_grad():
                    diagnostic_clean = self.model(batch.images).detach()
                teacher_prediction = teacher_entropy = None
                if self.teacher is not None:
                    teacher_adversarial_logits = self._diagnostic_teacher_adversarial_logits
                    if teacher_adversarial_logits is None:
                        with _evaluation_mode(self.teacher), torch.no_grad():
                            teacher_adversarial_logits = self.teacher(attack_result.adversarial).detach().float()
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
                    adversarial_images = attack_result.adversarial.index_select(0, positions).detach().cpu()
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
                self._diagnostic_teacher_adversarial_logits = None
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
                ],
                dtype=torch.float64,
                device=self.device,
            )
            self.global_step += 1
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak_allocated_bytes = torch.cuda.max_memory_allocated(self.device)
        else:
            peak_allocated_bytes = 0
        totals, observability = _reduce_epoch_observability(
            totals,
            local_seconds=time.perf_counter() - started_at,
            local_cuda_peak_allocated_bytes=peak_allocated_bytes,
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
            sampler = loader.sampler
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            if hasattr(loader.dataset, "set_epoch"):
                loader.dataset.set_epoch(epoch)
            train_metrics = self.train_epoch(loader)
            self._flush_sample_store()
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
            )
            save_checkpoint(self.output_dir / "last.pt", **common)
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
                "train_teacher_clean_forward_calls": train_metrics.get("teacher_clean_forward_calls", 0.0),
                "val_clean_accuracy": validation_metrics["clean_accuracy"],
                "val_pgd_accuracy": validation_metrics["pgd_accuracy"],
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
        if self.sample_store is not None:
            self.sample_store.load_state_dict(state.sample_state)
            self.sample_state = self.sample_store.state_dict()
        self.selection_metadata = state.selection_metadata
        return state
