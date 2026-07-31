"""Replicated stable-ID state for student-aware training policies."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch

from ard.signals.teacher_confidence import TeacherConfidenceBatch


@dataclass(frozen=True)
class SampleObservation:
    sample_id: int
    margin: float
    robust_correct: bool
    update: int
    rank: int = 0
    order: int = 0
    true_label: int | None = None
    teacher_clean_entropy: float | None = None
    teacher_clean_true_probability: float | None = None
    teacher_clean_max_wrong_probability: float | None = None
    teacher_clean_prediction: int | None = None
    teacher_clean_correct: bool | None = None
    teacher_adversarial_entropy: float | None = None
    teacher_adversarial_true_probability: float | None = None
    teacher_adversarial_max_wrong_probability: float | None = None
    teacher_adversarial_prediction: int | None = None
    teacher_adversarial_correct: bool | None = None


@dataclass
class SampleRecord:
    margin_ema: float
    seen: int
    robust_correct_count: int
    previous_robust_correct: bool | None
    forgetting_count: int
    last_update: int
    last_margin: float | None = None
    true_label: int | None = None
    teacher_clean_entropy: float | None = None
    teacher_clean_true_probability: float | None = None
    teacher_clean_max_wrong_probability: float | None = None
    teacher_clean_prediction: int | None = None
    teacher_clean_correct: bool | None = None
    teacher_adversarial_entropy: float | None = None
    teacher_adversarial_true_probability: float | None = None
    teacher_adversarial_max_wrong_probability: float | None = None
    teacher_adversarial_prediction: int | None = None
    teacher_adversarial_correct: bool | None = None

    @property
    def robust_correct_frequency(self) -> float:
        return self.robust_correct_count / self.seen if self.seen else 0.0


class SampleStateStore:
    """Sparse observations, deterministically merged once at each epoch boundary.

    The store is intentionally replicated on every rank.  Batch observations are
    only queued locally; ``merge_pending`` is the sole mutation path for records.
    """

    FORMAT_VERSION = 2

    def __init__(self, *, ema_decay: float = 0.9) -> None:
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        self.ema_decay = float(ema_decay)
        self.records: dict[int, SampleRecord] = {}
        self.pending: list[SampleObservation] = []
        self._next_order = 0

    def record_pending(
        self,
        *,
        sample_ids: torch.Tensor,
        margins: torch.Tensor,
        robust_correct: torch.Tensor,
        valid_mask: torch.Tensor,
        update: int,
        rank: int = 0,
        labels: torch.Tensor | None = None,
        teacher_clean: TeacherConfidenceBatch | None = None,
        teacher_adversarial: TeacherConfidenceBatch | None = None,
    ) -> None:
        """Queue valid detached observations; padded rows never enter state."""
        if any(value.ndim != 1 for value in (sample_ids, margins, robust_correct, valid_mask)):
            raise ValueError("sample observations must be one-dimensional")
        if not (sample_ids.shape == margins.shape == robust_correct.shape == valid_mask.shape):
            raise ValueError("sample observation vectors must have the same shape")
        if valid_mask.dtype != torch.bool:
            raise ValueError("sample observation valid_mask must be bool")
        ids = sample_ids.detach().to(device="cpu", dtype=torch.long)
        values = margins.detach().to(device="cpu", dtype=torch.float32)
        correct = robust_correct.detach().to(device="cpu", dtype=torch.bool)
        mask = valid_mask.detach().to(device="cpu", dtype=torch.bool)
        label_values = None if labels is None else labels.detach().to(device="cpu", dtype=torch.long)
        if label_values is not None and label_values.shape != ids.shape:
            raise ValueError("sample observation labels must match sample IDs")
        if (teacher_clean is None) != (teacher_adversarial is None):
            raise ValueError("teacher clean and adversarial observations must be provided together")
        if teacher_clean is not None and label_values is None:
            raise ValueError("teacher observations require true labels")

        def _teacher_values(
            values: TeacherConfidenceBatch | None,
        ) -> tuple[list[float], list[float], list[float], list[int], list[bool]] | None:
            if values is None:
                return None
            tensors = (
                values.entropy,
                values.true_probability,
                values.max_wrong_probability,
                values.prediction,
                values.correct,
            )
            if any(tensor.ndim != 1 or tensor.shape != ids.shape for tensor in tensors):
                raise ValueError("teacher observation vectors must match sample IDs")
            scalars = torch.stack(
                (
                    values.entropy.detach().float(),
                    values.true_probability.detach().float(),
                    values.max_wrong_probability.detach().float(),
                ),
                dim=1,
            ).to(device="cpu")
            if bool((~torch.isfinite(scalars[mask])).any()):
                raise FloatingPointError("cannot store non-finite teacher observations")
            if bool(((scalars[mask, 1:] < 0.0) | (scalars[mask, 1:] > 1.0)).any()):
                raise FloatingPointError("cannot store teacher probabilities outside [0,1]")
            return (
                scalars[:, 0].tolist(),
                scalars[:, 1].tolist(),
                scalars[:, 2].tolist(),
                values.prediction.detach().to(device="cpu", dtype=torch.long).tolist(),
                values.correct.detach().to(device="cpu", dtype=torch.bool).tolist(),
            )

        clean_values = _teacher_values(teacher_clean)
        adversarial_values = _teacher_values(teacher_adversarial)
        if bool((~torch.isfinite(values) & mask).any()):
            raise FloatingPointError("cannot store a non-finite robust margin")
        for position, (sample_id, margin, is_correct, valid) in enumerate(
            zip(ids.tolist(), values.tolist(), correct.tolist(), mask.tolist(), strict=True)
        ):
            if valid:
                clean = None if clean_values is None else tuple(values[position] for values in clean_values)
                adversarial = (
                    None if adversarial_values is None else tuple(values[position] for values in adversarial_values)
                )
                self.pending.append(
                    SampleObservation(
                        sample_id=int(sample_id),
                        margin=float(margin),
                        robust_correct=bool(is_correct),
                        update=int(update),
                        rank=int(rank),
                        order=self._next_order,
                        true_label=None if label_values is None else int(label_values[position]),
                        teacher_clean_entropy=None if clean is None else float(clean[0]),
                        teacher_clean_true_probability=None if clean is None else float(clean[1]),
                        teacher_clean_max_wrong_probability=None if clean is None else float(clean[2]),
                        teacher_clean_prediction=None if clean is None else int(clean[3]),
                        teacher_clean_correct=None if clean is None else bool(clean[4]),
                        teacher_adversarial_entropy=None if adversarial is None else float(adversarial[0]),
                        teacher_adversarial_true_probability=(None if adversarial is None else float(adversarial[1])),
                        teacher_adversarial_max_wrong_probability=(
                            None if adversarial is None else float(adversarial[2])
                        ),
                        teacher_adversarial_prediction=None if adversarial is None else int(adversarial[3]),
                        teacher_adversarial_correct=None if adversarial is None else bool(adversarial[4]),
                    )
                )
                self._next_order += 1

    @staticmethod
    def _coerce_observation(value: SampleObservation | Mapping[str, Any]) -> SampleObservation:
        if isinstance(value, SampleObservation):
            return value
        return SampleObservation(**dict(value))

    def pending_state(self) -> list[dict[str, Any]]:
        return [asdict(observation) for observation in self.pending]

    def merge_pending(self, pending_by_rank: Iterable[Iterable[SampleObservation | Mapping[str, Any]]]) -> None:
        """Apply each original sample at most once in a rank/order-stable order.

        A valid duplicate is not a new sample observation.  It can arise from a
        caller error or a distributed sampler edge case; rank then local order
        picks the canonical record deterministically rather than updating EMA or
        forgetting counts twice.
        """
        flattened = [self._coerce_observation(value) for shard in pending_by_rank for value in shard]
        flattened.sort(key=lambda item: (item.sample_id, item.rank, item.order))
        seen_ids: set[int] = set()
        for observation in flattened:
            if observation.sample_id in seen_ids:
                continue
            seen_ids.add(observation.sample_id)
            record = self.records.get(observation.sample_id)
            if record is None:
                self.records[observation.sample_id] = SampleRecord(
                    margin_ema=observation.margin,
                    seen=1,
                    robust_correct_count=int(observation.robust_correct),
                    previous_robust_correct=observation.robust_correct,
                    forgetting_count=0,
                    last_update=observation.update,
                    last_margin=observation.margin,
                    true_label=observation.true_label,
                    teacher_clean_entropy=observation.teacher_clean_entropy,
                    teacher_clean_true_probability=observation.teacher_clean_true_probability,
                    teacher_clean_max_wrong_probability=observation.teacher_clean_max_wrong_probability,
                    teacher_clean_prediction=observation.teacher_clean_prediction,
                    teacher_clean_correct=observation.teacher_clean_correct,
                    teacher_adversarial_entropy=observation.teacher_adversarial_entropy,
                    teacher_adversarial_true_probability=observation.teacher_adversarial_true_probability,
                    teacher_adversarial_max_wrong_probability=observation.teacher_adversarial_max_wrong_probability,
                    teacher_adversarial_prediction=observation.teacher_adversarial_prediction,
                    teacher_adversarial_correct=observation.teacher_adversarial_correct,
                )
                continue
            if (
                record.true_label is not None
                and observation.true_label is not None
                and record.true_label != observation.true_label
            ):
                raise ValueError("stable sample ID changed true label")
            record.margin_ema = self.ema_decay * record.margin_ema + (1.0 - self.ema_decay) * observation.margin
            record.seen += 1
            record.robust_correct_count += int(observation.robust_correct)
            if record.previous_robust_correct is True and not observation.robust_correct:
                record.forgetting_count += 1
            record.previous_robust_correct = observation.robust_correct
            record.last_update = observation.update
            record.last_margin = observation.margin
            for field in (
                "true_label",
                "teacher_clean_entropy",
                "teacher_clean_true_probability",
                "teacher_clean_max_wrong_probability",
                "teacher_clean_prediction",
                "teacher_clean_correct",
                "teacher_adversarial_entropy",
                "teacher_adversarial_true_probability",
                "teacher_adversarial_max_wrong_probability",
                "teacher_adversarial_prediction",
                "teacher_adversarial_correct",
            ):
                value = getattr(observation, field)
                if value is not None:
                    setattr(record, field, value)
        self.pending.clear()
        self._next_order = 0

    def margin_ema(self, sample_ids: torch.Tensor, *, default: float = 0.0) -> torch.Tensor:
        values = [
            self.records.get(int(sample_id), SampleRecord(default, 0, 0, None, 0, -1)).margin_ema
            for sample_id in sample_ids.detach().to(device="cpu", dtype=torch.long).tolist()
        ]
        return torch.tensor(values, dtype=torch.float32, device=sample_ids.device)

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.FORMAT_VERSION,
            "ema_decay": self.ema_decay,
            "records": {str(sample_id): asdict(record) for sample_id, record in sorted(self.records.items())},
            "pending": self.pending_state(),
            "next_order": self._next_order,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if set(state) != {"format_version", "ema_decay", "records", "pending", "next_order"}:
            raise ValueError("sample state has unexpected or missing keys")
        if state["format_version"] not in {1, self.FORMAT_VERSION}:
            raise ValueError("unsupported sample state format")
        if float(state["ema_decay"]) != self.ema_decay:
            raise ValueError("sample state ema_decay does not match configuration")
        raw_records = state["records"]
        if not isinstance(raw_records, Mapping):
            raise ValueError("sample state records must be a mapping")
        records: dict[int, SampleRecord] = {}
        for raw_id, raw_record in raw_records.items():
            sample_id = int(raw_id)
            record = SampleRecord(**dict(raw_record))
            if record.seen < 1 or record.robust_correct_count < 0 or record.robust_correct_count > record.seen:
                raise ValueError("sample state record counters are invalid")
            records[sample_id] = record
        pending = [self._coerce_observation(item) for item in state["pending"]]
        next_order = int(state["next_order"])
        if next_order < 0:
            raise ValueError("sample state next_order must be non-negative")
        self.records, self.pending, self._next_order = records, pending, next_order
