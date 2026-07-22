"""Replicated stable-ID state for student-aware training policies."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class SampleObservation:
    sample_id: int
    margin: float
    robust_correct: bool
    update: int
    rank: int = 0
    order: int = 0


@dataclass
class SampleRecord:
    margin_ema: float
    seen: int
    robust_correct_count: int
    previous_robust_correct: bool | None
    forgetting_count: int
    last_update: int

    @property
    def robust_correct_frequency(self) -> float:
        return self.robust_correct_count / self.seen if self.seen else 0.0


class SampleStateStore:
    """Sparse observations, deterministically merged once at each epoch boundary.

    The store is intentionally replicated on every rank.  Batch observations are
    only queued locally; ``merge_pending`` is the sole mutation path for records.
    """

    FORMAT_VERSION = 1

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
        if bool((~torch.isfinite(values) & mask).any()):
            raise FloatingPointError("cannot store a non-finite robust margin")
        for sample_id, margin, is_correct, valid in zip(
            ids.tolist(), values.tolist(), correct.tolist(), mask.tolist(), strict=True
        ):
            if valid:
                self.pending.append(
                    SampleObservation(
                        sample_id=int(sample_id),
                        margin=float(margin),
                        robust_correct=bool(is_correct),
                        update=int(update),
                        rank=int(rank),
                        order=self._next_order,
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
                )
                continue
            record.margin_ema = self.ema_decay * record.margin_ema + (1.0 - self.ema_decay) * observation.margin
            record.seen += 1
            record.robust_correct_count += int(observation.robust_correct)
            if record.previous_robust_correct is True and not observation.robust_correct:
                record.forgetting_count += 1
            record.previous_robust_correct = observation.robust_correct
            record.last_update = observation.update
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
        if state["format_version"] != self.FORMAT_VERSION:
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
