"""Low-overhead, pre-epoch telemetry for pure-ordering probes.

The telemetry is intentionally observational: it records the actual stable
IDs yielded by the train loader and joins them with a frozen pre-epoch risk
snapshot.  It never changes the sampler, model, optimizer, or sample state.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _rank(values: list[float]) -> list[float]:
    """Average-tie ranks, deterministic for finite values."""
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    result = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        rank = (position + end - 1) / 2.0
        for index in order[position:end]:
            result[index] = rank
        position = end
    return result


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    return _pearson(_rank(left), _rank(right))


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    lm, rm = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((a - lm) * (b - rm) for a, b in zip(left, right, strict=True))
    denominator = math.sqrt(
        sum((a - lm) ** 2 for a in left) * sum((b - rm) ** 2 for b in right)
    )
    return None if denominator == 0.0 else numerator / denominator


def _sha_risk(risk: Mapping[int, float]) -> str:
    payload = [[int(sample_id), float(risk[sample_id])] for sample_id in sorted(risk)]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def descriptor_summary(
    batches: list[dict[str, Any]], *, risk_snapshot_epoch: int, risk_snapshot_sha256: str
) -> dict[str, Any]:
    """Compute the preregistered D1--D6 descriptors from one epoch's batches.

    Risk is ``-margin_ema``.  D3 is the SD of per-batch high-risk fractions,
    D4 is lag-1 Pearson autocorrelation of batch mean risk, D5 is the longest
    run among top-20%-risk batches, and D6 is Spearman(batch position, mean
    risk).  All definitions are frozen by the probe registry.
    """
    means = [float(row["risk_mean"]) for row in batches]
    within = [float(row["risk_sd"]) for row in batches]
    high_fraction = [float(row["high_risk_fraction"]) for row in batches]
    if not batches:
        raise ValueError("cannot summarize an empty epoch")
    cutoff = max(1, math.ceil(len(batches) * 0.2))
    ranked = sorted(range(len(means)), key=lambda i: (-means[i], i))
    hard = {i for i in ranked[:cutoff]}
    longest = current = 0
    for index in range(len(means)):
        if index in hard:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    lag = _pearson(means[:-1], means[1:]) if len(means) >= 3 else None
    return {
        "schema_version": 1,
        "risk_definition": "-margin_ema",
        "risk_snapshot_epoch": int(risk_snapshot_epoch),
        "risk_snapshot_sha256": risk_snapshot_sha256,
        "batch_count": len(batches),
        "D1_batch_mean_risk_sd": statistics.pstdev(means) if len(means) > 1 else 0.0,
        "D2_within_batch_risk_sd_mean": statistics.fmean(within),
        "D3_high_risk_fraction_sd": statistics.pstdev(high_fraction) if len(high_fraction) > 1 else 0.0,
        "D4_lag1_batch_mean_risk_acf": lag,
        "D5_hard_batch_longest_run": longest,
        "D6_position_vs_batch_mean_risk_spearman": _spearman(list(range(len(means))), means),
    }


class OrderingTelemetry:
    """Collect actual batches and write compact JSONL telemetry."""

    def __init__(self, output_dir: Path, *, high_risk_fraction: float = 0.2) -> None:
        if not 0.0 < high_risk_fraction < 1.0:
            raise ValueError("high_risk_fraction must be between zero and one")
        self.output_dir = output_dir
        self.high_risk_fraction = high_risk_fraction
        self.batch_path = output_dir / "batch-order-metrics.jsonl"
        self.descriptor_path = output_dir / "ordering-descriptors.jsonl"
        self._epoch: int | None = None
        self._risk: dict[int, float] = {}
        self._high_ids: set[int] = set()
        self._rows: list[dict[str, Any]] = []
        self._order: list[int] = []
        self._snapshot_epoch = -1
        self._snapshot_sha = ""
        self.last_descriptor: dict[str, Any] | None = None

    def _begin(self, epoch: int, sample_store: Any) -> None:
        records = getattr(sample_store, "records", None)
        if not isinstance(records, Mapping) or not records:
            raise ValueError("ordering telemetry requires a populated sample-state store")
        risk: dict[int, float] = {}
        for sample_id, record in records.items():
            margin = float(getattr(record, "margin_ema"))
            if not math.isfinite(margin):
                raise ValueError("ordering telemetry risk snapshot contains non-finite margin_ema")
            risk[int(sample_id)] = -margin
        ordered = sorted(risk, key=lambda sample_id: (-risk[sample_id], sample_id))
        count = max(1, math.ceil(len(ordered) * self.high_risk_fraction))
        self._risk = risk
        self._high_ids = set(ordered[:count])
        self._epoch = int(epoch)
        self._rows = []
        self._order = []
        self._snapshot_epoch = epoch - 1
        self._snapshot_sha = _sha_risk(risk)

    def on_batch(self, epoch: int, batch_index: int, batch: Any, sample_store: Any) -> None:
        if self._epoch != epoch:
            self._begin(epoch, sample_store)
        ids = [int(value) for value in batch.sample_ids.detach().cpu().tolist()]
        mask = getattr(batch, "state_update_mask", None)
        valid = [True] * len(ids) if mask is None else [bool(value) for value in mask.detach().cpu().tolist()]
        valid_ids = [sample_id for sample_id, is_valid in zip(ids, valid, strict=True) if is_valid]
        missing = [sample_id for sample_id in valid_ids if sample_id not in self._risk]
        if missing:
            raise ValueError(f"ordering telemetry encountered unknown stable IDs: {missing[:4]}")
        values = [self._risk[sample_id] for sample_id in valid_ids]
        self._order.extend(valid_ids)
        self._rows.append(
            {
                "epoch": int(epoch),
                "batch_index": int(batch_index),
                "sample_ids": ids,
                "valid_mask": valid,
                "valid_sample_ids": valid_ids,
                "risk_mean": statistics.fmean(values) if values else 0.0,
                "risk_sd": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "high_risk_fraction": (
                    sum(sample_id in self._high_ids for sample_id in valid_ids) / len(valid_ids)
                    if valid_ids
                    else 0.0
                ),
                "risk_snapshot_epoch": self._snapshot_epoch,
                "risk_snapshot_sha256": self._snapshot_sha,
            }
        )

    def finish_epoch(self, epoch: int) -> None:
        if self._epoch != epoch:
            raise ValueError(f"cannot finish unobserved telemetry epoch {epoch}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.batch_path.open("a", encoding="utf-8") as handle:
            for row in self._rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        descriptors = descriptor_summary(
            self._rows, risk_snapshot_epoch=self._snapshot_epoch, risk_snapshot_sha256=self._snapshot_sha
        )
        descriptors.update(
            {
                "epoch": int(epoch),
                "order_permutation_sha256": hashlib.sha256(
                    json.dumps(self._order, separators=(",", ":")).encode()
                ).hexdigest(),
                "valid_sample_count": len(self._order),
            }
        )
        self.last_descriptor = descriptors
        with self.descriptor_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(descriptors, sort_keys=True, separators=(",", ":")) + "\n")
