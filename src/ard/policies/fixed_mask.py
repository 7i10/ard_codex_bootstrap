"""Immutable train-ID masks used by the registered factorial intervention screen."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch


class FixedMaskError(RuntimeError):
    """A mask cannot prove its exact train-only identity."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_ids_sha256(selected_ids: tuple[int, ...]) -> str:
    return hashlib.sha256(_canonical_json(list(selected_ids)).encode()).hexdigest()


@dataclass(frozen=True)
class FixedInterventionMask:
    """A binary lookup whose manifest proves fixed IDs and class budget."""

    selected_ids: frozenset[int]
    selected_ids_digest: str
    class_counts: Mapping[int, int]

    def values(self, sample_ids: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if sample_ids.ndim != 1:
            raise FixedMaskError("intervention mask sample IDs must be one-dimensional")
        values: list[float] = []
        for raw_id in sample_ids.detach().cpu().tolist():
            if isinstance(raw_id, bool) or not isinstance(raw_id, int):
                raise FixedMaskError("intervention mask received a non-integer stable sample ID")
            values.append(float(raw_id in self.selected_ids))
        return torch.tensor(values, device=device, dtype=dtype)


def load_fixed_intervention_mask(
    path: Path,
    *,
    expected_sha256: str,
    expected_selected_ids_sha256: str,
    expected_selected_count: int,
    expected_class_counts: Mapping[str, int],
    expected_provenance: Mapping[str, object],
    train_labels: Mapping[int, int],
    num_classes: int,
) -> FixedInterventionMask:
    """Load one strict, train-only mask and verify all configured identities."""
    if not path.is_file() or _sha256_file(path) != expected_sha256:
        raise FixedMaskError("intervention mask bytes do not match the configured SHA-256")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FixedMaskError("intervention mask is not readable canonical JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "namespace",
        "num_classes",
        "selected_ids",
        "selected_ids_sha256",
        "selected_count",
        "selected_class_counts",
        "provenance",
    }:
        raise FixedMaskError("intervention mask has unexpected or missing fields")
    if payload["schema_version"] != 1 or payload["namespace"] != "train" or payload["num_classes"] != num_classes:
        raise FixedMaskError("intervention mask is not an exact train-namespace class-compatible input")
    raw_ids = payload["selected_ids"]
    if not isinstance(raw_ids, list) or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_ids):
        raise FixedMaskError("intervention mask selected_ids must be integer IDs")
    selected_ids = tuple(raw_ids)
    if tuple(sorted(selected_ids)) != selected_ids or len(set(selected_ids)) != len(selected_ids):
        raise FixedMaskError("intervention mask selected IDs must be sorted and unique")
    if any(sample_id not in train_labels for sample_id in selected_ids):
        raise FixedMaskError("intervention mask contains an ID outside the exact train partition")
    digest = selected_ids_sha256(selected_ids)
    if payload["selected_ids_sha256"] != digest or digest != expected_selected_ids_sha256:
        raise FixedMaskError("intervention mask selected ID digest does not match")
    if payload["selected_count"] != len(selected_ids) or len(selected_ids) != expected_selected_count:
        raise FixedMaskError("intervention mask selected count does not match")
    class_counts = Counter(train_labels[sample_id] for sample_id in selected_ids)
    canonical_counts = {str(class_id): class_counts[class_id] for class_id in sorted(class_counts)}
    if payload["selected_class_counts"] != canonical_counts or dict(expected_class_counts) != canonical_counts:
        raise FixedMaskError("intervention mask selected class counts do not match")
    provenance = payload["provenance"]
    if not isinstance(provenance, dict) or provenance != dict(expected_provenance):
        raise FixedMaskError("intervention mask provenance does not match the hash-bound arm configuration")
    if provenance.get("source") not in {
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
    }:
        raise FixedMaskError("intervention mask provenance source is forbidden or unknown")
    return FixedInterventionMask(
        selected_ids=frozenset(selected_ids),
        selected_ids_digest=digest,
        class_counts={int(class_id): count for class_id, count in canonical_counts.items()},
    )
