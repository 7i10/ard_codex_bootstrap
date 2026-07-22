"""Transparent aggregation for teacher/run comparisons."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

_TRAINING_SEED_FIELDS = frozenset(
    {
        "split",
        "model_init",
        "data_order",
        "augmentation",
        "train_attack",
        "evaluation_attack",
        "qualitative_panel",
    }
)


def _canonical_identity(value: object) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    materialized = [float(value) for value in values]
    if not materialized or any(not math.isfinite(value) for value in materialized):
        raise ValueError("aggregate values must be non-empty and finite")
    mean = sum(materialized) / len(materialized)
    variance = (
        0.0 if len(materialized) == 1 else sum((value - mean) ** 2 for value in materialized) / (len(materialized) - 1)
    )
    return {
        "count": len(materialized),
        "mean": mean,
        "std": math.sqrt(variance),
        "worst": min(materialized),
        "best": max(materialized),
    }


def summarize_checkpoint_groups(rows: Iterable[Mapping[str, Any]], *, metric: str) -> dict[str, dict[str, float | int]]:
    """Aggregate best/last separately and reject incompatible threat models."""
    groups: dict[str, list[float]] = {}
    threats: dict[str, set[str]] = {}
    identities: set[tuple[str, ...]] = set()
    checkpoint_counts: Counter[tuple[str, str]] = Counter()
    run_metadata: dict[str, tuple[str, ...]] = {}
    required = {
        "train_run_id",
        "training_seeds",
        "evaluation_seed",
        "dataset_identity",
        "training_dataset_identity",
        "student_identity",
        "method_identity",
        "teacher_identity",
        "training_protocol_identity",
        "evaluation_protocol_identity",
        "threat_hash",
        "checkpoint_alias",
        "checkpoint_filename",
        "checkpoint_sha256",
    }
    for row in rows:
        missing = required.difference(row)
        if missing:
            raise ValueError("canonical evaluation result is missing: " + ", ".join(sorted(missing)))
        checkpoint = str(row["checkpoint_alias"])
        if checkpoint not in {"best", "last"}:
            raise ValueError("checkpoint group must be best or last")
        training_seeds = row["training_seeds"]
        if (
            not isinstance(training_seeds, Mapping)
            or set(training_seeds) != _TRAINING_SEED_FIELDS
            or any(isinstance(value, bool) or not isinstance(value, int) for value in training_seeds.values())
        ):
            raise ValueError("canonical evaluation training_seeds must be the complete seven-field mapping")
        groups.setdefault(checkpoint, []).append(float(row[metric]))
        if not isinstance(row["checkpoint_filename"], str) or not isinstance(row["checkpoint_sha256"], str):
            raise ValueError("canonical evaluation checkpoint identity is invalid")
        if len(row["checkpoint_sha256"]) != 64 or any(
            character not in "0123456789abcdef" for character in row["checkpoint_sha256"]
        ):
            raise ValueError("canonical evaluation checkpoint SHA-256 is invalid")
        threat = row["threat_hash"]
        threats.setdefault(checkpoint, set()).add(str(threat))
        identities.add(
            tuple(
                _canonical_identity(row.get(key, {}))
                for key in (
                    "dataset_identity",
                    "training_dataset_identity",
                    "student_identity",
                    "method_identity",
                    "training_protocol_identity",
                    "evaluation_protocol_identity",
                )
            )
            + (str(threat), str(row.get("evaluation_seed", "")))
        )
        train_run_id = str(row.get("train_run_id", ""))
        metadata = (
            _canonical_identity(training_seeds),
            _canonical_identity(row.get("teacher_identity", {})),
            _canonical_identity(row.get("student_identity", {})),
            _canonical_identity(row.get("method_identity", {})),
            _canonical_identity(row.get("training_protocol_identity", {})),
            _canonical_identity(row.get("evaluation_protocol_identity", {})),
        )
        if train_run_id in run_metadata and run_metadata[train_run_id] != metadata:
            raise ValueError("same train_run_id has contradictory metadata")
        run_metadata[train_run_id] = metadata
        checkpoint_counts[(train_run_id, checkpoint)] += 1
    if any(len(models) != 1 for models in threats.values()) or len(set().union(*threats.values())) != 1:
        raise ValueError("cannot aggregate mixed threat models")
    if set(groups) != {"best", "last"}:
        raise ValueError("analysis requires exactly best and last checkpoint aliases")
    if len(identities) != 1:
        raise ValueError("cannot aggregate mixed experiment identities")
    if any(
        checkpoint_counts[(train_run_id, checkpoint)] != 1
        for train_run_id in run_metadata
        for checkpoint in ("best", "last")
    ):
        raise ValueError("each train run requires exactly one best and one last checkpoint")
    return {checkpoint: summarize(values) for checkpoint, values in sorted(groups.items())}
