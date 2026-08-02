"""Immutable L1--L4 inventory binding for H5 point analyses."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ard.analysis.signal_audit import sha256_file

CONTRACT = "h5_confirmatory_cohort_inventory_v1"
LABELS = ("L1", "L2", "L3", "L4")
EXPECTED = {
    "L1": (1, "bartoldson2024_adversarial_wrn94_16"),
    "L2": (1, "chen2021_ltd_wrn34_10"),
    "L3": (2, "bartoldson2024_adversarial_wrn94_16"),
    "L4": (2, "chen2021_ltd_wrn34_10"),
}


class HistoryCohortError(ValueError):
    pass


def _hex(value: object, length: int, name: str) -> str:
    if not isinstance(value, str) or len(value) != length or any(char not in "0123456789abcdef" for char in value):
        raise HistoryCohortError(f"cohort {name} must be a lowercase SHA-{length * 4}")
    return value


def load_cohort_inventory(path: Path) -> tuple[dict[str, dict[str, Any]], str]:
    """Read the exact four-run identity inventory and return its file hash."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HistoryCohortError("cohort inventory is unreadable") from exc
    if not isinstance(value, Mapping) or set(value) != {"schema_version", "contract", "runs"}:
        raise HistoryCohortError("cohort inventory schema drifted")
    runs = value.get("runs")
    if value.get("schema_version") != 1 or value.get("contract") != CONTRACT or not isinstance(runs, Mapping):
        raise HistoryCohortError("cohort inventory contract drifted")
    if set(runs) != set(LABELS):
        raise HistoryCohortError("cohort inventory must contain exactly L1--L4")
    parsed: dict[str, dict[str, Any]] = {}
    ids: set[str] = set()
    for label in LABELS:
        item = runs[label]
        if not isinstance(item, Mapping) or set(item) != {
            "run_id",
            "config_hash",
            "scientific_git_sha",
            "seed",
            "teacher_registry_id",
        }:
            raise HistoryCohortError("cohort run identity schema drifted")
        run_id = item.get("run_id")
        seed, teacher = item.get("seed"), item.get("teacher_registry_id")
        if not isinstance(run_id, str) or not run_id or run_id in ids or (seed, teacher) != EXPECTED[label]:
            raise HistoryCohortError("cohort run label/seed/teacher identity drifted")
        ids.add(run_id)
        parsed[label] = {
            "run_id": run_id,
            "config_hash": _hex(item.get("config_hash"), 64, "config_hash"),
            "scientific_git_sha": _hex(item.get("scientific_git_sha"), 40, "scientific_git_sha"),
            "seed": seed,
            "teacher_registry_id": teacher,
        }
    return parsed, sha256_file(path)


def bind_reports_to_cohort(
    *, inventory: Mapping[str, Mapping[str, Any]], reports: Mapping[str, Mapping[str, Any]]
) -> None:
    """Ensure reports cannot be relabeled, swapped, or mixed across trajectories."""
    if set(reports) != set(LABELS):
        raise HistoryCohortError("H5 point collection requires exactly L1--L4 reports")
    for label in LABELS:
        report = reports[label]
        identity = report.get("input_identity") if isinstance(report, Mapping) else None
        if not isinstance(identity, Mapping):
            raise HistoryCohortError("H5 report lacks cohort-bindable input identity")
        expected = inventory[label]
        observed = {
            "run_id": identity.get("run_id"),
            "config_hash": identity.get("config_hash"),
            "scientific_git_sha": identity.get("scientific_git_sha"),
            "seed": identity.get("seed"),
            "teacher_registry_id": identity.get("teacher_registry_id"),
        }
        if observed != dict(expected):
            raise HistoryCohortError(f"H5 report {label} does not match cohort identity")
