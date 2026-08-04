"""CPU-only exact-online state export for one preregistered PRE39 anchor.

This is intentionally distinct from the frozen H5 online-state contract.  It
exports one candidate anchor only after the replay-domain screen nominates it.
"""

from __future__ import annotations

import math
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ard.analysis.h4a_taxonomy import _domain_panel, _lineage
from ard.analysis.rslad_signal_replay import FEATURE_EPOCHS, canonical_json, repository_root_from_source
from ard.analysis.sample_stats import write_sample_parquet
from ard.analysis.signal_audit import sha256_file
from ard.engine.checkpoint import REQUIRED_KEYS


class Pre39OnlineStateError(ValueError):
    """Raised when one candidate state cannot prove exact online lineage."""


ANCHORS = (9, 14, 19, 24, 29, 34)
CONTRACT = "pre39_online_state_candidate_v1"


@dataclass(frozen=True)
class Pre39OnlineStateExport:
    lineage: dict[str, Any]
    rows: tuple[dict[str, Any], ...]


def _provenance() -> dict[str, Any]:
    root = repository_root_from_source()
    paths = {
        "pre39_online_state": Path(__file__).resolve(),
        "pre39_online_state_cli": root / "src/ard/cli/pre39_online_state.py",
    }
    try:
        relative = [str(path.relative_to(root)) for path in paths.values()]
        subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", *relative], check=True, capture_output=True
        )
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise Pre39OnlineStateError("PRE39 online candidate requires tracked source files and Git identity") from exc
    if len(sha) != 40 or dirty:
        raise Pre39OnlineStateError("PRE39 online candidate requires a tracked-clean revision")
    return {
        "git": {"sha": sha, "dirty": False},
        "source_files": {name: sha256_file(path) for name, path in paths.items()},
    }


def _read_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq

        return [dict(row) for row in pq.read_table(path).to_pylist()]
    except Exception as exc:  # pragma: no cover - Arrow exception varies
        raise Pre39OnlineStateError("schema-v2 feature replay is unreadable") from exc


def _finite(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not -1 <= float(value) <= 1
    ):
        raise Pre39OnlineStateError(f"{name} is outside contract")
    return float(value)


def _state_count(record: Mapping[str, Any], name: str) -> int:
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Pre39OnlineStateError(f"sample-state {name} is invalid")
    return value


def export_pre39_online_state(
    *,
    checkpoint: Path,
    feature_observations: Path,
    feature_lineage: Path,
    anchor: int,
    expected_count: int = 45_000,
    analysis_provenance: Mapping[str, Any] | None = None,
) -> Pre39OnlineStateExport:
    """Validate and export one inclusive online SampleStateStore snapshot."""
    if (
        anchor not in ANCHORS
        or isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count < 1
    ):
        raise Pre39OnlineStateError("candidate anchor/count contract is invalid")
    provenance = dict(_provenance() if analysis_provenance is None else analysis_provenance)
    try:
        feature_meta = _lineage(
            feature_lineage,
            feature_observations,
            key="feature_observations_sha256",
            expected_count=expected_count,
            protocol="feature_protocol",
        )
        feature = _domain_panel(
            _read_rows(feature_observations), epochs=FEATURE_EPOCHS, expected_count=expected_count, name="feature"
        )
    except ValueError as exc:
        raise Pre39OnlineStateError(str(exc)) from exc
    inventory = [
        item
        for item in feature_meta.get("checkpoints", [])
        if isinstance(item, Mapping) and item.get("epoch") == anchor
    ]
    if len(inventory) != 1 or not isinstance(inventory[0].get("sha256"), str):
        raise Pre39OnlineStateError("feature lineage lacks exactly one candidate checkpoint inventory entry")
    if not checkpoint.is_file() or sha256_file(checkpoint) != inventory[0]["sha256"]:
        raise Pre39OnlineStateError("candidate checkpoint SHA does not match feature lineage inventory")
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover - torch error varies
        raise Pre39OnlineStateError("candidate checkpoint is unreadable") from exc
    if (
        not isinstance(payload, Mapping)
        or REQUIRED_KEYS.difference(payload)
        or payload.get("epoch") != anchor
        or payload.get("epoch_boundary") != "end"
    ):
        raise Pre39OnlineStateError("candidate checkpoint format/epoch boundary drifted")
    run_id, config_hash, world_size = (
        payload.get("tracker_run_id"),
        payload.get("config_hash"),
        payload.get("world_size"),
    )
    if (
        not isinstance(run_id, str)
        or not isinstance(config_hash, str)
        or isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size < 1
    ):
        raise Pre39OnlineStateError("candidate checkpoint run/config/world identity is invalid")
    if run_id != feature_meta["run_id"] or config_hash != feature_meta["config_hash"]:
        raise Pre39OnlineStateError("candidate checkpoint and feature lineage run/config drifted")
    state = payload.get("sample_state")
    records = state.get("records") if isinstance(state, Mapping) else None
    if (
        not isinstance(state, Mapping)
        or state.get("format_version") != 3
        or state.get("pending") != []
        or not isinstance(records, Mapping)
    ):
        raise Pre39OnlineStateError("candidate checkpoint lacks complete format-v3 settled sample state")
    if len(records) != expected_count:
        raise Pre39OnlineStateError("candidate sample-state record count differs from expected_count")
    rows: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for raw_id, raw in records.items():
        if not isinstance(raw_id, str) or not raw_id.isdigit() or not isinstance(raw, Mapping):
            raise Pre39OnlineStateError("candidate sample-state stable ID/record is invalid")
        sample_id = int(raw_id)
        label = raw.get("true_label")
        if (
            sample_id in seen_ids
            or isinstance(label, bool)
            or not isinstance(label, int)
            or not 0 <= label < 10
            or sample_id not in feature[anchor]
            or feature[anchor][sample_id]["class_id"] != label
        ):
            raise Pre39OnlineStateError("candidate sample-state/feature sparse ID/class join drifted")
        seen, hits = _state_count(raw, "seen"), _state_count(raw, "robust_correct_count")
        current = raw.get("previous_robust_correct")
        if seen != anchor + 1 or hits > seen or not isinstance(current, bool):
            raise Pre39OnlineStateError("candidate inclusive seen/current correctness contract drifted")
        seen_ids.add(sample_id)
        rows.append(
            {
                "namespace": "train",
                "sample_id": sample_id,
                "class_id": label,
                "anchor_epoch": anchor,
                "robust_correct_count": hits,
                "robust_correct_frequency_inclusive": hits / seen,
                "margin_ema": _finite(raw.get("margin_ema"), name="margin EMA"),
                "last_margin": _finite(raw.get("last_margin"), name="last margin"),
                "current_robust_correct": current,
            }
        )
    if seen_ids != set(feature[anchor]):
        raise Pre39OnlineStateError("candidate sample-state and feature replay stable ID sets drifted")
    lineage = {
        "schema_version": 1,
        "contract": CONTRACT,
        "run_id": run_id,
        "config_hash": config_hash,
        "world_size": world_size,
        "scientific_git_sha": feature_meta["scientific_git_sha"],
        "seed": feature_meta.get("seed"),
        "teacher": feature_meta["teacher"],
        "dataset_identity": feature_meta["dataset_identity"],
        "attack_identity": feature_meta["attack_identity"],
        "anchor_epoch": anchor,
        "expected_count": expected_count,
        "row_count": len(rows),
        "checkpoint_sha256": inventory[0]["sha256"],
        "feature_observations_sha256": sha256_file(feature_observations),
        "feature_lineage_sha256": sha256_file(feature_lineage),
        "analysis_provenance": provenance,
    }
    return Pre39OnlineStateExport(lineage=lineage, rows=tuple(sorted(rows, key=lambda row: int(row["sample_id"]))))


def write_pre39_online_state(*, output_dir: Path, export: Pre39OnlineStateExport) -> dict[str, Path]:
    paths = {"observations": output_dir / "pre39-online-state.parquet", "lineage": output_dir / "lineage.json"}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("refusing to overwrite PRE39 online candidate export")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_sample_parquet(export.rows, paths["observations"])
    paths["lineage"].write_bytes(
        canonical_json({**export.lineage, "observations_sha256": sha256_file(paths["observations"])}) + b"\n"
    )
    return paths
