"""CPU-only, hash-bound online SampleStateStore anchors for revised H5."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ard.analysis.rslad_signal_replay import canonical_json, repository_root_from_source
from ard.analysis.sample_stats import write_sample_parquet
from ard.analysis.signal_audit import sha256_file


class HistoryOnlineStateError(ValueError):
    pass


ANCHORS = (39, 59, 79)
CONTRACT = "h5_online_state_anchor_v1"


@dataclass(frozen=True)
class OnlineStateExport:
    lineage: dict[str, Any]
    rows: tuple[dict[str, Any], ...]


def _provenance() -> dict[str, Any]:
    root = repository_root_from_source()
    paths = {
        "history_online_state": Path(__file__).resolve(),
        "history_online_state_cli": root / "src/ard/cli/history_online_state.py",
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
        raise HistoryOnlineStateError("online anchor exporter requires tracked source files and Git identity") from exc
    if len(sha) != 40 or dirty:
        raise HistoryOnlineStateError("online anchor exporter requires a tracked-clean revision")
    return {
        "git": {"sha": sha, "dirty": False},
        "source_files": {key: sha256_file(path) for key, path in paths.items()},
    }


def _sha(path: Path) -> str:
    if not path.is_file():
        raise HistoryOnlineStateError("checkpoint is missing")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: object, name: str, lo: float = -1.0, hi: float = 1.0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not lo <= float(value) <= hi
    ):
        raise HistoryOnlineStateError(f"{name} is outside contract")
    return float(value)


def _counter(record: Mapping[str, Any], field: str) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HistoryOnlineStateError(f"sample-state {field} must be a nonnegative integer")
    return value


def _lineage(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HistoryOnlineStateError("replay lineage is unreadable") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or not isinstance(value.get("checkpoints"), list)
    ):
        raise HistoryOnlineStateError("replay lineage contract is invalid")
    for key in (
        "run_id",
        "config_hash",
        "scientific_git_sha",
        "seed",
        "attack_identity",
        "dataset_identity",
        "teacher",
    ):
        if key not in value:
            raise HistoryOnlineStateError("replay lineage identity is incomplete")
    return value


def export_online_anchors(
    *,
    checkpoints: Mapping[int, Path],
    replay_lineage: Path,
    expected_count: int = 45000,
    analysis_provenance: Mapping[str, Any] | None = None,
) -> OnlineStateExport:
    """Export only the inclusive online state at 39/59/79; never infer images."""
    expected_epochs = ANCHORS
    if expected_count < 1 or tuple(sorted(checkpoints)) != expected_epochs:
        raise HistoryOnlineStateError("online state exporter checkpoint schedule does not match its declared mode")
    replay = _lineage(replay_lineage)
    inventory = {item.get("epoch"): item for item in replay["checkpoints"] if isinstance(item, Mapping)}
    if any(epoch not in inventory for epoch in expected_epochs):
        raise HistoryOnlineStateError("replay lineage lacks one required anchor checkpoint")
    rows: list[dict[str, Any]] = []
    reference_ids: set[int] | None = None
    reference_labels: dict[int, int] | None = None
    identity: tuple[str, str, int] | None = None
    checkpoint_meta = []
    for epoch in expected_epochs:
        path = checkpoints[epoch]
        digest = _sha(path)
        item = inventory[epoch]
        if item.get("sha256") != digest:
            raise HistoryOnlineStateError("checkpoint SHA does not match replay lineage inventory")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise HistoryOnlineStateError("checkpoint is unreadable") from exc
        if not isinstance(payload, Mapping) or payload.get("epoch") != epoch or payload.get("epoch_boundary") != "end":
            raise HistoryOnlineStateError("checkpoint epoch/boundary contract drifted")
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
        ):
            raise HistoryOnlineStateError("checkpoint run/config/world identity is invalid")
        if identity is None:
            identity = (run_id, config_hash, world_size)
        elif identity != (run_id, config_hash, world_size):
            raise HistoryOnlineStateError("anchor checkpoints do not share run/config/world identity")
        state = payload.get("sample_state")
        records = state.get("records") if isinstance(state, Mapping) else None
        if (
            not isinstance(state, Mapping)
            or state.get("format_version") != 3
            or state.get("pending") != []
            or not isinstance(records, Mapping)
        ):
            raise HistoryOnlineStateError("checkpoint lacks complete format-v3 epoch-boundary sample state")
        if len(records) != expected_count:
            raise HistoryOnlineStateError("sample-state record count differs from expected count")
        ids: set[int] = set()
        labels: dict[int, int] = {}
        for raw_id, raw in records.items():
            if not isinstance(raw_id, str) or not raw_id.isdigit() or not isinstance(raw, Mapping):
                raise HistoryOnlineStateError("sample-state stable ID/record is invalid")
            sample_id = int(raw_id)
            label = raw.get("true_label")
            if sample_id in ids or isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < 10:
                raise HistoryOnlineStateError("sample-state ID/label contract drifted")
            seen, hits = _counter(raw, "seen"), _counter(raw, "robust_correct_count")
            if seen != epoch + 1 or hits > seen or not isinstance(raw.get("previous_robust_correct"), bool):
                raise HistoryOnlineStateError("sample-state inclusive seen/correctness contract drifted")
            ids.add(sample_id)
            labels[sample_id] = label
            rows.append(
                {
                    "namespace": "train",
                    "sample_id": sample_id,
                    "anchor_epoch": epoch,
                    "true_label": label,
                    "robust_correct_count": hits,
                    "previous_robust_correct": raw["previous_robust_correct"],
                    "margin_ema": _finite(raw.get("margin_ema"), "margin EMA"),
                    "last_margin": _finite(raw.get("last_margin"), "last margin"),
                    "robust_correct_frequency_inclusive": hits / seen,
                }
            )
        if reference_ids is None:
            reference_ids, reference_labels = ids, labels
        elif ids != reference_ids or labels != reference_labels:
            raise HistoryOnlineStateError("anchor checkpoints do not share exact stable ID/class mapping")
        checkpoint_meta.append({"epoch": epoch, "sha256": digest})
    assert identity is not None
    if identity[:2] != (replay["run_id"], replay["config_hash"]):
        raise HistoryOnlineStateError("anchor checkpoint and replay lineage run/config identity drifted")
    lineage = {
        "schema_version": 1,
        "contract": CONTRACT,
        "run_id": identity[0],
        "config_hash": identity[1],
        "world_size": identity[2],
        "scientific_git_sha": replay["scientific_git_sha"],
        "seed": replay["seed"],
        "attack_identity": replay["attack_identity"],
        "dataset_identity": replay["dataset_identity"],
        "teacher": replay["teacher"],
        "expected_count": expected_count,
        "row_count": len(rows),
        "checkpoints": checkpoint_meta,
        "replay_lineage_sha256": sha256_file(replay_lineage),
        "analysis_provenance": dict(_provenance() if analysis_provenance is None else analysis_provenance),
    }
    return OnlineStateExport(
        lineage=lineage, rows=tuple(sorted(rows, key=lambda row: (int(row["anchor_epoch"]), int(row["sample_id"]))))
    )


def write_online_anchors(*, output_dir: Path, export: OnlineStateExport) -> dict[str, Path]:
    paths = {"observations": output_dir / "online-anchor-states.parquet", "lineage": output_dir / "lineage.json"}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("refusing to overwrite online anchor export")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_sample_parquet(export.rows, paths["observations"])
    paths["lineage"].write_bytes(
        canonical_json({**export.lineage, "observations_sha256": sha256_file(paths["observations"])}) + b"\n"
    )
    return paths
