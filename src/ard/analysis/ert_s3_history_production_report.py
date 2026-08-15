"""Aggregate the preregistered ERT history-production screen."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


class HistoryProductionReportError(RuntimeError):
    """The production screen report lineage or input contract is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise HistoryProductionReportError(f"expected JSON object: {path}")
    return value


def _transition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_id.setdefault(int(row["sample_id"]), []).append(row)
    switches = entries = exits = reentries = teacher_only = student_only = 0
    durations: list[int] = []
    active_fraction: list[float] = []
    epoch_active: Counter[int] = Counter()
    epoch_count: Counter[int] = Counter()
    for sequence in by_id.values():
        sequence.sort(key=lambda row: int(row["epoch"]))
        actions = [bool(row["action_active"]) for row in sequence]
        teacher = [bool(row["teacher_adv_correct"]) for row in sequence]
        states = [bool(row.get("history_state_active", row["current_active"])) for row in sequence]
        had_active = False
        run = 0
        for index, active in enumerate(actions):
            if active:
                run += 1
            elif run:
                durations.append(run)
                run = 0
            if index:
                changed = actions[index] != actions[index - 1]
                state_changed = states[index] != states[index - 1]
                teacher_changed = teacher[index] != teacher[index - 1]
                if changed:
                    switches += 1
                    entries += int(not actions[index - 1] and active)
                    exits += int(actions[index - 1] and not active)
                    if not actions[index - 1] and active and had_active:
                        reentries += 1
                    teacher_only += int(changed and teacher_changed and not state_changed)
                    student_only += int(changed and state_changed and not teacher_changed)
            had_active = had_active or active
        if run:
            durations.append(run)
    for row in rows:
        epoch = int(row["epoch"])
        epoch_count[epoch] += 1
        epoch_active[epoch] += int(bool(row["action_active"]))
    active_fraction = [epoch_active[e] / epoch_count[e] for e in sorted(epoch_count)]
    return {
        "switches": switches,
        "entries": entries,
        "exits": exits,
        "reentries": reentries,
        "student_state_only_switches": student_only,
        "teacher_only_switches": teacher_only,
        "teacher_only_switch_share": teacher_only / switches if switches else 0.0,
        "active_fraction_mean": sum(active_fraction) / len(active_fraction) if active_fraction else 0.0,
        "active_fraction_by_epoch": {str(e): epoch_active[e] / epoch_count[e] for e in sorted(epoch_count)},
        "median_active_duration": sorted(durations)[len(durations) // 2] if durations else 0,
        "max_active_duration": max(durations, default=0),
        "row_count": len(rows),
    }


def _endpoint(path: Path) -> dict[str, Any]:
    metadata = _json(path / "endpoint.json")
    rows_path = path / "endpoint-sample-stats.parquet"
    if metadata.get("rows_sha256") != _sha256(rows_path):
        raise HistoryProductionReportError(f"endpoint row hash mismatch: {path}")
    return metadata


def build_report(*, config_path: Path, training_root: Path, endpoint_root: Path, output: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("contract") != "ert_s3_history_production_v1":
        raise HistoryProductionReportError("wrong production config contract")
    arms = tuple(config.get("arms", ()))
    if arms != ("BASE", "INST075", "M3_075", "M3E2_075"):
        raise HistoryProductionReportError("production report requires the frozen four-arm screen")
    horizons = tuple(int(x) for x in config.get("horizons", ()))
    if horizons != (84, 89, 94):
        raise HistoryProductionReportError("production report requires horizons 84/89/94")
    source = __import__("ard.tracking.adapter", fromlist=["collect_git_state"]).collect_git_state(Path.cwd())
    if source.get("dirty") is not False or not isinstance(source.get("sha"), str):
        raise HistoryProductionReportError("report requires a clean source tree")
    report: dict[str, Any] = {
        "schema_version": 1,
        "contract": "ert_s3_history_production_report_v1",
        "source_git_sha": source["sha"],
        "config_sha256": _sha256(config_path),
        "training_root": str(training_root.resolve()),
        "endpoint_root": str(endpoint_root.resolve()),
        "seeds": {},
    }
    for run_key in ("L2", "L4"):
        run = config["runs"][run_key]
        parent = Path(str(run["parent_checkpoint"]))
        if not parent.is_file():
            raise HistoryProductionReportError(f"missing parent checkpoint: {parent}")
        seed_result: dict[str, Any] = {"parent_checkpoint_sha256": _sha256(parent), "arms": {}, "horizons": {}}
        for arm in arms:
            arm_dir = training_root / run_key / arm
            state_path = arm_dir / "dynamic-state.parquet"
            manifest_path = arm_dir / "dynamic-state-manifest.json"
            if not state_path.is_file() or not manifest_path.is_file():
                raise HistoryProductionReportError(f"missing training state artifact: {arm_dir}")
            rows = pq.read_table(state_path).to_pylist()
            seed_result["arms"][arm] = {
                "training_state": {"path": str(state_path.resolve()), "sha256": _sha256(state_path), "rows": len(rows)},
                "transition": _transition(rows),
                "manifest_sha256": _sha256(manifest_path),
            }
        for horizon in horizons:
            horizon_result: dict[str, Any] = {"arms": {}, "deltas_vs_base": {}}
            base: dict[str, dict[str, Any]] = {}
            for arm in arms:
                for split in ("train", "validation"):
                    directory = endpoint_root / run_key / arm / f"epoch-{horizon}" / split
                    base[arm, split] = _endpoint(directory)
                horizon_result["arms"][arm] = {
                    "train_clean_accuracy": base[arm, "train"]["clean_accuracy"],
                    "train_robust_accuracy": base[arm, "train"]["robust_accuracy"],
                    "validation_clean_accuracy": base[arm, "validation"]["clean_accuracy"],
                    "validation_robust_accuracy": base[arm, "validation"]["robust_accuracy"],
                }
            for arm in arms[1:]:
                arm_metrics = horizon_result["arms"][arm]
                base_metrics = horizon_result["arms"]["BASE"]
                horizon_result["deltas_vs_base"][arm] = {
                    "train_clean": arm_metrics["train_clean_accuracy"] - base_metrics["train_clean_accuracy"],
                    "train_robust": arm_metrics["train_robust_accuracy"] - base_metrics["train_robust_accuracy"],
                    "validation_clean": (
                        arm_metrics["validation_clean_accuracy"] - base_metrics["validation_clean_accuracy"]
                    ),
                    "validation_robust": (
                        arm_metrics["validation_robust_accuracy"] - base_metrics["validation_robust_accuracy"]
                    ),
                }
            seed_result["horizons"][str(horizon)] = horizon_result
        report["seeds"][run_key] = seed_result
    if output.exists():
        raise HistoryProductionReportError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["output_sha256"] = _sha256(output)
    return report
