"""Deterministic, resumable paired bootstrap for validated H5 point tasks."""

from __future__ import annotations

import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from ard.analysis.rslad_signal_replay import canonical_json, repository_root_from_source
from ard.analysis.signal_audit import _bootstrap_indices, binary_metrics, sha256_file

SEED = 2026073102
REPLICATES = 2000
CONTRACT = "h5_paired_bootstrap_v1"
_WORKER_TASK: dict[str, Any] | None = None


class HistoryBootstrapError(ValueError):
    pass


def _source_fingerprint() -> dict[str, str]:
    root = repository_root_from_source()
    paths = {
        "analysis": Path(__file__).resolve(),
        "cli": root / "src/ard/cli/history_bootstrap.py",
        "signal_audit": root / "src/ard/analysis/signal_audit.py",
    }
    if any(not path.is_file() for path in paths.values()):
        raise HistoryBootstrapError("bootstrap source tree is incomplete")
    return {name: sha256_file(path) for name, path in paths.items()}


def _finite(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(float(value)):
        raise HistoryBootstrapError("bootstrap task score must be finite")
    return float(value)


def _validate_task(task: object) -> dict[str, Any]:
    if not isinstance(task, dict):
        raise HistoryBootstrapError("bootstrap task must be an object")
    required = {"task_id", "run", "anchor", "outcome", "stratum", "point_gate_pass", "joint_primary_gate", "rows"}
    if set(task) != required or task["point_gate_pass"] is not True:
        raise HistoryBootstrapError("bootstrap task schema/point gate drifted")
    if (
        not isinstance(task["task_id"], str)
        or not isinstance(task["run"], str)
        or task["run"] not in {"L1", "L3"}
        or isinstance(task["anchor"], bool)
        or not isinstance(task["anchor"], int)
        or not isinstance(task["outcome"], str)
        or not isinstance(task["stratum"], str)
        or not isinstance(task["joint_primary_gate"], str)
        or not isinstance(task["rows"], list)
    ):
        raise HistoryBootstrapError("bootstrap task identity is invalid")
    ids: set[int] = set()
    rows: list[dict[str, Any]] = []
    for row in task["rows"]:
        if not isinstance(row, dict) or set(row) != {"sample_id", "class_id", "outcome", "baseline", "candidate"}:
            raise HistoryBootstrapError("bootstrap task row schema drifted")
        sample_id, class_id = row["sample_id"], row["class_id"]
        if (
            isinstance(sample_id, bool)
            or not isinstance(sample_id, int)
            or sample_id < 0
            or sample_id in ids
            or isinstance(class_id, bool)
            or not isinstance(class_id, int)
            or not 0 <= class_id < 10
            or row["outcome"] not in {0, 1, False, True}
        ):
            raise HistoryBootstrapError("bootstrap task sample/class/outcome contract drifted")
        ids.add(sample_id)
        rows.append(
            {
                "sample_id": sample_id,
                "class_id": class_id,
                "outcome": int(row["outcome"]),
                "baseline": _finite(row["baseline"]),
                "candidate": _finite(row["candidate"]),
            }
        )
    if not rows or not any(row["outcome"] for row in rows) or all(row["outcome"] for row in rows):
        raise HistoryBootstrapError("bootstrap task outcome must contain both classes")
    return {**task, "rows": rows}


def _validate_point_report(report: object) -> tuple[str, str, list[dict[str, Any]]]:
    if not isinstance(report, dict):
        raise HistoryBootstrapError("point report must be an object")
    contract, cohort = report.get("contract"), report.get("cohort_inventory_sha256")
    if contract not in {"h5_early_online_collection_v1", "h5_late_history_screen_collection_v2"}:
        raise HistoryBootstrapError("point report contract is not bootstrap-eligible")
    if not isinstance(cohort, str) or len(cohort) != 64 or any(char not in "0123456789abcdef" for char in cohort):
        raise HistoryBootstrapError("point report lacks a hash-bound cohort inventory")
    tasks = report.get("bootstrap_tasks")
    gate = report.get("primary_bootstrap_gate")
    if not isinstance(tasks, list) or not isinstance(gate, dict):
        raise HistoryBootstrapError("point report lacks exact bootstrap tasks/gate")
    parsed = [_validate_task(task) for task in tasks]
    ids = [task["task_id"] for task in parsed]
    if len(ids) != len(set(ids)):
        raise HistoryBootstrapError("point report has duplicate bootstrap task IDs")
    gate_runs = [(task["joint_primary_gate"], task["run"]) for task in parsed]
    if len(gate_runs) != len(set(gate_runs)):
        raise HistoryBootstrapError("point report has duplicate bootstrap gate/run tasks")
    if contract == "h5_early_online_collection_v1":
        expected: dict[str, set[str]] = {}
        for key, value in gate.items():
            if not isinstance(value, dict) or value.get("pass") is not True:
                continue
            expected[key] = {"L1", "L3"}
        actual: dict[str, set[str]] = {}
        for task in parsed:
            key = task["joint_primary_gate"]
            if key not in expected:
                raise HistoryBootstrapError("early task does not match a passing point gate")
            expected_stratum = (
                "online_anchor_correct" if task["outcome"] == "peak_failure" else "online_anchor_wrong"
            )
            expected_gate = f"epoch{task['anchor']}-{task['outcome']}"
            if (
                task["anchor"] not in {39, 59, 79}
                or task["outcome"] not in {"peak_failure", "non_recovery"}
                or task["stratum"] != expected_stratum
                or key != expected_gate
                or task["task_id"] != f"{task['run']}-{expected_gate}"
            ):
                raise HistoryBootstrapError("early bootstrap task identity drifted")
            actual.setdefault(key, set()).add(task["run"])
        if actual != expected:
            raise HistoryBootstrapError("early point gate/task pair is incomplete or forged")
        if report.get("status") != ("point_gate_pass_bootstrap_pending" if parsed else "no_go_point_gate"):
            raise HistoryBootstrapError("early point report status does not match its exact tasks")
    else:
        for task in parsed:
            if (
                task["anchor"] != 99
                or task["outcome"] != "future_forgetting"
                or task["stratum"] != "online_anchor_correct"
                or task["joint_primary_gate"] != "late_all_l1_l4_and_bartoldson"
                or task["task_id"] != f"late-{task['run']}-epoch99-future_forgetting"
            ):
                raise HistoryBootstrapError("late bootstrap task identity drifted")
        if gate.get("status") == "point_gate_pass_bootstrap_pending":
            if {task["run"] for task in parsed} != {"L1", "L3"} or len(parsed) != 2:
                raise HistoryBootstrapError("late passing point gate requires exactly paired L1/L3 tasks")
        elif parsed:
            raise HistoryBootstrapError("late no-go point report must not contain bootstrap tasks")
        if report.get("status") != gate.get("status"):
            raise HistoryBootstrapError("late point report status does not match its primary gate")
    return contract, cohort, parsed


def _task_fingerprint(task: dict[str, Any], point_sha: str) -> str:
    return hashlib.sha256(
        canonical_json(
            {
                "contract": CONTRACT,
                "seed": SEED,
                "replicates": REPLICATES,
                "point_sha": point_sha,
                "source": _source_fingerprint(),
                "task": task,
            }
        )
    ).hexdigest()


def _init_worker(task: dict[str, Any]) -> None:
    global _WORKER_TASK
    _WORKER_TASK = task


def _delta(replicate: int) -> tuple[int, float | None]:
    if _WORKER_TASK is None:  # pragma: no cover - executor/runner contract
        raise HistoryBootstrapError("bootstrap worker lacks its immutable task")
    rows = _WORKER_TASK["rows"]
    selected = _bootstrap_indices(rows, seed=SEED, replicate=replicate, cluster=False)
    labels = [int(rows[index]["outcome"]) for index in selected]
    if not any(labels) or all(labels):
        return replicate, None
    return replicate, binary_metrics(labels, [float(rows[index]["candidate"]) for index in selected])[
        "auroc"
    ] - binary_metrics(labels, [float(rows[index]["baseline"]) for index in selected])["auroc"]


def _atomic(path: Path, value: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(canonical_json(value) + b"\n")
    temp.replace(path)


def _final_gate(contract: str, results: list[dict[str, Any]], *, partial: bool) -> dict[str, Any]:
    if partial:
        return {"status": "pending_bootstrap"}
    by_key: dict[tuple[str, str], dict[str, float | None]] = {}
    for row in results:
        by_key.setdefault((row["joint_primary_gate"], row["run"]), {})["lower"] = row["lower"]
    grouped: dict[str, dict[str, float | None]] = {}
    for (key, label), value in by_key.items():
        grouped.setdefault(key, {})[label] = value.get("lower")
    passed = {
        key: all(isinstance(item.get(label), float) and item[label] > 0 for label in ("L1", "L3"))
        for key, item in grouped.items()
    }
    if contract == "h5_early_online_collection_v1":
        earliest: dict[str, str | None] = {}
        for outcome in ("peak_failure", "non_recovery"):
            anchors = sorted(key for key in passed if key.endswith(outcome))
            earliest[outcome] = next((key for key in anchors if passed[key]), None)
        routes = {
            outcome: {"status": "go" if anchor is not None else "no_go", "earliest_passing_anchor": anchor}
            for outcome, anchor in earliest.items()
        }
        return {"status": "any_route_go" if any(earliest.values()) else "no_go", "routes": routes, "pairs": passed}
    return {"status": "go" if passed and all(passed.values()) else "no_go", "pairs": passed}


def run_bootstrap(
    *, point_report: Path, output: Path, progress_dir: Path, workers: int = 1, max_replicates: int | None = None
) -> dict[str, Any]:
    """Run only fully validated paired tasks; worker count cannot alter draws."""
    if output.exists():
        raise FileExistsError("refusing to overwrite bootstrap report")
    if workers < 1 or (max_replicates is not None and max_replicates < 1):
        raise HistoryBootstrapError("workers and requested replicates must be positive")
    try:
        report = json.loads(point_report.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HistoryBootstrapError("point report is unreadable") from exc
    point_contract, cohort_sha256, tasks = _validate_point_report(report)
    point_sha = sha256_file(point_report)
    limit = REPLICATES if max_replicates is None else min(max_replicates, REPLICATES)
    progress_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for task in tasks:
        fingerprint = _task_fingerprint(task, point_sha)
        progress = progress_dir / f"{task['task_id']}.json"
        state = {"fingerprint": fingerprint, "completed": {}}
        if progress.exists():
            state = json.loads(progress.read_text())
            if state.get("fingerprint") != fingerprint:
                raise HistoryBootstrapError("bootstrap progress fingerprint mismatch")
        completed = {int(key): value for key, value in state.get("completed", {}).items()}
        missing = [replicate for replicate in range(limit) if replicate not in completed]
        _init_worker(task)
        pool = (
            ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(task,))
            if workers > 1
            else None
        )
        computed = pool.map(_delta, missing) if pool is not None else map(_delta, missing)
        try:
            for replicate, value in computed:
                completed[replicate] = value
                if len(completed) % 50 == 0:
                    _atomic(
                        progress, {"fingerprint": fingerprint, "completed": {str(k): v for k, v in completed.items()}}
                    )
        except BaseException:
            _atomic(progress, {"fingerprint": fingerprint, "completed": {str(k): v for k, v in completed.items()}})
            raise
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
        _atomic(progress, {"fingerprint": fingerprint, "completed": {str(k): v for k, v in completed.items()}})
        if limit == REPLICATES:
            deltas = sorted(float(value) for value in completed.values() if value is not None)
            results.append(
                {
                    "task_id": task["task_id"],
                    "run": task["run"],
                    "joint_primary_gate": task["joint_primary_gate"],
                    "replicates": len(deltas),
                    "lower": deltas[max(0, math.floor(0.025 * (len(deltas) - 1)))] if deltas else None,
                    "upper": deltas[min(len(deltas) - 1, math.ceil(0.975 * (len(deltas) - 1)))] if deltas else None,
                }
            )
    final = {
        "schema_version": 1,
        "contract": CONTRACT,
        "point_report_contract": point_contract,
        "point_report_sha256": point_sha,
        "cohort_inventory_sha256": cohort_sha256,
        "seed": SEED,
        "replicates": REPLICATES,
        "source_files": _source_fingerprint(),
        "results": results,
        "partial": limit != REPLICATES,
        "final_gate": _final_gate(point_contract, results, partial=limit != REPLICATES),
    }
    _atomic(output, final)
    return final
