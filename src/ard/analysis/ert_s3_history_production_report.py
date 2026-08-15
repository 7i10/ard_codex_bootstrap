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


def _transition(rows: list[dict[str, Any]], *, arm: str) -> dict[str, Any]:
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
        # BASE never acts, while INST075 uses the current-visit predicate as
        # its student-side state.  History arms expose the persistent state.
        if arm == "BASE":
            states = [False for _ in sequence]
        elif arm == "INST075":
            # `current_active` already includes the teacher gate.  Use the
            # student-only S3 observation to attribute teacher-induced flips.
            states = [bool(row["raw_s3_observation"]) for row in sequence]
        else:
            states = [bool(row["history_state_active"]) for row in sequence]
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


def _endpoint_rows(path: Path) -> dict[int, dict[str, Any]]:
    import pyarrow.parquet as pq

    rows = pq.read_table(path / "endpoint-sample-stats.parquet").to_pylist()
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        label = row.get("true_label")
        if not isinstance(sample_id, int) or sample_id in result or not isinstance(label, int):
            raise HistoryProductionReportError(f"invalid endpoint stable-ID table: {path}")
        result[sample_id] = row
    if not result:
        raise HistoryProductionReportError(f"empty endpoint stable-ID table: {path}")
    return result


def _paired_metrics(
    control: dict[int, dict[str, Any]], treatment: dict[int, dict[str, Any]], ids: list[int]
) -> dict[str, Any]:
    if not ids or not set(ids).issubset(control) or not set(ids).issubset(treatment):
        raise HistoryProductionReportError("paired endpoint cohort is not contained in both universes")
    robust = [int(treatment[i]["robust_correct"]) - int(control[i]["robust_correct"]) for i in ids]
    clean = [int(treatment[i]["clean_correct"]) - int(control[i]["clean_correct"]) for i in ids]
    adv_margin = [
        float(treatment[i]["adversarial_probability_margin"])
        - float(control[i]["adversarial_probability_margin"])
        for i in ids
    ]
    clean_margin = [
        float(treatment[i]["clean_probability_margin"]) - float(control[i]["clean_probability_margin"])
        for i in ids
    ]
    rescue = sum(value == 1 for value in robust)
    harm = sum(value == -1 for value in robust)
    n = len(ids)
    return {
        "count": n,
        "rescue_count": rescue,
        "harm_count": harm,
        "net_rescue_count": rescue - harm,
        "rescue_rate": rescue / n,
        "harm_rate": harm / n,
        "net_rescue_rate": (rescue - harm) / n,
        "robust_accuracy_delta": sum(robust) / n,
        "clean_accuracy_delta": sum(clean) / n,
        "adversarial_margin_delta": sum(adv_margin) / n,
        "clean_margin_delta": sum(clean_margin) / n,
    }


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
            capture_path = arm_dir / "routing-capture-mask.json"
            capture = _json(capture_path)
            expected_capture_arm = {
                "BASE": "baseline",
                "INST075": "instant",
                "M3_075": "majority3",
                "M3E2_075": "majority3_exit2",
            }[arm]
            if capture.get("arm") != expected_capture_arm or not isinstance(capture.get("selected_ids"), list):
                raise HistoryProductionReportError(f"invalid routing capture: {capture_path}")
            seed_result["arms"][arm] = {
                "training_state": {"path": str(state_path.resolve()), "sha256": _sha256(state_path), "rows": len(rows)},
                "transition": _transition(rows, arm=arm),
                "manifest_sha256": _sha256(manifest_path),
                "capture": {
                    "path": str(capture_path.resolve()),
                    "sha256": _sha256(capture_path),
                    "selected_count": len(capture["selected_ids"]),
                    "selected_ids_sha256": capture.get("selected_ids_sha256"),
                },
            }
            seed_result["arms"][arm]["capture_ids"] = [int(x) for x in capture["selected_ids"]]
        for horizon in horizons:
            horizon_result: dict[str, Any] = {"arms": {}, "deltas_vs_base": {}}
            base: dict[tuple[str, str], dict[str, Any]] = {}
            rows: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
            for arm in arms:
                for split in ("train", "validation"):
                    directory = endpoint_root / run_key / arm / f"epoch-{horizon}" / split
                    base[arm, split] = _endpoint(directory)
                    rows[arm, split] = _endpoint_rows(directory)
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
                selected_ids = seed_result["arms"][arm]["capture_ids"]
                horizon_result.setdefault("paired_effects", {})[arm] = {
                    "selected_train": _paired_metrics(rows["BASE", "train"], rows[arm, "train"], selected_ids),
                    "validation_all": _paired_metrics(
                        rows["BASE", "validation"], rows[arm, "validation"], sorted(rows["BASE", "validation"])
                    ),
                }
            seed_result["horizons"][str(horizon)] = horizon_result
        for arm in arms:
            seed_result["arms"][arm].pop("capture_ids", None)
        report["seeds"][run_key] = seed_result
    if output.exists():
        raise HistoryProductionReportError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["output_sha256"] = _sha256(output)
    return report
