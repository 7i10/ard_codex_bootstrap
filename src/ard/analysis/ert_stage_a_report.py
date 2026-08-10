"""Aggregate paired CE-PGD20 endpoint rows for the ERT Stage A screen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from ard.tracking.adapter import collect_git_state


class StageAReportError(RuntimeError):
    """Stage A endpoint inputs are incomplete or inconsistent."""


ARMS = ("C79", "ST1W", "ST1M", "ST1S", "ST2W", "ST2M", "ST2S", "ST3K1", "ST3K05", "ST3K0", "CW1", "CW2", "CW3")
MASK_KEYS = {
    "ST1W": "s3_t1_q10", "ST1M": "s3_t1_q10", "ST1S": "s3_t1_q10",
    "ST2W": "s3_t2_q10", "ST2M": "s3_t2_q10", "ST2S": "s3_t2_q10",
    "ST3K1": "s3_t3_q10", "ST3K05": "s3_t3_q10", "ST3K0": "s3_t3_q10",
    "CW1": "student_clean_wrong", "CW2": "student_clean_wrong", "CW3": "student_clean_wrong",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> dict[int, dict[str, Any]]:
    values = pq.read_table(path).to_pylist()
    result: dict[int, dict[str, Any]] = {}
    for row in values:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, int) or sample_id in result:
            raise StageAReportError(f"duplicate or invalid endpoint sample ID in {path}")
        result[sample_id] = row
    if not result:
        raise StageAReportError(f"empty endpoint table: {path}")
    return result


def _mask(path: Path, key: str) -> set[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("masks", {}).get(key)
    if payload.get("anchor_epoch") != 79 or not isinstance(raw, dict):
        raise StageAReportError(f"invalid Stage A mask bundle/key: {path}::{key}")
    values = raw.get("selected_ids")
    if not isinstance(values, list) or any(not isinstance(item, int) or isinstance(item, bool) for item in values):
        raise StageAReportError(f"invalid selected IDs: {path}::{key}")
    selected = set(values)
    if len(selected) != len(values):
        raise StageAReportError(f"duplicate selected IDs: {path}::{key}")
    return selected


def _rates(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        raise StageAReportError("empty cohort")
    count = len(rows)
    return {
        "count": count,
        "clean_accuracy": sum(bool(row["clean_correct"]) for row in rows) / count,
        "robust_accuracy": sum(bool(row["robust_correct"]) for row in rows) / count,
        "adversarial_probability_margin": sum(float(row["adversarial_probability_margin"]) for row in rows) / count,
        "clean_probability_margin": sum(float(row["clean_probability_margin"]) for row in rows) / count,
    }


def _paired(
    control: dict[int, dict[str, Any]], treatment: dict[int, dict[str, Any]], ids: set[int]
) -> dict[str, float | int]:
    if not ids.issubset(control) or not ids.issubset(treatment):
        raise StageAReportError("endpoint/control sample-ID universe does not contain the registered cohort")
    pairs = [(control[item], treatment[item]) for item in sorted(ids)]
    n = len(pairs)
    rescue = sum((not c["robust_correct"]) and t["robust_correct"] for c, t in pairs)
    harm = sum(c["robust_correct"] and (not t["robust_correct"]) for c, t in pairs)
    clean_rescue = sum((not c["clean_correct"]) and t["clean_correct"] for c, t in pairs)
    clean_harm = sum(c["clean_correct"] and (not t["clean_correct"]) for c, t in pairs)
    return {
        "count": n,
        "control_robust_accuracy": sum(c["robust_correct"] for c, _ in pairs) / n,
        "treatment_robust_accuracy": sum(t["robust_correct"] for _, t in pairs) / n,
        "robust_accuracy_delta": sum(t["robust_correct"] - c["robust_correct"] for c, t in pairs) / n,
        "rescue_count": rescue,
        "harm_count": harm,
        "net_rescue_count": rescue - harm,
        "rescue_rate": rescue / n,
        "harm_rate": harm / n,
        "net_rescue_rate": (rescue - harm) / n,
        "control_clean_accuracy": sum(c["clean_correct"] for c, _ in pairs) / n,
        "treatment_clean_accuracy": sum(t["clean_correct"] for _, t in pairs) / n,
        "clean_accuracy_delta": sum(t["clean_correct"] - c["clean_correct"] for c, t in pairs) / n,
        "clean_rescue_count": clean_rescue,
        "clean_harm_count": clean_harm,
        "clean_margin_delta": sum(
            float(t["clean_probability_margin"]) - float(c["clean_probability_margin"]) for c, t in pairs
        )
        / n,
        "adversarial_margin_delta": sum(
            float(t["adversarial_probability_margin"])
            - float(c["adversarial_probability_margin"])
            for c, t in pairs
        )
        / n,
    }


def build_report(*, endpoint_root: Path, mask_paths: dict[str, Path], output: Path) -> dict[str, Any]:
    source = collect_git_state(Path.cwd())
    if source.get("dirty") is not False or not isinstance(source.get("sha"), str):
        raise StageAReportError("Stage A report requires a clean source tree")
    report: dict[str, Any] = {
        "schema_version": 1,
        "contract": "ert_stage_a_treatment_report_v1",
        "source_git_sha": source["sha"],
        "endpoint_attack": None,
        "seeds": {},
        "inputs": {},
    }
    for seed in ("L2", "L4"):
        seed_rows: dict[str, dict[int, dict[str, Any]]] = {}
        seed_result: dict[str, Any] = {"arms": {}}
        for arm in ARMS:
            train_arm = arm if not (seed == "L2" and arm in {"ST1W", "ST2S"}) else f"{arm}-rerun"
            base = endpoint_root / seed / arm
            endpoint_json = base / "endpoint.json"
            rows_path = base / "endpoint-sample-stats.parquet"
            if not endpoint_json.is_file() or not rows_path.is_file():
                raise StageAReportError(f"missing endpoint artifact: {seed}/{arm}")
            metadata = json.loads(endpoint_json.read_text(encoding="utf-8"))
            rows = _rows(rows_path)
            seed_rows[arm] = rows
            report["inputs"][f"{seed}/{arm}"] = {
                "endpoint_json": str(endpoint_json.resolve()),
                "endpoint_json_sha256": _sha256(endpoint_json),
                "rows": str(rows_path.resolve()),
                "rows_sha256": _sha256(rows_path),
                "checkpoint_sha256": metadata.get("checkpoint_sha256"),
                "train_arm": train_arm,
            }
            if report["endpoint_attack"] is None:
                report["endpoint_attack"] = {
                    "identity": metadata.get("attack"),
                    "identity_sha256": metadata.get("attack_identity_sha256"),
                }
            elif report["endpoint_attack"]["identity_sha256"] != metadata.get("attack_identity_sha256"):
                raise StageAReportError("endpoint attack identity differs between arms")
        control = seed_rows["C79"]
        for arm in ARMS:
            if arm == "C79":
                seed_result["arms"][arm] = {
                    "overall": _rates(list(control.values())),
                    "selected": None,
                }
                continue
            key = MASK_KEYS[arm]
            selected = _mask(mask_paths[seed], key)
            cohort = [seed_rows[arm][item] for item in selected]
            nonselected = set(control) - selected
            seed_result["arms"][arm] = {
                "mask_key": key,
                "selected": {"treatment": _rates(cohort), "paired": _paired(control, seed_rows[arm], selected)},
                "nonselected_paired": _paired(control, seed_rows[arm], nonselected),
                "overall": _rates(list(seed_rows[arm].values())),
            }
        report["seeds"][seed] = seed_result
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # A file cannot contain its own SHA-256 without a recursive fixed point.
    # Keep the JSON payload immutable and publish the actual file digest in a
    # sidecar; the CLI also returns that digest for manifests/logs.
    output_sha256 = _sha256(output)
    output.with_name(output.name + ".sha256").write_text(output_sha256 + "\n", encoding="ascii")
    return {**report, "output_sha256": output_sha256}
