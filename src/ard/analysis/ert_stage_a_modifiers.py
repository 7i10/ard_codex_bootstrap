"""Read-only Stage A treatment-effect modifier analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from ard.tracking.adapter import collect_git_state


class StageAModifierError(RuntimeError):
    """Modifier inputs violate the Stage A lineage contract."""


ARMS_BY_STATE = {
    "T1": ("ST1W", "ST1M", "ST1S"),
    "T2": ("ST2W", "ST2M", "ST2S"),
    "T3": ("ST3K1", "ST3K05", "ST3K0"),
}
MASK_KEYS = {
    "T1": "s3_t1_q10",
    "T2": "s3_t2_q10",
    "T3": "s3_t3_q10",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> dict[int, dict[str, Any]]:
    rows = pq.read_table(path).to_pylist()
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        item = row.get("sample_id")
        if not isinstance(item, int) or item in result:
            raise StageAModifierError(f"invalid or duplicate sample_id: {path}")
        result[item] = row
    if not result:
        raise StageAModifierError(f"empty parquet: {path}")
    return result


def _selected(mask_path: Path, key: str) -> set[int]:
    payload = json.loads(mask_path.read_text(encoding="utf-8"))
    if payload.get("anchor_epoch") != 79:
        raise StageAModifierError(f"mask is not anchored at epoch 79: {mask_path}")
    values = payload.get("masks", {}).get(key, {}).get("selected_ids")
    if not isinstance(values, list):
        raise StageAModifierError(f"missing mask {key}: {mask_path}")
    result = set(values)
    if len(result) != len(values):
        raise StageAModifierError(f"duplicate IDs in mask {key}: {mask_path}")
    return result


def _tertiles(ids: set[int], rows: dict[int, dict[str, Any]], field: str) -> dict[str, set[int]]:
    ordered = sorted(ids, key=lambda item: (float(rows[item][field]), item))
    n = len(ordered)
    return {
        "low": set(ordered[: n // 3]),
        "middle": set(ordered[n // 3 : (2 * n) // 3]),
        "high": set(ordered[(2 * n) // 3 :]),
    }


def _effect(control: dict[int, dict[str, Any]], treatment: dict[int, dict[str, Any]], ids: set[int]) -> dict[str, Any]:
    if not ids or not ids.issubset(control) or not ids.issubset(treatment):
        raise StageAModifierError("modifier cohort is missing from endpoint sample universe")
    pairs = [(control[item], treatment[item]) for item in sorted(ids)]
    n = len(pairs)
    rescue = sum((not c["robust_correct"]) and t["robust_correct"] for c, t in pairs)
    harm = sum(c["robust_correct"] and (not t["robust_correct"]) for c, t in pairs)
    return {
        "count": n,
        "robust_accuracy_delta": sum(int(t["robust_correct"]) - int(c["robust_correct"]) for c, t in pairs) / n,
        "rescue_rate": rescue / n,
        "harm_rate": harm / n,
        "net_rescue_rate": (rescue - harm) / n,
        "clean_accuracy_delta": sum(int(t["clean_correct"]) - int(c["clean_correct"]) for c, t in pairs) / n,
        "adversarial_margin_delta": sum(
            float(t["adversarial_probability_margin"]) - float(c["adversarial_probability_margin"])
            for c, t in pairs
        ) / n,
    }


def build_modifier_report(
    *,
    endpoint_root: Path,
    state_paths: dict[str, Path],
    mask_paths: dict[str, Path],
    endpoint_report: Path,
    output: Path,
) -> dict[str, Any]:
    source = collect_git_state(Path.cwd())
    if source.get("dirty") is not False or not isinstance(source.get("sha"), str):
        raise StageAModifierError("modifier report requires a clean source tree")
    result: dict[str, Any] = {
        "schema_version": 1,
        "contract": "ert_stage_a_modifier_report_v1",
        "source_git_sha": source["sha"],
        "endpoint_report_sha256": _sha256(endpoint_report),
        "inputs": {},
        "seeds": {},
    }
    for seed in ("L2", "L4"):
        state_path = state_paths[seed]
        mask_path = mask_paths[seed]
        state = _rows(state_path)
        result["inputs"][seed] = {
            "state_table": str(state_path.resolve()),
            "state_table_sha256": _sha256(state_path),
            "mask": str(mask_path.resolve()),
            "mask_sha256": _sha256(mask_path),
        }
        seed_out: dict[str, Any] = {}
        control = _rows(endpoint_root / seed / "C79" / "endpoint-sample-stats.parquet")
        for state_name, arms in ARMS_BY_STATE.items():
            ids = _selected(mask_path, MASK_KEYS[state_name])
            if not ids.issubset(state):
                raise StageAModifierError(f"state table lacks selected IDs: {seed}/{state_name}")
            # A state label is a frozen epoch-79 modifier, not a selector.
            state_ids = {item for item in ids if state[item].get("teacher_state_q10") == state_name}
            if state_ids != ids:
                raise StageAModifierError(f"mask/state mismatch for {seed}/{state_name}")
            groups: dict[str, dict[str, set[int]]] = {
                "mT_clean": _tertiles(ids, state, "mT_clean"),
                "DeltaT": _tertiles(ids, state, "DeltaT"),
                "teacher_clean_correct": {
                    "correct": {item for item in ids if bool(state[item]["teacher_clean_correct"])},
                    "wrong": {item for item in ids if not bool(state[item]["teacher_clean_correct"])},
                },
            }
            state_out: dict[str, Any] = {
                "count": len(ids),
                "teacher_clean_correct_count": len(groups["teacher_clean_correct"]["correct"]),
                "arms": {},
            }
            for arm in arms:
                endpoint = _rows(endpoint_root / seed / arm / "endpoint-sample-stats.parquet")
                arm_out: dict[str, Any] = {}
                for modifier, bins in groups.items():
                    arm_out[modifier] = {
                        label: _effect(control, endpoint, subset) for label, subset in bins.items() if subset
                    }
                state_out["arms"][arm] = arm_out
            seed_out[state_name] = state_out
        result["seeds"][seed] = seed_out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = _sha256(output)
    output.with_name(output.name + ".sha256").write_text(digest + "\n", encoding="ascii")
    return {**result, "output_sha256": digest}
