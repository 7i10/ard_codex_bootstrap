"""Aggregate the frozen reliability-gated CleanCE intervention screen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from ard.tracking.adapter import collect_git_state


class ReliabilityGatedReportError(RuntimeError):
    """Raised when one of the paired endpoint artifacts is incomplete."""


ARMS = ("G0_BASE", "G1_CW_ALL_CE015", "G2_CW_R_CE20_CE015", "G3_CW_R_KL10_CE015")
HORIZONS = (84, 89, 94)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> dict[int, dict[str, Any]]:
    values = pq.read_table(path).to_pylist()
    result = {int(row["sample_id"]): row for row in values}
    if len(result) != len(values) or not result:
        raise ReliabilityGatedReportError(f"invalid endpoint stable-ID table: {path}")
    return result


def _overlay_ids(path: Path) -> set[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("masks", {}).get("student_clean_wrong")
    if payload.get("anchor_epoch") != 79 or not isinstance(raw, dict):
        raise ReliabilityGatedReportError(f"invalid selector overlay: {path}")
    ids = raw.get("selected_ids")
    if not isinstance(ids, list) or len(ids) != len(set(ids)):
        raise ReliabilityGatedReportError(f"invalid selector IDs: {path}")
    return {int(item) for item in ids}


def _paired(control: dict[int, dict[str, Any]], treatment: dict[int, dict[str, Any]], ids: set[int]) -> dict[str, Any]:
    if not ids or not ids.issubset(control) or not ids.issubset(treatment):
        raise ReliabilityGatedReportError("paired endpoint cohort is incomplete")
    pairs = [(control[i], treatment[i]) for i in sorted(ids)]
    n = len(pairs)

    def rates(key: str) -> tuple[int, int]:
        rescue = sum(not bool(c[key]) and bool(t[key]) for c, t in pairs)
        harm = sum(bool(c[key]) and not bool(t[key]) for c, t in pairs)
        return rescue, harm

    robust_rescue, robust_harm = rates("robust_correct")
    clean_rescue, clean_harm = rates("clean_correct")
    return {
        "count": n,
        "robust_accuracy_delta": sum(bool(t["robust_correct"]) - bool(c["robust_correct"]) for c, t in pairs) / n,
        "clean_accuracy_delta": sum(bool(t["clean_correct"]) - bool(c["clean_correct"]) for c, t in pairs) / n,
        "robust_rescue": robust_rescue,
        "robust_harm": robust_harm,
        "robust_net_rescue": robust_rescue - robust_harm,
        "robust_rescue_rate": robust_rescue / n,
        "robust_harm_rate": robust_harm / n,
        "clean_rescue": clean_rescue,
        "clean_harm": clean_harm,
        "clean_net_rescue": clean_rescue - clean_harm,
        "clean_rescue_rate": clean_rescue / n,
        "clean_harm_rate": clean_harm / n,
        "clean_margin_delta": sum(
            float(t["clean_probability_margin"]) - float(c["clean_probability_margin"]) for c, t in pairs
        )
        / n,
        "robust_margin_delta": sum(
            float(t["adversarial_probability_margin"]) - float(c["adversarial_probability_margin"]) for c, t in pairs
        )
        / n,
    }


def build_report(
    *, endpoint_root: Path, training_dirs: dict[str, dict[str, Path]], selector_dirs: dict[str, Path], output: Path
) -> dict[str, Any]:
    source = collect_git_state(Path.cwd())
    if source.get("dirty") is not False or not isinstance(source.get("sha"), str):
        raise ReliabilityGatedReportError("report requires a clean source tree")
    report: dict[str, Any] = {
        "schema_version": 1,
        "contract": "ert_cw_reliability_gated_ce015_results_v1",
        "source_git_sha": source["sha"],
        "horizons": list(HORIZONS),
        "seeds": {},
        "inputs": {},
    }
    for run in ("L2", "L4"):
        bundles = json.loads((selector_dirs[run] / "selector-bundle.json").read_text(encoding="utf-8"))
        overlays = {name: _overlay_ids(selector_dirs[run] / f"{name}-overlay.json") for name in ("all", "ce20", "kl10")}
        rr = overlays["ce20"] & overlays["kl10"]
        ru = overlays["ce20"] - overlays["kl10"]
        ur = overlays["kl10"] - overlays["ce20"]
        uu = overlays["all"] - (rr | ru | ur)
        seed_report: dict[str, Any] = {"selector_bundle": bundles, "horizons": {}}
        for epoch in HORIZONS:
            arm_rows: dict[str, dict[int, dict[str, Any]]] = {}
            for arm in ARMS:
                path = training_dirs[run][arm] / "endpoint" / f"epoch-{epoch}" / "validation"
                meta_path, rows_path = path / "endpoint.json", path / "endpoint-sample-stats.parquet"
                if not meta_path.is_file() or not rows_path.is_file():
                    raise ReliabilityGatedReportError(f"missing endpoint: {run}/{arm}/epoch-{epoch}")
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("checkpoint_epoch") != epoch or meta.get("attack", {}).get("steps") != 20:
                    raise ReliabilityGatedReportError(f"endpoint identity mismatch: {meta_path}")
                arm_rows[arm] = _rows(rows_path)
                report["inputs"][f"{run}/{arm}/epoch-{epoch}"] = {
                    "meta_sha256": _sha256(meta_path),
                    "rows_sha256": _sha256(rows_path),
                    "checkpoint_sha256": meta.get("checkpoint_sha256"),
                    "attack_identity_sha256": meta.get("attack_identity_sha256"),
                }
            control = arm_rows["G0_BASE"]
            if set(control) != set(arm_rows["G1_CW_ALL_CE015"]):
                raise ReliabilityGatedReportError("endpoint stable-ID universe differs across arms")
            comparisons: dict[str, Any] = {}
            all_ids = set(control)
            cohorts = {
                "held_out": all_ids,
                "CW_all": overlays["all"],
                "CE20_R": overlays["ce20"],
                "KL10_R": overlays["kl10"],
                "RR": rr,
                "RU": ru,
                "UR": ur,
                "UU": uu,
                "CW_excluded_CE20": overlays["all"] - overlays["ce20"],
                "CW_excluded_KL10": overlays["all"] - overlays["kl10"],
                "non_CW": all_ids - overlays["all"],
            }
            for arm in ARMS[1:]:
                comparisons[arm] = {
                    cohort: _paired(control, arm_rows[arm], ids) for cohort, ids in cohorts.items() if ids
                }
            seed_report["horizons"][str(epoch)] = {"comparisons_vs_G0": comparisons}
        report["seeds"][run] = seed_report
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["output_sha256"] = _sha256(output)
    output.with_name(output.name + ".sha256").write_text(report["output_sha256"] + "\n", encoding="ascii")
    return report
