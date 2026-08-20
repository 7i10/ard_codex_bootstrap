"""Fixed-mask preparation for the Clean-Wrong reliability-gated screen.

The selector is deliberately an offline, pre-treatment operation.  It reads
the already hash-bound epoch-79 CE20 and KL10 replay tables and emits the
ordinary Stage-A overlay format consumed by the shared trainer.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from ard.analysis.ert_clean_wrong_broad_screen import fixed_clean_wrong_mask


class ReliabilityGatedError(RuntimeError):
    """Raised when a selector artifact does not satisfy its frozen contract."""


PARENT_SHA = {
    "L2": "ad43d72da2a02f205c65b96485379c9acb5fc2b07d6823d09820439aedc8f78c",
    "L4": "026a36d3fe057386fe19225fed23b56625ab23da80be3dd42cf3e478e5080bf1",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_replay(
    meta_path: Path, rows_path: Path, *, run: str, mask: dict[str, Any], attack: str
) -> dict[int, dict[str, Any]]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, Mapping) or meta.get("feature_epoch") != 79 or not meta.get("full_train_order_replayed"):
        raise ReliabilityGatedError(f"{attack} replay is not an epoch-79 full-order artifact: {meta_path}")
    if meta.get("checkpoint_sha256") != PARENT_SHA[run] or meta.get("mask_sha256") != mask["mask_sha256"]:
        raise ReliabilityGatedError(f"{attack} replay lineage does not match {run} parent/mask")
    if meta.get("rows_sha256") != _sha256(rows_path):
        raise ReliabilityGatedError(f"{attack} replay rows hash mismatch: {rows_path}")
    expected_contract = (
        "ert_clean_wrong_c0_ce_pgd20_features_v1" if attack == "CE20" else "ert_clean_wrong_c0_kl_pgd10_features_v1"
    )
    if meta.get("contract") != expected_contract:
        raise ReliabilityGatedError(f"unexpected {attack} replay contract")
    rows = pq.read_table(rows_path).to_pylist()
    expected = set(mask["selected_ids"])
    observed = {int(row["sample_id"]) for row in rows}
    if observed != expected or len(rows) != len(observed):
        raise ReliabilityGatedError(f"{attack} replay IDs do not exactly match the fixed Clean-Wrong cohort")
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        sample_id = int(row["sample_id"])
        margin = row.get("teacher_adv_margin")
        if not isinstance(margin, (float, int)) or not math.isfinite(float(margin)):
            raise ReliabilityGatedError(f"{attack} replay has invalid Teacher margin for {sample_id}")
        result[sample_id] = row
    return result


def _overlay(
    *, run: str, mask_path: Path, ids: list[int], labels: dict[int, int], selector: str, selector_meta: dict[str, Any]
) -> dict[str, Any]:
    selected_labels = [labels[sample_id] for sample_id in ids]
    counts: dict[str, int] = {}
    for label in selected_labels:
        counts[str(label)] = counts.get(str(label), 0) + 1
    return {
        "schema_version": 1,
        "contract": "ert_state_overlay_v1",
        "anchor_epoch": 79,
        "run": run,
        "source_mask_sha256": _sha256(mask_path),
        "selector": selector,
        "selector_meta": selector_meta,
        "masks": {
            "student_clean_wrong": {
                "selected_ids": ids,
                "selected_labels": selected_labels,
                "selected_class_counts": counts,
                "selection_rule": "teacher_adv_margin > 0 at fixed epoch-79 pre-treatment replay",
            }
        },
    }


def prepare_selector_bundle(
    *,
    run: str,
    mask_path: Path,
    ce_meta: Path,
    ce_rows: Path,
    kl_meta: Path,
    kl_rows: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if run not in PARENT_SHA:
        raise ReliabilityGatedError(f"unsupported run: {run}")
    mask = fixed_clean_wrong_mask(mask_path, run=run)
    ce = _load_replay(ce_meta, ce_rows, run=run, mask=mask, attack="CE20")
    kl = _load_replay(kl_meta, kl_rows, run=run, mask=mask, attack="KL10")
    ids = sorted(mask["selected_ids"])
    labels = {sample_id: int(ce[sample_id]["true_label"]) for sample_id in ids}
    ce_ids = {sample_id for sample_id in ids if float(ce[sample_id]["teacher_adv_margin"]) > 0.0}
    kl_ids = {sample_id for sample_id in ids if float(kl[sample_id]["teacher_adv_margin"]) > 0.0}
    rr, ru, ur, uu = set(), set(), set(), set()
    for sample_id in ids:
        if sample_id in ce_ids and sample_id in kl_ids:
            rr.add(sample_id)
        elif sample_id in ce_ids:
            ru.add(sample_id)
        elif sample_id in kl_ids:
            ur.add(sample_id)
        else:
            uu.add(sample_id)
    output_dir.mkdir(parents=True, exist_ok=False)
    selector_meta = {
        "anchor_epoch": 79,
        "parent_checkpoint_sha256": PARENT_SHA[run],
        "mask_sha256": mask["mask_sha256"],
        "threshold": "teacher_adv_margin > 0",
        "ce20_meta_sha256": _sha256(ce_meta),
        "ce20_rows_sha256": _sha256(ce_rows),
        "kl10_meta_sha256": _sha256(kl_meta),
        "kl10_rows_sha256": _sha256(kl_rows),
    }
    overlays: dict[str, str] = {}
    for name, selected in (("all", set(ids)), ("ce20", ce_ids), ("kl10", kl_ids)):
        payload = _overlay(
            run=run,
            mask_path=mask_path,
            ids=sorted(selected),
            labels=labels,
            selector=name,
            selector_meta=selector_meta,
        )
        path = output_dir / f"{name}-overlay.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        overlays[name] = str(path.resolve())
    intersection = {"RR": len(rr), "RU": len(ru), "UR": len(ur), "UU": len(uu)}
    bundle = {
        "schema_version": 1,
        "contract": "ert_cw_reliability_gated_ce015_selector_bundle_v1",
        "run": run,
        "parent_checkpoint_sha256": PARENT_SHA[run],
        "mask_sha256": mask["mask_sha256"],
        "clean_wrong_count": len(ids),
        "counts": {"CE20_reliable": len(ce_ids), "KL10_reliable": len(kl_ids), **intersection},
        "jaccard_ce20_kl10": len(ce_ids & kl_ids) / len(ce_ids | kl_ids) if ce_ids | kl_ids else 1.0,
        "selector_meta": selector_meta,
        "overlays": overlays,
        "overlay_sha256": {name: _sha256(Path(path)) for name, path in overlays.items()},
    }
    bundle_path = output_dir / "selector-bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    bundle["bundle_path"] = str(bundle_path.resolve())
    bundle["bundle_sha256"] = _sha256(bundle_path)
    return bundle
