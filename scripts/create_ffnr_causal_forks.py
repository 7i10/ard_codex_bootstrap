#!/usr/bin/env python3
# ruff: noqa: E501
"""Create the hash-bound epoch-79 five-arm FFNR causal pilot screen."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml

from ard.analysis.intervention_fork import build_parent_artifact_attestation, create_intervention_forks
from ard.config import load_config
from ard.engine.checkpoint import config_digest
from ard.policies import selected_ids_sha256
from ard.state import SampleStateStore


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return _sha256(path)


def _mask(
    *,
    path: Path,
    ids: list[int],
    labels: dict[int, int],
    source: str,
    parent_checkpoint_sha256: str,
    parent_sample_state_sha256: str,
    random_seed: int | None,
) -> dict[str, Any]:
    class_counts = Counter(labels[sample_id] for sample_id in ids)
    provenance: dict[str, Any] = {
        "source": source,
        "approved_selector_spec_sha256": None,
        "selector_spec_path": None,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "parent_sample_state_sha256": parent_sample_state_sha256,
        "random_seed": random_seed,
        "generator": "sha256-stratum-choice-v1" if random_seed is not None else None,
        "generator_version": "1" if random_seed is not None else None,
        "reference_history_mask_sha256": None,
        "reference_selected_count": None,
        "reference_selected_class_counts": None,
        "reference_history_selector_spec_sha256": None,
        "route": None,
        "anchor_robust_correct": None,
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "namespace": "train",
        "num_classes": 10,
        "selected_ids": ids,
        "selected_ids_sha256": selected_ids_sha256(tuple(ids)),
        "selected_count": len(ids),
        "selected_class_counts": {str(k): int(v) for k, v in sorted(class_counts.items())},
        "provenance": provenance,
    }
    _json(path, payload)
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "selected_ids_sha256": payload["selected_ids_sha256"],
        "selected_count": len(ids),
        "selected_class_counts": payload["selected_class_counts"],
        "provenance": provenance,
    }


def _prepare_parent(
    args: argparse.Namespace, payload: dict[str, Any], raw: dict[str, Any], run_manifest: Path
) -> dict[str, Any]:
    root = args.lineage_root / args.label
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "parent-manifest.json"
    shutil.copy2(run_manifest, manifest)
    sample_state = payload["sample_state"]
    store = SampleStateStore(ema_decay=float(raw["method"]["student_ema_decay"]))
    store.load_state_dict(sample_state)
    labels = {int(sample_id): int(record.true_label) for sample_id, record in store.records.items()}
    rows = [[sample_id, label] for sample_id, label in sorted(labels.items())]
    partition = root / "train-partition.json"
    ids_labels_sha = _canonical(rows)
    _json(
        partition, {"schema_version": 1, "namespace": "train", "ids_labels": rows, "ids_labels_sha256": ids_labels_sha}
    )
    checkpoint_sha = _sha256(args.parent)
    inventory = root / "artifact-inventory.json"
    inventory_sha = _json(
        inventory,
        {
            "schema_version": 1,
            "artifact": {
                "name": f"model-{payload['tracker_run_id']}-last",
                "version": "v15",
                "digest": f"local-{checkpoint_sha}",
                "checkpoint_sha256": checkpoint_sha,
            },
        },
    )
    attestation = root / "artifact-attestation.json"
    attestation_payload = build_parent_artifact_attestation(
        parent_manifest=manifest,
        artifact_inventory=inventory,
        checkpoint=args.parent,
    )
    attestation_sha = _json(attestation, attestation_payload)
    return {
        "checkpoint_sha256": checkpoint_sha,
        "raw_config_sha256": config_digest(raw),
        "git_sha": str(json.loads(manifest.read_text())["git"]["sha"]),
        "epoch": 79,
        "world_size": 1,
        "teacher_checkpoint_sha256": str(raw["teacher"]["checkpoint_sha256"]),
        "sample_state_records": len(labels),
        "sample_state_sha256": _canonical(sample_state),
        "train_partition_manifest": str(partition.resolve()),
        "train_partition_manifest_sha256": _sha256(partition),
        "train_partition_ids_labels_sha256": ids_labels_sha,
        "artifact_attestation": str(attestation.resolve()),
        "artifact_attestation_sha256": attestation_sha,
        "artifact_inventory": str(inventory.resolve()),
        "artifact_inventory_sha256": inventory_sha,
    }


def build(args: argparse.Namespace) -> None:
    raw = yaml.safe_load(args.parent_config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("parent config must be a mapping")
    payload = torch.load(args.parent, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("epoch") != 79:
        raise ValueError("parent must be an epoch-79 checkpoint")
    parent = _prepare_parent(args, payload, raw, args.parent_manifest)
    masks_root = args.mask_root / args.label
    masks_manifest = json.loads((args.mask_root / "manifest.json").read_text(encoding="utf-8"))
    run_masks = masks_manifest["runs"][args.label]
    store = SampleStateStore(ema_decay=float(raw["method"]["student_ema_decay"]))
    store.load_state_dict(payload["sample_state"])
    labels = {int(sample_id): int(record.true_label) for sample_id, record in store.records.items()}
    config_paths: list[Path] = []
    names = {
        "C79": None,
        "RA": "route_a_selected",
        "RAR": "route_a_random",
        "RB": "route_b_selected_q05",
        "RBR": "route_b_random_q05",
    }
    sources = {
        "RA": "ffnr_route_a_strong_ce_pgd20",
        "RAR": "ffnr_route_a_matched_random",
        "RB": "ffnr_route_b_strong_ce_pgd20",
        "RBR": "ffnr_route_b_matched_random",
    }
    selectors = {
        "C79": "none",
        "RA": "route_a_strong",
        "RAR": "route_a_matched_random",
        "RB": "route_b_strong",
        "RBR": "route_b_matched_random",
    }
    kinds = {
        "C79": "ordinary_rslad",
        "RA": "route_a_ce_anchor",
        "RAR": "route_a_ce_anchor",
        "RB": "route_b_ce_anchor",
        "RBR": "route_b_ce_anchor",
    }
    kd = {"C79": 0.5, "RA": 0.5, "RAR": 0.5, "RB": 1.0, "RBR": 1.0}
    mask_refs: dict[str, dict[str, Any]] = {}
    for arm, mask_name in names.items():
        child_raw = copy.deepcopy(raw)
        child_raw["output_dir"] = str((args.screen_root / arm).resolve())
        child_raw["tracking"]["run_id"] = f"ffnr-causal-{args.label.lower()}-{arm.lower()}-e79-84"
        child_raw["tracking"]["name"] = f"ffnr-causal-{args.label}-{arm}-epoch79-84"
        child_raw["tracking"]["group"] = f"ffnr-causal-{args.label}-epoch79-84"
        child_raw["training"]["epochs"] = 84
        intervention: dict[str, Any] = {
            "arm": arm,
            "selector": selectors[arm],
            "kind": kinds[arm],
            "parent": parent,
            "mask": None,
            "uniform_target_softening_rho": 0.5,
            "adversarial_kd_multiplier": kd[arm],
            "adversarial_ce_coefficient": 0.0 if arm == "C79" else 0.25,
        }
        if mask_name is not None:
            source = sources[arm]
            mask_path = masks_root / f"{mask_name}-registered.json"
            custom = run_masks[mask_name]
            ids = [int(value) for value in custom["selected_ids"]]
            mask_refs[arm] = _mask(
                path=mask_path,
                ids=ids,
                labels=labels,
                source=source,
                parent_checkpoint_sha256=parent["checkpoint_sha256"],
                parent_sample_state_sha256=parent["sample_state_sha256"],
                random_seed=20260809 if "random" in mask_name else None,
            )
            intervention["mask"] = mask_refs[arm]
        child_raw["intervention"] = intervention
        path = args.config_root / args.label / f"{arm}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(child_raw, sort_keys=False), encoding="utf-8")
        load_config(path)
        config_paths.append(path)
    args.screen_root.parent.mkdir(parents=True, exist_ok=True)
    if args.screen_root.exists():
        raise FileExistsError(args.screen_root)
    create_intervention_forks(
        parent_checkpoint=args.parent,
        parent_resolved_config=args.parent_config,
        parent_manifest=args.lineage_root / args.label / "parent-manifest.json",
        arm_config_paths=config_paths,
        root=Path.cwd(),
        git_state_collector=lambda _: {"sha": args.git_sha, "dirty": False},
    )
    print(args.screen_root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, choices=("L2", "L4"))
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-config", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--lineage-root", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--git-sha", required=True)
    args = parser.parse_args()
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
