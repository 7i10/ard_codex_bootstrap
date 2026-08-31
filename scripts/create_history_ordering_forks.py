#!/usr/bin/env python3
"""Materialize hash-bound epoch-99 forks for sample-keyed ordering runs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import torch

from ard.config import load_config, save_resolved_config
from ard.config.loader import resolved_config_dict
from ard.engine.checkpoint import REQUIRED_KEYS, config_digest
from ard.protocols import ensure_local_trainable
from ard.tracking.adapter import collect_git_state

PARENT_HASHES = {
    1: "360910a8a886cf904b206c9381cdf6eaa3e71d6150c0998224c7ab4307630835",
    2: "bb0c7c1ace81fd3df1b85660af265b91b1cefd6e91f3ce5d035b0d0c94f7aaf7",
}
PARENT_EPOCH = 99
HISTORY_PREDICTOR_SHA = "b6be23e7eaa31dd300ed21efb84746bba2270a4a044096c86d3be035c8abebbd"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_save(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(1, 2), required=True)
    parser.add_argument("--arm", choices=("NEW_CONTROL", "NEW_HISTORY"), required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    parent_path = args.parent.resolve()
    if not parent_path.is_file() or _sha256(parent_path) != PARENT_HASHES[args.seed]:
        raise ValueError("parent checkpoint is not the registered epoch-99 I100 boundary")
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    if not isinstance(parent, dict) or PARENT_EPOCH != parent.get("epoch") or parent.get("epoch_boundary") != "end":
        raise ValueError("parent payload is not the exact epoch-99 end boundary")
    missing = REQUIRED_KEYS.difference(parent)
    if missing:
        raise ValueError("parent checkpoint is incomplete: " + ", ".join(sorted(missing)))
    if parent.get("world_size") != 1 or parent.get("global_step") != 35_200:
        raise ValueError("parent world size/global step is inconsistent with the registered I100 parent")
    config = load_config(
        args.config,
        [
            "method.attack.random_start_keying=sample_keyed_v1",
            f"tracking.run_id=ert-rslad-history-ordering-{args.arm.lower()}-s{args.seed}",
            f"tracking.name=ert-rslad-history-ordering-{args.arm.lower()}-s{args.seed}",
            "tracking.artifact_retention=metrics_only",
            f"output_dir={args.output.resolve()}",
        ],
    )
    ensure_local_trainable(config.protocol.id)
    if config.protocol.id != "controlled_cifar10_r18_stagewise_augmentation_v1":
        raise ValueError("history ordering requires the registered stagewise I100 protocol")
    if config.seeds.model_init != args.seed or config.dataset.stagewise_switch_epoch != 100:
        raise ValueError("config seed/stagewise identity does not match this fork")
    git = collect_git_state(Path.cwd())
    if git.get("dirty") is not False or git.get("sha") != args.source_sha:
        raise ValueError("fork source SHA must be a clean current Git SHA")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite fork output: {output}")
    config_hash = config_digest(resolved_config_dict(config))
    run_id = config.tracking.run_id
    if not isinstance(run_id, str) or run_id == parent.get("tracker_run_id"):
        raise ValueError("child run identity must be distinct and explicit")
    metadata = parent.get("selection_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("parent selection metadata is missing")
    child_metadata = copy.deepcopy(dict(metadata))
    child_metadata["selected_epoch"] = None
    child_metadata["scope"] = "post_fork_best"
    for key in (
        "selected_clean_accuracy",
        "selected_pgd_accuracy",
        "last_epoch",
        "last_clean_accuracy",
        "last_pgd_accuracy",
    ):
        child_metadata.pop(key, None)
    transformed = copy.deepcopy(parent)
    transformed["config_hash"] = config_hash
    transformed["tracker_run_id"] = run_id
    transformed["best_metric"] = float("-inf")
    transformed["selection_metadata"] = child_metadata
    transformed["fork_lineage"] = {
        "kind": "stagewise_augmentation_fork_v1",
        "child_tracker_run_id": run_id,
        "child_config_sha256": config_hash,
        "parent_tracker_run_id": parent.get("tracker_run_id"),
        "parent_checkpoint_sha256": _sha256(parent_path),
        "parent_config_sha256": parent.get("config_hash"),
        "parent_payload_epoch": PARENT_EPOCH,
        "parent_completed_epoch_count": 100,
        "parent_git_sha": args.source_sha,
        "fork_git_sha": args.source_sha,
        "prefix_policy": "I100_CROPSHIFT_0_99_IDBH_WEAK_100_199",
        "ordering_policy": "history_balanced_v1" if args.arm == "NEW_HISTORY" else "epoch_shuffle_control",
        "history_predictor": "H2_margin_ema_direct_negative_rank",
        "history_predictor_sha256": HISTORY_PREDICTOR_SHA,
        "attack_random_start_keying": "sample_keyed_v1",
        "post_fork_best_scope": True,
    }
    output.mkdir(parents=True)
    _atomic_save(transformed, output / "last.pt")
    save_resolved_config(config, output / "resolved_config.yaml")
    lineage_path = output / "fork-lineage.json"
    lineage_path.write_text(json.dumps(transformed["fork_lineage"], sort_keys=True, indent=2) + "\n", encoding="utf-8")
    marker = {
        "schema_version": 1,
        "kind": "stagewise_augmentation_fork_v1",
        "status": "complete",
        "arm": args.arm,
        "seed": args.seed,
        "run_id": run_id,
        "config_hash": config_hash,
        "fork_checkpoint_sha256": _sha256(output / "last.pt"),
        "parent_checkpoint_sha256": _sha256(parent_path),
    }
    (output / "stagewise-fork-complete.json").write_text(
        json.dumps(marker, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output / "last.pt"), **marker}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
