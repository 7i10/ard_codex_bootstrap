#!/usr/bin/env python3
"""Create hash-bound stage-wise augmentation fork checkpoints."""

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
from ard.tracking import stable_run_id
from ard.tracking.adapter import collect_git_state

PARENTS = {
    (1, 50): Path(
        "/home/islab/workspace-local/shunsuke.naito/ard-runs/ard_codex_bootstrap/"
        "ert-rslad-single-switch-timing-v1/parents/seed1/s50/epoch-050.pt"
    ),
    (2, 50): Path(
        "/home/islab/workspace-local/shunsuke.naito/ard-runs/ard_codex_bootstrap/"
        "ert-rslad-single-switch-timing-v1/parents/seed2/s50/epoch-050.pt"
    ),
    (1, 75): Path(
        "/home/islab/workspace-local/shunsuke.naito/ard-runs/ard_codex_bootstrap/"
        "ert-rslad-single-switch-timing-v1/parents/seed1/s75/epoch-075.pt"
    ),
    (2, 75): Path(
        "/home/islab/workspace-local/shunsuke.naito/ard-runs/ard_codex_bootstrap/"
        "ert-rslad-single-switch-timing-v1/parents/seed2/s75/epoch-075.pt"
    ),
    (1, 100): Path(
        "/home/islab/workspace-local/shunsuke.naito/ard-runs/ard_codex_bootstrap/"
        "ert-rslad-stagewise-v1/seed1/s100/epoch-100.pt"
    ),
    (2, 100): Path(
        "/home/islab/workspace-local/shunsuke.naito/ard-runs/ard_codex_bootstrap/"
        "ert-rslad-stagewise-v1/seed2/s100/epoch-100.pt"
    ),
    (1, 150): Path(
        "/home/islab/workspace-local/shunsuke.naito/ard-runs/ard_codex_bootstrap/"
        "ert-rslad-stagewise-v1/seed1/s150/epoch-150.pt"
    ),
    (2, 150): Path(
        "/home/islab/workspace-local/shunsuke.naito/ard-runs/ard_codex_bootstrap/"
        "ert-rslad-stagewise-v1/seed2/s150/epoch-150.pt"
    ),
    (1, 125): Path(
        "/home/islab/workspace-local/shunsuke.naito/ard-runs/ard_codex_bootstrap/"
        "ert-rslad-single-switch-timing-v1/parents/seed1/s125/epoch-125.pt"
    ),
    (2, 125): Path(
        "/home/islab/workspace-local/shunsuke.naito/ard-runs/ard_codex_bootstrap/"
        "ert-rslad-single-switch-timing-v1/parents/seed2/s125/epoch-125.pt"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(1, 2), required=True)
    parser.add_argument("--switch", type=int, choices=(50, 75, 100, 125, 150), required=True)
    parser.add_argument("--late-policy", choices=("crop_re", "idbh_weak"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path, default=None, help="override the registered parent path")
    return parser


def _atomic_save(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = build_parser().parse_args()
    parent_path = (args.parent if args.parent is not None else PARENTS[(args.seed, args.switch)]).resolve()
    if not parent_path.is_file():
        raise FileNotFoundError(f"missing materialized parent: {parent_path}")
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    if not isinstance(parent, dict):
        raise ValueError("parent checkpoint must be a mapping")
    expected_epoch = args.switch - 1
    if parent.get("epoch") != expected_epoch or parent.get("epoch_boundary") != "end":
        raise ValueError("parent payload epoch does not match the requested switch boundary")
    missing = REQUIRED_KEYS.difference(parent)
    if missing:
        raise ValueError("parent checkpoint is incomplete: " + ", ".join(sorted(missing)))
    if parent.get("world_size") != 1 or parent.get("global_step") != (args.switch * 352):
        raise ValueError("parent world size/global step is inconsistent with the 45k single-rank source")

    config = load_config(args.config)
    ensure_local_trainable(config.protocol.id)
    if config.protocol.id != "controlled_cifar10_r18_stagewise_augmentation_v1":
        raise ValueError("stage-wise fork requires the registered stage-wise protocol")
    if config.seeds.model_init != args.seed:
        raise ValueError("config model seed does not match fork seed")
    if config.dataset.stagewise_switch_epoch != args.switch or config.dataset.stagewise_late_policy != args.late_policy:
        raise ValueError("config stage-wise policy does not match fork arguments")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite stage-wise fork output: {output}")
    if output == parent_path.resolve().parent:
        raise ValueError("stage-wise output must be distinct from its parent directory")
    git = collect_git_state(Path.cwd())
    if git.get("dirty") is not False or not isinstance(git.get("sha"), str):
        raise ValueError("stage-wise forks require a clean addressable source Git tree")
    config_hash = config_digest(resolved_config_dict(config))
    run_id = config.tracking.run_id or stable_run_id(config, config_hash=config_hash, git_sha=git["sha"])
    if run_id == parent.get("tracker_run_id"):
        raise ValueError("stage-wise child must have a distinct tracking identity")

    transformed = copy.deepcopy(parent)
    transformed["config_hash"] = config_hash
    transformed["tracker_run_id"] = run_id
    metadata = transformed.get("selection_metadata")
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
    transformed["best_metric"] = float("-inf")
    transformed["selection_metadata"] = child_metadata
    transformed["fork_lineage"] = {
        "kind": "stagewise_augmentation_fork_v1",
        "child_tracker_run_id": run_id,
        "child_config_sha256": config_hash,
        "parent_tracker_run_id": parent.get("tracker_run_id"),
        "parent_checkpoint_sha256": sha256(parent_path),
        "parent_payload_epoch": expected_epoch,
        "parent_completed_epoch_count": args.switch,
        "parent_config_sha256": parent.get("config_hash"),
        "parent_git_sha": "ffc217dd635462e1f14c93720561208db2d70254",
        "fork_git_sha": git["sha"],
        "switch_epoch": args.switch,
        "late_policy": args.late_policy,
        "prefix_policy": "cropshift",
        "post_fork_best_scope": True,
    }
    output.mkdir(parents=True)
    _atomic_save(transformed, output / "last.pt")
    save_resolved_config(config, output / "resolved_config.yaml")
    (output / "fork-lineage.json").write_text(
        json.dumps(transformed["fork_lineage"], sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    checkpoint_sha = sha256(output / "last.pt")
    (output / "stagewise-fork-complete.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "stagewise_augmentation_fork_v1",
                "status": "complete",
                "run_id": run_id,
                "config_hash": config_hash,
                "fork_checkpoint_sha256": checkpoint_sha,
                "parent_checkpoint_sha256": sha256(parent_path),
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output / "last.pt"), "sha256": checkpoint_sha, "run_id": run_id}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
