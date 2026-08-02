#!/usr/bin/env python3
"""Additive, explicit-registry W&B run tagging; dry-run is the default."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml


class WandbTagError(ValueError):
    pass


def load_registry(path: Path) -> tuple[dict[str, Any], tuple[tuple[str, tuple[str, ...]], ...]]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise WandbTagError("tag registry is unreadable") from exc
    if not isinstance(value, Mapping) or not isinstance(value.get("runs"), list):
        raise WandbTagError("tag registry must contain a top-level runs list")
    rows: list[tuple[str, tuple[str, ...]]] = []
    for entry in value["runs"]:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("run_id"), str) or not entry["run_id"]:
            raise WandbTagError("each tag registry entry requires run_id")
        tags = entry.get("tags")
        if not isinstance(tags, list) or not tags or any(not isinstance(tag, str) or not tag for tag in tags):
            raise WandbTagError("each tag registry entry requires a non-empty string tags list")
        rows.append((entry["run_id"], tuple(sorted(set(tags)))))
    if not rows or len({run_id for run_id, _ in rows}) != len(rows):
        raise WandbTagError("tag registry run IDs must be non-empty and unique")
    return dict(value), tuple(rows)


def tag_registered_runs(
    *, registry_path: Path, fetch_run: Callable[[str], Any], apply: bool = False
) -> list[dict[str, Any]]:
    _, entries = load_registry(registry_path)
    report: list[dict[str, Any]] = []
    for run_id, desired in entries:
        run = fetch_run(run_id)
        if getattr(run, "id", None) != run_id:
            raise WandbTagError("W&B client returned a run with the wrong explicit ID")
        existing = tuple(sorted(set(getattr(run, "tags", ()))))
        merged = tuple(sorted(set(existing).union(desired)))
        changed = existing != merged
        if apply and changed:
            run.tags = merged
            run.update()
        report.append(
            {
                "run_id": run_id,
                "existing_tags": list(existing),
                "desired_tags": list(desired),
                "final_tags": list(merged),
                "changed": changed,
                "applied": bool(apply and changed),
            }
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT"))
    parser.add_argument("--apply", action="store_true", help="Apply additive tag changes; default is dry-run.")
    args = parser.parse_args(argv)
    registry, _ = load_registry(args.registry)
    entity, project = args.entity or registry.get("entity"), args.project or registry.get("project")
    if not isinstance(entity, str) or not entity or not isinstance(project, str) or not project:
        raise WandbTagError("registry or environment must provide entity and project")
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover
        raise WandbTagError("install ard[tracking] to tag W&B runs") from exc
    api = wandb.Api()
    report = tag_registered_runs(
        registry_path=args.registry,
        fetch_run=lambda run_id: api.run(f"{entity}/{project}/{run_id}"),
        apply=args.apply,
    )
    print(json.dumps({"dry_run": not args.apply, "runs": report}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
