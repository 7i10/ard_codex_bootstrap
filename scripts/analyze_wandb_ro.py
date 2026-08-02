#!/usr/bin/env python3
"""Read one explicit W&B cohort and cache exact robust-overfitting histories."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # Allow `python scripts/analyze_wandb_ro.py` from repository root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ard.analysis.wandb_history import WandbHistoryError, analyze_cohort, load_cohort


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/wandb-history"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    cohort, _ = load_cohort(args.cohort)
    entity = args.entity or cohort.get("entity")
    project = args.project or cohort.get("project")
    configured_epochs = cohort["expected_epochs"]
    assert isinstance(configured_epochs, int) and not isinstance(configured_epochs, bool)
    if args.epochs is not None and args.epochs != configured_epochs:
        raise WandbHistoryError("--epochs must equal cohort expected_epochs when explicitly supplied")
    if not isinstance(entity, str) or not entity or not isinstance(project, str) or not project:
        raise WandbHistoryError("cohort or environment must provide entity and project")
    api: Any | None = None

    def fetch_run(run_id: str) -> Any:
        nonlocal api
        if api is None:
            try:
                import wandb
            except ImportError as exc:  # pragma: no cover - environment boundary
                raise WandbHistoryError("install ard[tracking] to analyze W&B history") from exc
            api = wandb.Api()
        return api.run(f"{entity}/{project}/{run_id}")

    result = analyze_cohort(
        cohort_path=args.cohort,
        cache_dir=args.cache_dir,
        fetch_run=fetch_run,
        expected_epochs=configured_epochs,
        force=args.force,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
