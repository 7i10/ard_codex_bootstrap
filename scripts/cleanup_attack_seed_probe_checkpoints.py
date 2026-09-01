#!/usr/bin/env python3
"""Release redundant attack-seed training checkpoints after endpoint evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED_PREFIX = "attack-seed-"
CHECKPOINTS = ("last.pt", "epoch-114.pt")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if run_dir.name.startswith(EXPECTED_PREFIX) is False:
        raise ValueError(f"refusing cleanup outside an attack-seed run: {run_dir}")
    removed: list[str] = []
    for name in CHECKPOINTS:
        path = run_dir / name
        if path.is_file():
            path.unlink()
            removed.append(name)
    report = {"run_dir": str(run_dir), "removed": removed, "status": "completed"}
    output = run_dir / "checkpoint-cleanup.json"
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
