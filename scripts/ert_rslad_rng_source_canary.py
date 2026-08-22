#!/usr/bin/env python3
"""Run the bounded RNG-source isolation canary before GPU production."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ard.analysis.ert_rslad_rng_sources import run_seed_isolation_canary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_seed_isolation_canary(batches=args.batches)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
