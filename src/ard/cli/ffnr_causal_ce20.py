"""Replay one explicit FF/NR causal-pilot endpoint under common CE-PGD20."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ard.analysis.ffnr_causal_ce20 import HORIZONS, LABELS, CausalCE20Error, run_causal_endpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--label", required=True, choices=LABELS)
    parser.add_argument("--horizon", required=True, choices=HORIZONS, type=int)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    paths = run_causal_endpoint(
        config_path=args.config.resolve(),
        label=args.label,
        horizon=args.horizon,
        device=torch.device(args.device),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps({name: str(path) for name, path in sorted(paths.items())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CausalCE20Error as exc:
        raise SystemExit(str(exc)) from exc
