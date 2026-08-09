"""Run one condition of the preregistered CE/KL x PGD10/20 FFNR replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ard.analysis.ffnr_attack_factorial import CONDITIONS, FactorialReplayError, run_factorial


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--device", required=True)
    parser.add_argument("--epochs", type=int, nargs="+")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    paths = run_factorial(
        config_path=args.config.resolve(),
        condition=args.condition,
        device=device,
        requested_epochs=args.epochs,
    )
    print(json.dumps({key: str(value) for key, value in sorted(paths.items())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FactorialReplayError as exc:
        raise SystemExit(str(exc)) from exc
