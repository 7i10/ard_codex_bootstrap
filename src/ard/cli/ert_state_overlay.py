"""Create the read-only, hash-bound ERT epoch-79 CPU state overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ard.analysis.ert_state_overlay import ERTStateOverlayError, run_overlay


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    paths = run_overlay(config_path=args.config.resolve(), output_dir=args.output_dir.resolve())
    print(json.dumps({name: str(path) for name, path in sorted(paths.items())}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except ERTStateOverlayError as exc:
        raise SystemExit(str(exc)) from exc
