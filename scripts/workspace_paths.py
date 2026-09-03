#!/usr/bin/env python3
"""Read one field from the tracked ARD workspace registry.

This tiny stdlib-only helper is intended for shell lifecycle wrappers.  It
keeps their defaults in the same registry used by Python runtime code instead
of duplicating host-local absolute paths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO_ROOT / "configs" / "workspace" / "ard_workspace_v1.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("field")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args()
    values = json.loads(args.registry.read_text(encoding="utf-8"))
    value = values.get(args.field)
    if not isinstance(value, str) or not value:
        raise SystemExit(f"workspace registry has no non-empty string field: {args.field}")
    print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
