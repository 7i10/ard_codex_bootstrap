#!/usr/bin/env python3
"""Materialize a child config with the frozen sample-keyed attack contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = yaml.safe_load(args.input.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("method"), dict):
        raise ValueError("config is missing method mapping")
    attack = payload["method"].get("attack")
    if not isinstance(attack, dict):
        raise ValueError("config is missing method.attack mapping")
    attack["random_start_keying"] = "sample_keyed_v1"
    payload["method"]["attack"] = attack
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
