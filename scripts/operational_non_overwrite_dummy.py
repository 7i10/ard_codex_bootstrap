#!/usr/bin/env python3
"""Bounded public CLI used only to verify operational output ownership.

This command deliberately has the same fail-closed ownership rule as a
scientific non-overwriting CLI: its output directory must not exist before the
command starts.  It performs no model, dataset, checkpoint, or GPU work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def source_sha() -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "unable to resolve source SHA")
    return result.stdout.strip().lower()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--payload", required=True)
    parser.add_argument("--expected-source-sha")
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=0.0,
        help="bounded process lifetime for an operational host-confirmation check",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.hold_seconds <= 10.0:
        raise ValueError("--hold-seconds must be in [0, 10]")
    resolved_sha = source_sha()
    if args.expected_source_sha and resolved_sha != args.expected_source_sha.lower():
        raise ValueError("current source SHA differs from --expected-source-sha")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True)
    artifact = {
        "schema_version": 1,
        "kind": "operational_non_overwrite_dummy",
        "payload": args.payload,
        "payload_sha256": hashlib.sha256(args.payload.encode("utf-8")).hexdigest(),
        "source_sha": resolved_sha,
        "observed_hostname": socket.gethostname(),
    }
    (output / "artifact.json").write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.hold_seconds:
        time.sleep(args.hold_seconds)
    print(json.dumps({"status": "ok", "output_dir": str(output), "artifact": artifact}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
