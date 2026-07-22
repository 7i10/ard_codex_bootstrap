#!/usr/bin/env python3
"""Verify pinned RobustBench source evidence and local teacher checkpoint bytes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ard.models.teacher_registry import TeacherRegistry, TeacherRegistryError  # noqa: E402


def verify(*, root: Path, registry_id: str) -> dict[str, str]:
    registry = TeacherRegistry.load(root)
    spec = registry.spec(registry_id)
    registry.validate_external()
    checkpoint = registry.checkpoint_path(spec)
    registry.validate_local_checkpoint(spec, checkpoint)
    return {
        "registry_id": spec.registry_id,
        "upstream_model_id": spec.upstream_model_id,
        "external_commit": registry.repository_commit,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": spec.checkpoint_sha256 or "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--registry-id", required=True)
    args = parser.parse_args()
    try:
        report = verify(root=args.root.resolve(), registry_id=args.registry_id)
        print(" ".join(f"{key}={value}" for key, value in report.items()))
    except TeacherRegistryError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
