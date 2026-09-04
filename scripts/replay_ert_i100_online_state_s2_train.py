#!/usr/bin/env python3
"""Replay the e114 canonical raw-train CE-PGD20 state for one Online-S2 arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml

from ard.analysis.ert_i100_s2_longitudinal import CONTRACT, replay_canonical_state
from ard.config import load_config

ENDPOINT_ATTACK_SHA256 = "7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2"
TEACHER_SHA256 = "fc398a4890e6856b5dd80856076000ec9e2debdd12d9f78a66171b9ffc383983"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read required JSON artifact: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"required JSON artifact is not a mapping: {path}")
    return value


def _require_source(expected: str) -> None:
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    if actual != expected or dirty:
        raise SystemExit("canonical state replay requires the frozen clean production source")


def _parent_config(training_root: Path) -> Path:
    lineage = training_root / "resolved_config.yaml"
    if not lineage.is_file():
        raise SystemExit(f"training lineage is missing: {lineage}")
    payload = yaml.safe_load(lineage.read_text(encoding="utf-8"))
    raw = payload.get("parent_config") if isinstance(payload, dict) else None
    config = Path(raw) if isinstance(raw, str) else None
    if config is None or not config.is_file():
        raise SystemExit("training lineage does not bind an available parent config")
    return config


def _output_is_fresh(path: Path) -> bool:
    """Allow only orchestrator bookkeeping before the public replay creates output."""

    if not path.exists():
        return True
    return all(entry.name == "orchestration" for entry in path.iterdir())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", choices=("dev-1", "dev-2"), required=True)
    parser.add_argument("--arm", choices=("control", "pmp", "dbdp"), required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    _require_source(args.expected_source_sha)
    training_root = args.training_root.resolve()
    output = args.output.resolve()
    if not _output_is_fresh(output):
        raise SystemExit(f"refusing to overwrite canonical replay output: {output}")
    summary = _read_json(training_root / "arm-summary.json")
    if summary.get("seed") != args.seed or summary.get("arm") != args.arm:
        raise SystemExit("training arm summary does not match the registered canonical replay job")
    config_path = _parent_config(training_root)
    config = load_config(config_path)
    if (
        config.method.selection_attack is None
        or config.method.selection_attack.identity_sha256() != ENDPOINT_ATTACK_SHA256
    ):
        raise SystemExit("canonical replay endpoint attack identity drifted")
    if config.teacher is None or config.teacher.checkpoint_sha256 != TEACHER_SHA256:
        raise SystemExit("canonical replay Teacher identity drifted")
    horizon = _read_json(training_root / "horizon-checkpoints.json")
    descriptor = horizon.get("114")
    checkpoint = training_root / "checkpoints" / "epoch-114.pt"
    if (
        not isinstance(descriptor, dict)
        or not checkpoint.is_file()
        or descriptor.get("path") != str(checkpoint.resolve())
        or descriptor.get("sha256") != _sha256(checkpoint)
    ):
        raise SystemExit("canonical replay e114 checkpoint differs from the child horizon manifest")
    result = replay_canonical_state(
        config_path=config_path,
        checkpoint=checkpoint,
        expected_checkpoint_sha256=str(descriptor["sha256"]),
        expected_epoch=114,
        output_dir=output,
        device=torch.device(args.device),
    )
    if (
        result.get("contract") != CONTRACT
        or result.get("observation", {}).get("attack_identity_sha256") != ENDPOINT_ATTACK_SHA256
    ):
        raise SystemExit("canonical replay did not preserve the registered observation contract")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
