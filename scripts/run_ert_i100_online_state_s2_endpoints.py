#!/usr/bin/env python3
"""Run the registered held-out CE-PGD20 endpoints for one Online-S2 arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml

from ard.analysis.ert_stage_a_endpoint import evaluate_endpoint
from ard.config import load_config

ENDPOINT_ATTACK_SHA256 = "7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2"
HORIZONS = (104, 109, 114)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_clean_source(expected: str) -> None:
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    if actual != expected or dirty:
        raise SystemExit("endpoint replay requires the frozen clean production source")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read required JSON artifact: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"required JSON artifact is not a mapping: {path}")
    return value


def _parent_config(training_root: Path) -> Path:
    lineage = training_root / "resolved_config.yaml"
    if not lineage.is_file():
        raise SystemExit(f"training lineage is missing: {lineage}")
    payload = yaml.safe_load(lineage.read_text(encoding="utf-8"))
    value = payload.get("parent_config") if isinstance(payload, dict) else None
    config = Path(value) if isinstance(value, str) else None
    if config is None or not config.is_file():
        raise SystemExit("training lineage does not bind an available parent config")
    return config


def _output_is_fresh(path: Path) -> bool:
    """Allow only orchestrator bookkeeping before a public CLI creates output."""

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
    _require_clean_source(args.expected_source_sha)
    training_root = args.training_root.resolve()
    output = args.output.resolve()
    if not _output_is_fresh(output):
        raise SystemExit(f"refusing to overwrite endpoint output: {output}")
    arm_summary = _read_json(training_root / "arm-summary.json")
    if arm_summary.get("seed") != args.seed or arm_summary.get("arm") != args.arm:
        raise SystemExit("training arm summary does not match the registered endpoint job")
    config_path = _parent_config(training_root)
    config = load_config(config_path)
    if config.method.selection_attack is None:
        raise SystemExit("parent config lacks the registered CE-PGD20 endpoint attack")
    if config.method.selection_attack.identity_sha256() != ENDPOINT_ATTACK_SHA256:
        raise SystemExit("parent config endpoint attack identity drifted")
    horizon = _read_json(training_root / "horizon-checkpoints.json")
    if set(horizon) != {str(epoch) for epoch in HORIZONS}:
        raise SystemExit("training result lacks the exact e104/e109/e114 horizon checkpoint set")
    outputs = []
    for epoch in HORIZONS:
        descriptor = horizon[str(epoch)]
        checkpoint = training_root / "checkpoints" / f"epoch-{epoch}.pt"
        if (
            not checkpoint.is_file()
            or not isinstance(descriptor, dict)
            or descriptor.get("path") != str(checkpoint.resolve())
            or descriptor.get("sha256") != _sha256(checkpoint)
        ):
            raise SystemExit(f"registered horizon checkpoint differs for epoch {epoch}")
        outputs.append(
            evaluate_endpoint(
                config_path=config_path,
                checkpoint=checkpoint,
                output_dir=output / f"e{epoch}-validation",
                device=torch.device(args.device),
                expected_epoch=epoch,
                split="validation",
            )
        )
    summary = {
        "schema_version": 1,
        "contract": "ert_rslad_i100_online_state_s2_endpoint_v1",
        "seed": args.seed,
        "arm": args.arm,
        "source_git_sha": args.expected_source_sha,
        "training_root": str(training_root),
        "outputs": outputs,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
