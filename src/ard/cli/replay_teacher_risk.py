"""Replay historical teacher risk on student-crafted training adversarial inputs."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml

from ard.analysis.signal_audit import SignalAuditError, inventory_run_bundle, select_prospective_checkpoints
from ard.analysis.teacher_risk_replay import TeacherRiskReplayError, replay_envelope, repository_root_from_source
from ard.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Signal-audit configuration.")
    parser.add_argument(
        "--output", required=True, type=Path, help="New replay provenance JSON; must not already exist."
    )
    parser.add_argument("--device", required=True, help="Replay device, for example cuda:0 or cpu.")
    parser.add_argument("--batch-size", required=True, type=int, help="Deterministic replay batch size.")
    return parser


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise TeacherRiskReplayError(f"cannot read replay input {path}") from exc
    if not isinstance(parsed, dict):
        raise TeacherRiskReplayError(f"replay input {path} must be a mapping")
    return parsed


def _configured_path(config_dir: Path, value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise TeacherRiskReplayError(f"analysis config {name} must be a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else (config_dir / path).resolve()


def _selected_manifest(config: Mapping[str, Any], *, config_dir: Path) -> tuple[Path, dict[str, Any]]:
    raw_manifests = config.get("manifests")
    if not isinstance(raw_manifests, list):
        raise TeacherRiskReplayError("analysis config requires a manifests list")
    run_id = config.get("prospective_run_id")
    if not isinstance(run_id, str) or not run_id:
        raise TeacherRiskReplayError("analysis config requires prospective_run_id")
    matches: list[tuple[Path, dict[str, Any]]] = []
    for value in raw_manifests:
        path = _configured_path(config_dir, value, name="manifests")
        manifest = _load_mapping(path)
        if manifest.get("run_id") == run_id:
            matches.append((path, manifest))
    if len(matches) != 1:
        raise TeacherRiskReplayError("replay requires exactly one local manifest for the prospective run")
    return matches[0]


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise TeacherRiskReplayError("CUDA replay was requested but CUDA is unavailable")
    if device.type not in {"cpu", "cuda"}:
        raise TeacherRiskReplayError("replay device must be CPU or CUDA")
    return device


def _require_single_process() -> None:
    raw_world_size = os.environ.get("WORLD_SIZE", "1")
    try:
        world_size = int(raw_world_size)
    except ValueError as exc:
        raise TeacherRiskReplayError("WORLD_SIZE must be an integer for single-process replay") from exc
    if world_size != 1:
        raise TeacherRiskReplayError("teacher-risk replay requires WORLD_SIZE=1")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise TeacherRiskReplayError("teacher-risk replay must not run inside initialized distributed execution")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _require_single_process()
    if args.batch_size < 1:
        raise TeacherRiskReplayError("--batch-size must be positive")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite an existing replay provenance envelope")
    config_path = args.config.resolve()
    audit_config = _load_mapping(config_path)
    if audit_config.get("replay_batch_size") != args.batch_size:
        raise TeacherRiskReplayError("--batch-size must exactly match analysis replay_batch_size")
    requested_device = _device(args.device)
    if audit_config.get("replay_device_type") != requested_device.type:
        raise TeacherRiskReplayError("requested --device type must exactly match analysis replay_device_type")
    manifest_path, _ = _selected_manifest(audit_config, config_dir=config_path.parent)
    inventories = inventory_run_bundle(manifest_path)
    historical, _ = select_prospective_checkpoints(
        inventories,
        run_id=str(audit_config["prospective_run_id"]),
        historical_epoch=audit_config.get("historical_epoch"),
    )
    resolved_path = manifest_path.parent / "resolved_config.yaml"
    training_config = load_config(resolved_path)
    envelope = replay_envelope(
        audit_config=audit_config,
        training_config=training_config,
        historical=historical,
        device=requested_device,
        batch_size=args.batch_size,
        repository_root=repository_root_from_source(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(json.dumps(envelope, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n")
    print(json.dumps({"output": str(output), "rows": len(envelope["rows"])}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except (SignalAuditError, TeacherRiskReplayError) as exc:
        raise SystemExit(str(exc)) from exc
