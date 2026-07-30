"""Run the local, read-only seed-zero signal audit from an explicit config."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml

from ard.analysis.signal_audit import (
    CheckpointInventory,
    SignalAuditError,
    associate_wandb_versions,
    audit_report,
    inventory_run_bundle,
    load_final_sample_stats,
    logical_dataset_fingerprint,
    logical_dataset_identity,
    replay_protocol,
    select_prospective_checkpoints,
    write_audit_report,
)
from ard.config.schema import TrainingConfig, training_execution_identity
from ard.engine.checkpoint import config_digest


def _load(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise SignalAuditError(f"cannot parse audit input {path}") from exc
    if not isinstance(parsed, dict):
        raise SignalAuditError(f"audit input {path} must be an object")
    return parsed, hashlib.sha256(raw).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Resolved JSON/YAML audit configuration.")
    parser.add_argument("--output", required=True, type=Path, help="Canonical JSON report path.")
    parser.add_argument(
        "--teacher-risk-replay",
        type=Path,
        help="Override the configured replay provenance JSON without modifying the analysis config.",
    )
    return parser


def _run_manifest(manifests: list[Path], *, run_id: str) -> tuple[Path, dict[str, Any]]:
    matches = []
    for path in manifests:
        source, _ = _load(path)
        if source.get("run_id") == run_id:
            matches.append((path, source))
    if len(matches) != 1:
        raise SignalAuditError("prospective run must resolve to exactly one local manifest")
    return matches[0]


def _source_tree_clean_git_sha() -> str:
    root = Path(__file__).resolve().parents[3]
    try:
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"], check=True, capture_output=True, text=True
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SignalAuditError("formal replay validation requires a readable source-tree Git identity") from exc
    if dirty:
        raise SignalAuditError("formal replay validation requires a clean source-tree Git worktree")
    if len(sha) != 40 or any(character not in "0123456789abcdef" for character in sha):
        raise SignalAuditError("source-tree Git SHA must be an exact lowercase commit SHA")
    return sha


def _lineage_for_formal_replay(
    *,
    config: Mapping[str, Any],
    manifests: list[Path],
    final_sha256: str,
    run_id: str,
    config_hash: str,
    historical: CheckpointInventory,
    require_clean_replay_git: bool,
) -> dict[str, Any]:
    manifest_path, manifest = _run_manifest(manifests, run_id=run_id)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not any(
        isinstance(item, Mapping) and item.get("type") == "sample-stats" and item.get("sha256") == final_sha256
        for item in artifacts
    ):
        raise SignalAuditError("final Parquet SHA is not bound to a sample-stats artifact in the selected manifest")
    resolved_path = manifest_path.parent / "resolved_config.yaml"
    resolved, _ = _load(resolved_path)
    if config_digest(resolved) != config_hash:
        raise SignalAuditError("resolved config SHA does not match manifest/checkpoint config identity")
    teacher = manifest.get("teacher")
    resolved_teacher = resolved.get("teacher")
    method = resolved.get("method")
    seeds = resolved.get("seeds")
    if not isinstance(teacher, Mapping) or not isinstance(resolved_teacher, Mapping) or not isinstance(method, Mapping):
        raise SignalAuditError("selected manifest/resolved config lacks teacher or method identity")
    teacher_sha = teacher.get("checkpoint_actual_sha256") or teacher.get("checkpoint_sha256")
    if _hex(teacher_sha, "manifest teacher checkpoint SHA") != _hex(
        resolved_teacher.get("checkpoint_sha256"), "resolved teacher checkpoint SHA"
    ):
        raise SignalAuditError("manifest and resolved config teacher checkpoint SHA differ")
    if config.get("method_id") != method.get("id"):
        raise SignalAuditError("analysis method_id does not match resolved training method")
    analysis_teacher = config.get("teacher")
    if not isinstance(analysis_teacher, Mapping) or analysis_teacher.get("registry_id") != resolved_teacher.get(
        "registry_id"
    ):
        raise SignalAuditError("analysis teacher registry_id does not match resolved training teacher")
    seed = seeds.get("model_init") if isinstance(seeds, Mapping) else None
    if config.get("training_seed") != seed or manifest.get("training_seed") != seed:
        raise SignalAuditError("analysis/resolved/manifest training seed identities differ")
    attack = method.get("attack")
    declared_identity = config.get("threat_identity", config.get("attack_identity"))
    if not isinstance(attack, Mapping) or not isinstance(declared_identity, Mapping):
        raise SignalAuditError("resolved config and analysis config require attack identities")
    train_expected_count = config.get("train_expected_count", 45000)
    derived_dataset_fingerprint = logical_dataset_fingerprint(resolved, train_expected_count=train_expected_count)
    if config.get("dataset_fingerprint") != derived_dataset_fingerprint:
        raise SignalAuditError("analysis dataset_fingerprint does not match the resolved training dataset")
    if json.dumps(declared_identity, sort_keys=True, separators=(",", ":")) != json.dumps(
        attack, sort_keys=True, separators=(",", ":")
    ):
        raise SignalAuditError("analysis threat/attack identity does not match the resolved training method attack")
    replay_batch_size = config.get("replay_batch_size")
    if isinstance(replay_batch_size, bool) or not isinstance(replay_batch_size, int) or replay_batch_size < 1:
        raise SignalAuditError("analysis replay_batch_size must be a positive integer")
    replay_device_type = config.get("replay_device_type")
    if replay_device_type != "cuda":
        raise SignalAuditError("analysis replay_device_type must be exactly cuda")
    checkpoint_payload = torch.load(historical.path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint_payload, Mapping):
        raise SignalAuditError("selected replay checkpoint is unreadable")
    global_step, train_attack_seed = checkpoint_payload.get("global_step"), seeds.get("train_attack")
    if (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step < 0
        or isinstance(train_attack_seed, bool)
        or not isinstance(train_attack_seed, int)
    ):
        raise SignalAuditError("selected replay checkpoint/global train attack seed are invalid")
    checkpoint_world_size = checkpoint_payload.get("world_size")
    if (
        isinstance(checkpoint_world_size, bool)
        or not isinstance(checkpoint_world_size, int)
        or checkpoint_world_size < 1
    ):
        raise SignalAuditError("selected replay checkpoint world_size is invalid")
    training_mapping = resolved.get("training")
    if not isinstance(training_mapping, Mapping):
        raise SignalAuditError("resolved training config is missing execution identity")
    try:
        execution_identity = training_execution_identity(
            training=TrainingConfig.model_validate(training_mapping), world_size=checkpoint_world_size
        )
    except ValueError as exc:
        raise SignalAuditError("selected replay checkpoint/config training execution identity is invalid") from exc
    return {
        "teacher_checkpoint_sha256": _hex(teacher_sha, "teacher checkpoint SHA"),
        "dataset_fingerprint": derived_dataset_fingerprint,
        "dataset_identity": logical_dataset_identity(resolved, train_expected_count=train_expected_count)["dataset"],
        "threat_or_attack_identity": attack,
        "replay_protocol": replay_protocol(
            batch_size=replay_batch_size,
            attack_seed_base=train_attack_seed + 1_000_003 * global_step,
            device_type=replay_device_type,
        ),
        "replay_git_sha": _source_tree_clean_git_sha() if require_clean_replay_git else None,
        "checkpoint_training": {"world_size": checkpoint_world_size, "execution_identity": execution_identity},
    }


def _hex(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SignalAuditError(f"{name} must be a lowercase SHA-256")
    return value


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config, config_source_hash = _load(args.config)
    config_dir = args.config.parent

    def configured_path(value: object, *, name: str) -> Path:
        if not isinstance(value, str) or not value:
            raise SignalAuditError(f"audit config {name} must be a non-empty path")
        return (config_dir / value).resolve() if not Path(value).is_absolute() else Path(value)

    try:
        manifests = [configured_path(value, name="manifests") for value in config["manifests"]]
    except (KeyError, TypeError) as exc:
        raise SignalAuditError("audit config requires a manifests list") from exc
    if "final_parquet" in config:
        final_path = configured_path(config["final_parquet"], name="final_parquet")
        final_hash = hashlib.sha256(final_path.read_bytes()).hexdigest()
        final_rows = load_final_sample_stats(
            final_path,
            expected_count=int(config.get("train_expected_count", 45000)),
            num_classes=int(config.get("num_classes", 10)),
            stored_risk_kind=str(config.get("stored_risk_kind", "joint")),
        )
    else:
        final_path = configured_path(config.get("final_rows"), name="final_rows")
        final_source, final_hash = _load(final_path)
        final_rows = final_source.get("rows")
        if not isinstance(final_rows, list):
            raise SignalAuditError("prepared final_rows input requires a rows list")
    inventories = tuple(item for manifest in manifests for item in inventory_run_bundle(manifest))
    prospective_run_id = config.get("prospective_run_id")
    if not isinstance(prospective_run_id, str) or "historical_epoch" not in config:
        raise SignalAuditError("audit config requires prospective_run_id and historical_epoch")
    historical, final = select_prospective_checkpoints(
        inventories, run_id=prospective_run_id, historical_epoch=config["historical_epoch"]
    )
    wandb_hashes: dict[str, str] = {}
    if "wandb_inventory" in config:
        wandb_path = configured_path(config["wandb_inventory"], name="wandb_inventory")
        wandb_source, wandb_hash = _load(wandb_path)
        wandb_rows = wandb_source.get("artifacts")
        if not isinstance(wandb_rows, list):
            raise SignalAuditError("W&B inventory input requires an artifacts list")
        selected_rows = [
            row for row in wandb_rows if isinstance(row, Mapping) and row.get("run_id") == prospective_run_id
        ]
        selected = associate_wandb_versions((historical, final), selected_rows)
        if (selected[0].wandb_version, selected[1].wandb_version) != ("v19", "v39"):
            raise SignalAuditError("selected historical/final checkpoints require exact W&B versions v19/v39")
        bindings = {(item.run_id, item.publication_order): item for item in selected}
        inventories = tuple(bindings.get((item.run_id, item.publication_order), item) for item in inventories)
        wandb_hashes[str(wandb_path)] = wandb_hash
    manifest_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in manifests}
    replay_requested = args.teacher_risk_replay is not None or "teacher_risk_replay" in config
    lineage = _lineage_for_formal_replay(
        config=config,
        manifests=manifests,
        final_sha256=final_hash,
        run_id=prospective_run_id,
        config_hash=historical.config_hash,
        historical=historical,
        require_clean_replay_git=replay_requested,
    )
    replay: dict[str, Any] | None = None
    replay_hashes: dict[str, str] = {}
    if args.teacher_risk_replay is not None:
        replay_path = args.teacher_risk_replay.resolve()
        replay, replay_hash = _load(replay_path)
        replay_hashes[str(replay_path)] = replay_hash
    elif "teacher_risk_replay" in config:
        replay_path = configured_path(config["teacher_risk_replay"], name="teacher_risk_replay")
        replay, replay_hash = _load(replay_path)
        replay_hashes[str(replay_path)] = replay_hash
    report = audit_report(
        config=config,
        inventories=inventories,
        final_rows=final_rows,
        input_hashes={
            "config_source": config_source_hash,
            "final_rows": final_hash,
            **manifest_hashes,
            **wandb_hashes,
            **replay_hashes,
        },
        teacher_risk_replay=replay,
        lineage=lineage,
    )
    write_audit_report(args.output, report)
    print(
        json.dumps(
            {"output": str(args.output), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
