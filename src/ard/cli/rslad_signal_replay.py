"""Replay a common RSLAD trajectory without modifying training artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml

from ard.analysis.rslad_signal_replay import (
    FEATURE_EPOCHS,
    OUTCOME_EPOCHS,
    RSLADSignalReplayError,
    build_feature_panel,
    build_outcome_panel,
    checkpoint_cache_identity,
    feature_replay_lineage,
    inventory_common_trajectory,
    inventory_feature_trajectory,
    join_feature_outcome_panels,
    load_cached_checkpoint,
    portable_cifar10_train_identity,
    predictive_audit,
    replay_checkpoint_rows,
    replay_lineage,
    runtime_identity,
    tracked_clean_analysis_provenance,
    validate_rslad_replay_attack,
    write_checkpoint_cache,
    write_feature_replay_outputs,
    write_replay_outputs,
)
from ard.analysis.signal_audit import inventory_run_bundle
from ard.analysis.teacher_risk_replay import build_replay_loader
from ard.config import load_config
from ard.engine.checkpoint import config_digest
from ard.models import build_teacher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Common-trajectory replay YAML/JSON.")
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="New output directory or cache-only resumable directory."
    )
    parser.add_argument("--device", required=True, help="Replay device, for example cuda:0.")
    parser.add_argument(
        "--feature-only",
        action="store_true",
        help="Replay only the epoch-4..99 feature panel; never read post-anchor outcome checkpoints.",
    )
    return parser


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RSLADSignalReplayError(f"cannot read common-trajectory replay config {path}") from exc
    if not isinstance(parsed, dict):
        raise RSLADSignalReplayError("common-trajectory replay config must be a mapping")
    return parsed


def saved_resolved_config_digests(path: Path) -> dict[str, str]:
    """Return distinct canonical-mapping and source-byte digests for saved config."""
    raw = path.read_bytes()
    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise RSLADSignalReplayError(f"cannot parse saved resolved config {path}") from exc
    if not isinstance(parsed, dict):
        raise RSLADSignalReplayError("saved resolved config must be a mapping")
    return {
        "mapping_sha256": config_digest(parsed),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _configured_path(root: Path, value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RSLADSignalReplayError(f"replay config {name} must be a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _require_int(config: Mapping[str, Any], name: str, *, minimum: int = 0) -> int:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RSLADSignalReplayError(f"replay config {name} must be an integer >= {minimum}")
    return value


def _require_single_process() -> None:
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as exc:
        raise RSLADSignalReplayError("WORLD_SIZE must be an integer for common-trajectory replay") from exc
    if world_size != 1 or (torch.distributed.is_available() and torch.distributed.is_initialized()):
        raise RSLADSignalReplayError("common-trajectory replay requires one non-distributed process")


def _replay_with_cache(
    *,
    cache_dir: Path,
    checkpoint: Any,
    training_config: Any,
    teacher: Any,
    loader: Any,
    device: torch.device,
    seed_domain: str,
    base_seed: int,
    expected_count: int,
    replay_batch_size: int,
    saved_resolved_config_mapping_sha256: str,
    saved_resolved_config_file_sha256: str,
    teacher_metadata: Mapping[str, Any],
    dataset_identity: Mapping[str, Any],
    analysis_provenance: Mapping[str, Any],
) -> Any:
    identity = checkpoint_cache_identity(
        checkpoint=checkpoint,
        training_config=training_config,
        seed_domain=seed_domain,
        base_seed=base_seed,
        expected_count=expected_count,
        device=device,
        replay_batch_size=replay_batch_size,
        saved_resolved_config_mapping_sha256=saved_resolved_config_mapping_sha256,
        saved_resolved_config_file_sha256=saved_resolved_config_file_sha256,
        teacher_metadata=teacher_metadata,
        dataset_identity=dataset_identity,
        analysis_provenance=analysis_provenance,
    )
    cached = load_cached_checkpoint(cache_dir=cache_dir, identity=identity)
    if cached is not None:
        return cached
    result = replay_checkpoint_rows(
        checkpoint=checkpoint,
        training_config=training_config,
        teacher=teacher,
        loader=loader,
        device=device,
        seed_domain=seed_domain,
        base_seed=base_seed,
    )
    return write_checkpoint_cache(cache_dir=cache_dir, identity=identity, result=result)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _require_single_process()
    config_path = args.config.resolve()
    config = _load_mapping(config_path)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(item.name != "checkpoint-cache" for item in output_dir.iterdir()):
        raise FileExistsError("output directory contains non-cache files; refusing to overwrite replay outputs")
    device = torch.device(args.device)
    if device.type != config.get("replay_device_type") or device.type not in {"cpu", "cuda"}:
        raise RSLADSignalReplayError("requested device type must exactly match replay_device_type")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RSLADSignalReplayError("CUDA replay was requested but CUDA is unavailable")
    analysis_provenance = tracked_clean_analysis_provenance()
    run_id = config.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise RSLADSignalReplayError("replay config requires a non-empty run_id")
    manifests = config.get("manifests")
    if not isinstance(manifests, list) or len(manifests) != 1:
        raise RSLADSignalReplayError("replay config requires exactly one local run-bundle manifest")
    manifest_path = _configured_path(config_path.parent, manifests[0], name="manifests")
    resolved_path = manifest_path.parent / "resolved_config.yaml"
    saved_resolved = _load_mapping(resolved_path)
    saved_digests = saved_resolved_config_digests(resolved_path)
    training_config = load_config(resolved_path)
    validate_rslad_replay_attack(training_config)
    if training_config.teacher is None:
        raise RSLADSignalReplayError("common-trajectory replay requires a registered teacher")
    inventory = (
        inventory_feature_trajectory(
            manifest_path,
            run_id=run_id,
            expected_config_hash=saved_digests["mapping_sha256"],
        )
        if args.feature_only
        else inventory_common_trajectory(
            inventory_run_bundle(manifest_path),
            run_id=run_id,
            expected_config_hash=saved_digests["mapping_sha256"],
        )
    )
    expected_count = _require_int(config, "train_expected_count", minimum=1)
    dataset_identity = portable_cifar10_train_identity(saved_resolved, expected_count=expected_count)
    batch_size = _require_int(config, "replay_batch_size", minimum=1)
    feature_seed = _require_int(config, "feature_attack_seed")
    loader = build_replay_loader(training_config, batch_size=batch_size)
    teacher = build_teacher(training_config.teacher, tier=training_config.tier).to(device)
    teacher_metadata = teacher.metadata.model_dump(mode="json")
    if (
        training_config.teacher.checkpoint_sha256 is None
        or teacher_metadata.get("checkpoint_sha256") != training_config.teacher.checkpoint_sha256
    ):
        raise RSLADSignalReplayError("built teacher checkpoint SHA does not match resolved teacher metadata")
    execution_runtime = runtime_identity(device)
    checkpoint_by_epoch = {item.epoch: item for item in inventory.checkpoints}
    cache_dir = output_dir / "checkpoint-cache"
    features = [
        _replay_with_cache(
            cache_dir=cache_dir,
            checkpoint=checkpoint_by_epoch[epoch],
            training_config=training_config,
            teacher=teacher,
            loader=loader,
            device=device,
            seed_domain="feature",
            base_seed=feature_seed,
            expected_count=expected_count,
            replay_batch_size=batch_size,
            saved_resolved_config_mapping_sha256=saved_digests["mapping_sha256"],
            saved_resolved_config_file_sha256=saved_digests["file_sha256"],
            teacher_metadata=teacher_metadata,
            dataset_identity=dataset_identity,
            analysis_provenance=analysis_provenance,
        )
        for epoch in FEATURE_EPOCHS
    ]
    feature_rows = tuple(row for item in features for row in item.rows)
    feature_panel = build_feature_panel(feature_rows, expected_count=expected_count)
    if args.feature_only:
        lineage = feature_replay_lineage(
            panel=inventory,
            training_config=training_config,
            expected_count=expected_count,
            replay_batch_size=batch_size,
            device_type=device.type,
            runtime=execution_runtime,
            feature_seed=feature_seed,
            saved_resolved_config_mapping_sha256=saved_digests["mapping_sha256"],
            saved_resolved_config_file_sha256=saved_digests["file_sha256"],
            teacher_metadata=teacher_metadata,
            dataset_identity=dataset_identity,
            analysis_provenance=analysis_provenance,
            feature_results=features,
            feature_panel=feature_panel,
        )
        paths = write_feature_replay_outputs(
            output_dir=output_dir,
            feature_observations=feature_rows,
            feature_panel=feature_panel,
            lineage=lineage,
        )
        print(json.dumps({name: str(path) for name, path in sorted(paths.items())}, sort_keys=True))
        return 0

    outcome_seed = _require_int(config, "outcome_attack_seed")
    if feature_seed == outcome_seed:
        raise RSLADSignalReplayError("feature_attack_seed and outcome_attack_seed must be independent")
    outcomes = [
        _replay_with_cache(
            cache_dir=cache_dir,
            checkpoint=checkpoint_by_epoch[epoch],
            training_config=training_config,
            teacher=teacher,
            loader=loader,
            device=device,
            seed_domain="outcome",
            base_seed=outcome_seed,
            expected_count=expected_count,
            replay_batch_size=batch_size,
            saved_resolved_config_mapping_sha256=saved_digests["mapping_sha256"],
            saved_resolved_config_file_sha256=saved_digests["file_sha256"],
            teacher_metadata=teacher_metadata,
            dataset_identity=dataset_identity,
            analysis_provenance=analysis_provenance,
        )
        for epoch in OUTCOME_EPOCHS
    ]
    outcome_rows = tuple(row for item in outcomes for row in item.rows)
    outcome_panel = build_outcome_panel(outcome_rows, expected_count=expected_count)
    joined = join_feature_outcome_panels(feature_panel, outcome_panel, expected_count=expected_count)
    report = predictive_audit(
        joined,
        split_seed=_require_int(config, "split_seed"),
        bootstrap_seed=_require_int(config, "bootstrap_seed"),
        bootstrap_replicates=_require_int(config, "bootstrap_replicates", minimum=1),
    )
    lineage = replay_lineage(
        panel=inventory,
        training_config=training_config,
        expected_count=expected_count,
        replay_batch_size=batch_size,
        device_type=device.type,
        runtime=execution_runtime,
        feature_seed=feature_seed,
        outcome_seed=outcome_seed,
        saved_resolved_config_mapping_sha256=saved_digests["mapping_sha256"],
        saved_resolved_config_file_sha256=saved_digests["file_sha256"],
        teacher_metadata=teacher_metadata,
        dataset_identity=dataset_identity,
        analysis_provenance=analysis_provenance,
        feature_results=features,
        outcome_results=outcomes,
    )
    paths = write_replay_outputs(
        output_dir=output_dir,
        feature_observations=feature_rows,
        outcome_observations=outcome_rows,
        feature_panel=feature_panel,
        outcome_panel=outcome_panel,
        lineage=lineage,
        report=report,
    )
    print(json.dumps({name: str(path) for name, path in sorted(paths.items())}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except RSLADSignalReplayError as exc:
        raise SystemExit(str(exc)) from exc
