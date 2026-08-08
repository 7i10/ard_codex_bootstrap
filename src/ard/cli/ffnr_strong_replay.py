"""Run the Chen-only FF/NR primary CE-PGD20 strong replay."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from ard.analysis.ffnr_strong_replay import (
    CONTRACT_ID,
    SEED_FORMULA,
    StrongReplayError,
    build_checkpoint_inventory_document,
    checkpoint_cache_identity,
    deterministic_replay_backend,
    load_cached_checkpoint,
    load_checkpoint_inventory_document,
    parse_replay_config,
    replay_checkpoint_rows,
    select_explicit_checkpoints,
    selection_attack_from_training,
    source_provenance,
    validate_epoch_universes,
    validate_selected_checkpoint_bytes,
    write_checkpoint_cache,
    write_checkpoint_inventory,
    write_outputs,
)
from ard.analysis.rslad_signal_replay import portable_cifar10_train_identity
from ard.analysis.signal_audit import sha256_file
from ard.analysis.teacher_risk_replay import build_replay_loader
from ard.config import load_config
from ard.engine.checkpoint import config_digest
from ard.models import build_teacher


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise StrongReplayError(f"cannot read strong replay config {path}") from exc
    if not isinstance(value, dict):
        raise StrongReplayError("strong replay config must be a mapping")
    return value


def _configured_path(root: Path, value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise StrongReplayError(f"strong replay config {name} must be a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _require_single_process() -> None:
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as exc:
        raise StrongReplayError("WORLD_SIZE must be an integer") from exc
    if world_size != 1 or (torch.distributed.is_available() and torch.distributed.is_initialized()):
        raise StrongReplayError("strong replay requires one non-distributed process")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        help="Strict subset of configured epochs; one epoch is the smoke mode.",
    )
    parser.add_argument(
        "--build-inventory-only",
        action="store_true",
        help="Hash/deserialise the full bundle once and atomically write config checkpoint_inventory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _require_single_process()
    config_path = args.config.resolve()
    launch = parse_replay_config(_load_mapping(config_path))
    device = torch.device(args.device)
    if device.type != launch["replay_device_type"] or (device.type == "cuda" and not torch.cuda.is_available()):
        raise StrongReplayError("requested replay device does not match an available configured device type")
    configured_epochs = tuple(launch["epochs"])
    requested_epochs = configured_epochs if args.epochs is None else tuple(args.epochs)
    if not requested_epochs or any(epoch not in configured_epochs for epoch in requested_epochs):
        raise StrongReplayError("--epochs must be a non-empty subset of configured epochs")
    if tuple(sorted(set(requested_epochs))) != requested_epochs:
        raise StrongReplayError("--epochs must be sorted and unique")
    inventory_path = _configured_path(config_path.parent, launch["checkpoint_inventory"], name="checkpoint_inventory")
    if args.build_inventory_only:
        document = build_checkpoint_inventory_document(
            manifest_path=_configured_path(config_path.parent, launch["manifest"], name="manifest"),
            run_id=launch["run_id"],
        )
        written = write_checkpoint_inventory(path=inventory_path, document=document)
        print(json.dumps({"checkpoint_inventory": str(written)}))
        return 0
    if args.output_dir is None:
        raise StrongReplayError("--output-dir is required unless --build-inventory-only is selected")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(item.name != "checkpoint-cache" for item in output_dir.iterdir()):
        raise StrongReplayError("strong replay output directory already exists; refusing to overwrite")
    manifest_path = _configured_path(config_path.parent, launch["manifest"], name="manifest")
    resolved_path = manifest_path.parent / "resolved_config.yaml"
    training_config = load_config(resolved_path)
    attack = selection_attack_from_training(training_config)
    provenance = source_provenance()
    inventory = load_checkpoint_inventory_document(
        path=inventory_path, manifest_path=manifest_path, run_id=launch["run_id"]
    )
    selected = select_explicit_checkpoints(inventory, run_id=launch["run_id"], epochs=requested_epochs)
    validate_selected_checkpoint_bytes(selected)
    saved_resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(saved_resolved, dict):
        raise StrongReplayError("saved resolved training config must be a mapping")
    resolved_hash = config_digest(saved_resolved)
    if any(item.config_hash != resolved_hash for item in selected):
        raise StrongReplayError("selected checkpoint config hash does not match saved resolved config")
    if training_config.teacher is None:
        raise StrongReplayError("strong replay requires a registered teacher")
    dataset_identity = portable_cifar10_train_identity(saved_resolved, expected_count=launch["train_expected_count"])
    with deterministic_replay_backend() as effective_backend:
        runtime: dict[str, Any] = {
            "device": str(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "deterministic_backend": asdict(effective_backend),
        }
        if device.type == "cuda":
            index = torch.cuda.current_device() if device.index is None else device.index
            runtime.update(
                {
                    "cuda_device_index": index,
                    "cuda_device_name": torch.cuda.get_device_name(index),
                    "cuda_device_capability": list(torch.cuda.get_device_capability(index)),
                }
            )
        teacher = build_teacher(training_config.teacher, tier=training_config.tier).to(device)
        if any(parameter.requires_grad for parameter in teacher.parameters()):
            raise StrongReplayError("built teacher is not frozen")
        loader = build_replay_loader(training_config, batch_size=launch["replay_batch_size"])
        teacher_metadata = teacher.metadata.model_dump(mode="json")
        cache_dir = output_dir / "checkpoint-cache"
        results = []
        identities = []
        for checkpoint in selected:
            identity = checkpoint_cache_identity(
                checkpoint=checkpoint,
                attack=attack,
                seed=launch["attack_seed"],
                replay_batch_size=launch["replay_batch_size"],
                expected_sample_count=launch["train_expected_count"],
                teacher_metadata=teacher_metadata,
                dataset_identity=dataset_identity,
                runtime=runtime,
                provenance=provenance,
            )
            identities.append(identity)
            cached = load_cached_checkpoint(cache_dir=cache_dir, identity=identity)
            results.append(
                cached
                if cached is not None
                else write_checkpoint_cache(
                    cache_dir=cache_dir,
                    identity=identity,
                    result=replay_checkpoint_rows(
                        checkpoint=checkpoint,
                        training_config=training_config,
                        teacher=teacher,
                        loader=loader,
                        device=device,
                        attack_seed_base=launch["attack_seed"],
                    ),
                )
            )
    expected_count = launch["train_expected_count"]
    if any(len(result.rows) != expected_count for result in results):
        raise StrongReplayError("replay row count does not match expected train partition count")
    stable_universe = validate_epoch_universes(results, expected_count=expected_count)
    lineage = {
        "contract": CONTRACT_ID,
        "schema_version": 1,
        "run_id": launch["run_id"],
        "semantic_role": launch["semantic_role"],
        "requested_epochs": list(requested_epochs),
        "mode": "one_checkpoint_smoke" if len(requested_epochs) == 1 else "multi_epoch_replay",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "checkpoint_inventory": str(inventory_path),
        "checkpoint_inventory_sha256": sha256_file(inventory_path),
        "saved_resolved_config": str(resolved_path),
        "saved_resolved_config_sha256": sha256_file(resolved_path),
        "saved_resolved_config_mapping_sha256": resolved_hash,
        "checkpoints": [
            {"epoch": item.epoch, "sha256": item.sha256, "path": item.path, "artifact_name": item.artifact_name}
            for item in selected
        ],
        "attack_identity": attack.identity(),
        "attack_identity_sha256": attack.identity_sha256(),
        "attack_seed_base": launch["attack_seed"],
        "seed_formula": SEED_FORMULA,
        "replay_batch_size": launch["replay_batch_size"],
        "train_expected_count": expected_count,
        "stable_id_class_universe": stable_universe,
        "runtime": runtime,
        "teacher": teacher_metadata,
        "dataset_identity": dataset_identity,
        "analysis_provenance": provenance,
        "checkpoint_cache_identities": identities,
        "results": [
            {
                "epoch": result.epoch,
                "checkpoint_sha256": result.checkpoint_sha256,
                "max_abs_delta": result.max_abs_delta,
            }
            for result in results
        ],
    }
    paths = write_outputs(output_dir=output_dir, results=results, lineage=lineage)
    print(json.dumps({name: str(path) for name, path in sorted(paths.items())}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except StrongReplayError as exc:
        raise SystemExit(str(exc)) from exc
