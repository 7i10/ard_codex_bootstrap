"""Prepare immutable epoch-39 history-routing v2 inputs and arm configs.

This operational helper deliberately does not contact W&B or start training.
The caller supplies the already-resolved artifact identity; all generated files
are hash-bound and can then be passed to :mod:`ard.cli.fork_intervention`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
import yaml

from ard.analysis.history_routing_v2 import build_history_routing_v2_bundle
from ard.analysis.intervention_fork import build_parent_artifact_attestation, create_intervention_forks
from ard.analysis.schedule_control_fork import parent_runtime_view
from ard.config import ExperimentConfig, load_config
from ard.config.loader import resolved_config_dict
from ard.engine.checkpoint import config_digest

ARMS: tuple[tuple[str, str, str, bool], ...] = (
    ("PF_TA", "peak_failure", "history", True),
    ("PF_R", "peak_failure", "random", True),
    ("NR_TA", "non_recovery", "history", False),
    ("NR_R", "non_recovery", "random", False),
)


class PrepareInputsError(RuntimeError):
    """Input lineage or destination is not safe to materialize."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(path: Path, name: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PrepareInputsError(f"{name} is unreadable") from exc
    if not isinstance(value, dict):
        raise PrepareInputsError(f"{name} must be a mapping")
    return value


def _json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrepareInputsError(f"{name} is unreadable") from exc
    if not isinstance(value, dict):
        raise PrepareInputsError(f"{name} must be a mapping")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _parent_identity(
    *, parent_checkpoint: Path, parent_config: Path, parent_manifest: Path
) -> tuple[dict[str, Any], dict[str, Any], ExperimentConfig, dict[str, Any]]:
    raw = _mapping(parent_config, "parent resolved config")
    raw_hash = config_digest(raw)
    manifest = _json(parent_manifest, "parent manifest")
    git = manifest.get("git")
    teacher = manifest.get("teacher")
    if not isinstance(git, dict) or git.get("dirty") not in (False, None) or not isinstance(git.get("sha"), str):
        raise PrepareInputsError("parent manifest Git state is dirty or unaddressable")
    if len(git["sha"]) != 40 or manifest.get("config_hash") != raw_hash:
        raise PrepareInputsError("parent manifest does not bind the untouched resolved config")
    if not isinstance(teacher, dict) or not isinstance(teacher.get("checkpoint_sha256"), str):
        raise PrepareInputsError("parent manifest teacher checkpoint SHA is missing")
    try:
        source, _migration = parent_runtime_view(raw)
    except Exception as exc:  # the helper has a narrower public error type than historical migration code
        raise PrepareInputsError("parent resolved config is not a runnable RSLAD source") from exc
    if source.method.id != "rslad" or source.protocol.id != "controlled_cifar10_r18_v1":
        raise PrepareInputsError("parent must be ordinary controlled CIFAR-10 RSLAD")
    if source.training.epochs != 200 or source.training.per_rank_batch_size != 128:
        raise PrepareInputsError("parent must use the fixed 200-epoch/128-batch protocol")
    if source.teacher is None or source.teacher.checkpoint_sha256 != teacher["checkpoint_sha256"]:
        raise PrepareInputsError("parent teacher lineage does not match the manifest")
    payload = torch.load(parent_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise PrepareInputsError("parent checkpoint must be a mapping")
    if (
        payload.get("epoch") != 39
        or payload.get("epoch_boundary") != "end"
        or payload.get("world_size") != 1
        or payload.get("config_hash") != raw_hash
        or payload.get("tracker_run_id") != manifest.get("run_id")
    ):
        raise PrepareInputsError("parent checkpoint is not the exact epoch-39 boundary")
    state = payload.get("sample_state")
    if not isinstance(state, dict) or len(state.get("records", {})) != 45_000:
        raise PrepareInputsError("parent checkpoint must contain all 45,000 sample-state records")
    return raw, manifest, source, payload


def _arm_raw(
    *,
    source: ExperimentConfig,
    arm: str,
    route: str,
    selector_kind: str,
    anchor_correct: bool,
    parent: dict[str, Any],
    mask_path: Path,
    bundle_path: Path,
    output_root: Path,
    entity: str,
    project: str,
    group: str,
    parent_run_id: str,
) -> dict[str, Any]:
    raw = resolved_config_dict(source)
    raw["protocol"] = {"id": "controlled_cifar10_r18_delayed_multistep_v1"}
    raw["scheduler"] = {"id": "multistep", "milestones": [120, 170], "gamma": 0.1, "step_at": "epoch_end"}
    child_dir = (output_root / arm).resolve()
    run_id = f"h2-{parent_run_id}-{arm.lower()}"
    raw["output_dir"] = str(child_dir)
    raw["tracking"] = {
        **dict(raw.get("tracking") or {}),
        "mode": "online",
        "entity": entity,
        "project": project,
        "group": group,
        "run_id": run_id,
        "name": run_id,
    }
    mask = _json(mask_path, f"{arm} mask")
    raw["intervention"] = {
        "arm": arm,
        "selector": selector_kind,
        "kind": "teacher_target_true_label_mix",
        "parent": parent,
        "mask": {
            "path": str(mask_path.resolve()),
            "sha256": sha256_file(mask_path),
            "selected_ids_sha256": mask["selected_ids_sha256"],
            "selected_count": mask["selected_count"],
            "selected_class_counts": mask["selected_class_counts"],
            "provenance": mask["provenance"],
        },
        "selector_bundle_path": str(bundle_path.resolve()),
        "selector_bundle_sha256": sha256_file(bundle_path),
    }
    # Validate before writing: this catches accidental field drift early.
    try:
        ExperimentConfig.model_validate(raw)
    except ValueError as exc:
        raise PrepareInputsError(f"generated {arm} config is invalid") from exc
    return raw


def prepare_inputs(
    *,
    parent_checkpoint: Path,
    parent_resolved_config: Path,
    parent_manifest: Path,
    train_partition_manifest: Path,
    artifact_name: str,
    artifact_version: str,
    artifact_digest: str,
    input_root: Path,
    screen_output_root: Path,
    wandb_entity: str,
    wandb_project: str,
    wandb_group: str,
    create_forks: bool = False,
) -> dict[str, Path]:
    if input_root.exists() or screen_output_root.exists():
        raise PrepareInputsError("refusing to overwrite an existing input or screen output root")
    for value, name in (
        (artifact_name, "artifact name"),
        (artifact_version, "artifact version"),
        (artifact_digest, "artifact digest"),
    ):
        if not value.strip():
            raise PrepareInputsError(f"{name} must be nonempty")
    if not train_partition_manifest.is_file():
        raise PrepareInputsError("train-partition manifest is missing")
    raw, manifest, source, payload = _parent_identity(
        parent_checkpoint=parent_checkpoint, parent_config=parent_resolved_config, parent_manifest=parent_manifest
    )
    checkpoint_sha = sha256_file(parent_checkpoint)
    partition_sha = sha256_file(train_partition_manifest)
    partition = _json(train_partition_manifest, "train-partition manifest")
    partition_ids_sha = partition.get("ids_labels_sha256")
    if not isinstance(partition_ids_sha, str):
        raise PrepareInputsError("train-partition manifest lacks ids_labels_sha256")
    parent_fields: dict[str, Any] = {
        "checkpoint_sha256": checkpoint_sha,
        "raw_config_sha256": config_digest(raw),
        "git_sha": manifest["git"]["sha"],
        "epoch": 39,
        "world_size": 1,
        "teacher_checkpoint_sha256": source.teacher.checkpoint_sha256 if source.teacher else "",
        "sample_state_records": 45000,
        "sample_state_sha256": hashlib.sha256(
            json.dumps(payload["sample_state"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
    }
    input_root.parent.mkdir(parents=True, exist_ok=True)
    created_input = False
    try:
        input_root.mkdir()
        created_input = True
        inventory = input_root / "artifact-inventory.json"
        _write_json(
            inventory,
            {
                "schema_version": 1,
                "artifact": {
                    "name": artifact_name,
                    "version": artifact_version,
                    "digest": artifact_digest,
                    "checkpoint_sha256": checkpoint_sha,
                },
            },
        )
        attestation = input_root / "artifact-attestation.json"
        _write_json(
            attestation,
            build_parent_artifact_attestation(
                parent_manifest=parent_manifest, artifact_inventory=inventory, checkpoint=parent_checkpoint
            ),
        )
        parent_fields.update(
            {
                "train_partition_manifest": str(train_partition_manifest.resolve()),
                "train_partition_manifest_sha256": partition_sha,
                "train_partition_ids_labels_sha256": partition_ids_sha,
                "artifact_attestation": str(attestation.resolve()),
                "artifact_attestation_sha256": sha256_file(attestation),
                "artifact_inventory": str(inventory.resolve()),
                "artifact_inventory_sha256": sha256_file(inventory),
            }
        )
        paths = build_history_routing_v2_bundle(
            parent_checkpoint=parent_checkpoint,
            parent_resolved_config=parent_resolved_config,
            parent_manifest=parent_manifest,
            train_partition_manifest=train_partition_manifest,
            train_partition_manifest_sha256=partition_sha,
            train_partition_ids_labels_sha256=partition_ids_sha,
            output_dir=input_root / "selector-bundle",
        )
        bundle = paths["bundle"]
        screen_output_root.parent.mkdir(parents=True, exist_ok=True)
        config_tmp = Path(tempfile.mkdtemp(prefix=".history-routing-v2-configs-", dir=screen_output_root.parent))
        configs: dict[str, Path] = {}
        try:
            for arm, route, kind, anchor_correct in ARMS:
                mask = paths[f"{route}_{kind}"]
                config_raw = _arm_raw(
                    source=source,
                    arm=arm,
                    route=route,
                    selector_kind="online_history" if kind == "history" else "class_state_count_matched_random",
                    anchor_correct=anchor_correct,
                    parent=parent_fields,
                    mask_path=mask,
                    bundle_path=bundle,
                    output_root=screen_output_root,
                    entity=wandb_entity,
                    project=wandb_project,
                    group=wandb_group,
                    parent_run_id=str(manifest["run_id"]),
                )
                path = config_tmp / f"{arm}.yaml"
                path.write_text(yaml.safe_dump(config_raw, sort_keys=False), encoding="utf-8")
                # Strict reload is part of preparation, not a training-side check.
                load_config(path)
                configs[arm] = path
            if create_forks:
                create_intervention_forks(
                    parent_checkpoint=parent_checkpoint,
                    parent_resolved_config=parent_resolved_config,
                    parent_manifest=parent_manifest,
                    arm_config_paths=[configs[name] for name, *_ in ARMS],
                    root=Path.cwd(),
                )
            else:
                screen_output_root.mkdir(parents=False)
            for arm, path in configs.items():
                shutil.copy2(path, screen_output_root / f"{arm}.yaml")
        finally:
            shutil.rmtree(config_tmp, ignore_errors=True)
        return {
            "input_root": input_root,
            "inventory": inventory,
            "attestation": attestation,
            "bundle": bundle,
            **{f"{arm}_config": screen_output_root / f"{arm}.yaml" for arm, *_ in ARMS},
        }
    except Exception:
        if created_input:
            shutil.rmtree(input_root, ignore_errors=True)
        if screen_output_root.is_dir() and not any(screen_output_root.iterdir()):
            screen_output_root.rmdir()
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-resolved-config", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--train-partition-manifest", type=Path, required=True)
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--artifact-version", required=True)
    parser.add_argument("--artifact-digest", required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--screen-output-root", type=Path, required=True)
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-group", required=True)
    parser.add_argument("--create-forks", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = prepare_inputs(
        parent_checkpoint=args.parent_checkpoint,
        parent_resolved_config=args.parent_resolved_config,
        parent_manifest=args.parent_manifest,
        train_partition_manifest=args.train_partition_manifest,
        artifact_name=args.artifact_name,
        artifact_version=args.artifact_version,
        artifact_digest=args.artifact_digest,
        input_root=args.input_root,
        screen_output_root=args.screen_output_root,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
        create_forks=args.create_forks,
    )
    print(json.dumps({key: str(value) for key, value in sorted(result.items())}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
