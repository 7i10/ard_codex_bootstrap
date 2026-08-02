"""Prepare immutable inputs for an epoch-79 schedule-control fork.

This is intentionally an operational tool, outside the public ``ard`` package.
It performs no training and never queries W&B: artifact identity is supplied by
the caller after an external, already-recorded lookup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
import yaml

from ard.analysis.schedule_control_fork import PARENT_EPOCH, ScheduleControlForkError, _parent_runtime_view, sha256_file
from ard.config import ExperimentConfig, save_resolved_config
from ard.engine.checkpoint import config_digest


class PrepareInputsError(RuntimeError):
    """Inputs do not describe one immutable epoch-79 parent."""


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _mapping(path: Path, name: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PrepareInputsError(f"{name} is unreadable") from exc
    if not isinstance(value, dict):
        raise PrepareInputsError(f"{name} must be a mapping")
    return value


def _json_mapping(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrepareInputsError(f"{name} is unreadable") from exc
    if not isinstance(value, dict):
        raise PrepareInputsError(f"{name} must be a mapping")
    return value


def _feature_partition(path: Path, lineage: dict[str, Any], epoch: int) -> list[list[int]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise PrepareInputsError("pyarrow is required to read feature observations") from exc
    if not path.is_file():
        raise PrepareInputsError("feature observations parquet is missing")
    table = pq.read_table(path, columns=["namespace", "sample_id", "class_id", "epoch", "observation_schema_version"])
    rows = list(
        zip(
            *(
                table.column(name).to_pylist()
                for name in ("namespace", "sample_id", "class_id", "epoch", "observation_schema_version")
            ),
            strict=True,
        )
    )
    if not rows or any(namespace != "train" for namespace, *_ in rows):
        raise PrepareInputsError("feature observations must use namespace=train")
    if any(
        isinstance(v, bool) or not isinstance(v, int)
        for _, sample_id, class_id, row_epoch, schema in rows
        for v in (sample_id, class_id, row_epoch, schema)
    ):
        raise PrepareInputsError("feature observations IDs, classes, epochs, and schema versions must be integers")
    if any(schema != 2 for *_, schema in rows):
        raise PrepareInputsError("feature observations must use observation_schema_version=2")
    epochs = sorted({row[3] for row in rows})
    if epoch not in epochs:
        raise PrepareInputsError(f"feature epoch {epoch} is absent")
    mappings: dict[int, list[tuple[int, int]]] = {}
    for current in epochs:
        selected = [(sample_id, class_id) for _, sample_id, class_id, row_epoch, _ in rows if row_epoch == current]
        if len(selected) != 45_000 or len({sample_id for sample_id, _ in selected}) != 45_000:
            raise PrepareInputsError("every feature epoch must contain exactly 45,000 unique train IDs")
        mappings[current] = sorted(selected)
    if any(mapping != mappings[epochs[0]] for mapping in mappings.values()):
        raise PrepareInputsError("feature epochs do not share one exact ID/class mapping")
    return [[sample_id, class_id] for sample_id, class_id in mappings[epoch]]


def _build_attestation(
    *, parent_manifest: Path, inventory_path: Path, inventory_final: Path, checkpoint: Path
) -> dict[str, object]:
    manifest = _json_mapping(parent_manifest, "parent manifest")
    git = manifest["git"]
    teacher = manifest["teacher"]
    inventory = _json_mapping(inventory_path, "artifact inventory")
    return {
        "schema_version": 1,
        "parent_manifest_path": str(parent_manifest.resolve()),
        "parent_manifest_sha256": sha256_file(parent_manifest),
        "artifact_inventory_path": str(inventory_final.resolve()),
        "artifact_inventory_sha256": sha256_file(inventory_path),
        "run_id": manifest["run_id"],
        "config_hash": manifest["config_hash"],
        "git_sha": git["sha"],
        "teacher_checkpoint_sha256": teacher["checkpoint_sha256"],
        "artifact": inventory["artifact"],
        "checkpoint_sha256": sha256_file(checkpoint),
    }


def prepare_inputs(
    *,
    parent_checkpoint: Path,
    parent_config: Path,
    parent_manifest: Path,
    feature_observations: Path,
    feature_lineage: Path,
    artifact_name: str,
    artifact_version: str,
    artifact_digest: str,
    output_dir: Path,
    child_output_dir: Path,
    child_run_id: str,
    child_group: str | None = None,
    feature_epoch: int = PARENT_EPOCH,
) -> Path:
    if output_dir.exists():
        raise PrepareInputsError("refusing to overwrite existing output directory")
    for value, name in (
        (artifact_name, "artifact name"),
        (artifact_version, "artifact version"),
        (artifact_digest, "artifact digest"),
    ):
        if not value.strip():
            raise PrepareInputsError(f"{name} must be nonempty")
    if not child_run_id.strip():
        raise PrepareInputsError("child run ID must be nonempty")
    if feature_epoch != PARENT_EPOCH:
        raise PrepareInputsError("schedule-control inputs must use epoch 79")
    if not parent_checkpoint.is_file() or not feature_observations.is_file():
        raise PrepareInputsError("parent checkpoint is missing")
    checkpoint_sha = sha256_file(parent_checkpoint)
    feature_observations_sha = sha256_file(feature_observations)
    lineage = _json_mapping(feature_lineage, "feature lineage")
    checkpoints = lineage.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise PrepareInputsError("feature lineage checkpoints must be a list")
    matches = [item for item in checkpoints if isinstance(item, dict) and item.get("epoch") == PARENT_EPOCH]
    if len(matches) != 1 or matches[0].get("sha256") != checkpoint_sha:
        raise PrepareInputsError("checkpoint does not match the exact epoch-79 feature-lineage item")
    if lineage.get("feature_observations_sha256") != feature_observations_sha:
        raise PrepareInputsError("feature observations bytes do not match feature lineage")
    manifest = _json_mapping(parent_manifest, "parent manifest")
    for key in ("run_id", "config_hash", "git", "teacher"):
        if key not in manifest:
            raise PrepareInputsError(f"parent manifest lacks {key}")
    git = manifest["git"]
    teacher = manifest["teacher"]
    if not isinstance(git, dict) or not git.get("sha") or git.get("dirty") is not False:
        raise PrepareInputsError("parent manifest Git identity must be clean and addressable")
    if not isinstance(teacher, dict) or not teacher.get("checkpoint_sha256"):
        raise PrepareInputsError("parent manifest teacher checkpoint SHA is missing")
    lineage_teacher = lineage.get("teacher")
    if (
        lineage.get("run_id") != manifest["run_id"]
        or lineage.get("config_hash") != manifest["config_hash"]
        or lineage.get("scientific_git_sha") != git["sha"]
        or not isinstance(lineage_teacher, dict)
        or lineage_teacher.get("checkpoint_sha256") != teacher["checkpoint_sha256"]
        or lineage.get("observation_schema_version") != 2
    ):
        raise PrepareInputsError("feature lineage does not match parent manifest identity")
    raw_config = _mapping(parent_config, "parent resolved config")
    raw_config_sha = config_digest(raw_config)
    if manifest.get("config_hash") != raw_config_sha:
        raise PrepareInputsError("parent config hash does not match manifest")
    try:
        parent_runtime, _ = _parent_runtime_view(raw_config)
        child_raw = json.loads(parent_runtime.model_dump_json())
        child_raw["protocol"] = {"id": "controlled_cifar10_r18_delayed_multistep_v1"}
        child_raw["scheduler"] = {
            "id": "multistep",
            "milestones": [120, 170],
            "gamma": 0.1,
            "step_at": "epoch_end",
        }
        child_raw["output_dir"] = str(child_output_dir.resolve())
        tracking = dict(child_raw.get("tracking") or {})
        tracking["run_id"] = child_run_id
        if child_group is not None:
            tracking["group"] = child_group
        child_raw["tracking"] = tracking
        child_config = ExperimentConfig.model_validate(child_raw)
    except (ScheduleControlForkError, TypeError, ValueError) as exc:
        raise PrepareInputsError("child delayed-schedule config is invalid") from exc
    payload = torch.load(parent_checkpoint, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("epoch") != PARENT_EPOCH
        or payload.get("tracker_run_id") != manifest["run_id"]
        or payload.get("config_hash") != raw_config_sha
        or payload.get("world_size") != 1
    ):
        raise PrepareInputsError("parent checkpoint is not the epoch-79 boundary")
    sample_state = payload.get("sample_state")
    if not isinstance(sample_state, dict) or not isinstance(sample_state.get("records"), dict):
        raise PrepareInputsError("parent checkpoint lacks sample-state records")
    records = sample_state["records"]
    if len(records) != 45_000:
        raise PrepareInputsError("parent sample state must contain exactly 45,000 records")
    labels = {}
    for key, record in records.items():
        try:
            sample_id = int(key)
            label = int(record["true_label"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PrepareInputsError("sample state records lack integer IDs/true labels") from exc
        labels[sample_id] = label
    rows = _feature_partition(feature_observations, lineage, feature_epoch)
    if dict(rows) != labels:
        raise PrepareInputsError("feature ID/class mapping does not match parent sample state")
    state_sha = _canonical_sha256(sample_state)
    partition = {
        "schema_version": 1,
        "namespace": "train",
        "ids_labels": rows,
        "ids_labels_sha256": _canonical_sha256([(a, b) for a, b in rows]),
    }
    artifact_inventory = {
        "schema_version": 1,
        "artifact": {
            "name": artifact_name,
            "version": artifact_version,
            "digest": artifact_digest,
            "checkpoint_sha256": checkpoint_sha,
        },
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".schedule-control-inputs-", dir=output_dir.parent))
    try:
        partition_path = staging / "train-partition.json"
        inventory_path = staging / "artifact-inventory.json"
        partition_path.write_text(json.dumps(partition, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        inventory_path.write_text(
            json.dumps(artifact_inventory, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
        )
        final_inventory = output_dir.resolve() / "artifact-inventory.json"
        final_attestation = output_dir.resolve() / "artifact-attestation.json"
        save_resolved_config(child_config, staging / "child-config.yaml")
        attestation = _build_attestation(
            parent_manifest=parent_manifest,
            inventory_path=inventory_path,
            inventory_final=final_inventory,
            checkpoint=parent_checkpoint,
        )
        attestation_path = staging / "artifact-attestation.json"
        attestation_path.write_text(json.dumps(attestation, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        inventory_record = {
            "schema_version": 1,
            "artifact": artifact_inventory["artifact"],
            "checkpoint": {"path": str(parent_checkpoint.resolve()), "sha256": checkpoint_sha, "epoch": PARENT_EPOCH},
        }
        (staging / "artifact-inventory.json").write_text(
            json.dumps(inventory_record, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        # Re-attest after the final inventory bytes are written.
        attestation = _build_attestation(
            parent_manifest=parent_manifest,
            inventory_path=inventory_path,
            inventory_final=final_inventory,
            checkpoint=parent_checkpoint,
        )
        attestation_path.write_text(json.dumps(attestation, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        spec_parent = {
            "checkpoint_sha256": checkpoint_sha,
            "raw_config_sha256": raw_config_sha,
            "git_sha": git["sha"],
            "epoch": PARENT_EPOCH,
            "world_size": 1,
            "teacher_checkpoint_sha256": teacher["checkpoint_sha256"],
            "sample_state_records": 45_000,
            "sample_state_sha256": state_sha,
            "train_partition_manifest": str((output_dir / "train-partition.json").resolve()),
            "train_partition_manifest_sha256": sha256_file(partition_path),
            "train_partition_ids_labels_sha256": partition["ids_labels_sha256"],
            "artifact_attestation": str(final_attestation),
            "artifact_attestation_sha256": sha256_file(attestation_path),
            "artifact_inventory": str(final_inventory),
            "artifact_inventory_sha256": sha256_file(inventory_path),
        }
        spec = {"schema_version": 1, "kind": "delayed_multistep_schedule_control_v1", "parent": spec_parent}
        (staging / "schedule-control-spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
        os.replace(staging, output_dir)
    except Exception:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir / "schedule-control-spec.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in (
        "parent-checkpoint",
        "parent-config",
        "parent-manifest",
        "feature-observations",
        "feature-lineage",
        "output-dir",
        "child-output-dir",
    ):
        parser.add_argument(f"--{flag}", required=True, type=Path)
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--artifact-version", required=True)
    parser.add_argument("--artifact-digest", required=True)
    parser.add_argument("--child-run-id", required=True)
    parser.add_argument("--child-group")
    parser.add_argument("--feature-epoch", type=int, default=PARENT_EPOCH)
    args = parser.parse_args()
    try:
        print(prepare_inputs(**vars(args)))
    except PrepareInputsError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
