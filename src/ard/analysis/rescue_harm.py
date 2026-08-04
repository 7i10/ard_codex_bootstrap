"""Hash-bound completed-v2 rescue/harm checkpoint replay and paired reports.

This module is deliberately read-only.  It evaluates fixed saved checkpoints
on the raw, unaugmented CIFAR-10 train partition and treats paired outcomes as
exploratory model-level moderation, never as unit-level causal effects.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ard.analysis.h4a_taxonomy import _domain_panel, _lineage
from ard.analysis.rslad_signal_replay import FEATURE_EPOCHS, canonical_json, runtime_identity
from ard.analysis.sample_stats import write_sample_parquet
from ard.analysis.signal_audit import CheckpointInventory, inventory_run_bundle, logical_dataset_identity, sha256_file
from ard.analysis.teacher_risk_replay import build_replay_loader, load_historical_student
from ard.attacks import AttackRequest, LinfPGD
from ard.config.loader import load_resolved_config_for_evaluation, resolved_config_dict
from ard.engine.checkpoint import REQUIRED_KEYS
from ard.policies import selected_ids_sha256


class RescueHarmError(ValueError):
    """Raised when a replay or paired report cannot prove its frozen inputs."""


EPOCHS = (99, 104, 109, 199)
ARMS = ("control", "PF_H", "PF_R", "NR_H", "NR_R")
SOURCE_ARM = {"control": "control", "PF_H": "PF_TA", "PF_R": "PF_R", "NR_H": "NR_TA", "NR_R": "NR_R"}
OBSERVATION_COLUMNS = (
    "namespace",
    "run_id",
    "arm",
    "seed",
    "epoch",
    "sample_id",
    "class_id",
    "clean_prediction",
    "clean_correct",
    "clean_probability_margin",
    "robust_prediction",
    "robust_correct",
    "robust_probability_margin",
)
CATEGORIES = ("rescued", "harmed", "stable_correct", "unchanged_failure")


@dataclass(frozen=True)
class Inventory:
    run_id: str
    arm: str
    seed: int
    teacher: Mapping[str, Any]
    config_hash: str
    checkpoints: tuple[CheckpointInventory, ...]


def _sha(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise RescueHarmError("expected lowercase SHA-256")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RescueHarmError(f"{name} must be an integer >= {minimum}")
    return value


def _json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RescueHarmError(f"{name} is unreadable") from exc
    if not isinstance(value, dict):
        raise RescueHarmError(f"{name} must be a JSON object")
    return value


def _hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _tracked_clean_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    paths = (Path(__file__).resolve(), root / "src/ard/cli/rescue_harm.py")
    try:
        relative = [str(path.relative_to(root)) for path in paths]
        subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", *relative], check=True, capture_output=True
        )
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise RescueHarmError("rescue/harm replay requires tracked source and Git identity") from exc
    if len(sha) != 40 or dirty:
        raise RescueHarmError("rescue/harm replay requires a tracked-clean revision")
    return {
        "git": {"sha": sha, "dirty": False},
        "source_files": {str(path.relative_to(root)): sha256_file(path) for path in paths},
    }


def _student_identity(config: Any) -> dict[str, Any]:
    return config.student.model_dump(mode="json")


def _teacher_identity(value: object) -> tuple[str | None, str]:
    if not isinstance(value, Mapping):
        raise RescueHarmError("teacher identity must be a mapping")
    registry, checkpoint = value.get("registry_id"), value.get("checkpoint_sha256")
    if registry is not None and not isinstance(registry, str):
        raise RescueHarmError("teacher registry identity is invalid")
    return registry, _sha(checkpoint)


def load_checkpoint_inventory(path: Path) -> Inventory:
    """Validate immutable bytes and exact payload epochs before GPU replay."""
    value = _json(path, name="checkpoint inventory")
    required = {"schema_version", "run_id", "arm", "seed", "teacher", "config_hash", "checkpoints"}
    if set(value) != required or value.get("schema_version") != 1:
        raise RescueHarmError("checkpoint inventory schema drifted")
    run_id, arm, seed = value["run_id"], value["arm"], value["seed"]
    if (
        not isinstance(run_id, str)
        or not run_id
        or arm not in ARMS
        or isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        raise RescueHarmError("checkpoint inventory run/arm/seed identity is invalid")
    if not isinstance(value["teacher"], Mapping):
        raise RescueHarmError("checkpoint inventory teacher identity is invalid")
    config_hash = _sha(value["config_hash"])
    raw = value["checkpoints"]
    if not isinstance(raw, list) or len(raw) != len(EPOCHS):
        raise RescueHarmError("checkpoint inventory requires exactly four fixed epochs")
    result: list[CheckpointInventory] = []
    for item in raw:
        if not isinstance(item, Mapping) or set(item) != {"epoch", "path", "sha256", "scientific_git_sha"}:
            raise RescueHarmError("checkpoint inventory entry schema drifted")
        epoch = _integer(item["epoch"], name="checkpoint epoch")
        checkpoint_path = Path(item["path"])
        if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != _sha(item["sha256"]):
            raise RescueHarmError("checkpoint inventory byte hash drifted")
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as exc:  # pragma: no cover - torch exception varies
            raise RescueHarmError("checkpoint payload is unreadable") from exc
        if not isinstance(payload, Mapping) or REQUIRED_KEYS.difference(payload):
            raise RescueHarmError("checkpoint payload lacks complete lineage")
        if (
            payload.get("epoch") != epoch
            or payload.get("config_hash") != config_hash
            or payload.get("tracker_run_id") != run_id
        ):
            raise RescueHarmError("checkpoint payload epoch/config/run lineage drifted")
        git_sha = item["scientific_git_sha"]
        if not isinstance(git_sha, str) or len(git_sha) not in {40, 64}:
            raise RescueHarmError("checkpoint inventory Git identity is invalid")
        result.append(
            CheckpointInventory(
                run_id=run_id,
                artifact_name=f"{arm}-epoch{epoch}",
                aliases=("last",),
                publication_order=epoch,
                path=str(checkpoint_path.resolve()),
                sha256=item["sha256"],
                epoch=epoch,
                sample_state_present=True,
                sample_state_count=0,
                config_hash=config_hash,
                scientific_git_sha=git_sha,
            )
        )
    if tuple(sorted(checkpoint.epoch for checkpoint in result)) != EPOCHS:
        raise RescueHarmError("checkpoint inventory epochs must be exactly 99/104/109/199")
    return Inventory(
        run_id=run_id,
        arm=arm,
        seed=seed,
        teacher=dict(value["teacher"]),
        config_hash=config_hash,
        checkpoints=tuple(sorted(result, key=lambda item: item.epoch)),
    )


def build_checkpoint_inventory(
    *, manifest: Path, resolved_config: Path, arm: str, seed: int, output: Path
) -> dict[str, Any]:
    """Freeze one completed local run bundle at the four paired snapshots."""
    if output.exists():
        raise FileExistsError("refusing to overwrite checkpoint inventory")
    if arm not in {"control", "PF_TA", "PF_R", "NR_TA", "NR_R"}:
        raise RescueHarmError("inventory arm must be control/PF_TA/PF_R/NR_TA/NR_R")
    entries = [item for item in inventory_run_bundle(manifest) if item.epoch in EPOCHS]
    if tuple(sorted(item.epoch for item in entries)) != EPOCHS:
        raise RescueHarmError("local run bundle lacks exactly the fixed 99/104/109/199 checkpoints")
    if len({item.run_id for item in entries}) != 1 or len({item.config_hash for item in entries}) != 1:
        raise RescueHarmError("fixed checkpoints do not share one run/config identity")
    evaluation = load_resolved_config_for_evaluation(resolved_config)
    if evaluation.raw_config_hash != entries[0].config_hash or evaluation.config.teacher is None:
        raise RescueHarmError("resolved config does not bind completed checkpoint inventory")
    value = {
        "schema_version": 1,
        "run_id": entries[0].run_id,
        "arm": arm,
        "seed": seed,
        "teacher": evaluation.config.teacher.model_dump(mode="json"),
        "config_hash": entries[0].config_hash,
        "checkpoints": [
            {
                "epoch": item.epoch,
                "path": str(Path(item.path).resolve()),
                "sha256": item.sha256,
                "scientific_git_sha": item.scientific_git_sha,
            }
            for item in sorted(entries, key=lambda item: item.epoch)
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json(value) + b"\n")
    return value


def _ce_pgd20(config: Any) -> None:
    attack = config.method.selection_attack
    if (
        attack is None
        or attack.loss != "ce"
        or attack.steps != 20
        or attack.norm != "linf"
        or attack.input_domain != "pixel_0_1"
        or attack.student_mode != "eval"
    ):
        raise RescueHarmError("rescue/harm replay requires the saved CE PGD-20 eval pixel-space selection attack")


def _primitives(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probabilities = F.softmax(logits.float(), dim=1)
    prediction = probabilities.argmax(dim=1)
    true = probabilities.gather(1, labels[:, None]).squeeze(1)
    masked = probabilities.clone()
    masked.scatter_(1, labels[:, None], float("-inf"))
    wrong = masked.max(dim=1).values
    return prediction, prediction.eq(labels), true - wrong


def replay_inventory(
    *,
    resolved_config: Path,
    inventory_path: Path,
    output_parquet: Path,
    output_lineage: Path,
    device: torch.device,
    batch_size: int,
    analysis_seed: int,
    epochs: Sequence[int] = EPOCHS,
) -> dict[str, Any]:
    """Replay all four immutable snapshots for one arm without teacher forwards."""
    if output_parquet.exists() or output_lineage.exists():
        raise FileExistsError("refusing to overwrite rescue/harm replay output")
    if device.type not in {"cpu", "cuda"} or (device.type == "cuda" and not torch.cuda.is_available()):
        raise RescueHarmError("requested replay device is unavailable")
    if batch_size < 1:
        raise RescueHarmError("replay batch_size must be positive")
    provenance = _tracked_clean_provenance()
    inventory = load_checkpoint_inventory(inventory_path)
    evaluation = load_resolved_config_for_evaluation(resolved_config)
    config = evaluation.config
    if config.dataset.name != "cifar10" or config.dataset.split != "train":
        raise RescueHarmError("rescue/harm replay is restricted to CIFAR-10 raw train data")
    if evaluation.raw_config_hash != inventory.config_hash or config.teacher is None:
        raise RescueHarmError("resolved config does not match immutable checkpoint inventory")
    if inventory.teacher != config.teacher.model_dump(mode="json"):
        raise RescueHarmError("checkpoint inventory teacher does not match resolved config teacher")
    _ce_pgd20(config)
    loader = build_replay_loader(config, batch_size=batch_size)
    if len(loader.dataset) != 45_000:
        raise RescueHarmError("raw CIFAR-10 replay loader must expose exactly 45,000 stable train IDs")
    rows: list[dict[str, Any]] = []
    attack = LinfPGD(config.method.selection_attack)
    selected_epochs = tuple(sorted(set(epochs)))
    if not selected_epochs or any(epoch not in EPOCHS for epoch in selected_epochs):
        raise RescueHarmError("replay epoch selection must be a non-empty subset of fixed snapshots")
    for checkpoint in (item for item in inventory.checkpoints if item.epoch in selected_epochs):
        student, _ = load_historical_student(
            checkpoint, config=config, device=device, expected_config_hash=inventory.config_hash
        )
        student.eval()
        for batch_index, raw_batch in enumerate(loader):
            batch = raw_batch.to(device)
            generator = torch.Generator(device=device).manual_seed(analysis_seed + 1_000_003 * batch_index)
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                clean_logits = student(batch.images.float()).float()
            result = attack.generate(
                AttackRequest(inputs=batch.images, labels=batch.labels, student=student, generator=generator)
            )
            if result.max_abs_delta > float(config.method.selection_attack.epsilon_value) + 1e-7:
                raise RescueHarmError("CE PGD replay violated its pixel-space Linf bound")
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                robust_logits = student(result.adversarial.float()).float()
                clean_prediction, clean_correct, clean_margin = _primitives(clean_logits, batch.labels)
                robust_prediction, robust_correct, robust_margin = _primitives(robust_logits, batch.labels)
            rows.extend(
                {
                    "namespace": "train",
                    "run_id": inventory.run_id,
                    "arm": inventory.arm,
                    "seed": inventory.seed,
                    "epoch": checkpoint.epoch,
                    "sample_id": int(sample_id),
                    "class_id": int(class_id),
                    "clean_prediction": int(cp),
                    "clean_correct": bool(cc),
                    "clean_probability_margin": float(cm),
                    "robust_prediction": int(rp),
                    "robust_correct": bool(rc),
                    "robust_probability_margin": float(rm),
                }
                for sample_id, class_id, cp, cc, cm, rp, rc, rm in zip(
                    batch.sample_ids.tolist(),
                    batch.labels.tolist(),
                    clean_prediction.tolist(),
                    clean_correct.tolist(),
                    clean_margin.tolist(),
                    robust_prediction.tolist(),
                    robust_correct.tolist(),
                    robust_margin.tolist(),
                    strict=True,
                )
            )
        student.zero_grad(set_to_none=True)
    expected = len(loader.dataset)
    if len(rows) != expected * len(selected_epochs) or any(
        sum(row["epoch"] == epoch for row in rows) != expected for epoch in selected_epochs
    ):
        raise RescueHarmError("replay lacks exact stable-ID coverage at one or more epochs")
    write_sample_parquet(sorted(rows, key=lambda row: (int(row["epoch"]), int(row["sample_id"]))), output_parquet)
    lineage = {
        "schema_version": 1,
        "contract": "completed_v2_rescue_harm_replay_v1",
        "observations_sha256": sha256_file(output_parquet),
        "run_id": inventory.run_id,
        "arm": inventory.arm,
        "seed": inventory.seed,
        "teacher": inventory.teacher,
        "config_sha256": inventory.config_hash,
        "source_resolved_config_sha256": sha256_file(resolved_config),
        "checkpoint_inventory_sha256": sha256_file(inventory_path),
        "checkpoints": [
            {"epoch": item.epoch, "sha256": item.sha256}
            for item in inventory.checkpoints
            if item.epoch in selected_epochs
        ],
        "dataset_identity": logical_dataset_identity(resolved_config_dict(config), train_expected_count=expected),
        "attack_identity": config.method.selection_attack.model_dump(mode="json"),
        "analysis_seed": analysis_seed,
        "student_identity": _student_identity(config),
        "runtime": runtime_identity(device),
        "row_count": len(rows),
        "analysis_provenance": provenance,
    }
    output_lineage.parent.mkdir(parents=True, exist_ok=True)
    output_lineage.write_bytes(canonical_json(lineage) + b"\n")
    return lineage


def _read_observations(
    path: Path, lineage_path: Path, *, arm: str
) -> tuple[dict[str, Any], dict[int, dict[int, dict[str, Any]]]]:
    lineage = _json(lineage_path, name=f"{arm} lineage")
    if (
        lineage.get("contract") != "completed_v2_rescue_harm_replay_v1"
        or lineage.get("arm") != arm
        or lineage.get("observations_sha256") != sha256_file(path)
    ):
        raise RescueHarmError(f"{arm} observation lineage drifted")
    for key in (
        "seed",
        "config_sha256",
        "dataset_identity",
        "attack_identity",
        "analysis_seed",
        "teacher",
        "student_identity",
    ):
        if key not in lineage:
            raise RescueHarmError(f"{arm} observation lineage is incomplete")
    if not isinstance(lineage["teacher"], Mapping) or not isinstance(lineage["student_identity"], Mapping):
        raise RescueHarmError(f"{arm} observation lineage model identity is invalid")
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path)
    except Exception as exc:  # pragma: no cover
        raise RescueHarmError(f"{arm} observations are unreadable") from exc
    if tuple(table.column_names) != OBSERVATION_COLUMNS:
        raise RescueHarmError(f"{arm} observation schema drifted")
    panels = {epoch: {} for epoch in EPOCHS}
    for row in table.to_pylist():
        epoch, sample_id, class_id = row.get("epoch"), row.get("sample_id"), row.get("class_id")
        if (
            epoch not in panels
            or not isinstance(sample_id, int)
            or not isinstance(class_id, int)
            or row.get("namespace") != "train"
            or sample_id in panels[epoch]
        ):
            raise RescueHarmError(f"{arm} stable-ID/epoch contract drifted")
        if (
            row.get("run_id") != lineage.get("run_id")
            or row.get("arm") != arm
            or row.get("seed") != lineage.get("seed")
        ):
            raise RescueHarmError(f"{arm} row identity drifted")
        if not isinstance(row.get("robust_correct"), bool) or not isinstance(row.get("clean_correct"), bool):
            raise RescueHarmError(f"{arm} correctness schema drifted")
        panels[epoch][sample_id] = dict(row)
    expected = lineage.get("row_count")
    if (
        not isinstance(expected, int)
        or expected != sum(len(panel) for panel in panels.values())
        or any(not panel for panel in panels.values())
    ):
        raise RescueHarmError(f"{arm} row count drifted")
    reference = panels[EPOCHS[0]]
    if any(
        set(panel) != set(reference)
        or any(panel[sample_id]["class_id"] != reference[sample_id]["class_id"] for sample_id in reference)
        for panel in panels.values()
    ):
        raise RescueHarmError(f"{arm} stable ID/class join drifted")
    return lineage, panels


def merge_epoch_replays(
    *, inputs: Mapping[int, tuple[Path, Path]], output_parquet: Path, output_lineage: Path
) -> dict[str, Any]:
    """Combine four independently replayed snapshots into one formal arm panel."""
    if output_parquet.exists() or output_lineage.exists():
        raise FileExistsError("refusing to overwrite merged rescue/harm replay output")
    if set(inputs) != set(EPOCHS):
        raise RescueHarmError("merge requires exactly one single-epoch input for 99/104/109/199")
    rows_by_epoch: dict[int, list[dict[str, Any]]] = {}
    lineages: dict[int, dict[str, Any]] = {}
    identity_keys = (
        "run_id",
        "arm",
        "seed",
        "teacher",
        "config_sha256",
        "dataset_identity",
        "attack_identity",
        "analysis_seed",
        "student_identity",
        "analysis_provenance",
    )
    for epoch in EPOCHS:
        path, lineage_path = inputs[epoch]
        lineage = _json(lineage_path, name=f"epoch-{epoch} replay lineage")
        if (
            lineage.get("contract") != "completed_v2_rescue_harm_replay_v1"
            or lineage.get("observations_sha256") != sha256_file(path)
            or lineage.get("row_count") is None
        ):
            raise RescueHarmError("single-epoch replay lineage drifted")
        checkpoints = lineage.get("checkpoints")
        if not isinstance(checkpoints, list) or len(checkpoints) != 1 or checkpoints[0].get("epoch") != epoch:
            raise RescueHarmError("single-epoch replay lineage checkpoint identity drifted")
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(path)
        except Exception as exc:  # pragma: no cover
            raise RescueHarmError("single-epoch observations are unreadable") from exc
        if tuple(table.column_names) != OBSERVATION_COLUMNS:
            raise RescueHarmError("single-epoch observation schema drifted")
        rows = [dict(row) for row in table.to_pylist()]
        if not rows or len(rows) != lineage["row_count"] or any(row.get("epoch") != epoch for row in rows):
            raise RescueHarmError("single-epoch observation count/epoch drifted")
        ids: set[int] = set()
        for row in rows:
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, int) or sample_id in ids or row.get("namespace") != "train":
                raise RescueHarmError("single-epoch stable-ID contract drifted")
            ids.add(sample_id)
            if (
                row.get("run_id") != lineage.get("run_id")
                or row.get("arm") != lineage.get("arm")
                or row.get("seed") != lineage.get("seed")
            ):
                raise RescueHarmError("single-epoch row identity drifted")
        rows_by_epoch[epoch], lineages[epoch] = rows, lineage
    reference = lineages[EPOCHS[0]]
    if any(any(lineages[epoch].get(key) != reference.get(key) for key in identity_keys) for epoch in EPOCHS[1:]):
        raise RescueHarmError("single-epoch replay identity drifted")
    reference_rows = {int(row["sample_id"]): int(row["class_id"]) for row in rows_by_epoch[EPOCHS[0]]}
    if any(
        {int(row["sample_id"]): int(row["class_id"]) for row in rows_by_epoch[epoch]} != reference_rows
        for epoch in EPOCHS[1:]
    ):
        raise RescueHarmError("single-epoch stable ID/class join drifted")
    rows = [row for epoch in EPOCHS for row in sorted(rows_by_epoch[epoch], key=lambda item: int(item["sample_id"]))]
    write_sample_parquet(rows, output_parquet)
    lineage = {
        **{
            key: value
            for key, value in reference.items()
            if key not in {"observations_sha256", "row_count", "checkpoints", "runtime"}
        },
        "observations_sha256": sha256_file(output_parquet),
        "checkpoints": [lineages[epoch]["checkpoints"][0] for epoch in EPOCHS],
        "runtime": {"merged_from_single_epoch_replays": True},
        "row_count": len(rows),
    }
    output_lineage.parent.mkdir(parents=True, exist_ok=True)
    output_lineage.write_bytes(canonical_json(lineage) + b"\n")
    return lineage


def _resolve_mask_path(bundle: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise RescueHarmError("selector mask path is invalid")
    declared = Path(value)
    if declared.is_file():
        return declared
    fallback = bundle.parent / declared.name
    if fallback.is_file():
        return fallback
    raise RescueHarmError("selector mask path is unavailable locally and has no bundle-relative fallback")


def _mask_bundle(path: Path, *, feature: Mapping[int, Mapping[str, Any]]) -> dict[str, set[int]]:
    """Load actual epoch-39 selector masks; eligibility comes only from feature state."""
    value = _json(path, name="epoch39 selector bundle")
    required = {"schema_version", "kind", "parent", "selection", "mask_paths"}
    if (
        set(value) != required
        or value.get("schema_version") != 1
        or value.get("kind") != "history_routing_v2_online_selector_v1"
    ):
        raise RescueHarmError("selector bundle schema/version drifted")
    parent, selection, paths = value["parent"], value["selection"], value["mask_paths"]
    if (
        not isinstance(parent, Mapping)
        or parent.get("epoch") != 39
        or not isinstance(selection, Mapping)
        or not isinstance(paths, Mapping)
    ):
        raise RescueHarmError("selector bundle parent/selection lineage drifted")
    route_specs = {
        "PF_H": ("peak_failure", "history", True),
        "PF_R": ("peak_failure", "random", True),
        "NR_H": ("non_recovery", "history", False),
        "NR_R": ("non_recovery", "random", False),
    }
    result: dict[str, set[int]] = {}
    labels = {sample_id: int(row["class_id"]) for sample_id, row in feature.items()}
    bundle_sha = sha256_file(path)
    for arm, (route, kind, anchor_correct) in route_specs.items():
        pair = paths.get(route)
        metadata = selection.get(route)
        if not isinstance(pair, Mapping) or not isinstance(metadata, Mapping):
            raise RescueHarmError("selector bundle route metadata drifted")
        mask_path = _resolve_mask_path(path, pair.get(kind))
        mask = _json(mask_path, name=f"{arm} selector mask")
        expected_keys = {
            "schema_version",
            "namespace",
            "num_classes",
            "selected_ids",
            "selected_ids_sha256",
            "selected_count",
            "selected_class_counts",
            "provenance",
        }
        ids = mask.get("selected_ids")
        if (
            set(mask) != expected_keys
            or mask.get("schema_version") != 1
            or mask.get("namespace") != "train"
            or mask.get("num_classes") != 10
            or not isinstance(ids, list)
        ):
            raise RescueHarmError("selector mask schema drifted")
        if (
            tuple(ids) != tuple(sorted(ids))
            or len(ids) != len(set(ids))
            or any(isinstance(item, bool) or not isinstance(item, int) for item in ids)
        ):
            raise RescueHarmError("selector mask selected IDs are not sorted unique integers")
        selected = set(ids)
        if (
            not selected.issubset(labels)
            or mask.get("selected_count") != len(ids)
            or mask.get("selected_ids_sha256") != selected_ids_sha256(tuple(ids))
        ):
            raise RescueHarmError("selector mask selected IDs/count/hash drifted")
        counts = {
            str(class_id): sum(labels[sample_id] == class_id for sample_id in ids)
            for class_id in sorted(set(labels.values()))
        }
        counts = {key: value for key, value in counts.items() if value}
        if (
            mask.get("selected_class_counts") != counts
            or metadata.get("selected_count") != len(ids)
            or metadata.get("selected_class_counts") != counts
        ):
            raise RescueHarmError("selector mask class/count metadata drifted")
        provenance = mask.get("provenance")
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("route") != route
            or provenance.get("anchor_robust_correct") is not anchor_correct
            or provenance.get("parent_checkpoint_sha256") != parent.get("checkpoint_sha256")
            or provenance.get("parent_sample_state_sha256") != parent.get("sample_state_sha256")
        ):
            raise RescueHarmError("selector mask parent/route provenance drifted")
        if kind == "history":
            if provenance.get("approved_selector_spec_sha256") != bundle_sha:
                raise RescueHarmError("history selector mask does not bind this bundle")
        elif provenance.get("reference_history_selector_spec_sha256") != bundle_sha:
            raise RescueHarmError("random selector mask does not bind this bundle")
        eligible = {sample_id for sample_id, row in feature.items() if bool(row["robust_correct"]) is anchor_correct}
        if not selected.issubset(eligible) or metadata.get("eligible_count") != len(eligible):
            raise RescueHarmError("selector mask does not match epoch39 feature-route eligibility")
        result[arm] = selected
    # These route populations are recomputed from epoch-39 robust correctness,
    # never from future outcome fields or a serialized eligibility list.
    result["PF"] = {sample_id for sample_id, row in feature.items() if bool(row["robust_correct"])}
    result["NR"] = set(feature) - result["PF"]
    return result


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    categories = {name: sum(row["category"] == name for row in rows) for name in CATEGORIES}
    count = len(rows)
    return {
        "count": count,
        "categories": categories,
        "net_rescue": categories["rescued"] - categories["harmed"],
        "net_rescue_rate": (categories["rescued"] - categories["harmed"]) / count if count else None,
    }


def _category(control: bool, arm: bool) -> str:
    return (
        "rescued"
        if not control and arm
        else "harmed"
        if control and not arm
        else "stable_correct"
        if arm
        else "unchanged_failure"
    )


def _feature_panel(
    path: Path, lineage_path: Path, *, expected_count: int
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    try:
        meta = _lineage(
            lineage_path,
            path,
            key="feature_observations_sha256",
            expected_count=expected_count,
            protocol="feature_protocol",
        )
        import pyarrow.parquet as pq

        raw = [dict(row) for row in pq.read_table(path).to_pylist()]
        panel = _domain_panel(raw, epochs=FEATURE_EPOCHS, expected_count=expected_count, name="feature")
    except ValueError as exc:
        raise RescueHarmError(str(exc)) from exc
    # Preserve the exact stored true-label probability only after validating
    # the schema-v2 record with H4a's frozen field/algebra checks.
    for row in raw:
        if row["epoch"] == 39:
            panel[39][row["sample_id"]]["teacher_clean_true_probability"] = float(row["teacher_clean_true_probability"])
    if not isinstance(meta.get("teacher"), Mapping):
        raise RescueHarmError("feature replay lacks teacher identity")
    return panel[39], meta


def report_rescue_harm(
    *,
    observations: Mapping[str, tuple[Path, Path]],
    mask_bundle: Path,
    feature_observations: Path,
    feature_lineage: Path,
    output: Path,
    expected_count: int,
) -> dict[str, Any]:
    """Report exhaustive paired categories; masks are frozen before moderators join."""
    if output.exists():
        raise FileExistsError("refusing to overwrite rescue/harm report")
    if set(observations) != set(ARMS):
        raise RescueHarmError("report requires control and all four PF/NR history/random arms")
    parsed = {arm: _read_observations(*observations[arm], arm=SOURCE_ARM[arm]) for arm in ARMS}
    control_meta, control = parsed["control"]
    identity_keys = ("seed", "dataset_identity", "attack_identity", "analysis_seed", "teacher", "student_identity")
    if any(any(parsed[arm][0].get(key) != control_meta.get(key) for key in identity_keys) for arm in ARMS[1:]):
        raise RescueHarmError("arm replay attack/student/teacher/seed/dataset identity drifted")
    ids = set(control[EPOCHS[0]])
    if len(ids) != expected_count or any(
        set(parsed[arm][1][EPOCHS[0]]) != ids
        or any(
            parsed[arm][1][EPOCHS[0]][sample_id]["class_id"] != control[EPOCHS[0]][sample_id]["class_id"]
            for sample_id in ids
        )
        for arm in ARMS[1:]
    ):
        raise RescueHarmError("arms do not share one exact stable-ID population")
    feature, feature_meta = _feature_panel(feature_observations, feature_lineage, expected_count=expected_count)
    if _teacher_identity(feature_meta.get("teacher")) != _teacher_identity(control_meta.get("teacher")):
        raise RescueHarmError("epoch39 feature replay teacher lineage identity drifted")
    if set(feature) != ids or any(
        feature[sample_id]["class_id"] != control[EPOCHS[0]][sample_id]["class_id"] for sample_id in ids
    ):
        raise RescueHarmError("epoch39 feature replay stable-ID/class join drifted")
    masks = _mask_bundle(mask_bundle, feature=feature)
    per_epoch: dict[str, Any] = {}
    for epoch in EPOCHS:
        epoch_report: dict[str, Any] = {}
        for arm in ARMS[1:]:
            rows = []
            for sample_id in sorted(ids):
                c, a, moderator = control[epoch][sample_id], parsed[arm][1][epoch][sample_id], feature[sample_id]
                selected = sample_id in masks[arm]
                route = "PF" if arm.startswith("PF_") else "NR"
                eligible = sample_id in masks[route]
                rows.append(
                    {
                        "category": _category(bool(c["robust_correct"]), bool(a["robust_correct"])),
                        "selected": selected,
                        "eligible": eligible,
                        "route": route,
                        "selection": "history" if arm.endswith("_H") else "random",
                        "teacher_clean_correct": bool(moderator["teacher_clean_correct"]),
                        "teacher_adversarial_correct": bool(moderator["teacher_adversarial_correct"]),
                        "teacher_clean_to_adversarial_flip": bool(moderator["teacher_prediction_flip"]),
                        "true_label_mix_l1_distance": 1 - float(moderator["teacher_clean_true_probability"]),
                    }
                )
            if sum(_summary(rows)["categories"].values()) != len(rows):
                raise RescueHarmError("rescue/harm categories are not exhaustive")
            groups = {
                "all": rows,
                "selected": [row for row in rows if row["selected"]],
                "non_selected": [row for row in rows if not row["selected"]],
                "eligible": [row for row in rows if row["eligible"]],
                "non_eligible": [row for row in rows if not row["eligible"]],
                "teacher_clean_correct": [row for row in rows if row["teacher_clean_correct"]],
                "teacher_clean_wrong": [row for row in rows if not row["teacher_clean_correct"]],
                "teacher_adversarial_correct": [row for row in rows if row["teacher_adversarial_correct"]],
                "teacher_adversarial_wrong": [row for row in rows if not row["teacher_adversarial_correct"]],
                "teacher_clean_to_adversarial_flip": [row for row in rows if row["teacher_clean_to_adversarial_flip"]],
                "teacher_clean_to_adversarial_stable_prediction": [
                    row for row in rows if not row["teacher_clean_to_adversarial_flip"]
                ],
            }
            epoch_report[arm] = {
                "route": route,
                "selection": "history" if arm.endswith("_H") else "random",
                "categories": {name: _summary(group) for name, group in groups.items()},
                "true_label_mix_l1_distance": {
                    name: _float_summary([float(row["true_label_mix_l1_distance"]) for row in group])
                    for name, group in groups.items()
                },
            }
        per_epoch[str(epoch)] = epoch_report
    result = {
        "schema_version": 1,
        "contract": "completed_v2_rescue_harm_report_v1",
        "exploratory_model_level_moderation_not_identifiable_unit_causal_effect": True,
        "epochs": list(EPOCHS),
        "input_identity": {
            "arm_lineage_sha256": {arm: sha256_file(observations[arm][1]) for arm in ARMS},
            "mask_bundle_sha256": sha256_file(mask_bundle),
            "feature_lineage_sha256": sha256_file(feature_lineage),
            "attack_identity": control_meta["attack_identity"],
        },
        "epochs_report": per_epoch,
        "diagnostics": {
            "kl_js": "not_available_without_full_distribution",
            "gradient": "not_available_without_full_distribution",
            "official_test": "not_used",
        },
        "true_label_mix_l1_formula": (
            "||p_teacher - (0.5*p_teacher + 0.5*one_hot(y))||_1 "
            "= 1 - p_teacher_clean(y); exact for all samples"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json(result) + b"\n")
    return result


def _float_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }
