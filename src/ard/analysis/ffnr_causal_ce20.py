"""Read-only common CE-PGD20 endpoints for the completed FF/NR causal pilot.

The training-state ``sample-stats`` files use the KL-PGD10 training attack and
are deliberately not accepted as endpoint observations here.  Every outcome in
this module is produced by a new eval-mode, pixel-space CE-PGD20 replay of an
explicit saved child checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from ard.analysis.ffnr_strong_replay import (
    EXPECTED_STABLE_ID_CLASS_UNIVERSE_SHA256,
    StrongReplayError,
    deterministic_replay_backend,
    replay_checkpoint_rows,
    stable_id_class_universe,
)
from ard.analysis.sample_stats import write_sample_parquet
from ard.analysis.signal_audit import CheckpointInventory, canonical_json, sha256_file
from ard.analysis.teacher_risk_replay import build_replay_loader
from ard.config import load_config
from ard.engine.checkpoint import REQUIRED_KEYS, config_digest
from ard.models import build_teacher
from ard.policies import selected_ids_sha256

CONTRACT_ID = "ffnr_causal_ce20_v1"
ARMS = ("C79", "RA", "RAR", "RB", "RBR")
LABELS = ("L2", "L4")
HORIZONS = (84, 89, 94)
# The W&B checkpoint inventory is authoritative: the three saved payload
# epochs are 84/89/93, with horizon 94 ending at zero-based epoch 93.
HORIZON_TO_CHECKPOINT_EPOCH = {84: 84, 89: 89, 94: 93}
ARM_MASK_SOURCE = {
    "RA": "ffnr_route_a_strong_ce_pgd20",
    "RAR": "ffnr_route_a_matched_random",
    "RB": "ffnr_route_b_strong_ce_pgd20",
    "RBR": "ffnr_route_b_matched_random",
}

ENDPOINT_COLUMNS = (
    "label",
    "arm",
    "horizon",
    "epoch",
    "namespace",
    "sample_id",
    "class_id",
    "student_clean_correct",
    "student_robust_correct",
    "student_clean_probability_margin",
    "student_adversarial_probability_margin",
    "student_adversarial_ce",
    "route_a_selected",
    "route_a_random",
    "route_b_selected",
    "route_b_random",
)


class CausalCE20Error(StrongReplayError):
    """Raised when a causal endpoint cannot prove all of its inputs."""


def ce_pgd20_attack_identity() -> dict[str, object]:
    """Return the complete, frozen common endpoint attack identity."""
    return {
        "norm": "linf",
        "input_domain": "pixel_0_1",
        "epsilon": "8/255",
        "epsilon_value": 8.0 / 255.0,
        "step_size": "2/255",
        "step_size_value": 2.0 / 255.0,
        "steps": 20,
        "random_start": True,
        "loss": "ce",
        "kl_target": None,
        "temperature": 1.0,
        "temperature_squared": True,
        "student_mode": "eval",
        "teacher_mode": "eval",
    }


def _read_yaml(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CausalCE20Error(f"{name} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise CausalCE20Error(f"{name} must be a mapping")
    return value


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CausalCE20Error(f"{name} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise CausalCE20Error(f"{name} must be a JSON object")
    return value


def _path(root: Path, value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CausalCE20Error(f"{name} must be a non-empty path")
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (root / candidate).resolve()


def parse_causal_config(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the intentionally small, explicit endpoint launcher schema."""
    required = {
        "schema_version",
        "contract",
        "expected_count",
        "stable_id_class_universe_sha256",
        "checkpoint_inventory",
        "horizons",
        "replay_batch_size",
        "attack_seed",
        "replay_device_type",
        "bootstrap",
        "s2_overlay",
        "runs",
    }
    if set(mapping) != required:
        raise CausalCE20Error(f"causal CE20 config keys must be exactly {sorted(required)}")
    if mapping.get("schema_version") != 1 or mapping.get("contract") != CONTRACT_ID:
        raise CausalCE20Error("causal CE20 config schema/contract mismatch")
    if (
        mapping.get("expected_count") != 45_000
        or mapping.get("stable_id_class_universe_sha256") != EXPECTED_STABLE_ID_CLASS_UNIVERSE_SHA256
    ):
        raise CausalCE20Error("causal CE20 config stable-ID universe drifted")
    if tuple(mapping.get("horizons", ())) != HORIZONS:
        raise CausalCE20Error("causal CE20 horizons must be exactly 84/89/94")
    for name in ("replay_batch_size", "attack_seed"):
        value = mapping.get(name)
        minimum = 1 if name == "replay_batch_size" else 0
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise CausalCE20Error(f"causal CE20 config {name} is invalid")
    if mapping.get("replay_device_type") not in {"cpu", "cuda"}:
        raise CausalCE20Error("causal CE20 replay_device_type must be cpu or cuda")
    bootstrap = mapping.get("bootstrap")
    if not isinstance(bootstrap, Mapping) or set(bootstrap) != {"replicates", "seed", "strata"}:
        raise CausalCE20Error("causal CE20 bootstrap schema drifted")
    if bootstrap.get("strata") != "class_id" or any(
        isinstance(bootstrap.get(name), bool) or not isinstance(bootstrap.get(name), int) or bootstrap[name] < 0
        for name in ("replicates", "seed")
    ):
        raise CausalCE20Error("causal CE20 bootstrap must use a non-negative fixed class-stratified seed")
    if mapping.get("s2_overlay") is not None:
        raise CausalCE20Error("S2 overlay must be null until a frozen source schema is registered")
    runs = mapping.get("runs")
    if not isinstance(runs, Mapping) or set(runs) != set(LABELS):
        raise CausalCE20Error("causal CE20 config requires exactly L2/L4")
    for label in LABELS:
        value = runs[label]
        if not isinstance(value, Mapping) or set(value) != set(ARMS):
            raise CausalCE20Error(f"causal CE20 {label} requires exactly C79/RA/RAR/RB/RBR")
        for arm in ARMS:
            entry = value[arm]
            if not isinstance(entry, Mapping) or set(entry) != {
                "manifest",
                "resolved_config",
                "validation_metrics",
            }:
                raise CausalCE20Error(f"causal CE20 {label}.{arm} schema drifted")
    return dict(mapping)


def load_causal_config(path: Path) -> dict[str, Any]:
    parsed = parse_causal_config(_read_yaml(path, name="causal CE20 config"))
    root = path.parent
    inventory_path = _path(root, parsed["checkpoint_inventory"], name="checkpoint_inventory")
    inventory = _read_json(inventory_path, name="checkpoint inventory")
    if (
        set(inventory) != {"contract", "artifacts"}
        or inventory.get("contract") != "ffnr_causal_ce20_checkpoint_inventory_v1"
    ):
        raise CausalCE20Error("checkpoint inventory contract/schema drifted")
    artifacts = inventory.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(LABELS) * len(ARMS) * len(HORIZONS):
        raise CausalCE20Error("checkpoint inventory must bind exactly thirty endpoint checkpoints")
    checkpoint_paths: dict[tuple[str, str, int], tuple[Path, str]] = {}
    for item in artifacts:
        if not isinstance(item, Mapping) or set(item) != {
            "arm",
            "artifact",
            "epoch",
            "path",
            "seed",
            "sha256",
            "size",
            "version",
        }:
            raise CausalCE20Error("checkpoint inventory artifact schema drifted")
        label, arm, epoch = item["seed"], item["arm"], item["epoch"]
        if label not in LABELS or arm not in ARMS or epoch not in set(HORIZON_TO_CHECKPOINT_EPOCH.values()):
            raise CausalCE20Error("checkpoint inventory artifact identity drifted")
        key = (label, arm, int(epoch))
        if key in checkpoint_paths or not isinstance(item.get("sha256"), str) or len(item["sha256"]) != 64:
            raise CausalCE20Error("checkpoint inventory has duplicate or invalid SHA-256 entries")
        artifact_path = Path(item["path"])
        if not artifact_path.is_file():
            raise CausalCE20Error("checkpoint inventory artifact is missing")
        checkpoint_paths[key] = (artifact_path.resolve(), item["sha256"])
    runs: dict[str, Any] = {}
    for label, by_arm in parsed["runs"].items():
        runs[label] = {}
        for arm, entry in by_arm.items():
            runs[label][arm] = {
                "manifest": _path(root, entry["manifest"], name=f"runs.{label}.{arm}.manifest"),
                "resolved_config": _path(root, entry["resolved_config"], name=f"runs.{label}.{arm}.resolved_config"),
                "checkpoint_paths": {
                    horizon: checkpoint_paths[(label, arm, HORIZON_TO_CHECKPOINT_EPOCH[horizon])]
                    for horizon in HORIZONS
                },
                "validation_metrics": _path(
                    root, entry["validation_metrics"], name=f"runs.{label}.{arm}.validation_metrics"
                ),
            }
    return {
        **parsed,
        "checkpoint_inventory": inventory_path,
        "checkpoint_inventory_sha256": sha256_file(inventory_path),
        "runs": runs,
    }


def _load_mask(raw: Mapping[str, Any], *, arm: str) -> set[int]:
    if arm == "C79":
        if raw.get("intervention", {}).get("mask") is not None:
            raise CausalCE20Error("C79 must not carry an intervention mask")
        return set()
    intervention = raw.get("intervention")
    if (
        not isinstance(intervention, Mapping)
        or intervention.get("arm") != arm
        or not isinstance(intervention.get("mask"), Mapping)
    ):
        raise CausalCE20Error(f"{arm} lacks its registered fixed mask")
    spec = intervention["mask"]
    path = _path(Path.cwd(), spec.get("path"), name=f"{arm} mask path")
    if sha256_file(path) != spec.get("sha256"):
        raise CausalCE20Error(f"{arm} fixed mask SHA-256 drifted")
    value = _read_json(path, name=f"{arm} fixed mask")
    required = {
        "schema_version",
        "namespace",
        "num_classes",
        "provenance",
        "selected_class_counts",
        "selected_count",
        "selected_ids",
        "selected_ids_sha256",
    }
    if set(value) != required or value.get("schema_version") != 1 or value.get("namespace") != "train":
        raise CausalCE20Error(f"{arm} fixed mask schema drifted")
    provenance = value.get("provenance")
    ids = value.get("selected_ids")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("source") != ARM_MASK_SOURCE[arm]
        or not isinstance(ids, list)
    ):
        raise CausalCE20Error(f"{arm} fixed mask provenance drifted")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in ids) or ids != sorted(set(ids)):
        raise CausalCE20Error(f"{arm} fixed mask IDs are invalid")
    selected = set(ids)
    ids_sha = selected_ids_sha256(tuple(ids))
    if (
        spec.get("selected_count") != len(selected)
        or spec.get("selected_ids_sha256") != ids_sha
        or value.get("selected_ids_sha256") != ids_sha
    ):
        raise CausalCE20Error(f"{arm} fixed mask count/ID digest drifted")
    return selected


def _arm_input(*, label: str, arm: str, entry: Mapping[str, Any], horizon: int) -> dict[str, Any]:
    raw = _read_yaml(entry["resolved_config"], name=f"{label}.{arm} resolved config")
    config = load_config(entry["resolved_config"])
    manifest = _read_json(entry["manifest"], name=f"{label}.{arm} manifest")
    if manifest.get("status") != "completed" or manifest.get("run_id") != config.tracking.run_id:
        raise CausalCE20Error(f"{label}.{arm} completed manifest/run identity drifted")
    checkpoint_path, checkpoint_sha256 = entry["checkpoint_paths"][horizon]
    if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != checkpoint_sha256:
        raise CausalCE20Error(f"{label}.{arm} horizon-{horizon} checkpoint is missing")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # torch error classes vary
        raise CausalCE20Error(f"{label}.{arm} checkpoint is unreadable") from exc
    epoch = HORIZON_TO_CHECKPOINT_EPOCH[horizon]
    expected_hash = config_digest(raw)
    if not isinstance(payload, Mapping) or REQUIRED_KEYS.difference(payload) or payload.get("epoch") != epoch:
        raise CausalCE20Error(f"{label}.{arm} does not bind registered endpoint checkpoint epoch {epoch}")
    if payload.get("config_hash") != expected_hash or payload.get("tracker_run_id") != config.tracking.run_id:
        raise CausalCE20Error(f"{label}.{arm} checkpoint config/run lineage drifted")
    selection_attack = config.method.selection_attack
    if (
        selection_attack is None
        or selection_attack.identity() != ce_pgd20_attack_identity()
        or selection_attack.trace_step_losses
    ):
        raise CausalCE20Error(f"{label}.{arm} selection attack is not exact eval-mode pixel CE-PGD20")
    intervention = raw.get("intervention")
    if (
        not isinstance(intervention, Mapping)
        or intervention.get("arm") != arm
        or not isinstance(intervention.get("parent"), Mapping)
    ):
        raise CausalCE20Error(f"{label}.{arm} intervention parent lineage is missing")
    parent = dict(intervention["parent"])
    mask = _load_mask(raw, arm=arm)
    inventory = CheckpointInventory(
        run_id=config.tracking.run_id,
        artifact_name=f"{arm}-horizon-{horizon}",
        aliases=("last",),
        publication_order=epoch,
        path=str(checkpoint_path),
        sha256=checkpoint_sha256,
        epoch=epoch,
        sample_state_present=True,
        sample_state_count=45_000,
        config_hash=expected_hash,
        scientific_git_sha=str(parent.get("git_sha", "")),
    )
    return {
        "raw": raw,
        "config": config,
        "manifest": manifest,
        "parent": parent,
        "mask": mask,
        "checkpoint": inventory,
        "validation_metrics": entry["validation_metrics"],
    }


def _validate_common_inputs(inputs: Mapping[str, Mapping[str, Any]]) -> None:
    reference = inputs["C79"]
    parent = reference["parent"]
    fields = ("checkpoint_sha256", "epoch", "teacher_checkpoint_sha256", "train_partition_ids_labels_sha256")
    if parent.get("epoch") != 79:
        raise CausalCE20Error("causal endpoint requires the exact epoch-79 common parent")
    teacher = reference["config"].teacher
    if teacher is None:
        raise CausalCE20Error("causal endpoint requires a registered frozen teacher")
    for arm, item in inputs.items():
        if any(item["parent"].get(field) != parent.get(field) for field in fields):
            raise CausalCE20Error(f"{arm} common-parent lineage drifted")
        if item["config"].teacher != teacher:
            raise CausalCE20Error(f"{arm} teacher identity drifted")
    if len(inputs["RA"]["mask"]) != len(inputs["RAR"]["mask"]) or len(inputs["RB"]["mask"]) != len(
        inputs["RBR"]["mask"]
    ):
        raise CausalCE20Error("selected and matched-random masks must be count matched")


def _tracked_clean_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    paths = (Path(__file__).resolve(), root / "src/ard/cli/ffnr_causal_ce20.py")
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
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CausalCE20Error("causal endpoint requires tracked source and readable Git identity") from exc
    if len(sha) != 40 or dirty:
        raise CausalCE20Error("causal endpoint requires a tracked-clean analysis revision")
    hashes = {str(path.relative_to(root)): sha256_file(path) for path in paths}
    return {
        "git": {"sha": sha, "dirty": False},
        "source_files": hashes,
        "source_sha256": hashlib.sha256(canonical_json(hashes)).hexdigest(),
    }


def _endpoint_rows(
    *, label: str, arm: str, horizon: int, rows: Sequence[Mapping[str, Any]], masks: Mapping[str, set[int]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        sample_id = int(row["sample_id"])
        result.append(
            {
                "label": label,
                "arm": arm,
                "horizon": horizon,
                "epoch": int(row["epoch"]),
                "namespace": row["namespace"],
                "sample_id": sample_id,
                "class_id": int(row["class_id"]),
                "student_clean_correct": bool(row["student_clean_correct"]),
                "student_robust_correct": bool(row["student_robust_correct"]),
                "student_clean_probability_margin": float(row["student_clean_probability_margin"]),
                "student_adversarial_probability_margin": float(row["student_adversarial_probability_margin"]),
                "student_adversarial_ce": float(row["student_adversarial_ce"]),
                "route_a_selected": sample_id in masks["RA"],
                "route_a_random": sample_id in masks["RAR"],
                "route_b_selected": sample_id in masks["RB"],
                "route_b_random": sample_id in masks["RBR"],
            }
        )
    return result


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _effect(
    control: Mapping[int, Mapping[str, Any]], treatment: Mapping[int, Mapping[str, Any]], ids: Sequence[int]
) -> dict[str, Any]:
    if not ids or any(item not in control or item not in treatment for item in ids):
        raise CausalCE20Error("paired endpoint stable-ID join is incomplete")
    robust_delta = [
        int(bool(treatment[item]["student_robust_correct"])) - int(bool(control[item]["student_robust_correct"]))
        for item in ids
    ]
    clean_delta = [
        int(bool(treatment[item]["student_clean_correct"])) - int(bool(control[item]["student_clean_correct"]))
        for item in ids
    ]
    rescue = [
        not bool(control[item]["student_robust_correct"]) and bool(treatment[item]["student_robust_correct"])
        for item in ids
    ]
    harm = [
        bool(control[item]["student_robust_correct"]) and not bool(treatment[item]["student_robust_correct"])
        for item in ids
    ]
    return {
        "n": len(ids),
        "control_robust_accuracy": _mean([float(bool(control[item]["student_robust_correct"])) for item in ids]),
        "treatment_robust_accuracy": _mean([float(bool(treatment[item]["student_robust_correct"])) for item in ids]),
        "robust_accuracy_delta": _mean(robust_delta),
        "control_clean_accuracy": _mean([float(bool(control[item]["student_clean_correct"])) for item in ids]),
        "treatment_clean_accuracy": _mean([float(bool(treatment[item]["student_clean_correct"])) for item in ids]),
        "clean_accuracy_delta": _mean(clean_delta),
        "rescue_count": sum(rescue),
        "rescue_rate": _mean([float(item) for item in rescue]),
        "harm_count": sum(harm),
        "harm_rate": _mean([float(item) for item in harm]),
        "net_rescue_count": sum(robust_delta),
        "net_rescue_rate": _mean(robust_delta),
        "clean_margin_delta": _mean(
            [
                float(treatment[item]["student_clean_probability_margin"])
                - float(control[item]["student_clean_probability_margin"])
                for item in ids
            ]
        ),
        "adversarial_margin_delta": _mean(
            [
                float(treatment[item]["student_adversarial_probability_margin"])
                - float(control[item]["student_adversarial_probability_margin"])
                for item in ids
            ]
        ),
    }


def _bootstrap_difference(
    *,
    control: Mapping[int, Mapping[str, Any]],
    selected: Mapping[int, Mapping[str, Any]],
    random_arm: Mapping[int, Mapping[str, Any]],
    selected_ids: set[int],
    random_ids: set[int],
    replicates: int,
    seed: int,
) -> dict[str, Any] | None:
    if replicates == 0:
        return None
    by_class: dict[int, tuple[list[int], list[int]]] = {}
    for class_id in range(10):
        left = sorted(item for item in selected_ids if int(control[item]["class_id"]) == class_id)
        right = sorted(item for item in random_ids if int(control[item]["class_id"]) == class_id)
        if len(left) != len(right):
            raise CausalCE20Error("class-stratified selected/random masks are not matched")
        by_class[class_id] = (left, right)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(replicates):
        left_values: list[int] = []
        right_values: list[int] = []
        for left, right in by_class.values():
            left_values.extend(
                int(bool(selected[item]["student_robust_correct"])) - int(bool(control[item]["student_robust_correct"]))
                for item in (rng.choice(left) for _ in left)
            )
            right_values.extend(
                int(bool(random_arm[item]["student_robust_correct"]))
                - int(bool(control[item]["student_robust_correct"]))
                for item in (rng.choice(right) for _ in right)
            )
        draws.append(_mean(left_values) - _mean(right_values))
    draws.sort()
    point = (
        _effect(control, selected, sorted(selected_ids))["robust_accuracy_delta"]
        - _effect(control, random_arm, sorted(random_ids))["robust_accuracy_delta"]
    )
    return {
        "replicates": replicates,
        "seed": seed,
        "strata": "class_id",
        "point": point,
        "ci95": [draws[int(0.025 * (replicates - 1))], draws[int(0.975 * (replicates - 1))]],
    }


def _validation(path: Path, *, endpoint_epoch: int) -> dict[str, float] | None:
    if not path.is_file():
        return None
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path, columns=["epoch", "val_clean_accuracy", "val_pgd_accuracy"])
    except Exception as exc:
        raise CausalCE20Error(f"validation metrics are unreadable: {path}") from exc
    rows = [row for row in table.to_pylist() if row.get("epoch") == endpoint_epoch]
    if len(rows) != 1:
        raise CausalCE20Error("validation metrics lack exactly one endpoint epoch")
    row = rows[0]
    if any(
        isinstance(row.get(name), bool)
        or not isinstance(row.get(name), (int, float))
        or not math.isfinite(float(row[name]))
        for name in ("val_clean_accuracy", "val_pgd_accuracy")
    ):
        raise CausalCE20Error("validation metrics are non-finite")
    return {"clean_accuracy": float(row["val_clean_accuracy"]), "pgd20_accuracy": float(row["val_pgd_accuracy"])}


def summarize_endpoint(
    *,
    label: str,
    horizon: int,
    rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    masks: Mapping[str, set[int]],
    bootstrap: Mapping[str, int],
    validation_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Compute paired model-level effects; never label them unit-level causality."""
    by_arm = {arm: {int(row["sample_id"]): row for row in rows_by_arm[arm]} for arm in ARMS}
    reference = by_arm["C79"]
    if any(set(rows) != set(reference) for rows in by_arm.values()):
        raise CausalCE20Error("endpoint arms do not share one exact stable-ID universe")
    universe = sorted(reference)
    report: dict[str, Any] = {
        "label": label,
        "horizon": horizon,
        "endpoint_epoch": HORIZON_TO_CHECKPOINT_EPOCH[horizon],
        "overall": {},
        "routes": {},
        "validation": {},
        "s2_overlay": {"available": False, "reason": "no frozen S2 source schema is registered"},
    }
    for arm in ARMS:
        report["overall"][arm] = _effect(reference, by_arm[arm], universe)
        value = _validation(validation_paths[arm], endpoint_epoch=HORIZON_TO_CHECKPOINT_EPOCH[horizon])
        if value is not None:
            report["validation"][arm] = value
    for route, selected_arm, random_arm in (("A", "RA", "RAR"), ("B", "RB", "RBR")):
        selected_ids, random_ids = sorted(masks[selected_arm]), sorted(masks[random_arm])
        selected_effect = _effect(reference, by_arm[selected_arm], selected_ids)
        random_effect = _effect(reference, by_arm[random_arm], random_ids)
        nonselected = sorted(set(universe).difference(masks[selected_arm]))
        contrast = {
            name: selected_effect[name] - random_effect[name]
            for name in (
                "robust_accuracy_delta",
                "clean_accuracy_delta",
                "net_rescue_rate",
                "clean_margin_delta",
                "adversarial_margin_delta",
            )
        }
        report["routes"][route] = {
            "selected": {"arm": selected_arm, **selected_effect},
            "random": {"arm": random_arm, **random_effect},
            "selected_minus_random": contrast,
            "selected_nonselected_spillover": {
                "arm": selected_arm,
                **_effect(reference, by_arm[selected_arm], nonselected),
            },
            "bootstrap_selected_minus_random": _bootstrap_difference(
                control=reference,
                selected=by_arm[selected_arm],
                random_arm=by_arm[random_arm],
                selected_ids=masks[selected_arm],
                random_ids=masks[random_arm],
                replicates=int(bootstrap["replicates"]),
                seed=int(bootstrap["seed"]),
            ),
        }
    return report


def _atomic_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def run_causal_endpoint(
    *, config_path: Path, label: str, horizon: int, device: torch.device, output_dir: Path
) -> dict[str, Path]:
    """Replay five explicit child checkpoints and write one immutable endpoint."""
    launch = load_causal_config(config_path)
    if label not in LABELS or horizon not in HORIZONS:
        raise CausalCE20Error("requested label/horizon is outside the frozen causal endpoint contract")
    if device.type != launch["replay_device_type"] or (device.type == "cuda" and not torch.cuda.is_available()):
        raise CausalCE20Error("requested device does not match the frozen endpoint runtime")
    if output_dir.exists():
        raise CausalCE20Error("causal CE20 output directory already exists; refusing to overwrite")
    inputs = {arm: _arm_input(label=label, arm=arm, entry=launch["runs"][label][arm], horizon=horizon) for arm in ARMS}
    _validate_common_inputs(inputs)
    provenance = _tracked_clean_provenance()
    control_config = inputs["C79"]["config"]
    teacher = build_teacher(control_config.teacher, tier=control_config.tier).to(device)  # type: ignore[arg-type]
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise CausalCE20Error("causal endpoint teacher must be frozen")
    loader = build_replay_loader(control_config, batch_size=launch["replay_batch_size"])
    masks = {arm: set(inputs[arm]["mask"]) for arm in ARMS if arm != "C79"}
    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    replay_meta: dict[str, Any] = {}
    with deterministic_replay_backend():
        for index, arm in enumerate(ARMS):
            result = replay_checkpoint_rows(
                checkpoint=inputs[arm]["checkpoint"],
                training_config=inputs[arm]["config"],
                teacher=teacher,
                loader=loader,
                device=device,
                attack_seed_base=launch["attack_seed"] + index * 10_000_019,
            )
            stable_id_class_universe(result.rows, expected_count=launch["expected_count"])
            rows_by_arm[arm] = _endpoint_rows(label=label, arm=arm, horizon=horizon, rows=result.rows, masks=masks)
            replay_meta[arm] = {
                "checkpoint": asdict(inputs[arm]["checkpoint"]),
                "max_abs_delta": result.max_abs_delta,
                "mask_count": len(inputs[arm]["mask"]),
            }
    report = summarize_endpoint(
        label=label,
        horizon=horizon,
        rows_by_arm=rows_by_arm,
        masks=masks,
        bootstrap=launch["bootstrap"],
        validation_paths={arm: inputs[arm]["validation_metrics"] for arm in ARMS},
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = [row for arm in ARMS for row in rows_by_arm[arm]]
    if any(set(row) != set(ENDPOINT_COLUMNS) for row in rows):
        raise CausalCE20Error("causal endpoint output schema drifted")
    observations = write_sample_parquet(rows, output_dir / "causal-ce20-observations.parquet")
    lineage = {
        "schema_version": 1,
        "contract": CONTRACT_ID,
        "label": label,
        "horizon": horizon,
        "endpoint_epoch": HORIZON_TO_CHECKPOINT_EPOCH[horizon],
        "attack_identity": ce_pgd20_attack_identity(),
        "attack_identity_sha256": hashlib.sha256(canonical_json(ce_pgd20_attack_identity())).hexdigest(),
        "attack_seed": launch["attack_seed"],
        "seed_rule": "attack_seed + 10000019*arm_index + 1000003*batch_index",
        "expected_count": launch["expected_count"],
        "checkpoint_inventory": str(launch["checkpoint_inventory"]),
        "checkpoint_inventory_sha256": launch["checkpoint_inventory_sha256"],
        "masks": {
            arm: {"count": len(ids), "ids_sha256": selected_ids_sha256(tuple(sorted(ids)))}
            for arm, ids in masks.items()
        },
        "arms": replay_meta,
        "analysis_provenance": provenance,
        "observations_sha256": sha256_file(observations),
        "row_count": len(rows),
    }
    return {
        "observations": observations,
        "report": _atomic_json(output_dir / "report.json", report),
        "lineage": _atomic_json(output_dir / "lineage.json", lineage),
    }
