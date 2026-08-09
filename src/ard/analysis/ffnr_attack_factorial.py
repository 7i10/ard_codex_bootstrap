"""Hash-bound CE/KL x PGD10/20 FF/NR replay for Chen L2/L4."""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import torch
import yaml

from ard.analysis.ffnr_strong_replay import (
    CONTRACT_ID as STRONG_CONTRACT_ID,
)
from ard.analysis.ffnr_strong_replay import (
    StrongReplayError,
    checkpoint_cache_identity,
    deterministic_replay_backend,
    load_cached_checkpoint,
    load_checkpoint_inventory_document,
    replay_checkpoint_rows,
    select_explicit_checkpoints,
    write_checkpoint_cache,
    write_outputs,
)
from ard.analysis.rslad_signal_replay import portable_cifar10_train_identity
from ard.analysis.signal_audit import canonical_json, sha256_file
from ard.analysis.teacher_risk_replay import build_replay_loader
from ard.config import load_config
from ard.config.schema import AttackConfig
from ard.engine.checkpoint import config_digest
from ard.models import build_teacher

CONTRACT_ID = "ffnr_attack_factorial_v1"
CONDITIONS = ("ce_pgd10", "ce_pgd20", "kl_pgd10", "kl_pgd20")


class FactorialReplayError(StrongReplayError):
    """Raised when a factorial replay input or lineage is invalid."""


def factorial_attack(condition: str) -> AttackConfig:
    """Return one preregistered pixel-space attack identity."""
    if condition not in CONDITIONS:
        raise FactorialReplayError(f"unknown factorial condition: {condition}")
    loss: Literal["ce", "kl"] = "kl" if condition.startswith("kl_") else "ce"
    target: Literal["teacher_clean"] | None = "teacher_clean" if loss == "kl" else None
    return AttackConfig(
        norm="linf",
        input_domain="pixel_0_1",
        epsilon="8/255",
        step_size="2/255",
        steps=20 if condition.endswith("20") else 10,
        random_start=True,
        loss=loss,
        kl_target=target,
        temperature=1.0,
        temperature_squared=True,
        student_mode="eval",
        teacher_mode="eval",
        trace_step_losses=False,
    )


def _sha256_mapping(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise FactorialReplayError(f"cannot read factorial config {path}") from exc
    if not isinstance(value, dict):
        raise FactorialReplayError("factorial config must be a mapping")
    return value


def _configured_path(config_path: Path, value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise FactorialReplayError(f"factorial config {name} must be a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def parse_factorial_config(mapping: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "contract",
        "run_id",
        "manifest",
        "checkpoint_inventory",
        "epochs",
        "train_expected_count",
        "replay_batch_size",
        "attack_seed",
        "replay_device_type",
        "output_root",
    }
    if set(mapping) != required:
        raise FactorialReplayError(f"factorial config keys must be exactly {sorted(required)}")
    if mapping["schema_version"] != 1 or mapping["contract"] != CONTRACT_ID:
        raise FactorialReplayError("factorial config schema/contract mismatch")
    if not isinstance(mapping["run_id"], str) or not mapping["run_id"]:
        raise FactorialReplayError("factorial run_id must be non-empty")
    epochs = mapping["epochs"]
    if (
        not isinstance(epochs, list)
        or not epochs
        or tuple(sorted(set(epochs))) != tuple(epochs)
        or any(isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0 for epoch in epochs)
    ):
        raise FactorialReplayError("factorial epochs must be sorted non-negative integers")
    for name in ("train_expected_count", "replay_batch_size"):
        if isinstance(mapping[name], bool) or not isinstance(mapping[name], int) or mapping[name] < 1:
            raise FactorialReplayError(f"factorial config {name} is invalid")
    if (
        isinstance(mapping["attack_seed"], bool)
        or not isinstance(mapping["attack_seed"], int)
        or mapping["attack_seed"] < 0
    ):
        raise FactorialReplayError("factorial attack_seed is invalid")
    if mapping["replay_device_type"] not in {"cpu", "cuda"}:
        raise FactorialReplayError("factorial replay_device_type must be cpu or cuda")
    return dict(mapping)


def _provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    paths = {
        "factorial_module": Path(__file__).resolve(),
        "strong_replay_module": root / "src/ard/analysis/ffnr_strong_replay.py",
        "factorial_cli": root / "src/ard/cli/ffnr_attack_factorial.py",
    }
    relative = [str(path.relative_to(root)) for path in paths.values()]
    subprocess.run(["git", "-C", str(root), "ls-files", "--error-unmatch", *relative], check=True, capture_output=True)
    sha = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(sha) != 40 or dirty:
        raise FactorialReplayError("factorial replay requires a tracked-clean Git revision")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    return {"git": {"sha": sha, "dirty": False}, "source_files": hashes, "source_sha256": _sha256_mapping(hashes)}


def run_factorial(
    *,
    config_path: Path,
    condition: str,
    device: torch.device,
    requested_epochs: Sequence[int] | None = None,
) -> dict[str, Path]:
    """Run one factorial condition against the frozen Chen checkpoint inventory."""
    launch = parse_factorial_config(_load_mapping(config_path))
    attack = factorial_attack(condition)
    if device.type != launch["replay_device_type"] or (device.type == "cuda" and not torch.cuda.is_available()):
        raise FactorialReplayError("requested replay device does not match the configured runtime")
    configured_epochs = tuple(launch["epochs"])
    epochs = configured_epochs if requested_epochs is None else tuple(requested_epochs)
    if not epochs or any(epoch not in configured_epochs for epoch in epochs) or tuple(sorted(set(epochs))) != epochs:
        raise FactorialReplayError("requested factorial epochs must be an ordered configured subset")
    manifest = _configured_path(config_path, launch["manifest"], name="manifest")
    inventory_path = _configured_path(config_path, launch["checkpoint_inventory"], name="checkpoint_inventory")
    resolved = manifest.parent / "resolved_config.yaml"
    training_config = load_config(resolved)
    saved = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(saved, dict):
        raise FactorialReplayError("saved resolved config is invalid")
    selected = select_explicit_checkpoints(
        load_checkpoint_inventory_document(path=inventory_path, manifest_path=manifest, run_id=launch["run_id"]),
        run_id=launch["run_id"],
        epochs=epochs,
    )
    resolved_hash = config_digest(saved)
    if any(item.config_hash != resolved_hash for item in selected):
        raise FactorialReplayError("factorial checkpoint config hash mismatch")
    if training_config.teacher is None:
        raise FactorialReplayError("factorial replay requires a registered teacher")
    dataset_identity = portable_cifar10_train_identity(saved, expected_count=launch["train_expected_count"])
    provenance = _provenance()
    with deterministic_replay_backend() as backend:
        teacher = build_teacher(training_config.teacher, tier=training_config.tier).to(device)
        if any(parameter.requires_grad for parameter in teacher.parameters()):
            raise FactorialReplayError("factorial replay teacher must be frozen")
        loader = build_replay_loader(training_config, batch_size=launch["replay_batch_size"])
        runtime: dict[str, Any] = {
            "device": str(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "deterministic_backend": asdict(backend),
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
        teacher_metadata = teacher.metadata.model_dump(mode="json")
        output = _configured_path(config_path, launch["output_root"], name="output_root") / condition
        results = []
        identities = []
        cache_dir = output / "checkpoint-cache"
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
                        attack_config=attack,
                    ),
                )
            )
    lineage = {
        "contract": CONTRACT_ID,
        "schema_version": 1,
        "strong_replay_contract": STRONG_CONTRACT_ID,
        "run_id": launch["run_id"],
        "condition": condition,
        "requested_epochs": list(epochs),
        "attack_identity": attack.identity(),
        "attack_identity_sha256": attack.identity_sha256(),
        "attack_seed_base": launch["attack_seed"],
        "replay_batch_size": launch["replay_batch_size"],
        "train_expected_count": launch["train_expected_count"],
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "checkpoint_inventory": str(inventory_path),
        "checkpoint_inventory_sha256": sha256_file(inventory_path),
        "saved_resolved_config": str(resolved),
        "saved_resolved_config_sha256": sha256_file(resolved),
        "checkpoints": [{"epoch": item.epoch, "sha256": item.sha256, "path": item.path} for item in selected],
        "analysis_provenance": provenance,
        "checkpoint_cache_identities": identities,
        "runtime": runtime,
        "teacher": teacher_metadata,
    }
    return write_outputs(output_dir=output, results=results, lineage=lineage)
