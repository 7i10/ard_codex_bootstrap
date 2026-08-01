"""Read-only common-trajectory RSLAD signal replay and predictive audit.

This module is intentionally separate from :mod:`teacher_risk_replay`: that
module is the formal epoch-99 teacher-risk replay contract.  Here, periodic
RSLAD ``last`` checkpoints are replayed in two independent random-start seed
domains to construct a clearly labelled checkpoint-panel proxy outcome.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ard.analysis.sample_stats import write_sample_parquet
from ard.attacks import AttackRequest, LinfPGD
from ard.config import ExperimentConfig
from ard.config.schema import training_execution_identity
from ard.engine.checkpoint import REQUIRED_KEYS
from ard.signals import RobustMarginSignal

from .signal_audit import (
    CheckpointInventory,
    SignalAuditError,
    _bootstrap_indices,
    _fit_logistic,
    _predict_logistic,
    _sha256_mapping,
    binary_metrics,
    canonical_json,
    deterministic_hash_split,
    join_samples,
    logical_dataset_identity,
    namespaced_samples,
    sha256_file,
)
from .teacher_risk_replay import load_historical_student


class RSLADSignalReplayError(SignalAuditError):
    """Raised when common-trajectory replay cannot establish its identity."""


PERIODIC_EPOCHS = tuple(range(4, 200, 5))
FEATURE_EPOCHS = tuple(epoch for epoch in PERIODIC_EPOCHS if epoch <= 99)
OUTCOME_EPOCHS = tuple(epoch for epoch in PERIODIC_EPOCHS if epoch >= 99)
PANEL_EMA_BETA = 0.9**5
SEED_DOMAINS = ("feature", "outcome")
DOMAIN_SEED_FORMULA = (
    "attack_seed_base=sha256(canonical_json(['rslad-common-trajectory-v1',base_seed,domain]))[:8]&((1<<63)-1); "
    "batch_seed=attack_seed_base+1000003*batch_index"
)


@dataclass(frozen=True)
class TemporalPanelInventory:
    """Hash-bound, exactly scheduled periodic-last checkpoints for one run."""

    run_id: str
    config_hash: str
    scientific_git_sha: str
    world_size: int
    checkpoints: tuple[CheckpointInventory, ...]


@dataclass(frozen=True)
class ReplayCheckpointResult:
    """One checkpoint's scalar observations and the attack-bound evidence."""

    epoch: int
    seed_domain: str
    attack_seed_base: int
    max_abs_delta: float
    rows: tuple[dict[str, Any], ...]


def repository_root_from_source() -> Path:
    """Return the repository owning this analysis implementation."""
    return Path(__file__).resolve().parents[3]


def semantic_source_paths() -> dict[str, Path]:
    """Return local source paths whose bytes define replay/audit semantics."""
    root = repository_root_from_source()
    return {
        "analysis_module": Path(__file__).resolve(),
        "cli_module": root / "src/ard/cli/rslad_signal_replay.py",
        "pgd": root / "src/ard/attacks/pgd.py",
        "teacher_risk_replay": root / "src/ard/analysis/teacher_risk_replay.py",
        "signal_audit": root / "src/ard/analysis/signal_audit.py",
        "robust_margin": root / "src/ard/signals/robust_margin.py",
        "model_registry": root / "src/ard/models/registry.py",
        "model_teacher": root / "src/ard/models/teacher.py",
        "teacher_registry": root / "src/ard/models/teacher_registry.py",
    }


def source_hashes() -> dict[str, str]:
    """Hash every local source path that changes replay or audit semantics."""
    paths = semantic_source_paths()
    if any(not path.is_file() for path in paths.values()):
        raise RSLADSignalReplayError("common-trajectory analysis source tree is incomplete")
    return {name: sha256_file(path) for name, path in paths.items()}


def verify_semantic_sources_tracked(*, root: Path, paths: Mapping[str, Path]) -> None:
    """Fail closed if any hashed semantic source is outside the Git index."""
    try:
        relative_paths = [str(path.resolve().relative_to(root.resolve())) for path in paths.values()]
        subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", *relative_paths],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise RSLADSignalReplayError("common-trajectory semantic source file is not Git-tracked") from exc


def tracked_clean_analysis_provenance() -> dict[str, Any]:
    """Bind execution to the current clean tracked analysis revision."""
    root = repository_root_from_source()
    paths = semantic_source_paths()
    verify_semantic_sources_tracked(root=root, paths=paths)
    try:
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RSLADSignalReplayError("common-trajectory replay requires a readable tracked Git identity") from exc
    if len(sha) != 40 or any(character not in "0123456789abcdef" for character in sha) or dirty:
        raise RSLADSignalReplayError("common-trajectory replay requires a tracked-clean analysis Git revision")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    return {"git": {"sha": sha, "dirty": False}, "source_files": hashes, "source_sha256": _sha256_mapping(hashes)}


def _require_int(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RSLADSignalReplayError(f"{name} must be an integer >= {minimum}")
    return value


def validate_exact_epoch_schedule(
    checkpoints: Sequence[CheckpointInventory], *, expected_epochs: Sequence[int] = PERIODIC_EPOCHS
) -> tuple[CheckpointInventory, ...]:
    """Reject missing, duplicate, or extra periodic ``last`` epochs exactly."""
    expected = tuple(expected_epochs)
    if expected != tuple(sorted(set(expected))) or not expected:
        raise RSLADSignalReplayError("expected periodic epochs must be a non-empty sorted unique sequence")
    by_epoch: dict[int, list[CheckpointInventory]] = defaultdict(list)
    for checkpoint in checkpoints:
        if not checkpoint.periodic_last:
            raise RSLADSignalReplayError("exact temporal inventory accepts only periodic last checkpoints")
        by_epoch[checkpoint.epoch].append(checkpoint)
    actual = tuple(sorted(by_epoch))
    if actual != expected:
        raise RSLADSignalReplayError(f"periodic last epochs are {actual}, expected exactly {expected}")
    if any(len(items) != 1 for items in by_epoch.values()):
        raise RSLADSignalReplayError("each required periodic last epoch must have exactly one checkpoint")
    return tuple(by_epoch[epoch][0] for epoch in expected)


def inventory_common_trajectory(
    inventory: Sequence[CheckpointInventory], *, run_id: str, expected_config_hash: str | None = None
) -> TemporalPanelInventory:
    """Freeze the 40-checkpoint RSLAD panel and its execution identity.

    ``inventory_run_bundle`` has already verified content-addressed checkpoint
    bytes.  This second layer rejects a cadence that is merely similar to the
    planned panel, and checks the world-size stored in every checkpoint.
    """
    if not isinstance(run_id, str) or not run_id:
        raise RSLADSignalReplayError("common-trajectory inventory requires a non-empty run_id")
    selected = [item for item in inventory if item.run_id == run_id and item.periodic_last]
    checkpoints = validate_exact_epoch_schedule(selected)
    if len({item.sha256 for item in checkpoints}) != len(checkpoints):
        raise RSLADSignalReplayError("periodic checkpoint hashes must be unique across epochs")
    config_hashes = {item.config_hash for item in checkpoints}
    git_shas = {item.scientific_git_sha for item in checkpoints}
    if len(config_hashes) != 1 or len(git_shas) != 1:
        raise RSLADSignalReplayError("periodic checkpoints must share config and scientific Git identities")
    config_hash = next(iter(config_hashes))
    if expected_config_hash is not None and config_hash != expected_config_hash:
        raise RSLADSignalReplayError("periodic checkpoint config hash does not match resolved training config")
    world_sizes: set[int] = set()
    for checkpoint in checkpoints:
        try:
            payload = torch.load(checkpoint.path, map_location="cpu", weights_only=False)
        except Exception as exc:  # pragma: no cover - inventory has normally already deserialized it
            raise RSLADSignalReplayError(
                "periodic checkpoint is unreadable during execution identity validation"
            ) from exc
        if not isinstance(payload, Mapping):
            raise RSLADSignalReplayError("periodic checkpoint payload is not a mapping")
        if payload.get("epoch") != checkpoint.epoch or payload.get("tracker_run_id") != run_id:
            raise RSLADSignalReplayError("periodic checkpoint payload does not match immutable inventory")
        world_sizes.add(_require_int(payload.get("world_size"), name="checkpoint world_size", minimum=1))
    if len(world_sizes) != 1:
        raise RSLADSignalReplayError("periodic checkpoints must share one checkpoint world_size")
    return TemporalPanelInventory(
        run_id=run_id,
        config_hash=config_hash,
        scientific_git_sha=next(iter(git_shas)),
        world_size=next(iter(world_sizes)),
        checkpoints=checkpoints,
    )


def inventory_feature_trajectory(
    manifest_path: Path, *, run_id: str, expected_config_hash: str | None = None
) -> TemporalPanelInventory:
    """Read only the pre-anchor checkpoints needed for feature-only replay.

    The manifest order is the immutable artifact-publication order.  We use it
    solely to select the first 20 periodic ``last`` artifacts, then prove that
    their payload epochs are exactly 4, 9, ..., 99.  In particular, this
    function must not hash or deserialize any post-anchor checkpoint: those
    bytes are prospective outcomes and are forbidden inputs to L3 features.
    """
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RSLADSignalReplayError("run-bundle manifest is unreadable") from exc
    if not isinstance(manifest, Mapping):
        raise RSLADSignalReplayError("run-bundle manifest must be a mapping")
    if manifest.get("run_id") != run_id:
        raise RSLADSignalReplayError("feature replay manifest run ID does not match the configured run")
    config_hash = manifest.get("config_hash")
    if not isinstance(config_hash, str) or len(config_hash) != 64:
        raise RSLADSignalReplayError("feature replay manifest config hash is invalid")
    if expected_config_hash is not None and config_hash != expected_config_hash:
        raise RSLADSignalReplayError("feature replay manifest config hash does not match resolved training config")
    git = manifest.get("git")
    scientific_git_sha = git.get("sha") if isinstance(git, Mapping) else None
    if not isinstance(scientific_git_sha, str) or len(scientific_git_sha) not in {40, 64}:
        raise RSLADSignalReplayError("feature replay manifest scientific Git SHA is invalid")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise RSLADSignalReplayError("feature replay manifest artifacts must be a list")

    periodic_entries = [
        (publication_order, entry)
        for publication_order, entry in enumerate(artifacts)
        if isinstance(entry, Mapping)
        and entry.get("type") == "model"
        and isinstance(entry.get("aliases"), list)
        and "last" in entry["aliases"]
    ]
    if len(periodic_entries) != len(PERIODIC_EPOCHS):
        raise RSLADSignalReplayError("feature replay requires exactly the immutable 40-checkpoint panel")
    selected = periodic_entries[: len(FEATURE_EPOCHS)]
    if len(selected) != len(FEATURE_EPOCHS):  # defensive if the panel constant changes
        raise RSLADSignalReplayError("feature replay checkpoint-panel selection is incomplete")

    inventory: list[CheckpointInventory] = []
    world_sizes: set[int] = set()
    bundle = manifest_path.parent.resolve()
    for publication_order, entry in selected:
        name, aliases, expected_sha = entry.get("name"), entry.get("aliases"), entry.get("sha256")
        local_path, source_path = entry.get("local_path"), entry.get("path")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(aliases, list)
            or not all(isinstance(alias, str) for alias in aliases)
            or len(set(aliases)) != len(aliases)
            or not isinstance(expected_sha, str)
            or len(expected_sha) != 64
            or not isinstance(local_path, str)
            or not isinstance(source_path, str)
        ):
            raise RSLADSignalReplayError("feature replay periodic artifact metadata is invalid")
        checkpoint_path = (bundle / local_path / Path(source_path).name).resolve()
        if bundle not in checkpoint_path.parents or not checkpoint_path.is_file():
            raise RSLADSignalReplayError("feature replay checkpoint is missing or escapes its run bundle")
        actual_sha = sha256_file(checkpoint_path)
        if actual_sha != expected_sha:
            raise RSLADSignalReplayError("feature replay checkpoint hash does not match its manifest artifact")
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as exc:  # pragma: no cover - torch serialization exception types vary
            raise RSLADSignalReplayError("feature replay checkpoint is unreadable") from exc
        if not isinstance(payload, Mapping) or REQUIRED_KEYS.difference(payload):
            raise RSLADSignalReplayError("feature replay checkpoint lacks the complete checkpoint contract")
        epoch = payload.get("epoch")
        if (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or payload.get("epoch_boundary") != "end"
            or payload.get("tracker_run_id") != run_id
            or payload.get("config_hash") != config_hash
        ):
            raise RSLADSignalReplayError("feature replay checkpoint identity or epoch boundary is invalid")
        world_size = _require_int(payload.get("world_size"), name="feature replay checkpoint world_size", minimum=1)
        world_sizes.add(world_size)
        inventory.append(
            CheckpointInventory(
                run_id=run_id,
                artifact_name=name,
                aliases=tuple(aliases),
                publication_order=publication_order,
                path=str(checkpoint_path),
                sha256=actual_sha,
                epoch=epoch,
                sample_state_present=isinstance(payload.get("sample_state"), Mapping) and bool(payload["sample_state"]),
                sample_state_count=(
                    len(payload["sample_state"].get("records", {}))
                    if isinstance(payload.get("sample_state"), Mapping)
                    and isinstance(payload["sample_state"].get("records", {}), Mapping)
                    else 0
                ),
                config_hash=config_hash,
                scientific_git_sha=scientific_git_sha,
            )
        )
    checkpoints = validate_exact_epoch_schedule(inventory, expected_epochs=FEATURE_EPOCHS)
    if len(world_sizes) != 1:
        raise RSLADSignalReplayError("feature replay checkpoints must share one checkpoint world size")
    return TemporalPanelInventory(
        run_id=run_id,
        config_hash=config_hash,
        scientific_git_sha=scientific_git_sha,
        world_size=next(iter(world_sizes)),
        checkpoints=checkpoints,
    )


def validate_rslad_replay_attack(config: ExperimentConfig) -> None:
    """Require the immutable RSLAD KL teacher-clean PGD-10 threat model."""
    attack = config.method.attack
    expected = {
        "norm": "linf",
        "input_domain": "pixel_0_1",
        "epsilon": "8/255",
        "epsilon_value": 8.0 / 255.0,
        "step_size": "2/255",
        "step_size_value": 2.0 / 255.0,
        "steps": 10,
        "random_start": True,
        "loss": "kl",
        "kl_target": "teacher_clean",
        "temperature": 1.0,
        "temperature_squared": True,
        "student_mode": "eval",
        "teacher_mode": "eval",
    }
    observed = attack.model_dump(mode="json")
    if config.method.id != "rslad" or {key: observed.get(key) for key in expected} != expected:
        raise RSLADSignalReplayError("common-trajectory replay requires exact RSLAD KL teacher_clean PGD-10")


def portable_cifar10_train_identity(resolved_config: Mapping[str, Any], *, expected_count: int) -> dict[str, Any]:
    """Require the fixed 45,000-source-ID CIFAR-10 train partition."""
    if expected_count != 45000:
        raise RSLADSignalReplayError("common-trajectory replay requires expected_count=45000")
    identity = logical_dataset_identity(resolved_config, train_expected_count=expected_count)
    dataset = identity["dataset"]
    if not isinstance(dataset, Mapping) or dataset.get("name") != "cifar10" or dataset.get("split") != "train":
        raise RSLADSignalReplayError("common-trajectory replay is restricted to the CIFAR-10 train partition")
    return identity


def runtime_identity(device: torch.device) -> dict[str, Any]:
    """Record the executable runtime and selected device without guessing hardware."""
    identity: dict[str, Any] = {
        "selected_device": str(device),
        "device_type": device.type,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RSLADSignalReplayError("CUDA runtime identity requested while CUDA is unavailable")
        index = torch.cuda.current_device() if device.index is None else device.index
        identity["cuda_device_index"] = index
        identity["cuda_device_name"] = torch.cuda.get_device_name(index)
        identity["cuda_device_capability"] = list(torch.cuda.get_device_capability(index))
    return identity


def domain_seed(*, base_seed: int, domain: str) -> int:
    """Derive a fixed, non-overlapping random-start panel for one seed domain."""
    _require_int(base_seed, name="base_seed")
    if domain not in SEED_DOMAINS:
        raise RSLADSignalReplayError(f"attack seed domain must be one of {SEED_DOMAINS}")
    digest = hashlib.sha256(canonical_json(["rslad-common-trajectory-v1", base_seed, domain])).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def domain_replay_protocol(*, base_seed: int, domain: str, device_type: str) -> dict[str, Any]:
    """Exact protocol for a hashed seed domain, not the epoch-99 replay formula."""
    _require_int(base_seed, name="base_seed")
    if domain not in SEED_DOMAINS or device_type not in {"cpu", "cuda"}:
        raise RSLADSignalReplayError("domain replay protocol has an invalid seed domain or device type")
    return {
        "base_seed": base_seed,
        "seed_domain": domain,
        "seed_formula": DOMAIN_SEED_FORMULA,
        "generator_per_batch": True,
        "precision": "fp32",
        "device_type": device_type,
        "backend": f"torch-{device_type}",
    }


def checkpoint_cache_identity(
    *,
    checkpoint: CheckpointInventory,
    training_config: ExperimentConfig,
    seed_domain: str,
    base_seed: int,
    expected_count: int,
    device: torch.device,
    replay_batch_size: int,
    saved_resolved_config_mapping_sha256: str,
    saved_resolved_config_file_sha256: str,
    teacher_metadata: Mapping[str, Any],
    dataset_identity: Mapping[str, Any],
    analysis_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Identity that must match before a checkpoint replay cache is trusted."""
    seed = domain_seed(base_seed=base_seed, domain=seed_domain)
    return {
        "schema_version": 1,
        "source_files": source_hashes(),
        "checkpoint": {
            "run_id": checkpoint.run_id,
            "epoch": checkpoint.epoch,
            "sha256": checkpoint.sha256,
            "config_hash": checkpoint.config_hash,
        },
        "attack_identity": training_config.method.attack.identity(),
        "expected_count": expected_count,
        "saved_resolved_config_mapping_sha256": saved_resolved_config_mapping_sha256,
        "saved_resolved_config_file_sha256": saved_resolved_config_file_sha256,
        "teacher": dict(teacher_metadata),
        "dataset_identity": dict(dataset_identity),
        "analysis_provenance": dict(analysis_provenance),
        "protocol": {
            **domain_replay_protocol(base_seed=base_seed, domain=seed_domain, device_type=device.type),
            "batch_size": _require_int(replay_batch_size, name="replay_batch_size", minimum=1),
            "runtime": runtime_identity(device),
        },
        "attack_seed_base": seed,
    }


def _cache_paths(cache_dir: Path, identity: Mapping[str, Any]) -> tuple[Path, Path]:
    key = _sha256_mapping(identity)
    checkpoint = identity["checkpoint"]
    assert isinstance(checkpoint, Mapping)
    stem = f"{identity['protocol']['seed_domain']}-epoch{checkpoint['epoch']}-{key}"
    return cache_dir / f"{stem}.parquet", cache_dir / f"{stem}.json"


def load_cached_checkpoint(*, cache_dir: Path, identity: Mapping[str, Any]) -> ReplayCheckpointResult | None:
    """Load only a complete, hash-bound checkpoint cache; reject any partial cache."""
    parquet_path, metadata_path = _cache_paths(cache_dir, identity)
    temporary_paths = (parquet_path.with_suffix(parquet_path.suffix + ".tmp"), metadata_path.with_suffix(".json.tmp"))
    if not parquet_path.exists() and not metadata_path.exists():
        for path in temporary_paths:
            if path.exists() and path.is_file():
                path.unlink()
            elif path.exists():
                raise RSLADSignalReplayError("checkpoint replay cache temporary entry is not a regular file")
        return None
    if not parquet_path.is_file() or not metadata_path.is_file():
        for path in (parquet_path, metadata_path):
            if path.exists() and path.is_file():
                path.unlink()
            elif path.exists():
                raise RSLADSignalReplayError("checkpoint replay cache entry is not a regular file")
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        import pyarrow.parquet as pq

        rows = pq.read_table(parquet_path).to_pylist()
    except Exception as exc:
        raise RSLADSignalReplayError("checkpoint replay cache is unreadable") from exc
    if not isinstance(metadata, Mapping) or canonical_json(metadata.get("identity")) != canonical_json(identity):
        raise RSLADSignalReplayError("checkpoint replay cache identity does not match this replay")
    if metadata.get("parquet_sha256") != sha256_file(parquet_path):
        raise RSLADSignalReplayError("checkpoint replay cache Parquet hash does not match metadata")
    ordered = tuple(sorted((dict(row) for row in rows), key=lambda row: int(row["sample_id"])))
    if metadata.get("rows_sha256") != _sha256_mapping(ordered):
        raise RSLADSignalReplayError("checkpoint replay cache row hash does not match metadata")
    expected_count = _require_int(identity.get("expected_count"), name="cache expected_count", minimum=1)
    namespaced_samples(ordered, namespace="train", expected_count=expected_count)
    max_abs_delta = metadata.get("max_abs_delta")
    if (
        isinstance(max_abs_delta, bool)
        or not isinstance(max_abs_delta, (int, float))
        or not math.isfinite(max_abs_delta)
    ):
        raise RSLADSignalReplayError("checkpoint replay cache max_abs_delta is invalid")
    checkpoint = identity["checkpoint"]
    protocol = identity["protocol"]
    assert isinstance(checkpoint, Mapping) and isinstance(protocol, Mapping)
    return ReplayCheckpointResult(
        epoch=int(checkpoint["epoch"]),
        seed_domain=str(protocol["seed_domain"]),
        attack_seed_base=int(identity["attack_seed_base"]),
        max_abs_delta=float(max_abs_delta),
        rows=ordered,
    )


def write_checkpoint_cache(
    *, cache_dir: Path, identity: Mapping[str, Any], result: ReplayCheckpointResult
) -> ReplayCheckpointResult:
    """Atomically publish one verified checkpoint cache only after its replay completes."""
    parquet_path, metadata_path = _cache_paths(cache_dir, identity)
    if parquet_path.exists() or metadata_path.exists():
        raise FileExistsError("refusing to overwrite existing checkpoint replay cache")
    checkpoint = identity["checkpoint"]
    protocol = identity["protocol"]
    assert isinstance(checkpoint, Mapping) and isinstance(protocol, Mapping)
    if (
        result.epoch != checkpoint["epoch"]
        or result.seed_domain != protocol["seed_domain"]
        or result.attack_seed_base != identity["attack_seed_base"]
    ):
        raise RSLADSignalReplayError("checkpoint replay result does not match its cache identity")
    ordered = tuple(sorted(result.rows, key=lambda row: int(row["sample_id"])))
    expected_count = _require_int(identity.get("expected_count"), name="cache expected_count", minimum=1)
    namespaced_samples(ordered, namespace="train", expected_count=expected_count)
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_sample_parquet(ordered, parquet_path)
    metadata = {
        "identity": identity,
        "parquet_sha256": sha256_file(parquet_path),
        "rows_sha256": _sha256_mapping(ordered),
        "max_abs_delta": result.max_abs_delta,
    }
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_bytes(canonical_json(metadata) + b"\n")
    temporary.replace(metadata_path)
    return ReplayCheckpointResult(
        epoch=result.epoch,
        seed_domain=result.seed_domain,
        attack_seed_base=result.attack_seed_base,
        max_abs_delta=result.max_abs_delta,
        rows=ordered,
    )


def _normalized_entropy(logits: torch.Tensor) -> torch.Tensor:
    logits = logits.detach().float()
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise RSLADSignalReplayError("teacher logits must have shape [batch, class>=2]")
    probabilities = F.softmax(logits, dim=1)
    normalized = -(probabilities * F.log_softmax(logits, dim=1)).sum(dim=1) / math.log(logits.shape[1])
    if (
        not torch.isfinite(normalized).all()
        or bool((normalized < -1e-6).any())
        or bool((normalized > 1.0 + 1e-6).any())
    ):
        raise RSLADSignalReplayError("normalized teacher entropy must be finite in [0, 1]")
    return normalized


def replay_checkpoint_rows(
    *,
    checkpoint: CheckpointInventory,
    training_config: ExperimentConfig,
    teacher: nn.Module,
    loader: Iterable[Any],
    device: torch.device,
    seed_domain: str,
    base_seed: int,
) -> ReplayCheckpointResult:
    """Replay both signals from one exact student-crafted adversarial input."""
    validate_rslad_replay_attack(training_config)
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise RSLADSignalReplayError("common-trajectory replay requires frozen teacher parameters")
    attack_seed_base = domain_seed(base_seed=base_seed, domain=seed_domain)
    student, _ = load_historical_student(
        checkpoint,
        config=training_config,
        device=device,
        expected_config_hash=checkpoint.config_hash,
    )
    teacher.eval()
    student.eval()
    attack = LinfPGD(training_config.method.attack)
    margin_signal = RobustMarginSignal()
    rows: list[dict[str, Any]] = []
    max_abs_delta = 0.0
    for batch_index, raw_batch in enumerate(loader):
        batch = raw_batch.to(device)
        generator = torch.Generator(device=device).manual_seed(attack_seed_base + 1_000_003 * batch_index)
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            teacher_clean_logits = teacher(batch.images.float()).detach().float()
        result = attack.generate(
            AttackRequest(
                inputs=batch.images,
                labels=batch.labels,
                student=student,
                teacher=teacher,
                target_logits=teacher_clean_logits,
                generator=generator,
            )
        )
        max_abs_delta = max(max_abs_delta, result.max_abs_delta)
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            student_logits = student(result.adversarial.float()).detach().float()
            teacher_logits = teacher(result.adversarial.float()).detach().float()
            entropy = _normalized_entropy(teacher_logits)
            margin = margin_signal.compute(
                student_adv_logits=student_logits,
                labels=batch.labels,
                valid_mask=torch.ones_like(batch.labels, dtype=torch.bool),
            ).values
            student_risk = (1.0 - margin) / 2.0
            robust_correct = student_logits.argmax(dim=1).eq(batch.labels)
        if not torch.isfinite(margin).all() or not torch.isfinite(student_risk).all():
            raise RSLADSignalReplayError("student margin/risk must be finite FP32 values")
        if bool((student_risk < -1e-6).any()) or bool((student_risk > 1.0 + 1e-6).any()):
            raise RSLADSignalReplayError("student probability-margin risk must be in [0, 1]")
        for parameter in teacher.parameters():
            if parameter.requires_grad or parameter.grad is not None:
                raise RSLADSignalReplayError("common-trajectory replay populated a teacher parameter gradient")
        for sample_id, class_id, entropy_value, margin_value, risk_value, correct in zip(
            batch.sample_ids.tolist(),
            batch.labels.tolist(),
            entropy.tolist(),
            margin.tolist(),
            student_risk.tolist(),
            robust_correct.tolist(),
            strict=True,
        ):
            rows.append(
                {
                    "namespace": "train",
                    "sample_id": int(sample_id),
                    "class_id": int(class_id),
                    "epoch": checkpoint.epoch,
                    "teacher_entropy_normalized": float(entropy_value),
                    "student_probability_margin": float(margin_value),
                    "student_margin_risk": float(risk_value),
                    "robust_correct": bool(correct),
                }
            )
        student.zero_grad(set_to_none=True)
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise RSLADSignalReplayError("replayed checkpoint contains duplicate stable source sample IDs")
    epsilon = training_config.method.attack.epsilon_value
    assert epsilon is not None
    if max_abs_delta > epsilon + 1e-7:
        raise RSLADSignalReplayError("common-trajectory replay violated pixel-space Linf projection")
    return ReplayCheckpointResult(
        epoch=checkpoint.epoch,
        seed_domain=seed_domain,
        attack_seed_base=attack_seed_base,
        max_abs_delta=max_abs_delta,
        rows=tuple(rows),
    )


def _panel_by_epoch(
    rows: Sequence[Mapping[str, Any]], *, expected_epochs: Sequence[int], expected_count: int
) -> dict[int, dict[Any, Mapping[str, Any]]]:
    by_epoch: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        epoch = _require_int(row.get("epoch"), name="row epoch")
        by_epoch[epoch].append(row)
    if tuple(sorted(by_epoch)) != tuple(expected_epochs):
        raise RSLADSignalReplayError("replay rows do not contain the exact required checkpoint panel")
    panels: dict[int, dict[Any, Mapping[str, Any]]] = {}
    reference: dict[Any, Mapping[str, Any]] | None = None
    for epoch in expected_epochs:
        panel = namespaced_samples(by_epoch[epoch], namespace="train", expected_count=expected_count)
        if reference is not None:
            for key, prior, current in join_samples(reference, panel):
                if prior.get("class_id") != current.get("class_id"):
                    raise RSLADSignalReplayError(
                        f"class_id changed for sample {key.sample_id} across replay checkpoints"
                    )
        panels[epoch] = panel
        reference = panel
    return panels


def build_feature_panel(rows: Sequence[Mapping[str, Any]], *, expected_count: int) -> tuple[dict[str, Any], ...]:
    """Build frozen epoch-99 current-state and preceding-history features."""
    panels = _panel_by_epoch(rows, expected_epochs=FEATURE_EPOCHS, expected_count=expected_count)
    output: list[dict[str, Any]] = []
    historical_epochs = FEATURE_EPOCHS[:-1]
    for key in sorted(panels[FEATURE_EPOCHS[0]]):
        ema: float | None = None
        historical_ema: float | None = None
        class_id: int | None = None
        entropy: float | None = None
        robust_correct_count = 0
        for epoch in FEATURE_EPOCHS:
            row = panels[epoch][key]
            margin = float(row["student_probability_margin"])
            risk = float(row["student_margin_risk"])
            value = float(row["teacher_entropy_normalized"])
            robust_correct = row.get("robust_correct")
            if not all(math.isfinite(item) for item in (margin, risk, value)) or not -1.0 <= margin <= 1.0:
                raise RSLADSignalReplayError("feature signal values must be finite and valid probability margins")
            if not 0.0 <= risk <= 1.0 or not 0.0 <= value <= 1.0:
                raise RSLADSignalReplayError("feature risks and normalized entropy must be in [0, 1]")
            if not isinstance(robust_correct, bool):
                raise RSLADSignalReplayError("feature robust_correct must be boolean")
            if not math.isclose(risk, (1.0 - margin) / 2.0, rel_tol=0.0, abs_tol=1e-7):
                raise RSLADSignalReplayError("student margin risk does not match its probability-margin definition")
            ema = margin if ema is None else PANEL_EMA_BETA * ema + (1.0 - PANEL_EMA_BETA) * margin
            if epoch in historical_epochs:
                historical_ema = (
                    margin
                    if historical_ema is None
                    else PANEL_EMA_BETA * historical_ema + (1.0 - PANEL_EMA_BETA) * margin
                )
                robust_correct_count += int(robust_correct)
            class_id = int(row["class_id"])
            entropy = value
        assert ema is not None and historical_ema is not None and class_id is not None and entropy is not None
        epoch99 = panels[FEATURE_EPOCHS[-1]][key]
        correctness_frequency = robust_correct_count / len(historical_epochs)
        output.append(
            {
                "namespace": key.namespace,
                "sample_id": key.sample_id,
                "class_id": class_id,
                "feature_epoch": FEATURE_EPOCHS[-1],
                "teacher_entropy_normalized": entropy,
                "student_robust_correct_epoch99": int(epoch99["robust_correct"]),
                "student_robust_correct_frequency": correctness_frequency,
                # These explicit names are the frozen Phase-B feature contract.
                # The panel names remain as stable aliases for existing outputs.
                "student_margin_historical_ema": historical_ema,
                "student_margin_historical_risk": (1.0 - historical_ema) / 2.0,
                "student_margin_instantaneous_epoch99": float(epoch99["student_probability_margin"]),
                "student_margin_panel_ema": ema,
                "student_margin_panel_risk": (1.0 - ema) / 2.0,
                "student_margin_epoch99": float(epoch99["student_probability_margin"]),
                "student_margin_risk_epoch99": float(epoch99["student_margin_risk"]),
            }
        )
    return tuple(output)


def build_outcome_panel(rows: Sequence[Mapping[str, Any]], *, expected_count: int) -> tuple[dict[str, Any], ...]:
    """Build the checkpoint-panel correct-to-wrong outcome through epoch 199."""
    panels = _panel_by_epoch(rows, expected_epochs=OUTCOME_EPOCHS, expected_count=expected_count)
    output: list[dict[str, Any]] = []
    for key in sorted(panels[OUTCOME_EPOCHS[0]]):
        previous: bool | None = None
        forgot = False
        transition_count = 0
        prospective_correctness: list[bool] = []
        class_id: int | None = None
        for epoch in OUTCOME_EPOCHS:
            row = panels[epoch][key]
            correct = row.get("robust_correct")
            if not isinstance(correct, bool):
                raise RSLADSignalReplayError("outcome robust_correct must be boolean")
            if previous is True and not correct:
                forgot = True
                transition_count += 1
            previous = correct
            if epoch > OUTCOME_EPOCHS[0]:
                prospective_correctness.append(correct)
            class_id = int(row["class_id"])
        assert previous is not None and class_id is not None and prospective_correctness
        output.append(
            {
                "namespace": key.namespace,
                "sample_id": key.sample_id,
                "class_id": class_id,
                "outcome_start_epoch": OUTCOME_EPOCHS[0],
                "outcome_end_epoch": OUTCOME_EPOCHS[-1],
                "checkpoint_panel_forgetting": int(forgot),
                "checkpoint_panel_transition_count": transition_count,
                "final_robust_error": int(not previous),
                "persistent_wrong": int(not any(prospective_correctness)),
                "post_anchor_robust_correct_frequency": sum(prospective_correctness) / len(prospective_correctness),
            }
        )
    return tuple(output)


def join_feature_outcome_panels(
    features: Sequence[Mapping[str, Any]], outcomes: Sequence[Mapping[str, Any]], *, expected_count: int
) -> tuple[dict[str, Any], ...]:
    """Join strict stable-ID feature and outcome panels without positional alignment."""
    feature_index = namespaced_samples(features, namespace="train", expected_count=expected_count)
    outcome_index = namespaced_samples(outcomes, namespace="train", expected_count=expected_count)
    joined: list[dict[str, Any]] = []
    for key, feature, outcome in join_samples(feature_index, outcome_index):
        if feature.get("class_id") != outcome.get("class_id"):
            raise RSLADSignalReplayError(f"feature/outcome class_id mismatch for sample {key.sample_id}")
        joined.append({**feature, **{name: value for name, value in outcome.items() if name not in feature}})
    return tuple(joined)


def _paired_metric_intervals(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_scores: Sequence[float],
    candidate_scores: Sequence[float],
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    if replicates < 1 or len(rows) != len(baseline_scores) or len(rows) != len(candidate_scores):
        raise RSLADSignalReplayError("paired bootstrap requires aligned scores and positive replicates")
    values: dict[str, list[float]] = {"auroc": [], "auprc": [], "log_loss": []}
    for replicate in range(replicates):
        selected = _bootstrap_indices(rows, seed=seed, replicate=replicate, cluster=True)
        targets = [int(rows[index]["outcome"]) for index in selected]
        try:
            baseline = binary_metrics(targets, [baseline_scores[index] for index in selected])
            candidate = binary_metrics(targets, [candidate_scores[index] for index in selected])
        except SignalAuditError:
            continue
        for metric in values:
            values[metric].append(candidate[metric] - baseline[metric])
    if not values["auroc"]:
        raise RSLADSignalReplayError("paired bootstrap produced no class-complete held-out replicate")
    intervals: dict[str, Any] = {}
    for metric, samples in values.items():
        samples.sort()
        intervals[metric] = {
            "lower": samples[max(0, math.floor(0.025 * (len(samples) - 1)))],
            "upper": samples[min(len(samples) - 1, math.ceil(0.975 * (len(samples) - 1)))],
        }
    return {"replicates": len(values["auroc"]), "class_stratified": True, "metrics": intervals}


CALIBRATION_BIN_COUNT = 10
CALIBRATION_BIN_DEFINITION = "equal_width_[lower,upper)_except_final_[0.9,1.0]"


def _calibration_summary(targets: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    """Return deterministic fixed-bin calibration diagnostics for one held-out model."""
    if len(targets) != len(scores) or not targets:
        raise RSLADSignalReplayError("calibration requires aligned non-empty targets and scores")
    bins: list[dict[str, Any]] = []
    ece = 0.0
    brier = sum((float(score) - int(target)) ** 2 for target, score in zip(targets, scores, strict=True)) / len(targets)
    for index in range(CALIBRATION_BIN_COUNT):
        lower = index / CALIBRATION_BIN_COUNT
        upper = (index + 1) / CALIBRATION_BIN_COUNT
        selected = [
            item
            for item, score in enumerate(scores)
            if (lower <= score <= upper if index == CALIBRATION_BIN_COUNT - 1 else lower <= score < upper)
        ]
        count = len(selected)
        mean_prediction = sum(float(scores[item]) for item in selected) / count if count else None
        observed_prevalence = sum(int(targets[item]) for item in selected) / count if count else None
        absolute_gap = (
            abs(mean_prediction - observed_prevalence)
            if mean_prediction is not None and observed_prevalence is not None
            else 0.0
        )
        contribution = count / len(targets) * absolute_gap
        ece += contribution
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_prediction": mean_prediction,
                "observed_prevalence": observed_prevalence,
                "absolute_gap": absolute_gap,
                "ece_contribution": contribution,
            }
        )
    return {
        "brier_score": brier,
        "expected_calibration_error": ece,
        "bin_count": CALIBRATION_BIN_COUNT,
        "bin_definition": CALIBRATION_BIN_DEFINITION,
        "bins": bins,
    }


def _secondary_outcome_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize Phase-B secondary outcomes on the primary held-out sample IDs."""
    required = {
        "checkpoint_panel_transition_count",
        "final_robust_error",
        "persistent_wrong",
        "post_anchor_robust_correct_frequency",
    }
    if not all(required.issubset(row) for row in rows):
        return {"available": False, "reason": "secondary_outcome_primitives_absent"}
    transition_counts: list[int] = []
    final_errors: list[int] = []
    persistent_wrongs: list[int] = []
    correctness_frequencies: list[float] = []
    for row in rows:
        transition_count = row["checkpoint_panel_transition_count"]
        final_error = row["final_robust_error"]
        persistent_wrong = row["persistent_wrong"]
        correctness_frequency = row["post_anchor_robust_correct_frequency"]
        if isinstance(transition_count, bool) or not isinstance(transition_count, int) or transition_count < 0:
            raise RSLADSignalReplayError("checkpoint-panel transition count must be a non-negative integer")
        if final_error not in {0, 1, False, True} or persistent_wrong not in {0, 1, False, True}:
            raise RSLADSignalReplayError("secondary binary outcomes must be binary")
        if (
            isinstance(correctness_frequency, bool)
            or not isinstance(correctness_frequency, (int, float))
            or not math.isfinite(float(correctness_frequency))
            or not 0.0 <= float(correctness_frequency) <= 1.0
        ):
            raise RSLADSignalReplayError("post-anchor robust correctness frequency must be finite in [0, 1]")
        transition_counts.append(transition_count)
        final_errors.append(int(final_error))
        persistent_wrongs.append(int(persistent_wrong))
        correctness_frequencies.append(float(correctness_frequency))
    return {
        "available": True,
        "sample_count": len(rows),
        "checkpoint_panel_transition_count": {
            "mean": sum(transition_counts) / len(transition_counts),
            "min": min(transition_counts),
            "max": max(transition_counts),
            "positive_prevalence": sum(value > 0 for value in transition_counts) / len(transition_counts),
        },
        "final_robust_error": {"prevalence": sum(final_errors) / len(final_errors)},
        "persistent_wrong": {"prevalence": sum(persistent_wrongs) / len(persistent_wrongs)},
        "post_anchor_robust_correct_frequency": {
            "mean": sum(correctness_frequencies) / len(correctness_frequencies),
            "min": min(correctness_frequencies),
            "max": max(correctness_frequencies),
        },
    }


def _history_gate_decision(delta_report: Mapping[str, Any]) -> str:
    """Apply the frozen allocation rule without silently shifting boundaries."""
    try:
        point = delta_report["point_estimates"]
        bounds = delta_report["bootstrap_95"]["metrics"]
        delta_auroc = float(point["auroc"])
        delta_log_loss = float(point["log_loss"])
        lower_auroc = float(bounds["auroc"]["lower"])
        upper_auroc = float(bounds["auroc"]["upper"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RSLADSignalReplayError("history gate report is incomplete") from exc
    if delta_auroc >= 0.02 and lower_auroc > 0.0 and delta_log_loss < 0.0:
        return "go"
    if upper_auroc < 0.01 or delta_log_loss >= 0.0:
        return "no_go"
    return "inconclusive"


def predictive_audit(
    rows: Sequence[Mapping[str, Any]], *, split_seed: int, bootstrap_seed: int, bootstrap_replicates: int
) -> dict[str, Any]:
    """Compare frozen Phase-B predictors on one held-out class-stratified split."""
    prepared: list[dict[str, Any]] = []
    for row in rows:
        entropy = float(row["teacher_entropy_normalized"])
        risk = float(row["student_margin_panel_risk"])
        historical_risk = float(row.get("student_margin_historical_risk", risk))
        instantaneous_risk = float(row["student_margin_risk_epoch99"])
        current_correctness = row.get("student_robust_correct_epoch99")
        if current_correctness is None:
            # Direct callers of the pre-Phase-B public audit retain their
            # entropy/student report compatibility.  Feature panels always
            # provide the measured epoch-99 robust-correctness primitive.
            current_correctness = int(instantaneous_risk < 0.5)
        correctness_frequency = row.get("student_robust_correct_frequency", 0.5)
        outcome = row.get("checkpoint_panel_forgetting")
        if (
            not math.isfinite(entropy)
            or not math.isfinite(risk)
            or not math.isfinite(historical_risk)
            or not math.isfinite(instantaneous_risk)
            or not 0.0 <= entropy <= 1.0
            or not 0.0 <= risk <= 1.0
            or not 0.0 <= historical_risk <= 1.0
            or not 0.0 <= instantaneous_risk <= 1.0
        ):
            raise RSLADSignalReplayError("predictive features must be finite [0, 1] values")
        if current_correctness not in {0, 1, False, True}:
            raise RSLADSignalReplayError("epoch-99 robust correctness must be binary")
        if not isinstance(correctness_frequency, (int, float)) or isinstance(correctness_frequency, bool):
            raise RSLADSignalReplayError("historical correctness frequency must be a finite [0, 1] value")
        correctness_frequency = float(correctness_frequency)
        if not math.isfinite(correctness_frequency) or not 0.0 <= correctness_frequency <= 1.0:
            raise RSLADSignalReplayError("historical correctness frequency must be a finite [0, 1] value")
        if outcome not in {0, 1, False, True}:
            raise RSLADSignalReplayError("checkpoint-panel forgetting outcome must be binary")
        prepared.append(
            {
                **row,
                "outcome": int(outcome),
                "teacher_entropy": entropy,
                "current_correctness": int(current_correctness),
                "instantaneous_margin_risk": instantaneous_risk,
                "historical_margin_risk": historical_risk,
                "historical_correctness_frequency": correctness_frequency,
            }
        )
    train_ids, held_out_ids = deterministic_hash_split(prepared, seed=split_seed, held_out_fraction=0.2)
    train_set, held_out_set = set(train_ids), set(held_out_ids)
    train = [row for row in prepared if int(row["sample_id"]) in train_set]
    held_out = [row for row in prepared if int(row["sample_id"]) in held_out_set]
    if not train or not held_out:
        raise RSLADSignalReplayError("fixed class-stratified split produced an empty partition")
    train_targets = [row["outcome"] for row in train]

    feature_sets: dict[str, tuple[str, ...]] = {
        "teacher_entropy": ("teacher_entropy",),
        "current_correctness": ("current_correctness",),
        "instantaneous_margin": ("instantaneous_margin_risk",),
        "history_only": ("historical_correctness_frequency", "historical_margin_risk"),
        "current_only": ("current_correctness", "instantaneous_margin_risk"),
        "current_plus_history": (
            "current_correctness",
            "instantaneous_margin_risk",
            "historical_correctness_frequency",
            "historical_margin_risk",
        ),
        "history_plus_teacher": (
            "historical_correctness_frequency",
            "historical_margin_risk",
            "teacher_entropy",
        ),
        "history_teacher_interaction": (
            "historical_correctness_frequency",
            "historical_margin_risk",
            "teacher_entropy",
            "historical_correctness_x_teacher_entropy",
            "historical_margin_risk_x_teacher_entropy",
        ),
    }
    for item in prepared:
        item["historical_correctness_x_teacher_entropy"] = (
            item["historical_correctness_frequency"] * item["teacher_entropy"]
        )
        item["historical_margin_risk_x_teacher_entropy"] = item["historical_margin_risk"] * item["teacher_entropy"]

    scores: dict[str, list[float]] = {}
    for name, features in feature_sets.items():
        fit = _fit_logistic([[float(row[feature]) for feature in features] for row in train], train_targets)
        scores[name] = _predict_logistic(fit, [[float(row[feature]) for feature in features] for row in held_out])
    targets = [row["outcome"] for row in held_out]
    models: dict[str, dict[str, Any]] = {
        name: {**binary_metrics(targets, values), "calibration": _calibration_summary(targets, values)}
        for name, values in scores.items()
    }

    def delta_report(*, baseline: str, candidate: str) -> dict[str, Any]:
        paired = _paired_metric_intervals(
            held_out,
            baseline_scores=scores[baseline],
            candidate_scores=scores[candidate],
            seed=bootstrap_seed,
            replicates=bootstrap_replicates,
        )
        return {
            "baseline": baseline,
            "candidate": candidate,
            "point_estimates": {
                metric: float(models[candidate][metric]) - float(models[baseline][metric])
                for metric in ("auroc", "auprc", "log_loss")
            },
            "bootstrap_95": paired,
        }

    current_baselines = ("current_correctness", "instantaneous_margin", "current_only")
    best_current_baseline = max(
        current_baselines,
        key=lambda name: (float(models[name]["auroc"]), -float(models[name]["log_loss"]), name),
    )
    history_minus_best_current = delta_report(baseline=best_current_baseline, candidate="history_only")
    history_decision = _history_gate_decision(history_minus_best_current)
    history_minus_entropy = delta_report(baseline="teacher_entropy", candidate="history_only")
    return {
        "outcome": "checkpoint_panel_forgetting",
        "held_out_fraction": 0.2,
        "train_sample_ids": train_ids,
        "held_out_sample_ids": held_out_ids,
        "split_identity": {
            "method": "true_class_stratified_deterministic_hash",
            "seed": split_seed,
            "train_sample_ids": train_ids,
            "held_out_sample_ids": held_out_ids,
        },
        "model_features": {name: list(features) for name, features in feature_sets.items()},
        "secondary_outcomes": _secondary_outcome_summary(held_out),
        "models": {
            **models,
            # Legacy aliases retain the original entropy/student report shape.
            "entropy": models["teacher_entropy"],
            "student": models["history_only"],
        },
        "student_minus_entropy": history_minus_entropy,
        "history_minus_instantaneous_margin": delta_report(baseline="instantaneous_margin", candidate="history_only"),
        "history_minus_current_correctness": delta_report(baseline="current_correctness", candidate="history_only"),
        "current_plus_history_minus_current_only": delta_report(
            baseline="current_only", candidate="current_plus_history"
        ),
        "history_minus_best_current_state": history_minus_best_current,
        "history_plus_teacher_minus_history": delta_report(baseline="history_only", candidate="history_plus_teacher"),
        "history_teacher_interaction_minus_history_teacher": delta_report(
            baseline="history_plus_teacher", candidate="history_teacher_interaction"
        ),
        "history_go_no_go": {
            "status": history_decision,
            "criteria": {
                "minimum_delta_auroc": 0.02,
                "paired_auroc_lower_ci_gt": 0.0,
                "log_loss_improves": True,
            },
            "baseline": best_current_baseline,
        },
        "proxy_decision": {
            "status": {"go": "student_better_proxy", "no_go": "entropy_better_proxy"}.get(
                history_decision, "inconclusive_proxy"
            ),
            "gating_metrics": ["auroc", "log_loss"],
            "secondary_metrics": ["auprc"],
            "scope": (
                "sample-conditional checkpoint-panel proxy, not training-seed uncertainty or intervention evidence"
            ),
        },
        "sensitivity": {
            "student_feature": "epoch99_instantaneous_margin_risk",
            "metrics": models["instantaneous_margin"],
        },
    }


def replay_lineage(
    *,
    panel: TemporalPanelInventory,
    training_config: ExperimentConfig,
    expected_count: int,
    replay_batch_size: int,
    device_type: str,
    runtime: Mapping[str, Any],
    feature_seed: int,
    outcome_seed: int,
    saved_resolved_config_mapping_sha256: str,
    saved_resolved_config_file_sha256: str,
    teacher_metadata: Mapping[str, Any],
    dataset_identity: Mapping[str, Any],
    analysis_provenance: Mapping[str, Any],
    feature_results: Sequence[ReplayCheckpointResult],
    outcome_results: Sequence[ReplayCheckpointResult],
) -> dict[str, Any]:
    """Return canonical, complete replay provenance without self-referential hashes."""
    if device_type not in {"cpu", "cuda"}:
        raise RSLADSignalReplayError("replay device_type must be cpu or cuda")
    if feature_seed == outcome_seed:
        raise RSLADSignalReplayError("feature and outcome replay base seeds must be independent")
    return {
        "schema_version": 1,
        "analysis_provenance": dict(analysis_provenance),
        "run_id": panel.run_id,
        "config_hash": panel.config_hash,
        "saved_resolved_config_mapping_sha256": saved_resolved_config_mapping_sha256,
        "saved_resolved_config_file_sha256": saved_resolved_config_file_sha256,
        "scientific_git_sha": panel.scientific_git_sha,
        "checkpoint_training": {
            "world_size": panel.world_size,
            "execution_identity": training_execution_identity(
                training=training_config.training, world_size=panel.world_size
            ),
        },
        "attack_identity": training_config.method.attack.identity(),
        "train_expected_count": expected_count,
        "teacher": dict(teacher_metadata),
        "dataset_identity": dict(dataset_identity),
        "runtime": dict(runtime),
        "checkpoints": [asdict(item) for item in panel.checkpoints],
        "feature_protocol": {
            **domain_replay_protocol(base_seed=feature_seed, domain="feature", device_type=device_type),
            "batch_size": replay_batch_size,
            "epochs": list(FEATURE_EPOCHS),
            "panel_ema_beta": PANEL_EMA_BETA,
        },
        "outcome_protocol": {
            **domain_replay_protocol(base_seed=outcome_seed, domain="outcome", device_type=device_type),
            "batch_size": replay_batch_size,
            "epochs": list(OUTCOME_EPOCHS),
            "outcome": "checkpoint_panel_correct_to_wrong_through_epoch_199",
        },
        "feature_replays": [
            {"epoch": item.epoch, "attack_seed_base": item.attack_seed_base, "max_abs_delta": item.max_abs_delta}
            for item in feature_results
        ],
        "outcome_replays": [
            {"epoch": item.epoch, "attack_seed_base": item.attack_seed_base, "max_abs_delta": item.max_abs_delta}
            for item in outcome_results
        ],
    }


def feature_replay_lineage(
    *,
    panel: TemporalPanelInventory,
    training_config: ExperimentConfig,
    expected_count: int,
    replay_batch_size: int,
    device_type: str,
    runtime: Mapping[str, Any],
    feature_seed: int,
    saved_resolved_config_mapping_sha256: str,
    saved_resolved_config_file_sha256: str,
    teacher_metadata: Mapping[str, Any],
    dataset_identity: Mapping[str, Any],
    analysis_provenance: Mapping[str, Any],
    feature_results: Sequence[ReplayCheckpointResult],
    feature_panel: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind L3's pre-anchor feature panel to its exact epoch-99 parent.

    This deliberately has no outcome protocol, outcome replay, or predictive
    report.  The selector can therefore treat its output as an input-only
    feature source rather than accidentally interpreting it as prospective
    intervention evidence.
    """
    if expected_count != 45000:
        raise RSLADSignalReplayError("feature-only replay requires exactly 45,000 stable train IDs")
    if device_type not in {"cpu", "cuda"}:
        raise RSLADSignalReplayError("feature-only replay device_type must be cpu or cuda")
    if tuple(item.epoch for item in panel.checkpoints) != FEATURE_EPOCHS:
        raise RSLADSignalReplayError("feature-only replay requires the exact epoch-4..99 feature schedule")
    if (
        training_config.protocol.id != "controlled_cifar10_r18_v1"
        or training_config.training.epochs != 200
        or training_config.training.per_rank_batch_size != 128
        or training_config.training.global_batch_size != 128
        or training_config.observation.profile != "teacher_response"
    ):
        raise RSLADSignalReplayError("feature-only replay requires the controlled observed-RSLAD parent protocol")
    if training_config.teacher is None or training_config.teacher.registry_id is None:
        raise RSLADSignalReplayError("feature-only replay requires a registered robust teacher parent")
    anchor = panel.checkpoints[-1]
    try:
        payload = torch.load(anchor.path, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover - torch serialization exception types vary
        raise RSLADSignalReplayError("feature-only parent checkpoint is unreadable") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("epoch") != FEATURE_EPOCHS[-1]
        or payload.get("epoch_boundary") != "end"
        or payload.get("tracker_run_id") != panel.run_id
        or payload.get("config_hash") != panel.config_hash
        or _require_int(payload.get("world_size"), name="feature-only parent world_size", minimum=1) != panel.world_size
    ):
        raise RSLADSignalReplayError("feature-only parent checkpoint lineage does not match the feature panel")
    sample_state = payload.get("sample_state")
    records = sample_state.get("records") if isinstance(sample_state, Mapping) else None
    if not isinstance(records, Mapping) or len(records) != expected_count:
        raise RSLADSignalReplayError("feature-only parent checkpoint lacks exactly 45,000 stable sample-state records")
    try:
        parent_ids = {
            int(sample_id)
            for sample_id in records
            if not isinstance(sample_id, bool) and isinstance(sample_id, (str, int))
        }
    except ValueError as exc:
        raise RSLADSignalReplayError("feature-only parent sample-state IDs are invalid") from exc
    feature_ids = {
        row.get("sample_id")
        for row in feature_panel
        if row.get("namespace") == "train"
        and isinstance(row.get("sample_id"), int)
        and not isinstance(row.get("sample_id"), bool)
    }
    if len(parent_ids) != expected_count or parent_ids != feature_ids:
        raise RSLADSignalReplayError("feature-only panel stable IDs do not exactly match the epoch-99 parent state")
    parent_sample_state_sha256 = hashlib.sha256(canonical_json(sample_state)).hexdigest()
    return {
        "schema_version": 1,
        "kind": "l3_checkpoint_panel_feature_source_v1",
        "analysis_provenance": dict(analysis_provenance),
        "run_id": panel.run_id,
        "config_hash": panel.config_hash,
        "scientific_git_sha": panel.scientific_git_sha,
        "teacher_registry_id": training_config.teacher.registry_id,
        "seed": training_config.seeds.model_init,
        "parent_epoch": FEATURE_EPOCHS[-1],
        "parent_checkpoint_sha256": anchor.sha256,
        "parent_sample_state_sha256": parent_sample_state_sha256,
        "parent_raw_config_sha256": saved_resolved_config_mapping_sha256,
        "saved_resolved_config_file_sha256": saved_resolved_config_file_sha256,
        "checkpoint_training": {
            "world_size": panel.world_size,
            "execution_identity": training_execution_identity(
                training=training_config.training, world_size=panel.world_size
            ),
        },
        "attack_identity": training_config.method.attack.identity(),
        "train_expected_count": expected_count,
        "teacher": dict(teacher_metadata),
        "dataset_identity": dict(dataset_identity),
        "runtime": dict(runtime),
        "checkpoints": [asdict(item) for item in panel.checkpoints],
        "feature_protocol": {
            **domain_replay_protocol(base_seed=feature_seed, domain="feature", device_type=device_type),
            "batch_size": replay_batch_size,
            "epochs": list(FEATURE_EPOCHS),
            "panel_ema_beta": PANEL_EMA_BETA,
        },
        "feature_replays": [
            {"epoch": item.epoch, "attack_seed_base": item.attack_seed_base, "max_abs_delta": item.max_abs_delta}
            for item in feature_results
        ],
    }


def write_feature_replay_outputs(
    *,
    output_dir: Path,
    feature_observations: Sequence[Mapping[str, Any]],
    feature_panel: Sequence[Mapping[str, Any]],
    lineage: Mapping[str, Any],
) -> dict[str, Path]:
    """Write only L3 feature artifacts; outcomes and reports are forbidden."""
    paths = {
        "feature_observations": output_dir / "feature-observations.parquet",
        "feature_panel": output_dir / "feature-panel.parquet",
        "lineage": output_dir / "lineage.json",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("refusing to overwrite an existing feature-only replay output")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_sample_parquet(
        sorted(feature_observations, key=lambda row: (int(row["epoch"]), int(row["sample_id"]))),
        paths["feature_observations"],
    )
    write_sample_parquet(sorted(feature_panel, key=lambda row: int(row["sample_id"])), paths["feature_panel"])
    paths["lineage"].write_bytes(
        canonical_json(
            {
                **lineage,
                "feature_observations_sha256": sha256_file(paths["feature_observations"]),
                "feature_panel_sha256": sha256_file(paths["feature_panel"]),
            }
        )
        + b"\n"
    )
    return paths


def write_replay_outputs(
    *,
    output_dir: Path,
    feature_observations: Sequence[Mapping[str, Any]],
    outcome_observations: Sequence[Mapping[str, Any]],
    feature_panel: Sequence[Mapping[str, Any]],
    outcome_panel: Sequence[Mapping[str, Any]],
    lineage: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Path]:
    """Write new genuine Parquet files and canonical JSON, never overwrite outputs."""
    paths = {
        "feature_observations": output_dir / "feature-observations.parquet",
        "outcome_observations": output_dir / "outcome-observations.parquet",
        "feature_panel": output_dir / "feature-panel.parquet",
        "outcome_panel": output_dir / "outcome-panel.parquet",
        "lineage": output_dir / "lineage.json",
        "report": output_dir / "predictive-audit.json",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("refusing to overwrite an existing common-trajectory replay output")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_sample_parquet(
        sorted(feature_observations, key=lambda row: (int(row["epoch"]), int(row["sample_id"]))),
        paths["feature_observations"],
    )
    write_sample_parquet(
        sorted(outcome_observations, key=lambda row: (int(row["epoch"]), int(row["sample_id"]))),
        paths["outcome_observations"],
    )
    write_sample_parquet(sorted(feature_panel, key=lambda row: int(row["sample_id"])), paths["feature_panel"])
    write_sample_parquet(sorted(outcome_panel, key=lambda row: int(row["sample_id"])), paths["outcome_panel"])
    artifact_hashes = {name: sha256_file(path) for name, path in paths.items() if name not in {"lineage", "report"}}
    paths["report"].write_bytes(canonical_json(report) + b"\n")
    paths["lineage"].write_bytes(
        canonical_json(
            {
                **lineage,
                "output_parquet_sha256": artifact_hashes,
                "predictive_audit_sha256": sha256_file(paths["report"]),
            }
        )
        + b"\n"
    )
    return paths
