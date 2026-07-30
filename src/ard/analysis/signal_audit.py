"""Fail-closed, seed-zero signal audit primitives.

The module deliberately has no W&B dependency.  It consumes only a local run
bundle and explicit feature rows; in particular, it never reconstructs a
historical teacher signal from final sample statistics.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch

from ard.engine.checkpoint import REQUIRED_KEYS


class SignalAuditError(ValueError):
    """Raised when audit lineage, schema, or temporal evidence is incomplete."""


REPLAY_SEED_FORMULA = "train_attack+1000003*checkpoint_global_step+1000003*batch_index"


def replay_protocol(*, batch_size: int, attack_seed_base: int, device_type: str) -> dict[str, Any]:
    """Canonical scientific protocol for random-start historical replay."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise SignalAuditError("replay protocol batch_size must be a positive integer")
    if isinstance(attack_seed_base, bool) or not isinstance(attack_seed_base, int):
        raise SignalAuditError("replay protocol attack_seed_base must be an integer")
    if device_type not in {"cpu", "cuda"}:
        raise SignalAuditError("replay protocol device_type must be cpu or cuda")
    return {
        "batch_size": batch_size,
        "seed_formula": REPLAY_SEED_FORMULA,
        "attack_seed_base": attack_seed_base,
        "generator_per_batch": True,
        "precision": "fp32",
        "device_type": device_type,
        "backend": f"torch-{device_type}",
    }


def canonical_json(value: object) -> bytes:
    """Return the sole JSON representation used for hashes and artifacts."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def logical_dataset_identity(resolved_config: Mapping[str, Any], *, train_expected_count: int) -> dict[str, Any]:
    """Return host-independent scientific dataset/split identity.

    ``root`` and ``download`` only describe storage/transport.  They are
    deliberately excluded.  CIFAR uses the same portable content fingerprints
    and version identifiers as saved-checkpoint evaluation, rather than an
    optional local ``content_sha256`` field.
    """
    dataset = resolved_config.get("dataset")
    seeds = resolved_config.get("seeds")
    training = resolved_config.get("training")
    if not isinstance(dataset, Mapping) or not isinstance(seeds, Mapping) or not isinstance(training, Mapping):
        raise SignalAuditError("resolved config requires dataset, seeds, and training mappings")
    split_seed = seeds.get("split")
    validation_fraction = training.get("validation_fraction")
    if isinstance(split_seed, bool) or not isinstance(split_seed, int):
        raise SignalAuditError("resolved seeds.split must be an integer")
    if isinstance(validation_fraction, bool) or not isinstance(validation_fraction, (int, float)):
        raise SignalAuditError("resolved training.validation_fraction must be numeric")
    if isinstance(train_expected_count, bool) or not isinstance(train_expected_count, int) or train_expected_count < 1:
        raise SignalAuditError("train_expected_count must be a positive integer")
    dataset_identity = {key: value for key, value in dataset.items() if key not in {"root", "download"}}
    portable_cifar = {
        "cifar10": {
            "version": "torchvision-cifar10",
            "content_fingerprint": "c58f30108f718f92721af3b95e74349a",
        },
        "cifar100": {
            "version": "torchvision-cifar100",
            "content_fingerprint": "eb9058c3a382ffc7106e4002c42a8d85",
        },
    }.get(dataset_identity.get("name"))
    if portable_cifar is not None:
        dataset_identity.pop("content_sha256", None)
        dataset_identity.update(portable_cifar)
    return {
        "dataset": dataset_identity,
        "split_seed": split_seed,
        "validation_fraction": float(validation_fraction),
        "train_expected_count": train_expected_count,
    }


def logical_dataset_fingerprint(resolved_config: Mapping[str, Any], *, train_expected_count: int) -> str:
    """SHA-256 of :func:`logical_dataset_identity`, not a dataset-byte hash."""
    return _sha256_mapping(logical_dataset_identity(resolved_config, train_expected_count=train_expected_count))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _analysis_source_hashes() -> dict[str, str]:
    """Hash both report semantics and the CLI that binds its formal inputs."""
    analysis_path = Path(__file__).resolve()
    cli_path = analysis_path.parents[1] / "cli" / "signal_audit.py"
    return {
        "analysis_module": sha256_file(analysis_path),
        "cli_module": sha256_file(cli_path),
    }


def _expected_replay_source_hashes() -> dict[str, str]:
    analysis_path = Path(__file__).resolve().parent / "teacher_risk_replay.py"
    cli_path = Path(__file__).resolve().parents[1] / "cli" / "replay_teacher_risk.py"
    return {"analysis_module": sha256_file(analysis_path), "cli_module": sha256_file(cli_path)}


def _sha256_mapping(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _hex_digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise SignalAuditError(f"{name} must be a lowercase SHA-256")
    return value


def _git_sha(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in {40, 64}
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise SignalAuditError(f"{name} must be an exact lowercase Git SHA")
    return value


@dataclass(frozen=True)
class CheckpointInventory:
    run_id: str
    artifact_name: str
    aliases: tuple[str, ...]
    publication_order: int
    path: str
    sha256: str
    epoch: int
    sample_state_present: bool
    sample_state_count: int
    config_hash: str
    scientific_git_sha: str
    wandb_version: str | None = None

    @property
    def periodic_last(self) -> bool:
        return "last" in self.aliases


def _local_artifact_file(bundle: Path, entry: Mapping[str, Any]) -> Path:
    local_path, original = entry.get("local_path"), entry.get("path")
    if not isinstance(local_path, str) or not isinstance(original, str):
        raise SignalAuditError("model artifact must contain local_path and path")
    root = bundle.resolve()
    candidate = (root / local_path / Path(original).name).resolve()
    if root not in candidate.parents or not candidate.is_file():
        raise SignalAuditError("content-addressed model artifact is missing or escapes its run bundle")
    return candidate


def _checkpoint_sample_state(payload: Mapping[str, Any]) -> tuple[bool, int]:
    state = payload.get("sample_state")
    if not isinstance(state, Mapping):
        return False, 0
    if not state:
        # Stateless methods deliberately serialize an exact empty mapping.
        return False, 0
    records = state.get("records")
    if not isinstance(records, Mapping):
        raise SignalAuditError("checkpoint sample_state.records must be a mapping")
    return True, len(records)


def inventory_run_bundle(manifest_path: Path) -> tuple[CheckpointInventory, ...]:
    """Verify locally copied model artifacts against their immutable manifest.

    Periodic ``last`` artifacts are retained in publication order and are the
    primary source for prospective analysis.  ``best`` artifacts are inventoried
    but cannot silently substitute for a missing historical ``last`` state.
    """
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SignalAuditError("run-bundle manifest is unreadable") from exc
    if not isinstance(manifest, Mapping):
        raise SignalAuditError("run-bundle manifest must be an object")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise SignalAuditError("manifest run_id is missing")
    config_hash = _hex_digest(manifest.get("config_hash"), name="manifest config_hash")
    git = manifest.get("git")
    if not isinstance(git, Mapping):
        raise SignalAuditError("manifest scientific Git identity is missing")
    scientific_git_sha = _git_sha(git.get("sha"), name="manifest scientific Git SHA")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise SignalAuditError("manifest artifacts must be a list")

    inventory: list[CheckpointInventory] = []
    inspected: dict[tuple[str, str], Mapping[str, Any]] = {}
    hashes_by_path: dict[str, str] = {}
    bundle = manifest_path.parent
    for publication_order, raw_entry in enumerate(artifacts):
        if not isinstance(raw_entry, Mapping) or raw_entry.get("type") != "model":
            continue
        name, aliases = raw_entry.get("name"), raw_entry.get("aliases")
        if not isinstance(name, str) or not name or not isinstance(aliases, list) or not aliases:
            raise SignalAuditError("model artifact requires a logical name and non-empty aliases")
        if any(not isinstance(alias, str) for alias in aliases) or len(set(aliases)) != len(aliases):
            raise SignalAuditError("model artifact aliases are invalid")
        expected_hash = _hex_digest(raw_entry.get("sha256"), name="model artifact SHA-256")
        checkpoint_path = _local_artifact_file(bundle, raw_entry)
        path_key = str(checkpoint_path.resolve())
        actual_hash = hashes_by_path.get(path_key)
        if actual_hash is None:
            actual_hash = sha256_file(checkpoint_path)
            hashes_by_path[path_key] = actual_hash
        if actual_hash != expected_hash:
            raise SignalAuditError("content-addressed model artifact hash mismatch")
        cache_key = (path_key, actual_hash)
        payload = inspected.get(cache_key)
        if payload is None:
            try:
                payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            except Exception as exc:  # torch reports several incompatible serialization errors
                raise SignalAuditError("model artifact checkpoint is unreadable") from exc
            if not isinstance(payload, Mapping):
                raise SignalAuditError(
                    "model artifact checkpoint schema does not match the complete checkpoint contract"
                )
            inspected[cache_key] = payload
        if REQUIRED_KEYS.difference(payload):
            raise SignalAuditError("model artifact checkpoint schema does not match the complete checkpoint contract")
        if payload.get("tracker_run_id") != run_id or payload.get("config_hash") != config_hash:
            raise SignalAuditError("checkpoint identity does not match manifest run/config identity")
        epoch = payload.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0 or payload.get("epoch_boundary") != "end":
            raise SignalAuditError("checkpoint epoch is invalid or not an epoch-boundary state")
        present, count = _checkpoint_sample_state(payload)
        inventory.append(
            CheckpointInventory(
                run_id=run_id,
                artifact_name=name,
                aliases=tuple(aliases),
                publication_order=publication_order,
                path=str(checkpoint_path),
                sha256=actual_hash,
                epoch=epoch,
                sample_state_present=present,
                sample_state_count=count,
                config_hash=config_hash,
                scientific_git_sha=scientific_git_sha,
            )
        )
    if not inventory:
        raise SignalAuditError("manifest contains no local model artifacts")
    return tuple(inventory)


def periodic_last_checkpoints(inventory: Sequence[CheckpointInventory]) -> tuple[CheckpointInventory, ...]:
    selected = tuple(item for item in inventory if item.periodic_last)
    if not selected:
        raise SignalAuditError("prospective audit requires a periodic last checkpoint; best-only fallback is forbidden")
    return selected


def select_prospective_checkpoints(
    inventory: Sequence[CheckpointInventory], *, run_id: str, historical_epoch: int
) -> tuple[CheckpointInventory, CheckpointInventory]:
    """Select one declared temporal pair without guessing a historical epoch."""
    if isinstance(historical_epoch, bool) or not isinstance(historical_epoch, int) or historical_epoch < 0:
        raise SignalAuditError("historical_epoch must be a non-negative zero-based checkpoint epoch")
    selected = periodic_last_checkpoints(tuple(item for item in inventory if item.run_id == run_id))
    historical = [item for item in selected if item.epoch == historical_epoch]
    if len(historical) != 1:
        raise SignalAuditError("prospective run requires exactly one explicit historical periodic last checkpoint")
    final_epoch = max(item.epoch for item in selected)
    final = [item for item in selected if item.epoch == final_epoch]
    if len(final) != 1:
        raise SignalAuditError("prospective run requires exactly one final periodic last checkpoint")
    if historical_epoch >= final_epoch:
        raise SignalAuditError("historical_epoch must precede the selected final checkpoint")
    pair = (historical[0], final[0])
    if any(not item.sample_state_present for item in pair):
        raise SignalAuditError("selected prospective checkpoints require SampleStateStore state")
    return pair


def validate_teacher_risk_replay(
    envelope: Mapping[str, Any],
    *,
    historical: CheckpointInventory,
    teacher_checkpoint_sha256: str,
    dataset_fingerprint: str,
    threat_or_attack_identity: Mapping[str, Any],
    dataset_identity: Mapping[str, Any] | None = None,
    expected_replay_protocol: Mapping[str, Any] | None = None,
    expected_git_sha: str | None = None,
    expected_checkpoint_training: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Validate the immutable provenance envelope for historical teacher replay."""
    required = {
        "run_id",
        "historical_epoch",
        "historical_checkpoint_sha256",
        "teacher_checkpoint_sha256",
        "dataset_fingerprint",
        "replay_output_sha256",
        "rows",
    }
    if expected_replay_protocol is not None:
        required |= {
            "dataset_identity",
            "replay_protocol",
            "replay_source_files",
            "replay_source_sha256",
            "git",
            "max_abs_delta",
            "checkpoint_training",
        }
    if not isinstance(envelope, Mapping) or not required.issubset(envelope):
        raise SignalAuditError("teacher-risk replay requires a complete provenance envelope")
    identities = [key for key in ("threat_identity", "attack_identity") if key in envelope]
    if len(identities) != 1 or not isinstance(envelope[identities[0]], Mapping):
        raise SignalAuditError("teacher-risk replay requires exactly one threat_identity or attack_identity")
    if envelope["run_id"] != historical.run_id or envelope["historical_epoch"] != historical.epoch:
        raise SignalAuditError("teacher-risk replay run ID or historical epoch does not match the selected checkpoint")
    if (
        _hex_digest(envelope["historical_checkpoint_sha256"], name="replay historical checkpoint SHA")
        != historical.sha256
    ):
        raise SignalAuditError("teacher-risk replay checkpoint SHA does not match the selected historical checkpoint")
    if (
        _hex_digest(envelope["teacher_checkpoint_sha256"], name="replay teacher checkpoint SHA")
        != teacher_checkpoint_sha256
    ):
        raise SignalAuditError("teacher-risk replay teacher checkpoint SHA does not match lineage")
    if envelope["dataset_fingerprint"] != dataset_fingerprint:
        raise SignalAuditError("teacher-risk replay dataset fingerprint does not match analysis lineage")
    if dataset_identity is not None:
        replay_dataset_identity = envelope.get("dataset_identity")
        if not isinstance(replay_dataset_identity, Mapping) or canonical_json(
            replay_dataset_identity
        ) != canonical_json(dataset_identity):
            raise SignalAuditError("teacher-risk replay portable dataset identity does not match analysis lineage")
    if expected_replay_protocol is not None and canonical_json(envelope.get("replay_protocol")) != canonical_json(
        expected_replay_protocol
    ):
        raise SignalAuditError("teacher-risk replay protocol does not match analysis lineage")
    if expected_replay_protocol is not None:
        source_files = envelope.get("replay_source_files")
        if not isinstance(source_files, Mapping) or set(source_files) != {"analysis_module", "cli_module"}:
            raise SignalAuditError("teacher-risk replay source files must declare analysis and CLI SHA-256 hashes")
        for name, digest in source_files.items():
            _hex_digest(digest, name=f"replay source hash {name}")
        if canonical_json(source_files) != canonical_json(_expected_replay_source_hashes()):
            raise SignalAuditError("teacher-risk replay source hashes do not match the local replay implementation")
        if _hex_digest(envelope.get("replay_source_sha256"), name="replay source combined SHA") != _sha256_mapping(
            source_files
        ):
            raise SignalAuditError("teacher-risk replay source combined SHA does not match source-file hashes")
        git = envelope.get("git")
        if not isinstance(git, Mapping) or set(git) != {"sha", "dirty"}:
            raise SignalAuditError("teacher-risk replay Git provenance is incomplete")
        _git_sha(git.get("sha"), name="replay Git SHA")
        if git.get("dirty") is not False:
            raise SignalAuditError("teacher-risk replay requires clean Git provenance")
        if expected_git_sha is not None and git["sha"] != _git_sha(expected_git_sha, name="expected replay Git SHA"):
            raise SignalAuditError("teacher-risk replay Git SHA does not match the local clean analysis lineage")
        if expected_checkpoint_training is not None and canonical_json(
            envelope.get("checkpoint_training")
        ) != canonical_json(expected_checkpoint_training):
            raise SignalAuditError("teacher-risk replay checkpoint training execution does not match analysis lineage")
        epsilon = _finite(threat_or_attack_identity.get("epsilon_value"), name="replay attack epsilon_value")
        max_abs_delta = _finite(envelope.get("max_abs_delta"), name="replay max_abs_delta")
        if max_abs_delta < 0.0 or max_abs_delta > epsilon + 1e-7:
            raise SignalAuditError("teacher-risk replay max_abs_delta exceeds the configured Linf bound")
    if canonical_json(envelope[identities[0]]) != canonical_json(threat_or_attack_identity):
        raise SignalAuditError("teacher-risk replay threat/attack identity does not match analysis lineage")
    rows = envelope["rows"]
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise SignalAuditError("teacher-risk replay rows must be a list of objects")
    if _hex_digest(envelope["replay_output_sha256"], name="replay output SHA") != _sha256_mapping(rows):
        raise SignalAuditError("teacher-risk replay output SHA does not match its rows")
    if any("teacher_risk" not in row or row["teacher_risk"] is None for row in rows):
        raise SignalAuditError("teacher-risk replay rows require teacher_risk")
    return tuple(rows)


def _md5_base64(path: Path) -> str:
    import base64

    digest = hashlib.md5(usedforsecurity=False)  # noqa: S324 - W&B manifest compatibility checksum, not security.
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


def associate_wandb_versions(
    inventory: Sequence[CheckpointInventory], wands_inventory: Sequence[Mapping[str, Any]]
) -> tuple[CheckpointInventory, ...]:
    """Validate read-only W&B periodic-last records against local immutable bytes.

    W&B records preserve API order.  Only ``last`` is associated this way:
    repeated ``best`` aliases may be content-deduplicated remotely, so local
    publication order is never used to invent their W&B versions.
    """
    remote: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in wands_inventory:
        required = {"run_id", "artifact_name", "version", "file_name", "file_md5", "size"}
        if not isinstance(row, Mapping) or set(row) != required:
            raise SignalAuditError("W&B inventory rows require the exact read-only artifact schema")
        run_id = row["run_id"]
        name = row["artifact_name"]
        version = row["version"]
        file_name = row["file_name"]
        md5 = row["file_md5"]
        size = row["size"]
        if not all(isinstance(value, str) and value for value in (run_id, name, version, file_name, md5)):
            raise SignalAuditError("W&B artifact identity fields must be non-empty strings")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise SignalAuditError("W&B artifact size must be a non-negative integer")
        remote[(run_id, name, file_name)].append(row)

    def version_number(row: Mapping[str, Any]) -> int:
        match = re.fullmatch(r"v([0-9]+)", str(row["version"]))
        if match is None:
            raise SignalAuditError("W&B artifact version must use the explicit vN format")
        return int(match.group(1))

    associated: list[CheckpointInventory] = []
    local_last: dict[tuple[str, str, str], list[CheckpointInventory]] = defaultdict(list)
    for item in inventory:
        if item.periodic_last:
            local_last[(item.run_id, item.artifact_name, Path(item.path).name)].append(item)
    for key, local_items in local_last.items():
        records = remote.get(key, [])
        if len(records) != len(local_items):
            raise SignalAuditError("W&B periodic last publication count does not match the local manifest")
        ordered_records = sorted(records, key=version_number)
        if len({version_number(record) for record in ordered_records}) != len(ordered_records):
            raise SignalAuditError("W&B periodic last versions must be unique")
        for item, record in zip(
            sorted(local_items, key=lambda value: value.publication_order), ordered_records, strict=True
        ):
            local = Path(item.path)
            if record["size"] != local.stat().st_size or record["file_md5"] != _md5_base64(local):
                raise SignalAuditError("W&B periodic last artifact size or MD5 does not match local content")
            associated.append(replace(item, wandb_version=str(record["version"])))
    local_keys = {(item.run_id, item.artifact_name, Path(item.path).name) for item in inventory}
    if any(key not in local_keys for key in remote):
        raise SignalAuditError("W&B inventory contains an artifact absent from the local manifest")
    versions = {(item.run_id, item.artifact_name, item.publication_order): item for item in associated}
    return tuple(versions.get((item.run_id, item.artifact_name, item.publication_order), item) for item in inventory)


@dataclass(frozen=True, order=True)
class SampleKey:
    namespace: str
    sample_id: int

    def __post_init__(self) -> None:
        if self.namespace not in {"train", "test"}:
            raise SignalAuditError("sample namespace must be explicit train or test")
        if isinstance(self.sample_id, bool) or not isinstance(self.sample_id, int) or self.sample_id < 0:
            raise SignalAuditError("sample ID must be a non-negative integer")


def namespaced_samples(
    rows: Iterable[Mapping[str, Any]], *, namespace: str, expected_count: int
) -> dict[SampleKey, Mapping[str, Any]]:
    """Validate stable IDs and the exact configured split cardinality."""
    if expected_count < 1:
        raise SignalAuditError("expected sample count must be positive")
    indexed: dict[SampleKey, Mapping[str, Any]] = {}
    for row in rows:
        if row.get("namespace") != namespace:
            raise SignalAuditError("row namespace does not match the selected dataset split")
        key = SampleKey(namespace, row.get("sample_id"))
        if key in indexed:
            raise SignalAuditError("sample IDs must be unique within a namespace")
        indexed[key] = row
    if len(indexed) != expected_count:
        raise SignalAuditError(f"{namespace} sample count is {len(indexed)}, expected {expected_count}")
    return indexed


def join_samples(
    left: Mapping[SampleKey, Any], right: Mapping[SampleKey, Any]
) -> tuple[tuple[SampleKey, Any, Any], ...]:
    namespaces = {key.namespace for key in left} | {key.namespace for key in right}
    if len(namespaces) != 1:
        raise SignalAuditError("joins across train/test sample namespaces are forbidden")
    if set(left) != set(right):
        raise SignalAuditError("sample joins require exactly matching stable-ID sets")
    return tuple((key, left[key], right[key]) for key in sorted(left))


def validate_sample_partitions(
    train_rows: Iterable[Mapping[str, Any]],
    test_rows: Iterable[Mapping[str, Any]],
    *,
    train_expected_count: int = 45000,
    test_expected_count: int = 10000,
) -> tuple[dict[SampleKey, Mapping[str, Any]], dict[SampleKey, Mapping[str, Any]]]:
    """Validate the separately namespaced training and official-test populations."""
    return (
        namespaced_samples(train_rows, namespace="train", expected_count=train_expected_count),
        namespaced_samples(test_rows, namespace="test", expected_count=test_expected_count),
    )


def deterministic_hash_split(
    rows: Sequence[Mapping[str, Any]], *, seed: int, held_out_fraction: float = 0.2
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Class-stratified deterministic split, independent of input order."""
    if not 0.0 < held_out_fraction < 1.0:
        raise SignalAuditError("held_out_fraction must be in (0, 1)")
    by_class: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        SampleKey(str(row.get("namespace")), row.get("sample_id"))
        label = row.get("class_id")
        if isinstance(label, bool) or not isinstance(label, int):
            raise SignalAuditError("analysis rows require integer class_id")
        by_class[label].append(row)
    train, held_out = [], []
    for label, items in sorted(by_class.items()):
        ranked = sorted(
            items,
            key=lambda row: hashlib.sha256(f"{seed}:{label}:{row['namespace']}:{row['sample_id']}".encode()).digest(),
        )
        hold_count = max(1, min(len(ranked) - 1, round(len(ranked) * held_out_fraction))) if len(ranked) > 1 else 0
        held_out.extend(int(row["sample_id"]) for row in ranked[:hold_count])
        train.extend(int(row["sample_id"]) for row in ranked[hold_count:])
    return tuple(sorted(train)), tuple(sorted(held_out))


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise SignalAuditError(f"{name} must be finite")
    return float(value)


def _binary(value: Any, *, name: str) -> int:
    if value not in {0, 1, False, True}:
        raise SignalAuditError(f"{name} must be binary")
    return int(value)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, value))))


def _fit_logistic(
    features: Sequence[Sequence[float]], targets: Sequence[int]
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    if not features or len(features) != len(targets) or len(set(targets)) != 2:
        raise SignalAuditError("prospective fit requires non-empty observations from both outcome classes")
    width = len(features[0])
    if any(len(row) != width for row in features):
        raise SignalAuditError("feature rows have inconsistent width")
    means = tuple(sum(row[col] for row in features) / len(features) for col in range(width))
    scales = tuple(
        max(1e-12, math.sqrt(sum((row[col] - means[col]) ** 2 for row in features) / len(features)))
        for col in range(width)
    )
    normalized = [[(value - means[col]) / scales[col] for col, value in enumerate(row)] for row in features]
    weights = [0.0] * (width + 1)
    for _ in range(400):
        gradient = [0.0] * len(weights)
        for row, target in zip(normalized, targets, strict=True):
            logit = weights[0] + sum(weight * value for weight, value in zip(weights[1:], row, strict=True))
            error = _sigmoid(logit) - target
            gradient[0] += error
            for col, value in enumerate(row):
                gradient[col + 1] += error * value
        scale = 1.0 / len(normalized)
        for col in range(len(weights)):
            weights[col] -= 0.15 * gradient[col] * scale + (0.001 * weights[col] if col else 0.0)
    return tuple(weights), means, scales


def _predict_logistic(
    fit: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]], features: Sequence[Sequence[float]]
) -> list[float]:
    weights, means, scales = fit
    return [
        _sigmoid(
            weights[0]
            + sum(
                weight * ((value - means[col]) / scales[col])
                for col, (weight, value) in enumerate(zip(weights[1:], row, strict=True))
            )
        )
        for row in features
    ]


def binary_metrics(targets: Sequence[int], scores: Sequence[float]) -> dict[str, float]:
    """Dependency-free AUROC, average precision, prevalence, and log loss."""
    if len(targets) != len(scores) or not targets:
        raise SignalAuditError("metric targets and scores must be equally non-empty")
    labels = [_binary(value, name="outcome") for value in targets]
    probabilities = [_finite(value, name="score") for value in scores]
    if any(value < 0.0 or value > 1.0 for value in probabilities):
        raise SignalAuditError("scores must be probabilities in [0, 1]")
    positives = sum(labels)
    if positives in {0, len(labels)}:
        raise SignalAuditError("AUROC/AUPRC require both outcome classes")
    ranked = sorted(zip(probabilities, labels), key=lambda pair: pair[0])
    rank_sum, index = 0.0, 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += average_rank * sum(label for _, label in ranked[index:end])
        index = end
    negatives = len(labels) - positives
    auroc = (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    ordered = sorted(zip(probabilities, labels), key=lambda pair: pair[0], reverse=True)
    hits = index = 0
    auprc = 0.0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        group_positives = sum(label for _, label in ordered[index:end])
        hits += group_positives
        # AP integrates one threshold group at a time, so tied scores do not
        # acquire a spurious order from input row order.
        auprc += group_positives * (hits / end) / positives
        index = end
    log_loss = -sum(
        target * math.log(max(score, 1e-15)) + (1 - target) * math.log(max(1.0 - score, 1e-15))
        for target, score in zip(labels, probabilities, strict=True)
    ) / len(labels)
    return {"auroc": auroc, "auprc": auprc, "prevalence": positives / len(labels), "log_loss": log_loss}


def _bootstrap_indices(rows: Sequence[Mapping[str, Any]], *, seed: int, replicate: int, cluster: bool) -> list[int]:
    if cluster:
        classes_by_sample: dict[int, set[int]] = defaultdict(set)
        for row in rows:
            classes_by_sample[int(row["sample_id"])].add(int(row["class_id"]))
        if any(len(classes) != 1 for classes in classes_by_sample.values()):
            raise SignalAuditError("a repeated sample cluster cannot span classes")
    groups: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[int(row["class_id"])].append(index)
    generator = random.Random(f"{seed}:{replicate}:{'cluster' if cluster else 'sample'}")
    selected: list[int] = []
    for _, indices in sorted(groups.items()):
        units: dict[int, list[int]] = defaultdict(list)
        for index in indices:
            units[int(rows[index]["sample_id"]) if cluster else index].append(index)
        choices = sorted(units)
        for _ in range(len(choices)):
            selected.extend(units[generator.choice(choices)])
    return selected


def bootstrap_metric_delta(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline: Sequence[float],
    candidate: Sequence[float],
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    if replicates < 1:
        raise SignalAuditError("bootstrap replicates must be positive")
    if len(rows) != len(baseline) or len(rows) != len(candidate):
        raise SignalAuditError("bootstrap rows/scores must align")
    cluster = len({int(row["sample_id"]) for row in rows}) != len(rows)
    deltas = []
    for replicate in range(replicates):
        selected = _bootstrap_indices(rows, seed=seed, replicate=replicate, cluster=cluster)
        targets = [int(rows[index]["outcome"]) for index in selected]
        try:
            candidate_auroc = binary_metrics(targets, [candidate[index] for index in selected])["auroc"]
            baseline_auroc = binary_metrics(targets, [baseline[index] for index in selected])["auroc"]
            deltas.append(candidate_auroc - baseline_auroc)
        except SignalAuditError:
            continue
    if not deltas:
        raise SignalAuditError("bootstrap produced no class-complete replicate")
    deltas.sort()
    return {
        "clustered_by_sample_id": cluster,
        "replicates": len(deltas),
        "lower": deltas[max(0, math.floor(0.025 * (len(deltas) - 1)))],
        "upper": deltas[min(len(deltas) - 1, math.ceil(0.975 * (len(deltas) - 1)))],
    }


def bootstrap_binary_metric_intervals(
    rows: Sequence[Mapping[str, Any]], *, scores: Sequence[float], seed: int, replicates: int
) -> dict[str, Any]:
    """Class-stratified, stable-sample-cluster bootstrap 95% intervals."""
    if replicates < 1 or len(rows) != len(scores):
        raise SignalAuditError("bootstrap metric rows/scores must align and use positive replicates")
    samples: dict[str, list[float]] = {"auroc": [], "auprc": [], "log_loss": [], "prevalence": []}
    for replicate in range(replicates):
        selected = _bootstrap_indices(rows, seed=seed, replicate=replicate, cluster=True)
        targets = [int(rows[index]["outcome"]) for index in selected]
        try:
            metrics = binary_metrics(targets, [scores[index] for index in selected])
        except SignalAuditError:
            continue
        for name, values in samples.items():
            values.append(metrics[name])
    if not samples["auroc"]:
        raise SignalAuditError("bootstrap produced no class-complete replicate")
    bounds: dict[str, Any] = {}
    for name, values in samples.items():
        values.sort()
        bounds[name] = {
            "lower": values[max(0, math.floor(0.025 * (len(values) - 1)))],
            "upper": values[min(len(values) - 1, math.ceil(0.975 * (len(values) - 1)))],
        }
    return {"clustered_by_sample_id": True, "replicates": len(samples["auroc"]), "metrics": bounds}


def _state_records(inventory: CheckpointInventory) -> Mapping[str, Mapping[str, Any]]:
    payload = torch.load(Path(inventory.path), map_location="cpu", weights_only=False)
    state = payload["sample_state"]
    records = state["records"]
    if not isinstance(records, Mapping):  # defensive after inventory validation
        raise SignalAuditError("checkpoint SampleStateStore records are invalid")
    return records


def _state_outcomes(
    historical: CheckpointInventory, final: CheckpointInventory, rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    prior, current = _state_records(historical), _state_records(final)
    output = []
    for row in rows:
        sample_id = str(SampleKey(str(row.get("namespace")), row.get("sample_id")).sample_id)
        if sample_id not in prior or sample_id not in current:
            raise SignalAuditError("prospective row is missing historical or final SampleStateStore state")
        previous, last = prior[sample_id], current[sample_id]
        for record in (previous, last):
            if not isinstance(record, Mapping):
                raise SignalAuditError("SampleStateStore record is invalid")
        forgetting = int(last["forgetting_count"]) - int(previous["forgetting_count"])
        seen, correct = int(last["seen"]), int(last["robust_correct_count"])
        final_correct = last.get("previous_robust_correct")
        if forgetting < 0 or seen < 1 or correct < 0 or correct > seen or not isinstance(final_correct, bool):
            raise SignalAuditError("SampleStateStore prospective counters are invalid")
        historical_margin = _finite(previous.get("margin_ema"), name="historical SampleStateStore margin_ema")
        if not -1.0 <= historical_margin <= 1.0:
            raise SignalAuditError("historical SampleStateStore margin_ema is outside [-1, 1]")
        if "student_risk" in row:
            prepared_student_risk = _finite(row["student_risk"], name="prepared student_risk")
            if not 0.0 <= prepared_student_risk <= 1.0:
                raise SignalAuditError("prepared student_risk must be in [0, 1]")
        result = dict(row)
        # Prepared student risk may be a final Parquet field.  It is never a
        # prospective feature; remove it after validating its declared range.
        result.pop("student_risk", None)
        result["historical_student_risk"] = (1.0 - historical_margin) / 2.0
        result["subsequent_forgetting_increment"] = forgetting
        result["final_robust_error"] = int(not final_correct)
        result["final_robust_correct_frequency"] = correct / seen
        output.append(result)
    return output


def load_final_sample_stats(
    path: Path,
    *,
    expected_count: int = 45000,
    num_classes: int = 10,
    stored_risk_kind: str = "joint",
    tolerance: float = 1e-6,
) -> list[dict[str, Any]]:
    """Build final association rows from genuine training sample-stat Parquet.

    The stored data itself has no split marker, so this loader assigns the
    explicit ``train`` namespace at the only permitted boundary.
    """
    if num_classes < 2 or tolerance < 0.0 or stored_risk_kind not in {"joint", "student"}:
        raise SignalAuditError("invalid final Parquet loader configuration")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - declared tracking extra
        raise SignalAuditError("final sample-stat Parquet requires pyarrow") from exc
    table = pq.read_table(path)
    required = {
        "sample_id",
        "true_label",
        "teacher_prediction",
        "teacher_entropy",
        "student_unlearnability",
        "joint_risk",
        "robust_correct",
    }
    if not required.issubset(table.column_names):
        raise SignalAuditError("final sample-stat Parquet is missing required signal columns")
    rows = table.select(sorted(required)).to_pylist()
    output: list[dict[str, Any]] = []
    log_classes = math.log(num_classes)
    for row in rows:
        sample_id = row["sample_id"]
        class_id = row["true_label"]
        if (
            isinstance(sample_id, bool)
            or not isinstance(sample_id, int)
            or isinstance(class_id, bool)
            or not isinstance(class_id, int)
        ):
            raise SignalAuditError("final sample-stat IDs and labels must be integers")
        teacher_prediction = row["teacher_prediction"]
        if (
            not 0 <= class_id < num_classes
            or isinstance(teacher_prediction, bool)
            or not isinstance(teacher_prediction, int)
            or not 0 <= teacher_prediction < num_classes
            or not isinstance(row["robust_correct"], bool)
        ):
            raise SignalAuditError("final sample-stat label/correctness is invalid")
        entropy = _finite(row["teacher_entropy"], name="teacher_entropy")
        student = _finite(row["student_unlearnability"], name="student_unlearnability")
        stored = _finite(row["joint_risk"], name="joint_risk")
        teacher = 1.0 - entropy / log_classes
        if not -tolerance <= teacher <= 1.0 + tolerance or not -tolerance <= student <= 1.0 + tolerance:
            raise SignalAuditError("final sample-stat risk is outside the documented [0, 1] range")
        teacher, student = min(1.0, max(0.0, teacher)), min(1.0, max(0.0, student))
        expected = student * teacher if stored_risk_kind == "joint" else student
        if not math.isclose(stored, expected, rel_tol=0.0, abs_tol=tolerance):
            raise SignalAuditError("stored joint_risk does not match the documented signal formula")
        rho = 0.5 * stored
        if not -tolerance <= rho <= 0.5 + tolerance:
            raise SignalAuditError("implied target rho is outside [0, 0.5]")
        output.append(
            {
                "namespace": "train",
                "sample_id": sample_id,
                "class_id": class_id,
                "teacher_entropy": entropy,
                "teacher_prediction": teacher_prediction,
                "teacher_correct": teacher_prediction == class_id,
                "teacher_risk": teacher,
                "student_risk": student,
                "stored_applied_risk": stored,
                "implied_rho": rho,
                "final_robust_error": int(not row["robust_correct"]),
            }
        )
    namespaced_samples(output, namespace="train", expected_count=expected_count)
    return output


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise SignalAuditError("quantiles require non-empty values and a probability in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    return {
        f"p{int(probability * 100):02d}": _quantile(values, probability)
        for probability in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    if not values:
        raise SignalAuditError("ranks require values")
    ranked = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ranked):
        end = start + 1
        while end < len(ranked) and ranked[end][1] == ranked[start][1]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for index, _ in ranked[start:end]:
            ranks[index] = average_rank
        start = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or not left:
        raise SignalAuditError("Spearman correlation requires equal non-empty vectors")
    left_rank, right_rank = _average_ranks(left), _average_ranks(right)
    left_mean, right_mean = sum(left_rank) / len(left_rank), sum(right_rank) / len(right_rank)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left_rank, right_rank, strict=True))
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left_rank))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right_rank))
    return None if left_scale == 0.0 or right_scale == 0.0 else numerator / (left_scale * right_scale)


def _risk_summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {"count": len(values), "mean": sum(values) / len(values), "quantiles": _quantiles(values)}


def _risk_deciles(rows: Sequence[Mapping[str, Any]], *, score_key: str) -> list[dict[str, Any]]:
    """Return exact deterministic rank deciles, tie-broken only by stable sample ID."""
    ranked = sorted(
        enumerate(rows),
        key=lambda item: (_finite(item[1][score_key], name=score_key), int(item[1]["sample_id"])),
    )
    bins: list[list[Mapping[str, Any]]] = [[] for _ in range(10)]
    for rank, (_, row) in enumerate(ranked):
        bins[min(9, (10 * rank) // len(ranked))].append(row)
    result = []
    for index, members in enumerate(bins, start=1):
        if not members:
            result.append(
                {"decile": index, "count": 0, "subsequent_forgetting_prevalence": None, "final_error_prevalence": None}
            )
            continue
        result.append(
            {
                "decile": index,
                "count": len(members),
                "subsequent_forgetting_prevalence": sum(
                    int(float(row["subsequent_forgetting_increment"]) > 0.0) for row in members
                )
                / len(members),
                "final_error_prevalence": sum(int(row["final_robust_error"]) for row in members) / len(members),
            }
        )
    return result


def artifact_only_temporal_diagnostics(
    rows: Sequence[Mapping[str, Any]], *, historical: CheckpointInventory | None, final: CheckpointInventory | None
) -> dict[str, Any]:
    """Report temporal student-state evidence without manufacturing teacher history."""
    if historical is None or final is None:
        return {"status": "insufficient_data", "reason": "missing selected temporal checkpoints"}
    state_rows = _state_outcomes(historical, final, rows)
    scores = [_finite(row["historical_student_risk"], name="historical_student_risk") for row in state_rows]
    outcomes = {}
    for name in ("subsequent_forgetting_increment", "final_robust_error"):
        targets = [int(float(row[name]) > 0.0) for row in state_rows]
        try:
            outcomes[name] = binary_metrics(targets, scores)
        except SignalAuditError:
            outcomes[name] = {"status": "insufficient_class_variation", "prevalence": sum(targets) / len(targets)}
    return {
        "status": "available",
        "historical_epoch": historical.epoch,
        "final_epoch": final.epoch,
        "outcomes": outcomes,
        "historical_student_risk_deciles": _risk_deciles(state_rows, score_key="historical_student_risk"),
    }


def prospective_prediction(
    rows: Sequence[Mapping[str, Any]],
    *,
    historical: CheckpointInventory | None,
    final: CheckpointInventory | None,
    split_seed: int,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    """Evaluate temporal predictions using replayed teacher and state-derived student risk.

    ``teacher_risk`` is an explicit historical replay input.  Student risk is
    derived only from the selected historical SampleStateStore margin and never
    from a prepared row (which may contain final-state diagnostics).
    """
    if historical is None or final is None:
        return {
            "analysis_type": "prospective_prediction",
            "decision": "insufficient_data",
            "reason": "missing periodic or final checkpoint",
        }
    # Presence is intentionally checked before state joins: final parquet risk is
    # not evidence of an available historical teacher-risk measurement.
    if any("teacher_risk" not in row or row["teacher_risk"] is None for row in rows):
        return {
            "analysis_type": "prospective_prediction",
            "decision": "insufficient_data",
            "reason": "periodic_teacher_risk_absent",
        }
    temporal_rows = _state_outcomes(historical, final, rows)
    train_ids, test_ids = deterministic_hash_split(temporal_rows, seed=split_seed)
    test_id_set = set(test_ids)
    train = [row for row in temporal_rows if int(row["sample_id"]) in set(train_ids)]
    held_out = [row for row in temporal_rows if int(row["sample_id"]) in test_id_set]
    if not train or not held_out:
        raise SignalAuditError("deterministic split produced an empty partition")
    outputs: dict[str, Any] = {"subsequent_forgetting_increment": {}, "final_robust_error": {}}
    for outcome_name, destination in outputs.items():
        prepared = []
        for row in temporal_rows:
            teacher = _finite(row["teacher_risk"], name="teacher_risk")
            student = _finite(row["historical_student_risk"], name="historical_student_risk")
            if not 0.0 <= teacher <= 1.0 or not 0.0 <= student <= 1.0:
                raise SignalAuditError("teacher/student risks must be in [0, 1]")
            outcome = int(float(row[outcome_name]) > 0.0)
            prepared.append({**row, "outcome": outcome, "teacher": teacher, "student": student})
        train_prepared = [row for row in prepared if int(row["sample_id"]) in set(train_ids)]
        test_prepared = [row for row in prepared if int(row["sample_id"]) in test_id_set]
        feature_sets = {
            "teacher_only": lambda item: [item["teacher"]],
            "student_only": lambda item: [item["student"]],
            "main_effects": lambda item: [item["teacher"], item["student"]],
            "main_effects_plus_product": lambda item: [
                item["teacher"],
                item["student"],
                item["teacher"] * item["student"],
            ],
        }
        scores: dict[str, list[float]] = {}
        for name, builder in feature_sets.items():
            fit = _fit_logistic([builder(row) for row in train_prepared], [row["outcome"] for row in train_prepared])
            scores[name] = _predict_logistic(fit, [builder(row) for row in test_prepared])
        destination["models"] = {}
        for name, values in scores.items():
            destination["models"][name] = {
                **binary_metrics([row["outcome"] for row in test_prepared], values),
                "bootstrap_95": bootstrap_binary_metric_intervals(
                    test_prepared,
                    scores=values,
                    seed=bootstrap_seed,
                    replicates=bootstrap_replicates,
                ),
            }
        interval = bootstrap_metric_delta(
            test_prepared,
            baseline=scores["teacher_only"],
            candidate=scores["main_effects_plus_product"],
            seed=bootstrap_seed,
            replicates=bootstrap_replicates,
        )
        baseline, augmented = destination["models"]["teacher_only"], destination["models"]["main_effects_plus_product"]
        delta = augmented["auroc"] - baseline["auroc"]
        if delta >= 0.02 and interval["lower"] > 0.0 and augmented["log_loss"] < baseline["log_loss"]:
            decision = "go"
        elif interval["upper"] < 0.01 or augmented["log_loss"] >= baseline["log_loss"]:
            decision = "no_go"
        else:
            decision = "inconclusive"
        destination["teacher_to_augmented_auroc"] = {"delta": delta, "bootstrap": interval, "decision": decision}
    # Forgetting is the preregistered primary target. Final error is retained as
    # a separate prospective outcome and cannot overwrite the decision.
    return {
        "analysis_type": "prospective_prediction",
        "decision": outputs["subsequent_forgetting_increment"]["teacher_to_augmented_auroc"]["decision"],
        "historical_epoch": historical.epoch,
        "final_epoch": final.epoch,
        "train_sample_ids": train_ids,
        "held_out_sample_ids": test_ids,
        "outcomes": outputs,
    }


def final_state_association(rows: Sequence[Mapping[str, Any]], *, rho_zero_threshold: float = 1e-6) -> dict[str, Any]:
    """Describe final-state association only; this never makes a Go/No-Go decision."""
    if not rows:
        raise SignalAuditError("final-state association rows are required")
    if not 0.0 <= rho_zero_threshold < 0.25:
        raise SignalAuditError("rho_zero_threshold must be in [0, 0.25)")
    targets = [_binary(row["final_robust_error"], name="final_robust_error") for row in rows]
    teacher = [_finite(row["teacher_risk"], name="teacher_risk") for row in rows]
    student = [_finite(row["student_risk"], name="student_risk") for row in rows]
    if any(not 0.0 <= value <= 1.0 for value in (*teacher, *student)):
        raise SignalAuditError("final-state teacher/student risks must be in [0, 1]")
    product = [teacher_value * student_value for teacher_value, student_value in zip(teacher, student, strict=True)]
    applied = [
        _finite(row.get("stored_applied_risk", value), name="stored_applied_risk")
        for row, value in zip(rows, product, strict=True)
    ]
    rho = [
        _finite(row.get("implied_rho", 0.5 * value), name="implied_rho")
        for row, value in zip(rows, applied, strict=True)
    ]
    if any(not 0.0 <= value <= 0.5 for value in rho):
        raise SignalAuditError("final-state implied rho must be in [0, 0.5]")
    teacher_correct = []
    entropy = []
    for row in rows:
        correct = row.get("teacher_correct")
        if not isinstance(correct, bool):
            raise SignalAuditError("final-state rows require derived teacher_correct")
        teacher_correct.append(correct)
        entropy.append(_finite(row["teacher_entropy"], name="teacher_entropy"))
    metrics = {
        "teacher_risk": binary_metrics(targets, teacher),
        "student_risk": binary_metrics(targets, student),
        "teacher_student_product": binary_metrics(targets, product),
    }
    top_count = max(1, math.ceil(len(rows) / 10))
    top_indices = sorted(range(len(rows)), key=lambda index: (-teacher[index], int(rows[index]["sample_id"])))[
        :top_count
    ]
    group_summaries = {}
    for label, expected in (("teacher_correct", True), ("teacher_wrong", False)):
        indices = [index for index, correct in enumerate(teacher_correct) if correct is expected]
        group_summaries[label] = {
            "count": len(indices),
            "teacher_risk": _risk_summary([teacher[index] for index in indices]),
            "student_risk": _risk_summary([student[index] for index in indices]),
            "applied_risk": _risk_summary([applied[index] for index in indices]),
            "teacher_entropy": _risk_summary([entropy[index] for index in indices]),
        }
    rho_max = 0.5
    near_zero = [value <= rho_zero_threshold for value in rho]
    near_max = [value >= rho_max - rho_zero_threshold for value in rho]
    return {
        "analysis_type": "final_state_association",
        "contemporaneous_only": True,
        "exploratory_only": True,
        "metrics": metrics,
        "spearman_teacher_student_risk": {"value": _spearman(teacher, student), "tie_aware": True},
        "risk_quantiles": {
            "teacher_risk": _quantiles(teacher),
            "student_risk": _quantiles(student),
            "applied_risk": _quantiles(applied),
            "implied_rho": _quantiles(rho),
        },
        "rho_distribution": {
            "rho_max": rho_max,
            "near_zero_threshold": rho_zero_threshold,
            "near_zero_fraction": sum(near_zero) / len(rho),
            "intermediate_fraction": sum(
                not zero and not maximum for zero, maximum in zip(near_zero, near_max, strict=True)
            )
            / len(rho),
            "near_rho_max_fraction": sum(near_max) / len(rho),
        },
        "teacher_correctness": {
            **group_summaries,
            "top_teacher_risk_decile_count": top_count,
            "teacher_wrong_fraction_in_top_teacher_risk_decile": sum(
                not teacher_correct[index] for index in top_indices
            )
            / top_count,
        },
    }


def audit_report(
    *,
    config: Mapping[str, Any],
    inventories: Sequence[CheckpointInventory],
    final_rows: Sequence[Mapping[str, Any]],
    input_hashes: Mapping[str, str],
    teacher_risk_replay: Mapping[str, Any] | None = None,
    lineage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose byte-stable, structurally distinct audit results from explicit inputs."""
    for name, digest in input_hashes.items():
        _hex_digest(digest, name=f"input hash {name}")
    train_count = int(config.get("train_expected_count", 45000))
    final_index = namespaced_samples(final_rows, namespace="train", expected_count=train_count)
    prospective_run_id = config.get("prospective_run_id")
    if not isinstance(prospective_run_id, str) or not prospective_run_id:
        raise SignalAuditError("audit config requires prospective_run_id")
    if "historical_epoch" not in config:
        raise SignalAuditError("audit config requires an explicit historical_epoch")
    historical, final = select_prospective_checkpoints(
        inventories, run_id=prospective_run_id, historical_epoch=config["historical_epoch"]
    )
    if historical.sample_state_count != train_count or final.sample_state_count != train_count:
        raise SignalAuditError("selected SampleStateStore count does not match the configured train population")
    if historical.wandb_version is None or final.wandb_version is None:
        prospective = {
            "analysis_type": "prospective_prediction",
            "decision": "insufficient_data",
            "reason": "selected_checkpoint_wandb_version_absent",
        }
    elif teacher_risk_replay is None or lineage is None:
        prospective = {
            "analysis_type": "prospective_prediction",
            "decision": "insufficient_data",
            "reason": "teacher_risk_replay_provenance_absent",
        }
    else:
        teacher_sha = _hex_digest(lineage.get("teacher_checkpoint_sha256"), name="lineage teacher checkpoint SHA")
        dataset_fingerprint = lineage.get("dataset_fingerprint")
        identity = lineage.get("threat_or_attack_identity")
        if not isinstance(dataset_fingerprint, str) or not dataset_fingerprint or not isinstance(identity, Mapping):
            raise SignalAuditError("analysis lineage requires dataset fingerprint and threat/attack identity")
        prospective_rows = validate_teacher_risk_replay(
            teacher_risk_replay,
            historical=historical,
            teacher_checkpoint_sha256=teacher_sha,
            dataset_fingerprint=dataset_fingerprint,
            threat_or_attack_identity=identity,
            dataset_identity=lineage.get("dataset_identity")
            if isinstance(lineage.get("dataset_identity"), Mapping)
            else None,
            expected_replay_protocol=lineage.get("replay_protocol")
            if isinstance(lineage.get("replay_protocol"), Mapping)
            else None,
            expected_git_sha=lineage.get("replay_git_sha") if isinstance(lineage.get("replay_git_sha"), str) else None,
            expected_checkpoint_training=lineage.get("checkpoint_training")
            if isinstance(lineage.get("checkpoint_training"), Mapping)
            else None,
        )
        prospective_index = namespaced_samples(prospective_rows, namespace="train", expected_count=train_count)
        for key, final_row, replay_row in join_samples(final_index, prospective_index):
            if final_row.get("class_id") != replay_row.get("class_id"):
                raise SignalAuditError(f"final/replay class_id mismatch for sample {key.sample_id}")
        prospective = prospective_prediction(
            list(prospective_rows),
            historical=historical,
            final=final,
            split_seed=int(config.get("split_seed", 0)),
            bootstrap_seed=int(config.get("bootstrap_seed", 0)),
            bootstrap_replicates=int(config.get("bootstrap_replicates", 200)),
        )
    source_hashes = _analysis_source_hashes()
    return {
        "schema_version": 1,
        "analysis_source_sha256": _sha256_mapping(source_hashes),
        "analysis_source_files": source_hashes,
        "config_hash": _sha256_mapping(config),
        "input_hashes": dict(sorted(input_hashes.items())),
        "checkpoint_inventory": [asdict(item) for item in inventories],
        "final_state_association": final_state_association(
            list(final_rows), rho_zero_threshold=float(config.get("rho_zero_threshold", 1e-6))
        ),
        "artifact_only_temporal_diagnostics": artifact_only_temporal_diagnostics(
            list(final_rows), historical=historical, final=final
        ),
        "prospective_prediction": prospective,
    }


def write_audit_report(path: Path, report: Mapping[str, Any]) -> Path:
    """Write canonical JSON whose bytes depend only on resolved inputs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(report) + b"\n")
    return path
