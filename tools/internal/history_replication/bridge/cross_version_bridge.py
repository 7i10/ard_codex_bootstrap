"""Fail-closed, bounded evidence for the observation-runtime migration.

This module is deliberately not a training launcher.  ``emit`` makes one
small synthetic CUDA RSLAD checkpoint in an isolated interpreter; ``compare``
accepts only two such emissions and proves that the migration changed neither
the optimization checkpoint nor the format-v3 sample-state primitives.  It is
used once to decide whether an already-running legacy observed trajectory can
be bridged into the observation-profile runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import random
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from ard.attacks import LinfPGD
from ard.config.schema import AttackConfig, ModelConfig
from ard.data import (
    EpochShuffleSampler,
    IndexedDataset,
    SyntheticCIFAR,
    collate_indexed,
    stratified_train_validation_split,
)
from ard.engine.trainer import Trainer
from ard.models import build_student
from ard.objectives import RSLADObjective
from ard.policies import RSLADBaselinePolicy
from ard.state import SampleStateStore

EMISSION_SCHEMA_VERSION = 1
_ALLOWED_TELEMETRY = frozenset({"teacher_clean_forward_calls", "teacher_adversarial_forward_calls"})
_IDENTITY_KEYS = frozenset({"config_hash", "tracker_run_id"})
_FIXTURE = "synthetic-cifar8-rslad-pgd1-random-start-epochs2"
_FIXTURE_SAMPLE_IDS = frozenset({"0", "1", "2", "6", "7"})
_RUNTIME_SOURCE_PATHS = (
    "src/ard/attacks/pgd.py",
    "src/ard/config/schema.py",
    "src/ard/data/datasets.py",
    "src/ard/data/indexed.py",
    "src/ard/engine/checkpoint.py",
    "src/ard/engine/trainer.py",
    "src/ard/models/registry.py",
    "src/ard/objectives/rslad.py",
    "src/ard/policies/rslad.py",
    "src/ard/state/sample_store.py",
)
_BRIDGE_SOURCE_KEYS = frozenset({"bridge_implementation", "bridge_wrapper"})
_SOURCE_HASH_KEYS = frozenset(_RUNTIME_SOURCE_PATHS).union(_BRIDGE_SOURCE_KEYS)
_SAMPLE_STATE_KEYS = frozenset({"format_version", "ema_decay", "records", "pending", "next_order"})
_RECORD_FIELD_TYPES: dict[str, tuple[type[object], ...]] = {
    "margin_ema": (float,),
    "seen": (int,),
    "robust_correct_count": (int,),
    "previous_robust_correct": (bool, type(None)),
    "forgetting_count": (int,),
    "last_update": (int,),
    "last_margin": (float, type(None)),
    "true_label": (int,),
    "teacher_clean_entropy": (float,),
    "teacher_clean_true_probability": (float,),
    "teacher_clean_max_wrong_probability": (float,),
    "teacher_clean_prediction": (int,),
    "teacher_clean_correct": (bool,),
    "teacher_adversarial_entropy": (float,),
    "teacher_adversarial_true_probability": (float,),
    "teacher_adversarial_max_wrong_probability": (float,),
    "teacher_adversarial_prediction": (int,),
    "teacher_adversarial_correct": (bool,),
    "first_robustly_learned_epoch": (int, type(None)),
    "current_correct_streak": (int,),
    "longest_correct_streak": (int,),
    "margin_mean": (float,),
    "margin_m2": (float,),
    "margin_time_sum": (float,),
    "margin_time_squared_sum": (float,),
    "margin_time_margin_sum": (float,),
    "history_statistics_complete": (bool,),
    "teacher_clean_to_adversarial_margin_response": (float,),
    "teacher_clean_to_adversarial_js_response": (float,),
}


class ObservationBridgeError(RuntimeError):
    """An emission or comparison is not suitable as migration evidence."""


def observation_kwargs_for(trainer_class: type[Any]) -> tuple[str, dict[str, Any]]:
    """Select exactly one supported observation API, rejecting ambiguity."""
    parameters = inspect.signature(trainer_class.__init__).parameters
    has_legacy = "observe_teacher_signals" in parameters
    has_profile = "observation_profile" in parameters
    if has_legacy == has_profile:
        candidates = ("observe_teacher_signals", "observation_profile")
        found = ", ".join(sorted(name for name in candidates if name in parameters))
        raise ObservationBridgeError("expected exactly one Trainer observation API; found " + (found or "neither"))
    if has_legacy:
        return "observe_teacher_signals", {"observe_teacher_signals": True}
    return "observation_profile", {"observation_profile": "teacher_response"}


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_root() -> Path:
    configured = os.environ.get("ARD_BRIDGE_RUNTIME_ROOT")
    if configured:
        root = Path(configured).resolve()
        if not (root / ".git").exists():
            raise ObservationBridgeError("ARD_BRIDGE_RUNTIME_ROOT is not a Git worktree")
        return root
    return Path(__file__).resolve().parents[4]


def _git_sha(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ObservationBridgeError("cannot resolve Git HEAD for bridge emission")
    sha = completed.stdout.strip()
    if len(sha) != 40:
        raise ObservationBridgeError("Git HEAD is not a full SHA")
    return sha


def _require_full_sha(value: str, *, label: str) -> str:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value.lower()):
        raise ObservationBridgeError(f"{label} must be a full 40-character Git SHA")
    return value.lower()


def _require_clean_worktree(root: Path) -> None:
    completed = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ObservationBridgeError("cannot inspect runtime worktree cleanliness")
    if completed.stdout:
        raise ObservationBridgeError("bridge emission requires a clean runtime worktree")


def _source_hashes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in _RUNTIME_SOURCE_PATHS:
        path = root / relative
        if not path.is_file():
            raise ObservationBridgeError(f"bridge source is missing: {relative}")
        result[relative] = _sha256_path(path)
    implementation = Path(__file__).resolve()
    wrapper_path = Path(os.environ.get("ARD_BRIDGE_WRAPPER_PATH", ""))
    if not wrapper_path.is_file():
        raise ObservationBridgeError("bridge wrapper path is missing or unreadable")
    result["bridge_implementation"] = _sha256_path(implementation)
    result["bridge_wrapper"] = _sha256_path(wrapper_path)
    if set(result) != _SOURCE_HASH_KEYS:
        raise ObservationBridgeError("bridge source hash manifest is incomplete")
    return result


def _assert_runtime_module_origins(root: Path) -> dict[str, str]:
    runtime_src = (root / "src").resolve()
    runtime_modules = {
        "attack": LinfPGD,
        "attack_config": AttackConfig,
        "data": SyntheticCIFAR,
        "indexed": IndexedDataset,
        "trainer": Trainer,
        "student": build_student,
        "objective": RSLADObjective,
        "policy": RSLADBaselinePolicy,
        "state": SampleStateStore,
    }
    origins: dict[str, str] = {}
    for label, symbol in runtime_modules.items():
        module = sys.modules.get(symbol.__module__)
        source = getattr(module, "__file__", None)
        if not isinstance(source, str):
            raise ObservationBridgeError(f"runtime module source is unavailable for {label}")
        resolved = Path(source).resolve()
        try:
            origins[label] = str(resolved.relative_to(runtime_src))
        except ValueError as exc:
            raise ObservationBridgeError(f"runtime module {label} did not resolve below runtime/src") from exc
    return origins


def _seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError as exc:  # numpy RNG is checkpointed, so do not silently omit it.
        raise ObservationBridgeError("NumPy is required for a bridge emission") from exc
    torch.cuda.manual_seed_all(seed)


def _cuda_environment(device: torch.device) -> dict[str, object]:
    index = torch.cuda.current_device() if device.index is None else device.index
    properties = torch.cuda.get_device_properties(index)
    uuid = getattr(properties, "uuid", None)
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_name": properties.name,
        "gpu_uuid": None if uuid is None else str(uuid),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _make_loaders(seed: int) -> tuple[DataLoader[Any], DataLoader[Any], EpochShuffleSampler]:
    dataset = IndexedDataset(SyntheticCIFAR(size=8, num_classes=3, image_size=4, seed=seed))
    train_dataset, validation_dataset = stratified_train_validation_split(dataset, validation_fraction=0.25, seed=seed)
    sampler = EpochShuffleSampler(len(train_dataset), seed=seed)
    validation_sampler = EpochShuffleSampler(len(validation_dataset), seed=seed, shuffle=False)
    return (
        DataLoader(train_dataset, batch_size=4, sampler=sampler, collate_fn=collate_indexed),
        DataLoader(validation_dataset, batch_size=4, sampler=validation_sampler, collate_fn=collate_indexed),
        sampler,
    )


def _atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_dump(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def emit(output: Path, *, expected_git_sha: str, device: str = "cuda:0") -> Path:
    """Emit one deterministic, bounded observed-RSLAD CUDA checkpoint wrapper."""
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda" or not torch.cuda.is_available():
        raise ObservationBridgeError("bridge emission requires an available CUDA device")
    if output.exists():
        raise ObservationBridgeError(f"refusing to overwrite existing bridge emission: {output}")
    root = _repo_root()
    expected_git_sha = _require_full_sha(expected_git_sha, label="expected Git SHA")
    _require_clean_worktree(root)
    if _git_sha(root) != expected_git_sha:
        raise ObservationBridgeError("runtime Git SHA does not match the requested bridge SHA")
    runtime_module_paths = _assert_runtime_module_origins(root)
    api_name, observation_kwargs = observation_kwargs_for(Trainer)
    original_deterministic = torch.are_deterministic_algorithms_enabled()
    original_benchmark = torch.backends.cudnn.benchmark
    original_cudnn_deterministic = torch.backends.cudnn.deterministic
    original_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    original_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        _seed_all(4)
        torch.cuda.reset_peak_memory_stats(resolved_device)
        torch.manual_seed(123)
        student = build_student(ModelConfig(architecture="fixture_cnn", num_classes=3), tier="smoke")
        torch.manual_seed(456)
        teacher = build_student(ModelConfig(architecture="fixture_cnn", num_classes=3), tier="smoke")
        optimizer = SGD(student.parameters(), lr=0.03, momentum=0.9)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="ard-observation-bridge-", dir=output.parent) as temporary_output:
            trainer = Trainer(
                model=student,
                teacher=teacher,
                optimizer=optimizer,
                scheduler=StepLR(optimizer, step_size=1, gamma=0.8),
                scaler=None,
                attack=LinfPGD(
                    AttackConfig(
                        loss="kl",
                        kl_target="teacher_clean",
                        epsilon="1/255",
                        step_size="1/255",
                        steps=1,
                        random_start=True,
                    )
                ),
                selection_attack=LinfPGD(
                    AttackConfig(
                        loss="ce",
                        epsilon="1/255",
                        step_size="1/255",
                        steps=1,
                        random_start=True,
                        student_mode="eval",
                        teacher_mode="eval",
                    )
                ),
                objective=RSLADObjective(),
                policy=RSLADBaselinePolicy(),
                sample_store=SampleStateStore(ema_decay=0.9),
                device=resolved_device,
                output_dir=Path(temporary_output),
                config_hash=f"observation-bridge-{api_name}",
                seed=4,
                evaluation_attack_seed=9,
                tracker_run_id=f"observation-bridge-{api_name}",
                **observation_kwargs,
            )
            train_loader, validation_loader, _ = _make_loaders(seed=4)
            history = trainer.fit(train_loader, validation_loader=validation_loader, epochs=2)
            checkpoint_path = trainer.output_dir / "last.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ObservationBridgeError("bridge trainer did not produce a mapping checkpoint")
        telemetry = {
            key.removeprefix("train_"): value
            for key, value in history[-1].items()
            if key.startswith("train_teacher_") and key.endswith("_forward_calls")
        }
        if not set(telemetry).issubset(_ALLOWED_TELEMETRY):
            raise ObservationBridgeError("bridge observed unrecognized forward-count telemetry")
        emission = {
            "bridge_schema_version": EMISSION_SCHEMA_VERSION,
            "metadata": {
                "git_sha": _git_sha(root),
                "command": list(sys.argv),
                "source_hashes": _source_hashes(root),
                "trainer_observation_api": api_name,
                "device": str(resolved_device),
                "fixture": _FIXTURE,
                "runtime_module_paths": runtime_module_paths,
                "environment": _cuda_environment(resolved_device),
            },
            "checkpoint": checkpoint,
            "telemetry": telemetry,
        }
        _atomic_torch_save(emission, output)
        _atomic_json_dump(
            {
                "bridge_schema_version": EMISSION_SCHEMA_VERSION,
                "emission": str(output),
                "emission_sha256": _sha256_path(output),
                "checkpoint_canonical_sha256": canonical_sha256(checkpoint),
                "metadata": emission["metadata"],
                "telemetry": telemetry,
            },
            output.with_suffix(output.suffix + ".json"),
        )
        return output
    finally:
        torch.use_deterministic_algorithms(original_deterministic)
        torch.backends.cudnn.benchmark = original_benchmark
        torch.backends.cudnn.deterministic = original_cudnn_deterministic
        torch.backends.cuda.matmul.allow_tf32 = original_matmul_tf32
        torch.backends.cudnn.allow_tf32 = original_cudnn_tf32


def _canonical_update(digest: hashlib._Hash, value: object) -> None:
    """Hash only explicitly supported checkpoint value types; unknown is an error."""
    if value is None:
        digest.update(b"none")
    elif isinstance(value, bool):
        digest.update(b"bool:1" if value else b"bool:0")
    elif isinstance(value, int):
        digest.update(f"int:{value}".encode())
    elif isinstance(value, float):
        digest.update(b"float:")
        digest.update(value.hex().encode())
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(f"str:{len(encoded)}:".encode())
        digest.update(encoded)
    elif isinstance(value, bytes):
        digest.update(f"bytes:{len(value)}:".encode())
        digest.update(value)
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(f"tensor:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    elif isinstance(value, Mapping):
        digest.update(f"map:{len(value)}:".encode())
        encoded_keys: list[tuple[bytes, object]] = []
        for key, mapped in value.items():
            key_digest = hashlib.sha256()
            _canonical_update(key_digest, key)
            encoded_keys.append((key_digest.digest(), mapped))
        for key_digest, mapped in sorted(encoded_keys, key=lambda item: item[0]):
            digest.update(key_digest)
            _canonical_update(digest, mapped)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        digest.update(f"sequence:{len(value)}:".encode())
        for item in value:
            _canonical_update(digest, item)
    else:
        try:
            import numpy as np

            if isinstance(value, np.ndarray):
                array = value.astype(value.dtype, copy=False)
                digest.update(f"ndarray:{array.dtype}:{tuple(array.shape)}:".encode())
                digest.update(array.tobytes(order="C"))
                return
        except ImportError:
            pass
        raise ObservationBridgeError(f"unsupported checkpoint value for canonical hashing: {type(value)!r}")


def canonical_sha256(value: object) -> str:
    digest = hashlib.sha256()
    _canonical_update(digest, value)
    return digest.hexdigest()


def _assert_equal(left: object, right: object, *, path: str) -> None:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape) or not torch.equal(left, right):
            raise ObservationBridgeError(f"payload mismatch at {path}")
        return
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            raise ObservationBridgeError(f"payload mapping keys mismatch at {path}")
        for key in sorted(left, key=repr):
            _assert_equal(left[key], right[key], path=f"{path}.{key}")
        return
    if (
        isinstance(left, Sequence)
        and isinstance(right, Sequence)
        and not isinstance(left, (str, bytes, bytearray))
        and not isinstance(right, (str, bytes, bytearray))
    ):
        if len(left) != len(right):
            raise ObservationBridgeError(f"payload sequence length mismatch at {path}")
        for index, (left_value, right_value) in enumerate(zip(left, right, strict=True)):
            _assert_equal(left_value, right_value, path=f"{path}[{index}]")
        return
    try:
        import numpy as np

        if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
            if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right):
                raise ObservationBridgeError(f"payload mismatch at {path}")
            return
    except ImportError:
        pass
    if type(left) is not type(right) or left != right:
        raise ObservationBridgeError(f"payload mismatch at {path}")


def _validate_sample_state(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ObservationBridgeError("checkpoint sample_state is not a mapping")
    if set(value) != _SAMPLE_STATE_KEYS or value.get("format_version") != 3:
        raise ObservationBridgeError("bridge requires format-v3 sample state")
    if type(value["ema_decay"]) is not float or value["ema_decay"] != 0.9:
        raise ObservationBridgeError("bridge sample state has an unexpected EMA decay")
    records = value.get("records")
    if not isinstance(records, Mapping) or set(records) != _FIXTURE_SAMPLE_IDS:
        raise ObservationBridgeError("bridge sample state has unexpected stable sample IDs")
    pending = value.get("pending")
    if not isinstance(pending, list) or pending:
        raise ObservationBridgeError("bridge sample state must have an empty pending queue")
    if type(value["next_order"]) is not int or value["next_order"] != 0:
        raise ObservationBridgeError("bridge sample state has unexpected pending order")
    for sample_id, record in records.items():
        if not isinstance(sample_id, str) or not isinstance(record, Mapping):
            raise ObservationBridgeError("bridge sample state records are malformed")
        if set(record) != set(_RECORD_FIELD_TYPES):
            raise ObservationBridgeError("bridge sample state record fields are incomplete or unexpected")
        for name, expected_types in _RECORD_FIELD_TYPES.items():
            primitive = record[name]
            if type(primitive) not in expected_types:
                raise ObservationBridgeError(f"bridge sample state record field has wrong type: {name}")
        if record["seen"] != 2 or not 0 <= record["robust_correct_count"] <= record["seen"]:
            raise ObservationBridgeError("bridge sample state record counts are invalid")
        if not 0 <= record["forgetting_count"] <= record["seen"] - 1:
            raise ObservationBridgeError("bridge sample state forgetting count is invalid")
        if not 0 <= record["current_correct_streak"] <= record["seen"]:
            raise ObservationBridgeError("bridge sample state current streak is invalid")
        if not record["current_correct_streak"] <= record["longest_correct_streak"] <= record["seen"]:
            raise ObservationBridgeError("bridge sample state longest streak is invalid")
        if record["last_update"] not in {2, 3}:
            raise ObservationBridgeError("bridge sample state last update is outside the two-epoch fixture")
        if record["history_statistics_complete"] is not True:
            raise ObservationBridgeError("bridge sample state history statistics are incomplete")
        _validate_record_ranges(record)
    return dict(value)


def _validate_record_ranges(record: Mapping[str, object]) -> None:
    import math

    bounded = {
        "margin_ema": (-1.0, 1.0),
        "last_margin": (-1.0, 1.0),
        "teacher_clean_entropy": (0.0, math.log(3.0)),
        "teacher_adversarial_entropy": (0.0, math.log(3.0)),
        "teacher_clean_true_probability": (0.0, 1.0),
        "teacher_clean_max_wrong_probability": (0.0, 1.0),
        "teacher_adversarial_true_probability": (0.0, 1.0),
        "teacher_adversarial_max_wrong_probability": (0.0, 1.0),
        "teacher_clean_to_adversarial_margin_response": (-2.0, 2.0),
        "teacher_clean_to_adversarial_js_response": (0.0, math.log(2.0)),
    }
    for name, (minimum, maximum) in bounded.items():
        value = record[name]
        if value is None and name == "last_margin":
            raise ObservationBridgeError("bridge sample state has an unobserved final margin")
        if not isinstance(value, float) or not math.isfinite(value) or not minimum <= value <= maximum:
            raise ObservationBridgeError(f"bridge sample state field is outside its finite range: {name}")
    for name in ("margin_mean", "margin_m2", "margin_time_sum", "margin_time_squared_sum", "margin_time_margin_sum"):
        value = record[name]
        if not isinstance(value, float) or not math.isfinite(value):
            raise ObservationBridgeError(f"bridge sample state history field is non-finite: {name}")
    if record["margin_m2"] < 0.0:
        raise ObservationBridgeError("bridge sample state margin accumulator is negative")


def _load_emission(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ObservationBridgeError(f"bridge emission does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("bridge_schema_version") != EMISSION_SCHEMA_VERSION:
        raise ObservationBridgeError("bridge emission schema is unsupported")
    metadata, checkpoint, telemetry = payload.get("metadata"), payload.get("checkpoint"), payload.get("telemetry")
    if not isinstance(metadata, Mapping) or not isinstance(checkpoint, Mapping) or not isinstance(telemetry, Mapping):
        raise ObservationBridgeError("bridge emission is incomplete")
    required_metadata = {
        "git_sha",
        "command",
        "source_hashes",
        "trainer_observation_api",
        "device",
        "fixture",
        "runtime_module_paths",
        "environment",
    }
    if set(metadata) != required_metadata:
        raise ObservationBridgeError("bridge emission metadata is incomplete or has unrecognized fields")
    if metadata["trainer_observation_api"] not in {"observe_teacher_signals", "observation_profile"}:
        raise ObservationBridgeError("bridge emission has an unknown Trainer observation API")
    if metadata["fixture"] != _FIXTURE:
        raise ObservationBridgeError("bridge emission has an unexpected fixture identity")
    _require_full_sha(str(metadata["git_sha"]), label="emission Git SHA")
    source_hashes = metadata["source_hashes"]
    if not isinstance(source_hashes, Mapping) or set(source_hashes) != _SOURCE_HASH_KEYS:
        raise ObservationBridgeError("bridge emission source hashes are incomplete")
    for key, value in source_hashes.items():
        if not isinstance(key, str) or not isinstance(value, str) or len(value) != 64:
            raise ObservationBridgeError("bridge emission contains an invalid source hash")
    module_paths = metadata["runtime_module_paths"]
    if not isinstance(module_paths, Mapping) or not module_paths:
        raise ObservationBridgeError("bridge emission is missing runtime module origins")
    if not isinstance(metadata["environment"], Mapping):
        raise ObservationBridgeError("bridge emission is missing environment metadata")
    if not set(telemetry).issubset(_ALLOWED_TELEMETRY):
        raise ObservationBridgeError("bridge emission has unrecognized telemetry")
    _validate_sample_state(checkpoint.get("sample_state"))
    return dict(payload)


def compare(
    left_path: Path,
    right_path: Path,
    *,
    attestation: Path,
    left_expected_git_sha: str,
    right_expected_git_sha: str,
) -> dict[str, Any]:
    """Compare two emissions and write a JSON attestation only on exact success."""
    if attestation.exists():
        raise ObservationBridgeError(f"refusing to overwrite existing attestation: {attestation}")
    if left_path.resolve() == right_path.resolve():
        raise ObservationBridgeError("left and right bridge emissions must be distinct files")
    left_expected_git_sha = _require_full_sha(left_expected_git_sha, label="left expected Git SHA")
    right_expected_git_sha = _require_full_sha(right_expected_git_sha, label="right expected Git SHA")
    if left_expected_git_sha == right_expected_git_sha:
        raise ObservationBridgeError("left and right expected Git SHAs must differ")
    left, right = _load_emission(left_path), _load_emission(right_path)
    left_metadata, right_metadata = left["metadata"], right["metadata"]
    assert isinstance(left_metadata, Mapping) and isinstance(right_metadata, Mapping)
    if left_metadata["git_sha"] != left_expected_git_sha or right_metadata["git_sha"] != right_expected_git_sha:
        raise ObservationBridgeError("emission Git SHA does not match its expected runtime SHA")
    if left_metadata["trainer_observation_api"] != "observe_teacher_signals":
        raise ObservationBridgeError("left emission must use the legacy observation API")
    if right_metadata["trainer_observation_api"] != "observation_profile":
        raise ObservationBridgeError("right emission must use the observation-profile API")
    if left_metadata["fixture"] != right_metadata["fixture"]:
        raise ObservationBridgeError("bridge emissions do not use an identical fixture")
    if _sha256_path(left_path) == _sha256_path(right_path):
        raise ObservationBridgeError("bridge emissions have identical output hashes")
    left_checkpoint, right_checkpoint = dict(left["checkpoint"]), dict(right["checkpoint"])
    left_state = _validate_sample_state(left_checkpoint.pop("sample_state", None))
    right_state = _validate_sample_state(right_checkpoint.pop("sample_state", None))
    for key in _IDENTITY_KEYS:
        if key not in left_checkpoint or key not in right_checkpoint:
            raise ObservationBridgeError(f"checkpoint is missing allowed identity field {key}")
        left_checkpoint.pop(key)
        right_checkpoint.pop(key)
    _assert_equal(left_checkpoint, right_checkpoint, path="checkpoint")
    _assert_equal(left_state, right_state, path="checkpoint.sample_state")
    result = {
        "bridge_schema_version": EMISSION_SCHEMA_VERSION,
        "result": "pass",
        "allowed_differences": {
            "checkpoint": sorted(_IDENTITY_KEYS),
            "telemetry": sorted(_ALLOWED_TELEMETRY),
            "metadata": [
                "git_sha",
                "command",
                "source_hashes",
                "trainer_observation_api",
                "device",
                "runtime_module_paths",
                "environment",
            ],
        },
        "left": {
            "path": str(left_path),
            "output_sha256": _sha256_path(left_path),
            "checkpoint_sha256": canonical_sha256(left["checkpoint"]),
            "metadata": dict(left["metadata"]),
            "sample_state_record_count": len(left_state["records"]),
        },
        "right": {
            "path": str(right_path),
            "output_sha256": _sha256_path(right_path),
            "checkpoint_sha256": canonical_sha256(right["checkpoint"]),
            "metadata": dict(right["metadata"]),
            "sample_state_record_count": len(right_state["records"]),
        },
        "equal_checkpoint_without_identity_sha256": canonical_sha256(left_checkpoint),
        "equal_sample_state_sha256": canonical_sha256(left_state),
    }
    _atomic_json_dump(result, attestation)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    emit_parser = subparsers.add_parser("emit", help="emit a bounded observed-RSLAD CUDA bridge payload")
    emit_parser.add_argument("--output", type=Path, required=True)
    emit_parser.add_argument("--expected-git-sha", required=True)
    emit_parser.add_argument("--device", default="cuda:0")
    compare_parser = subparsers.add_parser("compare", help="compare two bridge payloads exactly")
    compare_parser.add_argument("--left", type=Path, required=True)
    compare_parser.add_argument("--right", type=Path, required=True)
    compare_parser.add_argument("--attestation", type=Path, required=True)
    compare_parser.add_argument("--left-expected-git-sha", required=True)
    compare_parser.add_argument("--right-expected-git-sha", required=True)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "emit":
            print(emit(arguments.output, expected_git_sha=arguments.expected_git_sha, device=arguments.device))
        else:
            result = compare(
                arguments.left,
                arguments.right,
                attestation=arguments.attestation,
                left_expected_git_sha=arguments.left_expected_git_sha,
                right_expected_git_sha=arguments.right_expected_git_sha,
            )
            print(json.dumps(result, sort_keys=True))
    except ObservationBridgeError as exc:
        parser.exit(2, f"cross-version observation bridge: {exc}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the wrapper.
    raise SystemExit(main())
