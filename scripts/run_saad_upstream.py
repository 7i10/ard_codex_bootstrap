#!/usr/bin/env python3
"""Run the pinned full-SAAD oracle without importing it into ARD.

This launcher is intentionally an operational boundary, not a second training
engine.  It only accepts frozen Bartoldson/Chen SAAD profiles, stages symlinks
in a fresh output directory, and delegates Python import ordering to
``saad_runtime_bootstrap.py``.  The upstream checkout is never edited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SAAD_COMMIT = "295121c5d2eed827b5b2d6aa42307de809bdfada"
ROBUSTBENCH_COMMIT = "78fcc9e48a07a861268f295a777b975f25155964"
BARTOLDSON_SHA256 = "56bbad8ad748df86e67c24dba4f59a9e7d285e583251460b2ed154017a18cb0b"
CHEN_SHA256 = "fc398a4890e6856b5dd80856076000ec9e2debdd12d9f78a66171b9ffc383983"
CHEN_WRN34_20_SHA256 = "dbfc7cfe402d9ddf6cbe47c4809eab97fcccce7b6a254030cdca2640639cfa28"
UNMODIFIED_VARIANT = "unmodified_upstream"
PAPER_WD_VARIANT = "paper_weight_decay_5e4"
PAPER_WD_PATCH = ROOT / "patches" / "saad" / "0001-use-cli-weight-decay.patch"
TEACHER_PROFILES = {
    "Bartoldson2024Adversarial_WRN-94-16": {
        "checkpoint_sha256": BARTOLDSON_SHA256,
        "run_name": "full-saad-bartoldson-seed0",
        "variants": frozenset({UNMODIFIED_VARIANT}),
    },
    "Chen2021LTD_WRN34_10": {
        "checkpoint_sha256": CHEN_SHA256,
        "run_name": "full-saad-chen-seed0",
        "variants": frozenset({UNMODIFIED_VARIANT}),
    },
    "Chen2021LTD_WRN34_20": {
        "checkpoint_sha256": CHEN_WRN34_20_SHA256,
        "run_name": "full-saad-chen34-20-seed0",
        "variants": frozenset({UNMODIFIED_VARIANT, PAPER_WD_VARIANT}),
    },
}
CIFAR10_ARCHIVE_SHA256 = "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"
CIFAR10_EXTRACTED_SHA256 = {
    "batches.meta": "f962466ef690d46b226450fb9aadc74ba4bc64a76aa526b5827fe4bc5c7125cb",
    "data_batch_1": "54636561a3ce25bd3e19253c6b0d8538147b0ae398331ac4a2d86c6d987368cd",
    "data_batch_2": "766b2cef9fbc745cf056b3152224f7cf77163b330ea9a15f9392beb8b89bc5a8",
    "data_batch_3": "0f00d98ebfb30b3ec0ad19f9756dc2630b89003e10525f5e148445e82aa6a1f9",
    "data_batch_4": "3f7bb240661948b8f4d53e36ec720d8306f5668bd0071dcb4e6c947f78e9682b",
    "data_batch_5": "d91802434d8376bbaeeadf58a737e3a1b12ac839077e931237e0dcd43adcb154",
    "readme.html": "4d1c3fb199d6a183ae03f5162b469d7bc04edf2fad9547bd5f224271d52f98e5",
    "test_batch": "f53d8d457504f7cff4ea9e021afcf0e0ad8e24a91f3fc42091b8adef61157831",
}
PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
LOSS_TOKEN_RE = re.compile(r"loss:\s*([^\s|]+)(?=[\s|])")


class SAADLaunchError(RuntimeError):
    """The isolated upstream launcher contract was not satisfied."""


@dataclass(frozen=True)
class SAADConfig:
    runtime_python: Path
    pytorch_cuda_alloc_conf: str
    dataset_root: Path
    teacher_checkpoint: Path
    teacher_checkpoint_sha256: str
    teacher_name: str
    run_name: str
    output_dir: Path
    smoke_batch_size: int
    smoke_loss_events: int
    physical_gpu: int
    teacher_logit_contract: dict[str, Any]
    source_variant: str = UNMODIFIED_VARIANT
    source_patch: Path | None = None
    source_patch_sha256: str | None = None
    weight_decay: float = 0.0002


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _git(directory: Path, *arguments: str) -> str:
    completed = subprocess.run(["git", *arguments], cwd=directory, text=True, capture_output=True)
    if completed.returncode:
        raise SAADLaunchError(completed.stderr.strip() or "failed to inspect upstream checkout")
    return completed.stdout.strip()


def launch_source_files(config_path: Path, config: SAADConfig | None = None) -> dict[str, dict[str, str]]:
    """Hash every local source/config input that changes launch semantics."""
    sources = {
        "launcher": _file_identity(Path(__file__)),
        "runtime_bootstrap": _file_identity(ROOT / "scripts" / "saad_runtime_bootstrap.py"),
        "config": _file_identity(config_path),
        "runtime_lock": _file_identity(ROOT / "requirements" / "saad-upstream-runtime.lock"),
        "teacher_probe": _file_identity(ROOT / "scripts" / "saad_teacher_probe.py"),
    }
    if config is not None and config.source_patch is not None:
        sources["upstream_patch"] = _file_identity(config.source_patch)
    return sources


def launch_identity(
    *,
    config_path: Path,
    command: list[str],
    config: SAADConfig | None = None,
    entrypoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Hash every local input that changes an upstream launch's semantics."""
    return {
        "ard_git": {"head": _git(ROOT, "rev-parse", "HEAD"), "dirty": bool(_git(ROOT, "status", "--porcelain"))},
        "command_sha256": _canonical_sha256(command),
        "source_files": launch_source_files(config_path, config),
        "source_variant": config.source_variant if config is not None else UNMODIFIED_VARIANT,
        "entrypoint": entrypoint,
    }


def _checked_mapping(value: Any, *, context: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SAADLaunchError(f"{context} must be a mapping")
    unknown = set(value) - keys
    missing = keys - set(value)
    if unknown:
        raise SAADLaunchError(f"{context} has unknown keys: {sorted(unknown)}")
    if missing:
        raise SAADLaunchError(f"{context} is missing keys: {sorted(missing)}")
    return value


def load_config(path: Path) -> SAADConfig:
    """Parse the deliberately small, unknown-key-failing operational YAML."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise SAADLaunchError(f"unable to read upstream SAAD config {path}: {error}") from error
    if not isinstance(raw, dict):
        raise SAADLaunchError("config must be a mapping")
    required_root_keys = {
        "version",
        "runtime",
        "inputs",
        "output",
        "smoke",
        "gpu",
        "protocol",
        "teacher_logit_contract",
    }
    unknown_root_keys = set(raw) - required_root_keys - {"source"}
    missing_root_keys = required_root_keys - set(raw)
    if unknown_root_keys:
        raise SAADLaunchError(f"config has unknown keys: {sorted(unknown_root_keys)}")
    if missing_root_keys:
        raise SAADLaunchError(f"config is missing keys: {sorted(missing_root_keys)}")
    root = raw
    if root["version"] != 1:
        raise SAADLaunchError("config.version must be exactly 1")
    runtime = _checked_mapping(
        root["runtime"],
        context="runtime",
        keys={"python", "python_version", "pytorch_cuda_alloc_conf"},
    )
    inputs = _checked_mapping(
        root["inputs"],
        context="inputs",
        keys={"dataset_root", "cifar10_archive_sha256", "teacher_checkpoint", "teacher_checkpoint_sha256"},
    )
    output = _checked_mapping(root["output"], context="output", keys={"directory"})
    smoke = _checked_mapping(root["smoke"], context="smoke", keys={"batch_size", "loss_events"})
    gpu = _checked_mapping(root["gpu"], context="gpu", keys={"physical_id"})
    source = root.get("source")
    if source is None:
        source = {"variant": UNMODIFIED_VARIANT, "patch": None, "patch_sha256": None}
    source = _checked_mapping(source, context="source", keys={"variant", "patch", "patch_sha256"})
    teacher_logit_contract = _checked_mapping(
        root["teacher_logit_contract"],
        context="teacher_logit_contract",
        keys={
            "reference_torch",
            "candidate_torch",
            "fixed_input_count",
            "atol",
            "rtol",
            "require_argmax_equal",
        },
    )
    protocol = _checked_mapping(
        root["protocol"],
        context="protocol",
        keys={
            "method",
            "epochs",
            "batch_size",
            "seed",
            "student",
            "teacher_name",
            "dataset",
            "swa_epoch",
            "beta",
            "gamma",
            "igdm_alpha",
            "lambda_inner",
            "nowand",
            "lr",
            "momentum",
            "weight_decay",
            "milestones",
            "inner_steps",
            "epsilon",
            "step_size",
            "entropy_multiplier",
        },
    )
    frozen = {
        "method": "saad",
        "epochs": 200,
        "batch_size": 128,
        "seed": 0,
        "student": "RES-18",
        "dataset": "cifar10",
        "swa_epoch": 95,
        "beta": 0,
        "gamma": 1,
        "igdm_alpha": 1,
        "lambda_inner": 1,
        "nowand": 1,
        "lr": 0.1,
        "momentum": 0.9,
        "milestones": [100, 150],
        "inner_steps": 10,
        "epsilon": "8/255",
        "step_size": "2/255",
        "entropy_multiplier": 5,
    }
    teacher_name = protocol.get("teacher_name")
    if not isinstance(teacher_name, str):
        raise SAADLaunchError("protocol.teacher_name must be a string")
    profile = TEACHER_PROFILES.get(teacher_name)
    variant = source.get("variant")
    expected_weight_decay = 0.0005 if variant == PAPER_WD_VARIANT else 0.0002
    frozen["weight_decay"] = expected_weight_decay
    protocol_without_teacher = {key: value for key, value in protocol.items() if key != "teacher_name"}
    if profile is None:
        raise SAADLaunchError(f"teacher_name is not an allowed upstream profile: {teacher_name!r}")
    if variant not in {UNMODIFIED_VARIANT, PAPER_WD_VARIANT}:
        raise SAADLaunchError(f"unknown upstream source variant: {variant!r}")
    if variant not in profile["variants"]:
        raise SAADLaunchError(f"source variant {variant!r} is not approved for {teacher_name}")
    if protocol_without_teacher != frozen:
        changed = {
            key: {"expected": frozen[key], "actual": protocol_without_teacher.get(key)}
            for key in frozen
            if protocol_without_teacher.get(key) != frozen[key]
        }
        raise SAADLaunchError(f"protocol drift is forbidden: {json.dumps(changed, sort_keys=True)}")
    if runtime["python_version"] != "3.11.15":
        raise SAADLaunchError("runtime.python_version must be 3.11.15")
    if runtime["pytorch_cuda_alloc_conf"] != PYTORCH_CUDA_ALLOC_CONF:
        raise SAADLaunchError(f"runtime.pytorch_cuda_alloc_conf must be exactly {PYTORCH_CUDA_ALLOC_CONF!r}")
    if inputs["cifar10_archive_sha256"] != CIFAR10_ARCHIVE_SHA256:
        raise SAADLaunchError("CIFAR-10 archive hash differs from frozen input")
    if inputs["teacher_checkpoint_sha256"] != profile["checkpoint_sha256"]:
        raise SAADLaunchError(f"{teacher_name} checkpoint hash differs from frozen input")
    if not isinstance(smoke["batch_size"], int) or smoke["batch_size"] not in {16, 128}:
        raise SAADLaunchError("smoke.batch_size must be 16 or 128")
    if not isinstance(smoke["loss_events"], int) or not 2 <= smoke["loss_events"] <= 10:
        raise SAADLaunchError("smoke.loss_events must be within [2, 10]")
    expected_physical_gpu = 1 if variant == PAPER_WD_VARIANT else 0
    if gpu["physical_id"] != expected_physical_gpu:
        raise SAADLaunchError(
            f"gpu.physical_id must be {expected_physical_gpu} for source variant {variant!r}"
        )
    expected_logit_contract = {
        "reference_torch": "2.11.0+cu128",
        "candidate_torch": "2.4.1+cu121",
        "fixed_input_count": 4,
        "atol": 0.0001,
        "rtol": 0,
        "require_argmax_equal": True,
    }
    if teacher_logit_contract != expected_logit_contract:
        raise SAADLaunchError("teacher_logit_contract differs from the frozen cross-runtime contract")

    def project_path(value: Any, *, field: str) -> Path:
        if not isinstance(value, str) or not value:
            raise SAADLaunchError(f"{field} must be a non-empty path string")
        candidate = Path(value)
        return candidate if candidate.is_absolute() else ROOT / candidate

    patch_value = source.get("patch")
    patch_sha256 = source.get("patch_sha256")
    patch_path: Path | None = None
    if variant == UNMODIFIED_VARIANT:
        if patch_value is not None or patch_sha256 is not None:
            raise SAADLaunchError("unmodified upstream variant forbids a patch")
    else:
        patch_path = project_path(patch_value, field="source.patch")
        if patch_path.resolve() != PAPER_WD_PATCH.resolve():
            raise SAADLaunchError("paper weight-decay variant requires the approved patch path")
        if not patch_path.is_file() or not isinstance(patch_sha256, str) or _sha256(patch_path) != patch_sha256:
            raise SAADLaunchError("paper weight-decay patch SHA-256 mismatch")

    return SAADConfig(
        runtime_python=project_path(runtime["python"], field="runtime.python"),
        pytorch_cuda_alloc_conf=runtime["pytorch_cuda_alloc_conf"],
        dataset_root=project_path(inputs["dataset_root"], field="inputs.dataset_root"),
        teacher_checkpoint=project_path(inputs["teacher_checkpoint"], field="inputs.teacher_checkpoint"),
        teacher_checkpoint_sha256=inputs["teacher_checkpoint_sha256"],
        teacher_name=teacher_name,
        run_name=(
            f"{profile['run_name']}-paper-wd5e4" if variant == PAPER_WD_VARIANT else str(profile["run_name"])
        ),
        output_dir=project_path(output["directory"], field="output.directory"),
        smoke_batch_size=smoke["batch_size"],
        smoke_loss_events=smoke["loss_events"],
        physical_gpu=gpu["physical_id"],
        teacher_logit_contract=teacher_logit_contract,
        source_variant=variant,
        source_patch=patch_path,
        source_patch_sha256=patch_sha256,
        weight_decay=expected_weight_decay,
    )


def verified_checkout(root: Path, name: str, *, commit: str) -> Path:
    lock = yaml.safe_load((root / "external.lock.yaml").read_text(encoding="utf-8"))
    expected = lock.get("repositories", {}).get(name)
    if not isinstance(expected, dict):
        raise SAADLaunchError(f"external.lock.yaml has no {name} entry")
    checkout = root / ".external" / name
    if not checkout.is_dir():
        raise SAADLaunchError(f"pinned {name} checkout is absent")
    if expected.get("commit") != commit:
        raise SAADLaunchError(f"external.lock.yaml {name} commit is not frozen identity")
    if _git(checkout, "remote", "get-url", "origin") != expected.get("url"):
        raise SAADLaunchError(f"{name} origin differs from external.lock.yaml")
    if _git(checkout, "rev-parse", "HEAD") != commit:
        raise SAADLaunchError(f"{name} HEAD differs from frozen identity")
    if _git(checkout, "status", "--porcelain"):
        raise SAADLaunchError(f"refusing to launch dirty {name} checkout")
    return checkout.resolve()


def verified_saad_clone(root: Path = ROOT) -> Path:
    """Compatibility import used by the optional formula differential test."""
    return verified_checkout(root, "saad", commit=SAAD_COMMIT)


def verify_inputs(config: SAADConfig) -> dict[str, Any]:
    archive = config.dataset_root / "cifar-10-python.tar.gz"
    extracted_root = config.dataset_root / "cifar-10-batches-py"
    if not archive.is_file():
        raise SAADLaunchError("local CIFAR-10 archive is required; download is forbidden")
    if _sha256(archive) != CIFAR10_ARCHIVE_SHA256:
        raise SAADLaunchError("local CIFAR-10 archive hash differs from frozen input")
    extracted: dict[str, dict[str, str]] = {}
    for name, expected_hash in CIFAR10_EXTRACTED_SHA256.items():
        source = extracted_root / name
        if not source.is_file():
            raise SAADLaunchError(f"local CIFAR-10 extracted file is absent: {name}; download is forbidden")
        actual_hash = _sha256(source)
        if actual_hash != expected_hash:
            raise SAADLaunchError(f"local CIFAR-10 extracted file hash differs: {name}")
        extracted[name] = _file_identity(source)
    if (
        not config.teacher_checkpoint.is_file()
        or _sha256(config.teacher_checkpoint) != config.teacher_checkpoint_sha256
    ):
        raise SAADLaunchError(f"local {config.teacher_name} checkpoint hash differs from frozen input")
    return {
        "cifar10_archive": _file_identity(archive),
        "cifar10_extracted": extracted,
        "teacher_checkpoint": _file_identity(config.teacher_checkpoint),
    }


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _safe_symlink(link: Path, target: Path, *, stage_root: Path) -> None:
    resolved_stage = stage_root.resolve()
    resolved_target = target.resolve()
    if not target.exists() or _is_relative_to(resolved_target, resolved_stage):
        raise SAADLaunchError(f"unsafe staging target for {link.name}: {target}")
    if not _is_relative_to(link.parent.resolve(), resolved_stage):
        raise SAADLaunchError(f"staging link escapes stage root: {link}")
    if link.exists() or link.is_symlink():
        raise SAADLaunchError(f"staging link already exists: {link}")
    relative_target = os.path.relpath(resolved_target, link.parent.resolve())
    link.symlink_to(relative_target, target_is_directory=resolved_target.is_dir())
    if link.resolve() != resolved_target:
        raise SAADLaunchError(f"staging link resolved unexpectedly: {link}")


def stage_inputs(*, config: SAADConfig, saad: Path, output_dir: Path) -> Path:
    """Create a source-symlink stage without modifying either upstream checkout."""
    if output_dir.exists():
        raise SAADLaunchError(f"refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    stage = output_dir / "stage"
    stage.mkdir()
    # A directory of links (rather than a link to the checkout) gives RobustBench
    # its upstream-relative ``./models`` cache without writing into .external.
    for source in saad.iterdir():
        if source.name in {".git", "__pycache__", "models"}:
            continue
        if source.name == "saad.py" and config.source_variant == PAPER_WD_VARIANT:
            _write_paper_weight_decay_variant(source=source, destination=stage / source.name, config=config)
            continue
        _safe_symlink(stage / source.name, source, stage_root=stage)
    _safe_symlink(output_dir / "dataset", config.dataset_root, stage_root=output_dir)
    model_dir = stage / "models" / "cifar10" / "Linf"
    model_dir.mkdir(parents=True)
    _safe_symlink(model_dir / f"{config.teacher_name}.pt", config.teacher_checkpoint, stage_root=stage)
    return stage


def _write_paper_weight_decay_variant(*, source: Path, destination: Path, config: SAADConfig) -> None:
    """Materialize the one-line paper-aligned patch without editing upstream."""
    if config.source_patch is None or config.source_patch_sha256 is None:
        raise SAADLaunchError("paper weight-decay variant has no hash-bound patch")
    if _sha256(config.source_patch) != config.source_patch_sha256:
        raise SAADLaunchError("paper weight-decay patch drifted before staging")
    original = "optimizer = torch.optim.SGD(student.parameters(), lr=args.lr, momentum=0.9, weight_decay=2e-4)"
    replacement = "optimizer = torch.optim.SGD(student.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)"
    text = source.read_text(encoding="utf-8")
    if text.count(original) != 1 or replacement in text:
        raise SAADLaunchError("pinned upstream optimizer source no longer matches the approved patch preimage")
    patch_text = config.source_patch.read_text(encoding="utf-8")
    if f"-{original}" not in patch_text or f"+{replacement}" not in patch_text:
        raise SAADLaunchError("approved patch content does not describe the expected optimizer change")
    destination.write_text(text.replace(original, replacement), encoding="utf-8")


def entrypoint_identity(
    *, config: SAADConfig, saad: Path, staged_entrypoint: Path | None = None
) -> dict[str, Any]:
    """Bind the exact upstream and executed ``saad.py`` bytes."""
    upstream = saad / "saad.py"
    if not upstream.is_file():
        raise SAADLaunchError("pinned upstream saad.py is absent")
    original = "optimizer = torch.optim.SGD(student.parameters(), lr=args.lr, momentum=0.9, weight_decay=2e-4)"
    replacement = "optimizer = torch.optim.SGD(student.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)"
    upstream_text = upstream.read_text(encoding="utf-8")
    if upstream_text.count(original) != 1 or replacement in upstream_text:
        raise SAADLaunchError("pinned upstream optimizer source no longer matches the approved preimage")
    executed_text = upstream_text
    changed_lines = 0
    if config.source_variant == PAPER_WD_VARIANT:
        executed_text = upstream_text.replace(original, replacement)
        changed_lines = 1
    executed_sha256 = hashlib.sha256(executed_text.encode("utf-8")).hexdigest()
    if staged_entrypoint is not None:
        if not staged_entrypoint.is_file() or _sha256(staged_entrypoint) != executed_sha256:
            raise SAADLaunchError("staged saad.py bytes differ from the approved source variant")
        if config.source_variant == UNMODIFIED_VARIANT and (
            not staged_entrypoint.is_symlink() or staged_entrypoint.resolve() != upstream.resolve()
        ):
            raise SAADLaunchError("unmodified upstream variant must execute the original saad.py symlink")
    return {
        "variant": config.source_variant,
        "upstream": _file_identity(upstream),
        "executed_sha256": executed_sha256,
        "changed_lines": changed_lines,
        "patch_sha256": config.source_patch_sha256,
    }


def upstream_args(*, config: SAADConfig, batch_size: int) -> list[str]:
    """Return the entire frozen argument vector; callers cannot append drift."""
    if batch_size not in {16, 128}:
        raise SAADLaunchError("smoke batch must be 16 or 128")
    return [
        "--method",
        "saad",
        "--igdm_alpha",
        "1",
        "--beta",
        "0",
        "--gamma",
        "1",
        "--entropy_scale",
        "5",
        "--lambda_inner",
        "1",
        "--swa_epoch",
        "95",
        "--epochs",
        "200",
        "--batch",
        str(batch_size),
        "--lr",
        "0.1",
        "--momentum",
        "0.9",
        "--wd",
        str(config.weight_decay),
        "--teacher_name",
        config.teacher_name,
        "--student",
        "RES-18",
        "--dataset",
        "cifar10",
        "--depth",
        "0",
        "--widen_factor",
        "0",
        "--nowand",
        "1",
        "--wandb_name",
        config.run_name,
        "--seed",
        "0",
    ]


def build_runtime_command(
    *, config: SAADConfig, stage: Path, saad: Path, robustbench: Path, provenance: Path, batch_size: int
) -> list[str]:
    return [
        str(config.runtime_python),
        str(ROOT / "scripts" / "saad_runtime_bootstrap.py"),
        "--runtime-lock",
        str(ROOT / "requirements" / "saad-upstream-runtime.lock"),
        "--expected-python",
        "3.11.15",
        "--saad-stage",
        str(stage),
        "--saad-root",
        str(saad),
        "--robustbench-root",
        str(robustbench),
        "--provenance",
        str(provenance),
        "--",
        *upstream_args(config=config, batch_size=batch_size),
    ]


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@dataclass(frozen=True)
class SmokeResult:
    state: str
    loss_events: int
    returncode: int
    gpu_telemetry: dict[str, Any]
    invalid_loss_tokens: tuple[str, ...]


def classify_smoke(
    *,
    terminated_by_supervisor: bool,
    loss_events: int,
    requested_events: int,
    returncode: int,
    nonfinite_loss: bool = False,
) -> str:
    if nonfinite_loss:
        return "smoke_failure"
    if terminated_by_supervisor and requested_events <= loss_events <= 10:
        return "expected_smoke_termination"
    if returncode == 0:
        return "unexpected_smoke_completion"
    return "smoke_failure"


def observe_loss_chunk(buffer: str, chunk: bytes) -> tuple[str, tuple[float, ...], tuple[str, ...]]:
    """Parse complete upstream ``loss:`` tokens once across arbitrary stream chunks."""
    text = (buffer + chunk.decode("utf-8", errors="replace"))[-8192:]
    matches = list(LOSS_TOKEN_RE.finditer(text))
    if not matches:
        return text, (), ()
    values: list[float] = []
    invalid: list[str] = []
    for match in matches:
        token = match.group(1)
        try:
            value = float(token)
        except ValueError:
            invalid.append(token)
        else:
            if math.isfinite(value):
                values.append(value)
            else:
                invalid.append(token)
    # Keep the delimiter and suffix after the final consumed token.  A later
    # chunk cannot re-match this token, while a split next token is retained.
    return text[matches[-1].end() :], tuple(values), tuple(invalid)


def read_available_pipe_bytes(stream: Any) -> bytes:
    """Read promptly from a subprocess pipe; ``BufferedReader.read(n)`` waits for ``n`` bytes."""
    return os.read(stream.fileno(), 4096)


def runtime_environment(*, physical_gpu: int, pytorch_cuda_alloc_conf: str) -> dict[str, str]:
    """Prevent source-tree bytecode writes and ambient user-site imports."""
    if pytorch_cuda_alloc_conf != PYTORCH_CUDA_ALLOC_CONF:
        raise SAADLaunchError(f"PYTORCH_CUDA_ALLOC_CONF must be exactly {PYTORCH_CUDA_ALLOC_CONF!r}")
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(physical_gpu),
            "PYTORCH_CUDA_ALLOC_CONF": pytorch_cuda_alloc_conf,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONNOUSERSITE": "1",
            "WANDB_MODE": "disabled",
        }
    )
    return environment


class GPUSampler:
    """Best-effort physical-GPU telemetry, independent of CUDA device remapping."""

    def __init__(self, *, physical_gpu: int, interval_seconds: float = 0.5, snapshot_path: Path | None = None) -> None:
        self.physical_gpu = physical_gpu
        self.interval_seconds = interval_seconds
        self.snapshot_path = snapshot_path
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample_once(self) -> None:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--id",
                    str(self.physical_gpu),
                    "--query-gpu=index,memory.used,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                capture_output=True,
                timeout=5,
                check=False,
            )
            if completed.returncode:
                raise RuntimeError(completed.stderr.strip() or "nvidia-smi failed")
            fields = [item.strip() for item in completed.stdout.strip().split(",")]
            if len(fields) != 4:
                raise RuntimeError("unexpected nvidia-smi CSV row")
            index, memory_mib, utilization_percent, temperature_c = (int(item) for item in fields)
            if index != self.physical_gpu:
                raise RuntimeError(f"nvidia-smi returned physical GPU {index}, expected {self.physical_gpu}")
            self.samples.append(
                {
                    "unix": time.time(),
                    "physical_gpu": index,
                    "memory_mib": memory_mib,
                    "utilization_percent": utilization_percent,
                    "temperature_c": temperature_c,
                }
            )
            if self.snapshot_path is not None:
                _write_json_atomic(
                    self.snapshot_path,
                    {
                        "state": "running",
                        "updated_unix": time.time(),
                        "gpu": telemetry_summary(
                            physical_gpu=self.physical_gpu,
                            samples=self.samples,
                            errors=self.errors,
                        ),
                    },
                )
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
            self.errors.append(str(error))

    def _run(self) -> None:
        self._sample_once()
        while not self._stop.wait(self.interval_seconds):
            self._sample_once()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=self.interval_seconds + 1)
        return telemetry_summary(physical_gpu=self.physical_gpu, samples=self.samples, errors=self.errors)


def telemetry_summary(*, physical_gpu: int, samples: list[dict[str, Any]], errors: list[str]) -> dict[str, Any]:
    """Return a stable telemetry schema even when a short run has no sample."""
    peak_memory = max((sample["memory_mib"] for sample in samples), default=None)
    peak_utilization = max((sample["utilization_percent"] for sample in samples), default=None)
    peak_temperature = max((sample["temperature_c"] for sample in samples), default=None)
    return {
        "physical_gpu": physical_gpu,
        "samples": samples,
        "peak_memory_mib": peak_memory,
        "peak_utilization_percent": peak_utilization,
        "peak_temperature_c": peak_temperature,
        "errors": errors,
    }


def runtime_provenance(path: Path) -> dict[str, Any]:
    """Include the bootstrap's concrete import evidence in the final manifest."""
    if not path.is_file():
        return {"present": False}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SAADLaunchError(f"runtime provenance is unreadable: {error}") from error
    if not isinstance(value, dict):
        raise SAADLaunchError("runtime provenance is not a mapping")
    return {"present": True, "identity": _file_identity(path), "modules": value}


def verify_teacher_logit_evidence(config: SAADConfig, path: Path) -> dict[str, Any]:
    """Require measured cross-runtime evidence rather than a YAML assertion."""
    if not path.is_file():
        raise SAADLaunchError("teacher-logit evidence is absent")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SAADLaunchError(f"teacher-logit evidence is unreadable: {error}") from error
    if not isinstance(value, dict) or value.get("schema_version") != 1 or value.get("passed") is not True:
        raise SAADLaunchError("teacher-logit evidence is not a passing schema-v1 comparison")
    expected = {
        "teacher_name": config.teacher_name,
        "checkpoint_sha256": config.teacher_checkpoint_sha256,
        "contract": config.teacher_logit_contract,
        "probe_source_sha256": _sha256(ROOT / "scripts" / "saad_teacher_probe.py"),
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise SAADLaunchError(f"teacher-logit evidence identity mismatch: {key}")
    fixed_input = value.get("fixed_input")
    observed = value.get("observed")
    if (
        not isinstance(fixed_input, dict)
        or len(str(fixed_input.get("sha256", ""))) != 64
        or not isinstance(observed, dict)
        or not math.isfinite(float(observed.get("max_abs", math.nan)))
        or observed.get("argmax_equal") is not True
        or float(observed["max_abs"]) > float(config.teacher_logit_contract["atol"])
    ):
        raise SAADLaunchError("teacher-logit evidence has invalid measured values")
    return {"identity": _file_identity(path), "comparison": value}


def _read_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SAADLaunchError(f"{label} is unreadable: {error}") from error
    if not isinstance(value, dict):
        raise SAADLaunchError(f"{label} must be a JSON mapping")
    return value


def verify_smoke_evidence_bundle(
    *,
    config: SAADConfig,
    config_path: Path,
    inputs: dict[str, Any],
    teacher_evidence: dict[str, Any],
    manifest_paths: list[Path],
    saad: Path | None = None,
) -> dict[str, Any]:
    """Validate the two preregistered smokes before a heavy process exists."""
    if len(manifest_paths) != 2:
        raise SAADLaunchError("full execution requires exactly two smoke manifests")
    expected_sources = launch_source_files(config_path, config)
    expected_entrypoint = entrypoint_identity(config=config, saad=saad or (ROOT / ".external" / "saad"))
    expected_git = _git(ROOT, "rev-parse", "HEAD")
    if _git(ROOT, "status", "--porcelain"):
        raise SAADLaunchError("full execution requires a clean ARD Git tree")
    expected_quotas = {16: 2, 128: 10}
    observed: dict[int, dict[str, Any]] = {}
    runtime_identity: Any = None
    for raw_path in manifest_paths:
        path = raw_path.resolve()
        manifest = _read_json_mapping(path, label="smoke manifest")
        telemetry_path = path.parent / "telemetry.json"
        telemetry = _read_json_mapping(telemetry_path, label="smoke telemetry")
        args = manifest.get("upstream_args")
        if not isinstance(args, list) or "--batch" not in args:
            raise SAADLaunchError("smoke manifest has no batch identity")
        try:
            batch = int(args[args.index("--batch") + 1])
        except (ValueError, IndexError, TypeError) as error:
            raise SAADLaunchError("smoke manifest batch identity is malformed") from error
        if batch not in expected_quotas or batch in observed:
            raise SAADLaunchError("smoke manifests must contain unique batch 16 and 128 evidence")
        if args != upstream_args(config=config, batch_size=batch):
            raise SAADLaunchError(f"batch-{batch} smoke upstream command drift")
        terminal = manifest.get("terminal")
        if not isinstance(terminal, dict) or terminal.get("state") != "expected_smoke_termination":
            raise SAADLaunchError(f"batch-{batch} smoke did not pass")
        if (
            terminal.get("requested_smoke_loss_events") != expected_quotas[batch]
            or terminal.get("loss_events") != expected_quotas[batch]
        ):
            raise SAADLaunchError(f"batch-{batch} smoke loss quota mismatch")
        if terminal.get("invalid_loss_tokens") != []:
            raise SAADLaunchError(f"batch-{batch} smoke contains invalid losses")
        identity = manifest.get("launch_identity")
        if not isinstance(identity, dict) or identity.get("ard_git") != {"head": expected_git, "dirty": False}:
            raise SAADLaunchError(f"batch-{batch} smoke Git identity mismatch")
        if identity.get("source_files") != expected_sources:
            raise SAADLaunchError(f"batch-{batch} smoke source identity mismatch")
        if identity.get("source_variant") != config.source_variant:
            raise SAADLaunchError(f"batch-{batch} smoke source variant mismatch")
        if identity.get("entrypoint") != expected_entrypoint:
            raise SAADLaunchError(f"batch-{batch} smoke executed entrypoint mismatch")
        environment = manifest.get("environment")
        gpu = telemetry.get("gpu")
        if (
            manifest.get("physical_gpu") != config.physical_gpu
            or not isinstance(environment, dict)
            or environment.get("CUDA_VISIBLE_DEVICES") != str(config.physical_gpu)
            or not isinstance(gpu, dict)
            or gpu.get("physical_gpu") != config.physical_gpu
        ):
            raise SAADLaunchError(f"batch-{batch} smoke physical GPU identity mismatch")
        if manifest.get("inputs") != inputs:
            raise SAADLaunchError(f"batch-{batch} smoke input identity mismatch")
        if manifest.get("teacher_logit_evidence", {}).get("identity") != teacher_evidence["identity"]:
            raise SAADLaunchError(f"batch-{batch} teacher-logit evidence mismatch")
        if manifest.get("environment", {}).get("PYTORCH_CUDA_ALLOC_CONF") != config.pytorch_cuda_alloc_conf:
            raise SAADLaunchError(f"batch-{batch} allocator identity mismatch")
        provenance = manifest.get("runtime_provenance")
        if not isinstance(provenance, dict) or provenance.get("present") is not True:
            raise SAADLaunchError(f"batch-{batch} runtime provenance is absent")
        modules = provenance.get("modules")
        if isinstance(modules, dict):
            modules = {key: value for key, value in modules.items() if key != "saad_stage"}
        if runtime_identity is None:
            runtime_identity = modules
        elif modules != runtime_identity:
            raise SAADLaunchError("smoke runtime/import provenance differs across batches")
        gpu = telemetry.get("gpu")
        if not isinstance(gpu, dict) or gpu.get("errors") != []:
            raise SAADLaunchError(f"batch-{batch} GPU telemetry is invalid")
        peak = gpu.get("peak_memory_mib")
        temperature = gpu.get("peak_temperature_c")
        if not isinstance(peak, int) or not isinstance(temperature, int):
            raise SAADLaunchError(f"batch-{batch} GPU telemetry peaks are absent")
        if batch == 128 and peak > 22500:
            raise SAADLaunchError(f"batch-128 smoke peak memory {peak} MiB exceeds 22500 MiB")
        if temperature >= 80:
            raise SAADLaunchError(f"batch-{batch} smoke temperature is unsafe")
        observed[batch] = {
            "manifest": _file_identity(path),
            "telemetry": _file_identity(telemetry_path),
            "peak_memory_mib": peak,
            "peak_temperature_c": temperature,
        }
    if set(observed) != set(expected_quotas):
        raise SAADLaunchError("smoke evidence is incomplete")
    return {"batches": observed, "runtime_import_provenance": runtime_identity}


def supervise_smoke(
    command: list[str],
    *,
    cwd: Path,
    output_dir: Path,
    requested_events: int,
    physical_gpu: int,
    pytorch_cuda_alloc_conf: str,
) -> SmokeResult:
    """Stream immutable logs, then stop the complete upstream process group on loss quota."""
    stdout_path, stderr_path = output_dir / "stdout.log", output_dir / "stderr.log"
    losses: list[float] = []
    invalid_loss_tokens: list[str] = []
    lock = threading.Lock()

    def drain(stream: Any, destination: Path, *, observe: bool) -> None:
        text = ""
        with destination.open("xb") as handle:
            while True:
                chunk = read_available_pipe_bytes(stream)
                if not chunk:
                    break
                handle.write(chunk)
                if observe:
                    text, values, invalid = observe_loss_chunk(text, chunk)
                    if values or invalid:
                        with lock:
                            losses.extend(values)
                            invalid_loss_tokens.extend(invalid)

    sampler = GPUSampler(physical_gpu=physical_gpu)
    sampler.start()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        env=runtime_environment(physical_gpu=physical_gpu, pytorch_cuda_alloc_conf=pytorch_cuda_alloc_conf),
    )
    assert process.stdout is not None and process.stderr is not None
    workers = [
        threading.Thread(target=drain, args=(process.stdout, stdout_path), kwargs={"observe": True}, daemon=True),
        threading.Thread(target=drain, args=(process.stderr, stderr_path), kwargs={"observe": True}, daemon=True),
    ]
    for worker in workers:
        worker.start()
    terminated = False
    while process.poll() is None:
        with lock:
            seen = len(losses)
            saw_invalid_loss = bool(invalid_loss_tokens)
        if saw_invalid_loss or seen >= requested_events:
            os.killpg(process.pid, signal.SIGTERM)
            terminated = True
            break
        time.sleep(0.05)
    try:
        returncode = process.wait(timeout=30 if terminated else None)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        returncode = process.wait()
    for worker in workers:
        worker.join(timeout=5)
    with lock:
        count = len(losses)
        invalid_tokens = tuple(invalid_loss_tokens)
    state = classify_smoke(
        terminated_by_supervisor=terminated,
        loss_events=count,
        requested_events=requested_events,
        returncode=returncode,
        nonfinite_loss=bool(invalid_tokens),
    )
    return SmokeResult(
        state=state,
        loss_events=count,
        returncode=returncode,
        gpu_telemetry=sampler.stop(),
        invalid_loss_tokens=invalid_tokens,
    )


def select_execution(
    config: SAADConfig, *, mode: str, smoke_batch_size: int | None, output_dir: Path | None
) -> tuple[SAADConfig, int]:
    """Select only the preregistered smoke variants or the frozen full run."""
    if output_dir is None:
        raise SAADLaunchError("--output-dir is required for every executing upstream run")
    selected_output = output_dir if output_dir.is_absolute() else ROOT / output_dir
    if mode == "full":
        if smoke_batch_size is not None:
            raise SAADLaunchError("--smoke-batch-size is invalid in full mode")
        return replace(config, output_dir=selected_output), 128
    if smoke_batch_size not in {16, 128}:
        raise SAADLaunchError("smoke mode requires --smoke-batch-size 16 or 128")
    return replace(config, output_dir=selected_output), smoke_batch_size


def select_smoke_loss_events(config: SAADConfig, *, mode: str, execute: bool, smoke_loss_events: int | None) -> int:
    """Permit a bounded smoke quota override without mutating frozen YAML."""
    if smoke_loss_events is None:
        return config.smoke_loss_events
    if not execute:
        raise SAADLaunchError("--smoke-loss-events is execute-only")
    if mode != "smoke":
        raise SAADLaunchError("--smoke-loss-events is valid only in smoke mode")
    if not 2 <= smoke_loss_events <= 10:
        raise SAADLaunchError("--smoke-loss-events must be within [2, 10]")
    return smoke_loss_events


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--smoke-batch-size", choices=(16, 128), type=int)
    parser.add_argument("--smoke-loss-events", type=int)
    parser.add_argument("--teacher-logit-evidence", type=Path)
    parser.add_argument("--smoke-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute", action="store_true", help="execute only after printing/validating frozen identity")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if not args.execute and args.teacher_logit_evidence is not None:
        raise SAADLaunchError("--teacher-logit-evidence is execute-only")
    if args.execute and args.teacher_logit_evidence is None:
        raise SAADLaunchError("--teacher-logit-evidence is required for execution")
    if args.mode != "full" and args.smoke_manifest:
        raise SAADLaunchError("--smoke-manifest is valid only in full mode")
    if not args.execute and args.smoke_manifest:
        raise SAADLaunchError("--smoke-manifest is execute-only")
    requested_events = select_smoke_loss_events(
        config,
        mode=args.mode,
        execute=args.execute,
        smoke_loss_events=args.smoke_loss_events,
    )
    saad = verified_saad_clone(ROOT)
    robustbench = verified_checkout(ROOT, "robustbench", commit=ROBUSTBENCH_COMMIT)
    inputs = verify_inputs(config)
    teacher_logit_evidence = (
        verify_teacher_logit_evidence(config, args.teacher_logit_evidence.resolve())
        if args.teacher_logit_evidence is not None
        else None
    )
    heavy_gate_evidence = (
        verify_smoke_evidence_bundle(
            config=config,
            config_path=args.config.resolve(),
            inputs=inputs,
            teacher_evidence=teacher_logit_evidence,
            manifest_paths=args.smoke_manifest,
            saad=saad,
        )
        if args.execute and args.mode == "full" and teacher_logit_evidence is not None
        else None
    )
    if not args.execute:
        if args.mode == "full":
            if args.smoke_batch_size is not None:
                raise SAADLaunchError("--smoke-batch-size is invalid in full mode")
            batch = 128
        else:
            batch = args.smoke_batch_size or config.smoke_batch_size
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "command": upstream_args(config=config, batch_size=batch),
                    "inputs": inputs,
                    "requested_smoke_loss_events": requested_events if args.mode == "smoke" else None,
                },
                sort_keys=True,
            )
        )
        return 0
    config, batch = select_execution(
        config, mode=args.mode, smoke_batch_size=args.smoke_batch_size, output_dir=args.output_dir
    )
    stage = stage_inputs(config=config, saad=saad, output_dir=config.output_dir)
    executed_entrypoint = entrypoint_identity(config=config, saad=saad, staged_entrypoint=stage / "saad.py")
    provenance = config.output_dir / "runtime-provenance.json"
    command = build_runtime_command(
        config=config, stage=stage, saad=saad, robustbench=robustbench, provenance=provenance, batch_size=batch
    )
    identity = launch_identity(
        config_path=args.config.resolve(), command=command, config=config, entrypoint=executed_entrypoint
    )
    started = time.time()
    if args.mode == "full":
        _write_json_exclusive(
            config.output_dir / "launch-manifest.json",
            {
                "schema_version": 1,
                "state": "launched",
                "started_unix": started,
                "physical_gpu": config.physical_gpu,
                "command": command,
                "launch_identity": identity,
                "inputs": inputs,
                "teacher_logit_evidence": teacher_logit_evidence,
                "heavy_gate_evidence": heavy_gate_evidence,
                "limitations": {
                    "upstream_resume": False,
                    "upstream_best_last": False,
                    "upstream_test_each_epoch": True,
                    "upstream_final_swa_only": True,
                    "upstream_autoattack_in_process": True,
                },
            },
        )
    if args.mode == "smoke":
        result = supervise_smoke(
            command,
            cwd=stage,
            output_dir=config.output_dir,
            requested_events=requested_events,
            physical_gpu=config.physical_gpu,
            pytorch_cuda_alloc_conf=config.pytorch_cuda_alloc_conf,
        )
        terminal = {
            "state": result.state,
            "loss_events": result.loss_events,
            "requested_smoke_loss_events": requested_events,
            "invalid_loss_tokens": list(result.invalid_loss_tokens),
            "returncode": result.returncode,
        }
        gpu_telemetry = result.gpu_telemetry
    else:
        stdout = config.output_dir / "stdout.log"
        stderr = config.output_dir / "stderr.log"
        sampler = GPUSampler(
            physical_gpu=config.physical_gpu,
            interval_seconds=5.0,
            snapshot_path=config.output_dir / "live-telemetry.json",
        )
        sampler.start()
        with stdout.open("xb") as out, stderr.open("xb") as err:
            completed = subprocess.run(
                command,
                cwd=stage,
                stdout=out,
                stderr=err,
                env=runtime_environment(
                    physical_gpu=config.physical_gpu,
                    pytorch_cuda_alloc_conf=config.pytorch_cuda_alloc_conf,
                ),
                start_new_session=True,
            )
        gpu_telemetry = sampler.stop()
        terminal = {
            "state": "completed" if completed.returncode == 0 else "upstream_failure",
            "returncode": completed.returncode,
        }
    _write_json_exclusive(
        config.output_dir / "manifest.json",
        {
            "schema_version": 1,
            "mode": args.mode,
            "physical_gpu": config.physical_gpu,
            "upstream": {"saad_commit": SAAD_COMMIT, "robustbench_commit": ROBUSTBENCH_COMMIT},
            "source_variant": config.source_variant,
            "inputs": inputs,
            "teacher_logit_contract": config.teacher_logit_contract,
            "teacher_logit_evidence": teacher_logit_evidence,
            "heavy_gate_evidence": heavy_gate_evidence,
            "command": command,
            "launch_identity": identity,
            "upstream_args": upstream_args(config=config, batch_size=batch),
            "environment": {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
                "PYTHONNOUSERSITE": "1",
                "WANDB_MODE": "disabled",
                "PYTHONPATH": None,
                "CUDA_VISIBLE_DEVICES": str(config.physical_gpu),
                "PYTORCH_CUDA_ALLOC_CONF": config.pytorch_cuda_alloc_conf,
            },
            "runtime_provenance": runtime_provenance(provenance),
            "started_unix": started,
            "finished_unix": time.time(),
            "terminal": terminal,
            "limitations": {
                "upstream_resume": False,
                "upstream_best_last": False,
                "upstream_test_each_epoch": True,
                "final_autoattack_failure_is_training_success": False,
            },
            "runtime_decisions": {
                "rejected_python_3_8_20": (
                    "pinned RobustBench import fails under Python 3.8 because "
                    "robustarch_wide_resnet.py uses list[Callable]"
                ),
                "selected_python": "3.11.15",
            },
        },
    )
    _write_json_exclusive(config.output_dir / "telemetry.json", {"mode": args.mode, "gpu": gpu_telemetry, **terminal})
    if args.mode == "full":
        terminal_files = {
            name: _file_identity(config.output_dir / name)
            for name in ("stdout.log", "stderr.log", "telemetry.json", "runtime-provenance.json")
            if (config.output_dir / name).is_file()
        }
        _write_json_exclusive(
            config.output_dir / "terminal.json",
            {
                "schema_version": 1,
                "finished_unix": time.time(),
                "terminal": terminal,
                "launch_manifest": _file_identity(config.output_dir / "launch-manifest.json"),
                "files": terminal_files,
            },
        )
    print(json.dumps(terminal, sort_keys=True))
    return 0 if terminal["state"] in {"expected_smoke_termination", "completed"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
