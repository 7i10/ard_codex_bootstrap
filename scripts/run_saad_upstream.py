#!/usr/bin/env python3
"""Run the pinned full-SAAD oracle without importing it into ARD.

This launcher is intentionally an operational boundary, not a second training
engine.  It only accepts the frozen Bartoldson/SAAD protocol, stages symlinks
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
LOSS_TOKEN_RE = re.compile(r"loss:\s*([^\s|]+)(?=[\s|])")


class SAADLaunchError(RuntimeError):
    """The isolated upstream launcher contract was not satisfied."""


@dataclass(frozen=True)
class SAADConfig:
    runtime_python: Path
    dataset_root: Path
    teacher_checkpoint: Path
    output_dir: Path
    smoke_batch_size: int
    smoke_loss_events: int
    physical_gpu: int
    teacher_logit_contract: dict[str, Any]


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


def launch_identity(*, config_path: Path, command: list[str]) -> dict[str, Any]:
    """Hash every local input that changes an upstream launch's semantics."""
    source_files = {
        "launcher": _file_identity(Path(__file__)),
        "runtime_bootstrap": _file_identity(ROOT / "scripts" / "saad_runtime_bootstrap.py"),
        "config": _file_identity(config_path),
        "runtime_lock": _file_identity(ROOT / "requirements" / "saad-upstream-runtime.lock"),
    }
    return {
        "ard_git": {"head": _git(ROOT, "rev-parse", "HEAD"), "dirty": bool(_git(ROOT, "status", "--porcelain"))},
        "command_sha256": _canonical_sha256(command),
        "source_files": source_files,
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
    root = _checked_mapping(
        raw,
        context="config",
        keys={"version", "runtime", "inputs", "output", "smoke", "gpu", "protocol", "teacher_logit_contract"},
    )
    if root["version"] != 1:
        raise SAADLaunchError("config.version must be exactly 1")
    runtime = _checked_mapping(root["runtime"], context="runtime", keys={"python", "python_version"})
    inputs = _checked_mapping(
        root["inputs"],
        context="inputs",
        keys={"dataset_root", "cifar10_archive_sha256", "teacher_checkpoint", "teacher_checkpoint_sha256"},
    )
    output = _checked_mapping(root["output"], context="output", keys={"directory"})
    smoke = _checked_mapping(root["smoke"], context="smoke", keys={"batch_size", "loss_events"})
    gpu = _checked_mapping(root["gpu"], context="gpu", keys={"physical_id"})
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
            "measured_max_abs",
            "measured_mean_abs",
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
        "teacher_name": "Bartoldson2024Adversarial_WRN-94-16",
        "dataset": "cifar10",
        "swa_epoch": 95,
        "beta": 0,
        "gamma": 1,
        "igdm_alpha": 1,
        "lambda_inner": 1,
        "nowand": 1,
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 0.0002,
        "milestones": [100, 150],
        "inner_steps": 10,
        "epsilon": "8/255",
        "step_size": "2/255",
        "entropy_multiplier": 5,
    }
    if protocol != frozen:
        changed = {
            key: {"expected": frozen[key], "actual": protocol.get(key)}
            for key in frozen
            if protocol.get(key) != frozen[key]
        }
        raise SAADLaunchError(f"protocol drift is forbidden: {json.dumps(changed, sort_keys=True)}")
    if runtime["python_version"] != "3.11.15":
        raise SAADLaunchError("runtime.python_version must be 3.11.15")
    if inputs["cifar10_archive_sha256"] != CIFAR10_ARCHIVE_SHA256:
        raise SAADLaunchError("CIFAR-10 archive hash differs from frozen input")
    if inputs["teacher_checkpoint_sha256"] != BARTOLDSON_SHA256:
        raise SAADLaunchError("Bartoldson checkpoint hash differs from frozen input")
    if not isinstance(smoke["batch_size"], int) or smoke["batch_size"] not in {16, 128}:
        raise SAADLaunchError("smoke.batch_size must be 16 or 128")
    if not isinstance(smoke["loss_events"], int) or not 2 <= smoke["loss_events"] <= 10:
        raise SAADLaunchError("smoke.loss_events must be within [2, 10]")
    if gpu["physical_id"] != 0:
        raise SAADLaunchError("gpu.physical_id must be frozen Hamster GPU 0")
    expected_logit_contract = {
        "reference_torch": "2.11.0+cu128",
        "candidate_torch": "2.4.1+cu121",
        "fixed_input_count": 4,
        "atol": 0.0001,
        "rtol": 0,
        "require_argmax_equal": True,
        "measured_max_abs": 0.000080824,
        "measured_mean_abs": 0.000030991,
    }
    if teacher_logit_contract != expected_logit_contract:
        raise SAADLaunchError("teacher_logit_contract differs from the frozen cross-runtime contract")

    def project_path(value: Any, *, field: str) -> Path:
        if not isinstance(value, str) or not value:
            raise SAADLaunchError(f"{field} must be a non-empty path string")
        candidate = Path(value)
        return candidate if candidate.is_absolute() else ROOT / candidate

    return SAADConfig(
        runtime_python=project_path(runtime["python"], field="runtime.python"),
        dataset_root=project_path(inputs["dataset_root"], field="inputs.dataset_root"),
        teacher_checkpoint=project_path(inputs["teacher_checkpoint"], field="inputs.teacher_checkpoint"),
        output_dir=project_path(output["directory"], field="output.directory"),
        smoke_batch_size=smoke["batch_size"],
        smoke_loss_events=smoke["loss_events"],
        physical_gpu=gpu["physical_id"],
        teacher_logit_contract=teacher_logit_contract,
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
    if not config.teacher_checkpoint.is_file() or _sha256(config.teacher_checkpoint) != BARTOLDSON_SHA256:
        raise SAADLaunchError("local Bartoldson checkpoint hash differs from frozen input")
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
        _safe_symlink(stage / source.name, source, stage_root=stage)
    _safe_symlink(output_dir / "dataset", config.dataset_root, stage_root=output_dir)
    model_dir = stage / "models" / "cifar10" / "Linf"
    model_dir.mkdir(parents=True)
    _safe_symlink(model_dir / "Bartoldson2024Adversarial_WRN-94-16.pt", config.teacher_checkpoint, stage_root=stage)
    return stage


def upstream_args(*, batch_size: int) -> list[str]:
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
        "0.0002",
        "--teacher_name",
        "Bartoldson2024Adversarial_WRN-94-16",
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
        "full-saad-bartoldson-seed0",
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
        *upstream_args(batch_size=batch_size),
    ]


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")


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


def runtime_environment(*, physical_gpu: int) -> dict[str, str]:
    """Prevent source-tree bytecode writes and ambient user-site imports."""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(physical_gpu),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONNOUSERSITE": "1",
            "WANDB_MODE": "disabled",
        }
    )
    return environment


class GPUSampler:
    """Best-effort physical-GPU telemetry, independent of CUDA device remapping."""

    def __init__(self, *, physical_gpu: int, interval_seconds: float = 5.0) -> None:
        self.physical_gpu = physical_gpu
        self.interval_seconds = interval_seconds
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
        peak_memory = max((sample["memory_mib"] for sample in self.samples), default=None)
        peak_utilization = max((sample["utilization_percent"] for sample in self.samples), default=None)
        peak_temperature = max((sample["temperature_c"] for sample in self.samples), default=None)
        return {
            "physical_gpu": self.physical_gpu,
            "samples": self.samples,
            "peak_memory_mib": peak_memory,
            "peak_utilization_percent": peak_utilization,
            "peak_temperature_c": peak_temperature,
            "errors": self.errors,
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


def supervise_smoke(
    command: list[str], *, cwd: Path, output_dir: Path, requested_events: int, physical_gpu: int
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
                chunk = stream.read(4096)
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
        env=runtime_environment(physical_gpu=physical_gpu),
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--smoke-batch-size", choices=(16, 128), type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute", action="store_true", help="execute only after printing/validating frozen identity")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    saad = verified_saad_clone(ROOT)
    robustbench = verified_checkout(ROOT, "robustbench", commit=ROBUSTBENCH_COMMIT)
    inputs = verify_inputs(config)
    if not args.execute:
        if args.mode == "full":
            if args.smoke_batch_size is not None:
                raise SAADLaunchError("--smoke-batch-size is invalid in full mode")
            batch = 128
        else:
            batch = args.smoke_batch_size or config.smoke_batch_size
        print(
            json.dumps(
                {"mode": args.mode, "command": upstream_args(batch_size=batch), "inputs": inputs}, sort_keys=True
            )
        )
        return 0
    config, batch = select_execution(
        config, mode=args.mode, smoke_batch_size=args.smoke_batch_size, output_dir=args.output_dir
    )
    stage = stage_inputs(config=config, saad=saad, output_dir=config.output_dir)
    provenance = config.output_dir / "runtime-provenance.json"
    command = build_runtime_command(
        config=config, stage=stage, saad=saad, robustbench=robustbench, provenance=provenance, batch_size=batch
    )
    identity = launch_identity(config_path=args.config.resolve(), command=command)
    started = time.time()
    if args.mode == "smoke":
        result = supervise_smoke(
            command,
            cwd=stage,
            output_dir=config.output_dir,
            requested_events=config.smoke_loss_events,
            physical_gpu=config.physical_gpu,
        )
        terminal = {
            "state": result.state,
            "loss_events": result.loss_events,
            "invalid_loss_tokens": list(result.invalid_loss_tokens),
            "returncode": result.returncode,
        }
        gpu_telemetry = result.gpu_telemetry
    else:
        stdout = config.output_dir / "stdout.log"
        stderr = config.output_dir / "stderr.log"
        sampler = GPUSampler(physical_gpu=config.physical_gpu)
        sampler.start()
        with stdout.open("xb") as out, stderr.open("xb") as err:
            completed = subprocess.run(
                command, cwd=stage, stdout=out, stderr=err, env=runtime_environment(physical_gpu=config.physical_gpu)
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
            "inputs": inputs,
            "teacher_logit_contract": config.teacher_logit_contract,
            "command": command,
            "launch_identity": identity,
            "upstream_args": upstream_args(batch_size=batch),
            "environment": {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
                "PYTHONNOUSERSITE": "1",
                "WANDB_MODE": "disabled",
                "PYTHONPATH": None,
                "CUDA_VISIBLE_DEVICES": str(config.physical_gpu),
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
    print(json.dumps(terminal, sort_keys=True))
    return 0 if terminal["state"] in {"expected_smoke_termination", "completed"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
