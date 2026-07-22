#!/usr/bin/env python3
"""Explicitly acquire one allowlisted RobustBench checkpoint into an external cache.

This operator tool is deliberately separate from the ARD runtime.  It is the
only project path that permits the pinned RobustBench ``load_model`` downloader
to run; training and evaluation continue to require a hash-registered local
teacher cache.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ard.models.teacher_registry import (  # noqa: E402
    ROBUSTBENCH_REPOSITORY,
    TeacherRegistry,
    TeacherRegistryError,
    TeacherSpec,
    sha256_file,
)

ALLOWED_REGISTRY_IDS = frozenset({"chen2021_ltd_wrn34_10", "bartoldson2024_adversarial_wrn94_16"})
DEFAULT_MODEL_DIR = ROOT.parent / "datasets" / "ard" / "teachers" / "robustbench"


class AcquisitionError(RuntimeError):
    """An acquisition request or staged RobustBench download is unsafe."""


@dataclass(frozen=True)
class AcquisitionReport:
    registry_id: str
    upstream_model_id: str
    checkpoint: str
    checkpoint_sha256: str
    bytes: int
    parameter_count: int
    logits_shape: tuple[int, int]
    device: str


LoadModel = Callable[..., nn.Module]
RegistryLoader = Callable[[Path], TeacherRegistry]
LoadModelResolver = Callable[[Path, bool], LoadModel]


def expected_checkpoint_path(model_dir: Path, spec: TeacherSpec) -> Path:
    """Return the exact lowercase RobustBench path used by its pinned loader."""
    return model_dir / "cifar10" / "Linf" / f"{spec.upstream_model_id}.pt"


def acquire(
    *,
    root: Path,
    registry_id: str,
    model_dir: Path,
    allow_network: bool,
    device: torch.device = torch.device("cpu"),
    registry_loader: RegistryLoader = TeacherRegistry.load,
    load_model_resolver: LoadModelResolver | None = None,
) -> AcquisitionReport:
    """Download, validate, and atomically publish one known teacher checkpoint.

    The temporary model directory is a sibling of ``model_dir`` so a hard-link
    publication is necessarily same-filesystem and cannot replace an existing
    final path.  Any exception removes only this invocation's staging tree.
    """
    if not allow_network:
        raise AcquisitionError("network acquisition requires explicit --allow-network")
    if registry_id not in ALLOWED_REGISTRY_IDS:
        raise AcquisitionError(f"registry ID is not acquisition-allowlisted: {registry_id!r}")
    selected_root = root.resolve()
    registry = registry_loader(selected_root)
    try:
        spec = registry.spec(registry_id)
        registry.validate_external()
    except TeacherRegistryError as exc:
        raise AcquisitionError(str(exc)) from exc
    if spec.registry_id not in ALLOWED_REGISTRY_IDS:
        raise AcquisitionError(f"registry entry is not acquisition-allowlisted: {spec.registry_id!r}")
    expected_locked_sha = _locked_checkpoint_sha(spec)

    destination_root = model_dir.resolve()
    destination = expected_checkpoint_path(destination_root, spec)
    if destination.exists():
        raise AcquisitionError(f"refusing to overwrite existing RobustBench checkpoint: {destination}")
    destination_root.parent.mkdir(parents=True, exist_ok=True)
    resolver = load_model_resolver or _pinned_load_model
    loader = resolver(selected_root, False)
    stage_root = Path(tempfile.mkdtemp(prefix=".robustbench-stage-", dir=destination_root.parent))
    try:
        model = loader(
            model_name=spec.upstream_model_id,
            model_dir=stage_root,
            dataset="cifar10",
            threat_model="Linf",
        )
        parameter_count, logits_shape = _validate_loaded_model(model, spec, device)
        staged = expected_checkpoint_path(stage_root, spec)
        if not staged.is_file():
            raise AcquisitionError(f"pinned RobustBench loader did not create the expected checkpoint: {staged}")
        _fsync_file(staged)
        checkpoint_sha256 = sha256_file(staged)
        checkpoint_bytes = staged.stat().st_size
        if expected_locked_sha is not None and checkpoint_sha256 != expected_locked_sha:
            raise AcquisitionError(
                f"teacher checkpoint hash mismatch: expected {expected_locked_sha}, got {checkpoint_sha256}"
            )
        published = _publish_no_clobber(staged, destination)
        return AcquisitionReport(
            registry_id=spec.registry_id,
            upstream_model_id=spec.upstream_model_id,
            checkpoint=str(published),
            checkpoint_sha256=checkpoint_sha256,
            bytes=checkpoint_bytes,
            parameter_count=parameter_count,
            logits_shape=logits_shape,
            device=str(device),
        )
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def _locked_checkpoint_sha(spec: TeacherSpec) -> str | None:
    """Validate lock state before acquiring; verified entries cannot drift."""
    status = getattr(spec, "checkpoint_status", "missing")
    digest = getattr(spec, "checkpoint_sha256", None)
    if status == "missing" and digest is None:
        return None
    if (
        status == "verified"
        and isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    ):
        return digest
    raise AcquisitionError("teacher lock checkpoint identity is inconsistent; refusing acquisition")


def _validate_loaded_model(model: nn.Module, spec: TeacherSpec, device: torch.device) -> tuple[int, tuple[int, int]]:
    if not isinstance(model, nn.Module):
        raise AcquisitionError("pinned RobustBench load_model did not return a torch.nn.Module")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != spec.expected_parameter_count:
        raise AcquisitionError(
            f"teacher parameter count mismatch: expected {spec.expected_parameter_count}, got {parameter_count}"
        )
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(_fixed_pixels(device))
    if tuple(logits.shape) != (1, 10):
        raise AcquisitionError(f"teacher logits must have shape (1, 10), got {tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        raise AcquisitionError("teacher logits are non-finite")
    return parameter_count, (1, 10)


def _publish_no_clobber(staged: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if staged.stat().st_dev != destination.parent.stat().st_dev:
        raise AcquisitionError("staging and final checkpoint directory are not on the same filesystem")
    try:
        os.link(staged, destination)
    except FileExistsError as exc:
        raise AcquisitionError(f"refusing to overwrite existing RobustBench checkpoint: {destination}") from exc
    try:
        _fsync_directory(destination.parent)
    except Exception:
        _remove_our_published_checkpoint(destination, staged)
        raise
    return destination


def _remove_our_published_checkpoint(destination: Path, staged: Path) -> None:
    """Remove only the hard link made by this invocation after a post-link failure."""
    try:
        destination_stat = destination.stat()
        staged_stat = staged.stat()
    except FileNotFoundError:
        return
    if (destination_stat.st_dev, destination_stat.st_ino) == (staged_stat.st_dev, staged_stat.st_ino):
        destination.unlink()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fixed_pixels(device: torch.device) -> torch.Tensor:
    return torch.linspace(0.0, 1.0, steps=3 * 32 * 32, dtype=torch.float32, device=device).reshape(1, 3, 32, 32)


def _pinned_load_model(root: Path, local_only: bool) -> LoadModel:
    """Resolve ``load_model`` only from the verified external checkout."""
    checkout = (root / ".external" / ROBUSTBENCH_REPOSITORY).resolve()
    _reject_unverified_preloaded_modules(checkout)
    checkout_string = str(checkout)
    if checkout_string in sys.path:
        sys.path.remove(checkout_string)
    sys.path.insert(0, checkout_string)
    module = importlib.import_module("robustbench.utils")
    if not _module_is_under(module, checkout):
        raise AcquisitionError("pinned RobustBench utils module has unverifiable provenance")
    loader = getattr(module, "load_model", None)
    if not callable(loader):
        raise AcquisitionError("pinned RobustBench utils.load_model is unavailable")
    try:
        source = Path(inspect.getfile(loader)).resolve()
    except (OSError, TypeError) as exc:
        raise AcquisitionError("pinned RobustBench load_model has unverifiable provenance") from exc
    if not _path_is_under(source, checkout):
        raise AcquisitionError("pinned RobustBench load_model is not defined under the verified checkout")
    if local_only:
        _disable_robustbench_downloads(module)
    return loader


def _disable_robustbench_downloads(module: ModuleType) -> None:
    def reject_download(*_args: object, **_kwargs: object) -> None:
        raise AcquisitionError("local-only audit refuses a RobustBench download")

    setattr(module, "download_gdrive", reject_download)
    setattr(module, "download_gdrive_new", reject_download)
    timm = getattr(module, "timm", None)
    if timm is None or not hasattr(timm, "create_model"):
        raise AcquisitionError("pinned RobustBench utils has no verifiable timm.create_model route")
    setattr(timm, "create_model", reject_download)


def _reject_unverified_preloaded_modules(checkout: Path) -> None:
    for name, module in tuple(sys.modules.items()):
        if name != ROBUSTBENCH_REPOSITORY and not name.startswith(f"{ROBUSTBENCH_REPOSITORY}."):
            continue
        if module is None or not _module_is_under(module, checkout):
            raise AcquisitionError(f"preloaded RobustBench module has unverifiable provenance: {name}")


def _module_is_under(module: object, checkout: Path) -> bool:
    location = getattr(module, "__file__", None)
    return isinstance(location, str) and _path_is_under(Path(location).resolve(), checkout)


def _path_is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--registry-id", required=True, choices=sorted(ALLOWED_REGISTRY_IDS))
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", default="cpu", help="validation device; use cuda:0 only for the one-teacher audit")
    parser.add_argument("--allow-network", action="store_true", help="required opt-in for the pinned downloader")
    args = parser.parse_args()
    try:
        report = acquire(
            root=args.root,
            registry_id=args.registry_id,
            model_dir=args.model_dir,
            allow_network=args.allow_network,
            device=torch.device(args.device),
        )
    except AcquisitionError as exc:
        parser.error(str(exc))
    print(json.dumps(asdict(report), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
