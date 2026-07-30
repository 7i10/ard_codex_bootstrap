"""Explicit, provenance-checked AutoAttack adapter.

Full AutoAttack is deliberately outside the automated test suite.  The
adapter still records enough package identity for a saved result to be audited
without treating an arbitrary installed ``autoattack`` module as equivalent.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import random
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch
from torch import nn

EXPECTED_AUTOATTACK_COMMIT = "a39220048b3c9f2cca9a4d3a54604793c68eca7e"
# SHA-256 over sorted ``relative_path + NUL + bytes + NUL`` for all 13
# package-local ``*.py`` files at EXPECTED_AUTOATTACK_COMMIT.
EXPECTED_AUTOATTACK_SOURCE_SHA256 = "e74d6dab0e34faf840f1bdfe0f77e9ddcc5f753a7426cbaa54b11bf17f896487"


class AutoAttackUnavailable(RuntimeError):
    pass


class AutoAttackProvenanceError(RuntimeError):
    """The installed package cannot prove the pinned AutoAttack source."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_python_sources(package_root: Path) -> tuple[tuple[str, Path], ...]:
    if not package_root.is_dir():
        raise AutoAttackProvenanceError("AutoAttack package root is absent")
    sources = tuple(
        (path.relative_to(package_root).as_posix(), path)
        for path in sorted(package_root.rglob("*.py"))
        if path.is_file()
    )
    if not sources:
        raise AutoAttackProvenanceError("AutoAttack package contains no Python source")
    return sources


def _source_digest(sources: Iterable[tuple[str, Path]]) -> tuple[str, list[str]]:
    """Hash path/byte pairs, independent of the host install prefix."""
    digest = hashlib.sha256()
    relative_paths: list[str] = []
    for relative, path in sources:
        if not relative or relative.startswith("/") or ".." in Path(relative).parts:
            raise AutoAttackProvenanceError("AutoAttack source path is not safely relative")
        relative_paths.append(relative)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest(), relative_paths


def _direct_url(distribution: importlib.metadata.Distribution) -> dict[str, Any] | None:
    direct_url_path = Path(distribution.locate_file("direct_url.json"))
    if not direct_url_path.is_file():
        # Wheels conventionally put this alongside METADATA rather than at
        # site-packages root; ``files`` is portable across metadata backends.
        for item in distribution.files or ():
            if item.name == "direct_url.json":
                candidate = Path(distribution.locate_file(item))
                if candidate.is_file():
                    direct_url_path = candidate
                    break
        else:
            return None
    try:
        value = json.loads(direct_url_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AutoAttackProvenanceError("AutoAttack direct_url metadata is unreadable") from exc
    if not isinstance(value, dict):
        raise AutoAttackProvenanceError("AutoAttack direct_url metadata must be an object")
    return value


def _license_digests(distribution: importlib.metadata.Distribution) -> list[dict[str, str]]:
    candidates: list[tuple[str, Path]] = []
    for item in distribution.files or ():
        normalized = item.as_posix()
        name = item.name.lower()
        if name.startswith("license") or normalized.lower().endswith("/license") or "/licenses/" in normalized.lower():
            path = Path(distribution.locate_file(item))
            if path.is_file():
                candidates.append((normalized, path))
    return [
        {"path": relative, "sha256": _sha256_file(path)}
        for relative, path in sorted(candidates, key=lambda item: item[0])
    ]


def autoattack_provenance(
    module: Any,
    *,
    distribution: importlib.metadata.Distribution | None = None,
    expected_commit: str = EXPECTED_AUTOATTACK_COMMIT,
    expected_source_sha256: str = EXPECTED_AUTOATTACK_SOURCE_SHA256,
) -> dict[str, Any]:
    """Return and validate identity for a real installed AutoAttack package.

    VCS direct-url data is preferred when present.  A package manager may omit
    it, so the pinned deterministic source digest remains mandatory in every
    case.  A mismatched or absent proof fails before an evaluation begins.
    """
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str):
        raise AutoAttackProvenanceError("AutoAttack module has no source path")
    package_root = Path(module_file).resolve().parent
    source_sha256, source_paths = _source_digest(_relative_python_sources(package_root))
    if source_sha256 != expected_source_sha256:
        raise AutoAttackProvenanceError("AutoAttack Python source digest does not match the pinned commit")

    try:
        dist = distribution or importlib.metadata.distribution("autoattack")
    except importlib.metadata.PackageNotFoundError as exc:
        raise AutoAttackProvenanceError("AutoAttack distribution metadata is unavailable") from exc
    metadata_name = dist.metadata.get("Name")
    metadata_version = dist.version
    if (
        not isinstance(metadata_name, str)
        or not metadata_name
        or not isinstance(metadata_version, str)
        or not metadata_version
    ):
        raise AutoAttackProvenanceError("AutoAttack distribution metadata is incomplete")
    direct_url = _direct_url(dist)
    vcs_commit: str | None = None
    if direct_url is not None:
        vcs_info = direct_url.get("vcs_info")
        if not isinstance(vcs_info, dict) or vcs_info.get("vcs") != "git":
            raise AutoAttackProvenanceError("AutoAttack direct_url is not a Git VCS record")
        commit = vcs_info.get("commit_id")
        if not isinstance(commit, str) or commit != expected_commit:
            raise AutoAttackProvenanceError("AutoAttack direct_url commit does not match the pinned commit")
        vcs_commit = commit

    return {
        "mode": "installed",
        "distribution": {"name": metadata_name, "version": metadata_version},
        "direct_url": direct_url,
        "vcs_commit": vcs_commit,
        "expected_commit": expected_commit,
        "source_sha256": source_sha256,
        "source_paths": source_paths,
        "licenses": _license_digests(dist),
    }


def _injected_test_provenance(adapter_class: Callable[..., Any]) -> dict[str, Any]:
    return {
        "mode": "injected-test-adapter",
        "adapter": f"{adapter_class.__module__}.{adapter_class.__qualname__}",
        "not_production_provenance": True,
    }


def run_autoattack(
    *,
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    norm: str,
    epsilon: float,
    output_path: Path,
    seed: int,
    batch_size: int = 128,
    autoattack_cls: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run only when the standalone evaluation CLI explicitly requests it."""
    if norm != "linf":
        raise ValueError("AutoAttack adapter supports the validated Linf threat model only")
    if autoattack_cls is None:
        try:
            import autoattack
        except ImportError as exc:
            raise AutoAttackUnavailable("AutoAttack is optional; install it for a separate evaluation process") from exc
        provenance = autoattack_provenance(autoattack)
        version = provenance["distribution"]["version"]
        adapter_class: Callable[..., Any] = autoattack.AutoAttack
    else:
        provenance = _injected_test_provenance(autoattack_cls)
        version = "injected"
        adapter_class = autoattack_cls
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    was_training = model.training
    model.eval()
    try:
        adversary: Any = adapter_class(model, norm="Linf", eps=epsilon, version="standard", device=images.device)
        adversary.seed = seed
        adversarial = adversary.run_standard_evaluation(images, labels, bs=batch_size)
        with torch.no_grad():
            accuracy = model(adversarial).argmax(1).eq(labels).float().mean().item()
    finally:
        model.train(was_training)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "autoattack_accuracy": float(accuracy),
        "seed": seed,
        "norm": "Linf",
        "epsilon": epsilon,
        "attack_version": "standard",
        "version": version,
        "batch_size": batch_size,
        "provenance": provenance,
    }
    output_path.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    return result
