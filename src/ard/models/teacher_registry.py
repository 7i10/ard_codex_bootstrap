"""Pinned, local-only RobustBench teacher specifications and checkpoint contracts."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
import yaml
from torch import nn

from ard.config.schema import NormalizationConfig, TeacherConfig

SHA256_LENGTH = 64
ROBUSTBENCH_REPOSITORY = "robustbench"
NormalizationProfile = Literal[
    "fixture_unit",
    "cifar10_raw_identity",
    "cifar10_standard",
    "robustbench_cifar10_bartoldson_embedded",
    "cifar100_standard",
    "tiny_imagenet_standard",
    "custom",
]


class TeacherRegistryError(RuntimeError):
    """A pinned teacher, checkout, or local checkpoint violates its contract."""


@dataclass(frozen=True)
class TeacherThreat:
    norm: str
    epsilon: str
    input_domain: str


@dataclass(frozen=True)
class TeacherPreprocessing:
    owner: str
    profile: NormalizationProfile
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    provenance: str

    def normalization(self) -> NormalizationConfig:
        if self.profile != "custom":
            # Named ARD profiles own their canonical config provenance.  The
            # upstream-specific provenance remains immutable in this lock.
            return NormalizationConfig(profile=self.profile)
        return NormalizationConfig(
            profile=self.profile,
            mean=self.mean,
            std=self.std,
            provenance=self.provenance,
        )


@dataclass(frozen=True)
class FactorySpec:
    module: str
    symbol: str
    kwargs: Mapping[str, Any]


@dataclass(frozen=True)
class TeacherSpec:
    registry_id: str
    upstream_model_id: str
    architecture: str
    factory: FactorySpec
    dataset: str
    threat: TeacherThreat
    preprocessing: TeacherPreprocessing
    expected_parameter_count: int
    upstream_locator: str
    checkpoint_filename: str
    checkpoint_path: Path
    checkpoint_sha256: str | None
    checkpoint_status: str


@dataclass(frozen=True)
class TeacherRegistry:
    root: Path
    repository_url: str
    repository_commit: str
    repository_license_file: str
    repository_license_sha256: str
    specs: Mapping[str, TeacherSpec]

    @classmethod
    def load(cls, root: Path | None = None) -> TeacherRegistry:
        selected_root = (root or Path(__file__).resolve().parents[3]).resolve()
        lock_path = selected_root / "teachers.lock.yaml"
        try:
            raw = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise TeacherRegistryError(f"teacher lock is missing: {lock_path}") from exc
        except yaml.YAMLError as exc:
            raise TeacherRegistryError(f"teacher lock is invalid YAML: {exc}") from exc
        if not isinstance(raw, dict) or raw.get("version") != 1:
            raise TeacherRegistryError("teacher lock must have version: 1")
        repository = raw.get("repository")
        teachers = raw.get("teachers")
        if not isinstance(repository, dict) or not isinstance(teachers, dict) or not teachers:
            raise TeacherRegistryError("teacher lock requires repository and non-empty teachers mappings")
        url, commit = repository.get("url"), repository.get("commit")
        evidence = repository.get("license_evidence")
        root_license = evidence.get("repository_code_and_weights") if isinstance(evidence, dict) else None
        if (
            repository.get("name") != ROBUSTBENCH_REPOSITORY
            or not isinstance(url, str)
            or not isinstance(commit, str)
            or len(commit) != 40
            or any(char not in "0123456789abcdef" for char in commit)
            or not isinstance(root_license, dict)
            or root_license.get("status") != "verified"
            or not isinstance(root_license.get("file"), str)
            or not _valid_sha256(root_license.get("sha256"))
        ):
            raise TeacherRegistryError("teacher lock has invalid pinned RobustBench repository/license evidence")
        specs = {registry_id: _parse_spec(registry_id, entry) for registry_id, entry in teachers.items()}
        if set(specs) != {"chen2021_ltd_wrn34_10", "bartoldson2024_adversarial_wrn94_16"}:
            raise TeacherRegistryError("teacher lock must contain exactly the approved RobustBench registry IDs")
        return cls(
            root=selected_root,
            repository_url=url,
            repository_commit=commit,
            repository_license_file=root_license["file"],
            repository_license_sha256=root_license["sha256"],
            specs=specs,
        )

    def spec(self, registry_id: str) -> TeacherSpec:
        try:
            return self.specs[registry_id]
        except KeyError as exc:
            raise TeacherRegistryError(f"unknown teacher registry ID: {registry_id!r}") from exc

    def checkpoint_path(self, spec: TeacherSpec) -> Path:
        return self.root / spec.checkpoint_path

    def validate_config(self, config: TeacherConfig) -> TeacherSpec:
        if config.source != "robustbench" or config.registry_id is None:
            raise TeacherRegistryError("RobustBench validation requires a robustbench TeacherConfig with registry_id")
        spec = self.spec(config.registry_id)
        expected_path = self.checkpoint_path(spec).resolve()
        configured_path = config.checkpoint.resolve() if config.checkpoint is not None else None
        if (
            config.architecture != spec.architecture
            or config.num_classes != 10
            or config.preprocessing_owner != spec.preprocessing.owner
            or config.normalization != spec.preprocessing.normalization()
            or config.threat_norm != spec.threat.norm
            or config.threat_epsilon != spec.threat.epsilon
            or configured_path != expected_path
        ):
            raise TeacherRegistryError(f"teacher config does not exactly match registry entry {spec.registry_id}")
        if spec.checkpoint_status != "verified" or spec.checkpoint_sha256 is None:
            raise TeacherRegistryError(f"teacher checkpoint is not hash-registered: {spec.registry_id}")
        if config.checkpoint_sha256 != spec.checkpoint_sha256:
            raise TeacherRegistryError("teacher config checkpoint SHA does not match the project-owned lock SHA")
        return spec

    def validate_local_checkpoint(self, spec: TeacherSpec, path: Path) -> None:
        if not path.is_file():
            raise TeacherRegistryError(f"teacher checkpoint is missing; no download will be attempted: {path}")
        if spec.checkpoint_sha256 is None or spec.checkpoint_status != "verified":
            raise TeacherRegistryError(f"teacher checkpoint is not hash-registered: {spec.registry_id}")
        observed = sha256_file(path)
        if observed != spec.checkpoint_sha256:
            raise TeacherRegistryError(
                f"teacher checkpoint hash mismatch: expected {spec.checkpoint_sha256}, got {observed}"
            )

    def validate_external(self) -> None:
        external_lock = self.root / "external.lock.yaml"
        try:
            raw = yaml.safe_load(external_lock.read_text(encoding="utf-8"))
        except (FileNotFoundError, yaml.YAMLError) as exc:
            raise TeacherRegistryError("external.lock.yaml is required to validate RobustBench") from exc
        entry = raw.get("repositories", {}).get(ROBUSTBENCH_REPOSITORY) if isinstance(raw, dict) else None
        if not isinstance(entry, dict):
            raise TeacherRegistryError("external.lock.yaml does not pin RobustBench")
        evidence = entry.get("license_evidence")
        if (
            entry.get("url") != self.repository_url
            or entry.get("commit") != self.repository_commit
            or entry.get("license_file") != self.repository_license_file
            or not isinstance(evidence, dict)
            or evidence.get("sha256") != self.repository_license_sha256
            or entry.get("license_status") != "verified"
        ):
            raise TeacherRegistryError("external RobustBench lock does not match teacher lock evidence")
        checkout = self.root / ".external" / ROBUSTBENCH_REPOSITORY
        if not checkout.is_dir():
            raise TeacherRegistryError(f"pinned RobustBench checkout is missing: {checkout}")
        if _git(checkout, ["rev-parse", "--is-inside-work-tree"]) != "true":
            raise TeacherRegistryError(f"RobustBench checkout is not a Git work tree: {checkout}")
        if _git(checkout, ["remote", "get-url", "origin"]) != self.repository_url:
            raise TeacherRegistryError("RobustBench origin does not match the pinned repository URL")
        if _git(checkout, ["rev-parse", "HEAD"]) != self.repository_commit:
            raise TeacherRegistryError("RobustBench HEAD does not match the pinned commit")
        if _git(checkout, ["status", "--porcelain", "--untracked-files=all"]):
            raise TeacherRegistryError("RobustBench checkout is dirty")
        license_path = checkout / self.repository_license_file
        if not license_path.is_file() or sha256_file(license_path) != self.repository_license_sha256:
            raise TeacherRegistryError("RobustBench license evidence does not match the pinned checkout")

    def constructor(
        self, spec: TeacherSpec, *, resolver: Callable[[FactorySpec], Callable[..., nn.Module]] | None = None
    ) -> nn.Module:
        self.validate_external()
        factory = resolver(spec.factory) if resolver is not None else _external_factory(self.root, spec.factory)
        kwargs = _resolve_factory_kwargs(spec.factory.kwargs)
        model = factory(**kwargs)
        if not isinstance(model, nn.Module):
            raise TeacherRegistryError("teacher factory did not return a torch.nn.Module")
        observed = sum(parameter.numel() for parameter in model.parameters())
        if observed != spec.expected_parameter_count:
            raise TeacherRegistryError(
                f"teacher parameter count mismatch: expected {spec.expected_parameter_count}, got {observed}"
            )
        return model


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    """Accept one unambiguous state mapping and strip exact wrapper prefixes."""
    if not isinstance(payload, Mapping):
        raise TypeError("teacher checkpoint must be a state dictionary or contain model/state_dict")
    is_raw = all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in payload.items())
    wrapper_mappings = tuple(
        name for name in ("model", "state_dict") if name in payload and isinstance(payload[name], Mapping)
    )
    if is_raw:
        state = payload
    elif len(wrapper_mappings) == 1:
        state = payload[wrapper_mappings[0]]
    elif len(wrapper_mappings) == 2:
        raise ValueError("teacher checkpoint wrapper payload is ambiguous; both model and state_dict are mappings")
    else:
        raise TypeError("teacher checkpoint must contain a raw state mapping or model/state_dict mapping")
    if not isinstance(state, Mapping):
        raise TypeError("teacher checkpoint wrapper must contain a state dictionary")
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise TypeError("teacher state dictionary must map string keys to tensors")
        stripped = key
        while stripped.startswith("module.") or stripped.startswith("model."):
            stripped = stripped.split(".", 1)[1]
        if stripped in normalized:
            raise ValueError(f"teacher checkpoint prefix stripping causes key collision: {stripped}")
        normalized[stripped] = value
    return normalized


def _parse_spec(registry_id: str, entry: Any) -> TeacherSpec:
    if not isinstance(registry_id, str) or not isinstance(entry, dict):
        raise TeacherRegistryError("teacher entries must be named mappings")
    required = {
        "source",
        "upstream_model_id",
        "architecture",
        "factory",
        "dataset",
        "threat",
        "preprocessing",
        "expected_parameter_count",
        "upstream_locator",
        "checkpoint_filename",
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_status",
    }
    if set(entry) != required or entry["source"] != ROBUSTBENCH_REPOSITORY:
        raise TeacherRegistryError(f"teacher lock entry {registry_id!r} has an invalid schema")
    factory, threat, preprocessing = entry["factory"], entry["threat"], entry["preprocessing"]
    if not all(isinstance(value, dict) for value in (factory, threat, preprocessing)):
        raise TeacherRegistryError(f"teacher lock entry {registry_id!r} has invalid nested metadata")
    kwargs = factory.get("kwargs")
    checkpoint_path = Path(str(entry["checkpoint_path"]))
    if (
        checkpoint_path.is_absolute()
        or ".." in checkpoint_path.parts
        or checkpoint_path.name != entry["checkpoint_filename"]
    ):
        raise TeacherRegistryError(f"teacher lock entry {registry_id!r} has unsafe checkpoint path")
    sha = entry["checkpoint_sha256"]
    status = entry["checkpoint_status"]
    if (
        status not in {"missing", "verified"}
        or (status == "missing") != (sha is None)
        or (sha is not None and not _valid_sha256(sha))
    ):
        raise TeacherRegistryError(f"teacher lock entry {registry_id!r} has invalid checkpoint status/SHA")
    mean, std = tuple(preprocessing.get("mean", ())), tuple(preprocessing.get("std", ()))
    if len(mean) != 3 or len(std) != 3 or not all(isinstance(x, (float, int)) for x in (*mean, *std)):
        raise TeacherRegistryError(f"teacher lock entry {registry_id!r} has invalid preprocessing statistics")
    try:
        parsed_preprocessing = TeacherPreprocessing(
            owner=str(preprocessing["owner"]),
            profile=cast(NormalizationProfile, str(preprocessing["profile"])),
            mean=mean,
            std=std,
            provenance=str(preprocessing["provenance"]),
        )
        parsed_preprocessing.normalization()
    except (KeyError, ValueError) as exc:
        raise TeacherRegistryError(f"teacher lock entry {registry_id!r} has invalid preprocessing") from exc
    if threat != {"norm": "linf", "epsilon": "8/255", "input_domain": "pixel_0_1"}:
        raise TeacherRegistryError(
            f"teacher lock entry {registry_id!r} does not preserve the CIFAR-10 Linf 8/255 threat"
        )
    if not isinstance(kwargs, dict) or not isinstance(entry["expected_parameter_count"], int):
        raise TeacherRegistryError(f"teacher lock entry {registry_id!r} has invalid factory/count metadata")
    return TeacherSpec(
        registry_id=registry_id,
        upstream_model_id=str(entry["upstream_model_id"]),
        architecture=str(entry["architecture"]),
        factory=FactorySpec(module=str(factory.get("module")), symbol=str(factory.get("symbol")), kwargs=kwargs),
        dataset=str(entry["dataset"]),
        threat=TeacherThreat(**threat),
        preprocessing=parsed_preprocessing,
        expected_parameter_count=entry["expected_parameter_count"],
        upstream_locator=str(entry["upstream_locator"]),
        checkpoint_filename=str(entry["checkpoint_filename"]),
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=sha,
        checkpoint_status=status,
    )


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == SHA256_LENGTH and all(char in "0123456789abcdef" for char in value)


def _git(cwd: Path, args: list[str]) -> str:
    completed = subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True)
    if completed.returncode:
        raise TeacherRegistryError(completed.stderr.strip() or f"git {' '.join(args)} failed")
    return completed.stdout.strip()


def _external_factory(root: Path, factory: FactorySpec) -> Callable[..., nn.Module]:
    checkout = (root / ".external" / ROBUSTBENCH_REPOSITORY).resolve()
    _validate_loaded_robustbench_modules(checkout)
    external = str(checkout)
    if external in sys.path:
        sys.path.remove(external)
    sys.path.insert(0, external)
    module = importlib.import_module(factory.module)
    if not _module_is_under(module, checkout):
        raise TeacherRegistryError(f"pinned teacher module is not loaded from verified checkout: {factory.module}")
    resolved = getattr(module, factory.symbol, None)
    if not callable(resolved):
        raise TeacherRegistryError(f"pinned teacher factory is unavailable: {factory.module}.{factory.symbol}")
    try:
        symbol_path = Path(inspect.getfile(resolved)).resolve()
    except (OSError, TypeError) as exc:
        raise TeacherRegistryError(
            f"pinned teacher factory has unverifiable provenance: {factory.module}.{factory.symbol}"
        ) from exc
    if not _path_is_under(symbol_path, checkout):
        raise TeacherRegistryError(
            f"pinned teacher factory is not defined under verified checkout: {factory.module}.{factory.symbol}"
        )
    return resolved


def _validate_loaded_robustbench_modules(checkout: Path) -> None:
    for name, module in tuple(sys.modules.items()):
        if name != ROBUSTBENCH_REPOSITORY and not name.startswith(f"{ROBUSTBENCH_REPOSITORY}."):
            continue
        if module is None or not _module_is_under(module, checkout):
            raise TeacherRegistryError(f"preloaded RobustBench module has unverifiable provenance: {name}")


def _module_is_under(module: Any, checkout: Path) -> bool:
    module_path = getattr(module, "__file__", None)
    return isinstance(module_path, str) and _path_is_under(Path(module_path).resolve(), checkout)


def _path_is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_factory_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    resolved = dict(kwargs)
    if resolved.get("activation_fn") == "torch.nn.SiLU":
        resolved["activation_fn"] = torch.nn.SiLU
    return resolved
