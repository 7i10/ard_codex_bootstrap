"""Bounded local clean/PGD screening for one strict RobustBench teacher.

The runner deliberately has no W&B, AutoAttack, DDP, student, or training
dependencies.  It accepts injected builders so its contracts can be tested
without CIFAR data, a checkpoint, or network access.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Subset

from ard.attacks import AttackRequest, LinfPGD
from ard.config.schema import DatasetConfig, TeacherConfig
from ard.config.teacher_audit import (
    TeacherAuditConfig,
    TeacherAuditDatasetConfig,
    resolved_teacher_audit_config,
    save_resolved_teacher_audit_config,
)
from ard.data import IndexedBatch, build_dataset, collate_indexed
from ard.models.teacher import TeacherAdapter, build_teacher
from ard.models.teacher_registry import TeacherRegistry, TeacherRegistryError, TeacherSpec, sha256_file


class TeacherAuditError(RuntimeError):
    """A bounded teacher audit contract was violated."""


class RegistryLike(Protocol):
    @property
    def repository_commit(self) -> str: ...

    def spec(self, registry_id: str) -> TeacherSpec: ...

    def validate_external(self) -> None: ...

    def validate_config(self, config: TeacherConfig) -> TeacherSpec: ...

    def checkpoint_path(self, spec: TeacherSpec) -> Path: ...


DatasetBuilder = Callable[[TeacherAuditDatasetConfig], Dataset[tuple[torch.Tensor, int, int]]]
TeacherBuilder = Callable[[TeacherConfig], TeacherAdapter]
RegistryLoader = Callable[[Path], RegistryLike]
FailureInjector = Callable[[], None]


@dataclass(frozen=True)
class Accuracy:
    correct: int
    count: int
    accuracy: float


@dataclass(frozen=True)
class BackendFlags:
    deterministic_algorithms: bool
    cudnn_benchmark: bool
    cudnn_deterministic: bool
    cuda_matmul_allow_tf32: bool
    cudnn_allow_tf32: bool


@dataclass(frozen=True)
class AuditLineage:
    project_root: str
    project_git_sha: str
    project_git_status: str
    project_git_dirty: bool
    project_binary_diff_sha256: str
    project_untracked_sha256: dict[str, str]
    project_untracked_digest_sha256: str
    external_lock_sha256: str
    teachers_lock_sha256: str
    robustbench_locked_commit: str
    robustbench_observed_commit: str


LineageCollector = Callable[[Path], AuditLineage]


@dataclass(frozen=True)
class TeacherAuditReport:
    schema_version: int
    config_sha256: str
    registry_id: str
    upstream_model_id: str
    checkpoint_sha256: str
    preprocessing_owner: str
    preprocessing_profile: str | None
    threat_model: dict[str, str]
    teacher_metadata: dict[str, object]
    lineage: AuditLineage
    environment: dict[str, object]
    backend_flags: BackendFlags
    selected_source_ids: tuple[int, ...]
    selected_source_ids_sha256: str
    clean: Accuracy
    pgd: Accuracy
    attack_identity: dict[str, object]
    attack_identity_sha256: str
    device: dict[str, object]
    cuda_peak_allocated_bytes: int
    cuda_peak_reserved_bytes: int

    def result_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["selected_source_ids"] = list(self.selected_source_ids)
        return cast(dict[str, object], result)


def _default_dataset_builder(config: TeacherAuditDatasetConfig) -> Dataset[tuple[torch.Tensor, int, int]]:
    dataset = build_dataset(
        DatasetConfig(
            name=config.name,
            root=config.root,
            split=config.split,
            download=config.download,
            num_classes=10,
        )
    )
    return cast(Dataset[tuple[torch.Tensor, int, int]], dataset)


def _default_teacher_builder(config: TeacherConfig) -> TeacherAdapter:
    return build_teacher(config, tier="production")


def _default_registry_loader(project_root: Path) -> RegistryLike:
    return TeacherRegistry.load(project_root)


def _teacher_config(registry: RegistryLike, spec: TeacherSpec) -> TeacherConfig:
    return TeacherConfig(
        source="robustbench",
        registry_id=cast(Any, spec.registry_id),
        architecture=cast(Any, spec.architecture),
        num_classes=10,
        normalization=spec.preprocessing.normalization(),
        preprocessing_owner=cast(Any, spec.preprocessing.owner),
        checkpoint=registry.checkpoint_path(spec),
        checkpoint_sha256=spec.checkpoint_sha256,
        threat_norm=cast(Any, spec.threat.norm),
        threat_epsilon=spec.threat.epsilon,
    )


def _source_targets(dataset: Dataset[tuple[torch.Tensor, int, int]]) -> Sequence[int]:
    raw = getattr(dataset, "dataset", dataset)
    targets = getattr(raw, "targets", None)
    if not isinstance(targets, Sequence) or len(targets) != len(cast(Sequence[object], dataset)):
        raise TeacherAuditError("teacher audit requires indexed CIFAR source targets for stratified selection")
    labels: list[int] = []
    for label in targets:
        if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < 10:
            raise TeacherAuditError("teacher audit requires CIFAR-10 integer targets in [0, 9]")
        labels.append(label)
    return labels


def select_stratified_source_ids(
    dataset: Dataset[tuple[torch.Tensor, int, int]], *, max_samples: int, seed: int
) -> tuple[int, ...]:
    """Return a seed-fixed, class-balanced subset of immutable source IDs."""
    if max_samples < 10:
        raise TeacherAuditError("teacher audit max_samples must cover all 10 CIFAR-10 classes")
    labels = _source_targets(dataset)
    target_count = min(max_samples, len(cast(Sequence[object], dataset)))
    if target_count < 10:
        raise TeacherAuditError("teacher audit dataset must contain at least one sample for every class")
    source_ids_by_class: dict[int, list[int]] = defaultdict(list)
    for source_id, label in enumerate(labels):
        source_ids_by_class[label].append(source_id)
    if set(source_ids_by_class) != set(range(10)):
        raise TeacherAuditError("teacher audit official test split must expose every CIFAR-10 class")
    base, remainder = divmod(target_count, 10)
    generator = torch.Generator().manual_seed(seed)
    selected: list[int] = []
    for label in range(10):
        quota = base + (1 if label < remainder else 0)
        candidates = source_ids_by_class[label]
        if len(candidates) < quota:
            raise TeacherAuditError(f"teacher audit class {label} has fewer than its required stratified quota")
        positions = torch.randperm(len(candidates), generator=generator)[:quota].tolist()
        selected.extend(candidates[position] for position in positions)
    return tuple(sorted(selected))


def _source_ids_sha256(source_ids: Sequence[int]) -> str:
    payload = json.dumps(list(source_ids), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _device_identity(device: torch.device) -> dict[str, object]:
    if device.type == "cpu":
        return {"type": "cpu"}
    if device.type != "cuda" or not torch.cuda.is_available():
        raise TeacherAuditError("teacher audit requested CUDA but CUDA is unavailable")
    index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "type": "cuda",
        "index": index,
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "torch_cuda": torch.version.cuda or "unknown",
    }


def _validate_teacher(teacher: TeacherAdapter, spec: TeacherSpec, *, repository_commit: str) -> None:
    metadata = teacher.metadata
    if teacher.training or teacher.model.training:
        raise TeacherAuditError("teacher audit requires an eval-mode teacher")
    if any(parameter.requires_grad or parameter.grad is not None for parameter in teacher.parameters()):
        raise TeacherAuditError("teacher audit requires frozen teacher parameters with no gradients")
    expected = {
        "architecture": spec.architecture,
        "num_classes": 10,
        "normalization": spec.preprocessing.normalization(),
        "registry_id": spec.registry_id,
        "upstream_model_id": spec.upstream_model_id,
        "external_commit": repository_commit,
        "checkpoint_sha256": spec.checkpoint_sha256,
        "preprocessing_owner": spec.preprocessing.owner,
        "preprocessing_profile": spec.preprocessing.profile,
        "threat_model": {
            "norm": spec.threat.norm,
            "epsilon": spec.threat.epsilon,
            "input_domain": spec.threat.input_domain,
        },
    }
    actual = {
        "architecture": metadata.architecture,
        "num_classes": metadata.num_classes,
        "normalization": metadata.normalization,
        "registry_id": metadata.registry_id,
        "upstream_model_id": metadata.upstream_model_id,
        "external_commit": metadata.external_commit,
        "checkpoint_sha256": metadata.checkpoint_sha256,
        "preprocessing_owner": metadata.preprocessing_owner,
        "preprocessing_profile": metadata.preprocessing_profile,
        "threat_model": metadata.threat_model,
    }
    if actual != expected:
        raise TeacherAuditError("teacher metadata does not exactly match the registered RobustBench teacher")


def _validate_pixels(images: torch.Tensor) -> None:
    if not images.is_floating_point() or not torch.isfinite(images).all():
        raise TeacherAuditError("teacher audit dataset images must be finite floating-point pixel tensors")
    if images.numel() and (images.detach().amin() < 0 or images.detach().amax() > 1):
        raise TeacherAuditError("teacher audit dataset images must lie in pixel domain [0, 1]")


def _validate_logits(logits: torch.Tensor, labels: torch.Tensor) -> None:
    if logits.ndim != 2 or logits.shape != (labels.shape[0], 10) or not torch.isfinite(logits).all():
        raise TeacherAuditError("teacher audit received non-finite or incorrectly shaped logits")


def _accuracy(correct: int, count: int) -> Accuracy:
    if count <= 0:
        raise TeacherAuditError("teacher audit cannot report an empty sample set")
    return Accuracy(correct=correct, count=count, accuracy=correct / count)


def _git(root: Path, arguments: Sequence[str], *, binary: bool = False) -> str | bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True,
            text=not binary,
            check=False,
        )
    except OSError as exc:
        raise TeacherAuditError("project Git lineage is unavailable") from exc
    if completed.returncode != 0:
        raise TeacherAuditError("project Git lineage is unavailable")
    return completed.stdout


def _valid_commit(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def _sha256_regular_file(path: Path) -> str:
    try:
        initial_mode = path.lstat().st_mode
    except OSError as exc:
        raise TeacherAuditError(f"untracked lineage file is unavailable: {path}") from exc
    if not stat.S_ISREG(initial_mode):
        raise TeacherAuditError(f"untracked lineage path is not a regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TeacherAuditError(f"untracked lineage file is unreadable or unsafe: {path}") from exc
    digest = hashlib.sha256()
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise TeacherAuditError(f"untracked lineage path is not a regular file: {path}")
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _untracked_file_hashes(root: Path) -> dict[str, str]:
    raw = cast(bytes, _git(root, ("ls-files", "--others", "--exclude-standard", "-z"), binary=True))
    hashes: dict[str, str] = {}
    for encoded_path in raw.split(b"\0"):
        if not encoded_path:
            continue
        relative = Path(os.fsdecode(encoded_path))
        if relative.is_absolute() or ".." in relative.parts:
            raise TeacherAuditError("Git returned an unsafe untracked lineage path")
        normalized = relative.as_posix()
        if normalized in hashes:
            raise TeacherAuditError("Git returned a duplicate untracked lineage path")
        hashes[normalized] = _sha256_regular_file(root / relative)
    return dict(sorted(hashes.items()))


def collect_audit_lineage(project_root: Path) -> AuditLineage:
    """Collect exact local project/lock/upstream lineage without storing the diff."""
    root = project_root.resolve()
    top_level = cast(str, _git(root, ("rev-parse", "--show-toplevel"))).strip()
    if Path(top_level).resolve() != root:
        raise TeacherAuditError("project_root must be the exact Git repository root")
    project_sha = cast(str, _git(root, ("rev-parse", "HEAD"))).strip()
    if not _valid_commit(project_sha):
        raise TeacherAuditError("project Git SHA is unavailable or invalid")
    status = cast(str, _git(root, ("status", "--porcelain=v1", "--untracked-files=all"))).rstrip("\n")
    binary_diff = cast(bytes, _git(root, ("diff", "--binary", "HEAD"), binary=True))
    untracked_sha256 = _untracked_file_hashes(root)
    untracked_digest_sha256 = hashlib.sha256(
        json.dumps(untracked_sha256, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    external_lock = root / "external.lock.yaml"
    teachers_lock = root / "teachers.lock.yaml"
    if not external_lock.is_file() or not teachers_lock.is_file():
        raise TeacherAuditError("teacher audit requires external.lock.yaml and teachers.lock.yaml")
    try:
        external_raw = yaml.safe_load(external_lock.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise TeacherAuditError("external.lock.yaml is invalid YAML") from exc
    robustbench = external_raw.get("repositories", {}).get("robustbench") if isinstance(external_raw, dict) else None
    locked_commit = robustbench.get("commit") if isinstance(robustbench, dict) else None
    if not _valid_commit(locked_commit):
        raise TeacherAuditError("external.lock.yaml has no valid pinned RobustBench commit")
    checkout = root / ".external" / "robustbench"
    observed_commit = cast(str, _git(checkout, ("rev-parse", "HEAD"))).strip()
    if not _valid_commit(observed_commit) or observed_commit != locked_commit:
        raise TeacherAuditError("observed RobustBench commit does not match external.lock.yaml")
    return AuditLineage(
        project_root=str(root),
        project_git_sha=project_sha,
        project_git_status=status,
        project_git_dirty=bool(status),
        project_binary_diff_sha256=hashlib.sha256(binary_diff).hexdigest(),
        project_untracked_sha256=untracked_sha256,
        project_untracked_digest_sha256=untracked_digest_sha256,
        external_lock_sha256=sha256_file(external_lock),
        teachers_lock_sha256=sha256_file(teachers_lock),
        robustbench_locked_commit=locked_commit,
        robustbench_observed_commit=observed_commit,
    )


def _environment_identity() -> dict[str, object]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


@contextmanager
def _deterministic_backend() -> Iterator[BackendFlags]:
    previous = BackendFlags(
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
    )
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    enforced = BackendFlags(True, False, True, False, False)
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield enforced
    finally:
        torch.use_deterministic_algorithms(previous.deterministic_algorithms, warn_only=previous_warn_only)
        torch.backends.cudnn.benchmark = previous.cudnn_benchmark
        torch.backends.cudnn.deterministic = previous.cudnn_deterministic
        torch.backends.cuda.matmul.allow_tf32 = previous.cuda_matmul_allow_tf32
        torch.backends.cudnn.allow_tf32 = previous.cudnn_allow_tf32


def run_teacher_audit(
    config: TeacherAuditConfig,
    *,
    project_root: Path,
    dataset_builder: DatasetBuilder = _default_dataset_builder,
    teacher_builder: TeacherBuilder = _default_teacher_builder,
    registry_loader: RegistryLoader = _default_registry_loader,
    lineage_collector: LineageCollector = collect_audit_lineage,
) -> TeacherAuditReport:
    """Measure clean and exact configured PGD accuracy in one local process."""
    if config.run.device == "cuda" and not torch.cuda.is_available():
        raise TeacherAuditError("teacher audit requested CUDA but CUDA is unavailable")
    device = torch.device(config.run.device)
    selected_root = project_root.resolve()
    registry = registry_loader(selected_root)
    try:
        registry.validate_external()
        spec = registry.spec(config.teacher.registry_id)
        teacher_config = _teacher_config(registry, spec)
        if registry.validate_config(teacher_config) != spec:
            raise TeacherAuditError("teacher registry returned a mismatched validated specification")
    except TeacherRegistryError as exc:
        raise TeacherAuditError(str(exc)) from exc
    lineage = lineage_collector(selected_root)
    if lineage.robustbench_locked_commit != registry.repository_commit:
        raise TeacherAuditError("teacher registry commit does not match recorded RobustBench lineage")

    dataset = dataset_builder(config.dataset)
    selected_source_ids = select_stratified_source_ids(
        dataset, max_samples=config.run.max_samples, seed=config.run.seed
    )
    loader = DataLoader(
        Subset(dataset, list(selected_source_ids)),
        batch_size=config.run.batch_size,
        shuffle=False,
        num_workers=config.run.num_workers,
        collate_fn=collate_indexed,
    )
    with _deterministic_backend() as backend_flags:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        teacher = teacher_builder(teacher_config).to(device)
        teacher.eval()
        _validate_teacher(teacher, spec, repository_commit=registry.repository_commit)
        attack = LinfPGD(config.attack)
        clean_correct = pgd_correct = count = 0
        for batch_index, batch in enumerate(loader):
            if not isinstance(batch, IndexedBatch):
                raise TeacherAuditError("teacher audit loader must produce IndexedBatch values")
            if batch.state_update_mask is not None and not bool(batch.state_update_mask.all()):
                raise TeacherAuditError("teacher audit does not permit padded or duplicate source samples")
            if tuple(batch.sample_ids.tolist()) != selected_source_ids[count : count + len(batch.labels)]:
                raise TeacherAuditError("teacher audit loader changed the selected stable source ID order")
            batch = batch.to(device)
            _validate_pixels(batch.images)
            with torch.no_grad():
                clean_logits = teacher.logits(batch.images, require_input_grad=False).float()
            _validate_logits(clean_logits, batch.labels)
            clean_correct += int((clean_logits.argmax(dim=1) == batch.labels).sum().item())
            generator = torch.Generator(device=device).manual_seed(config.run.seed + batch_index)
            prior_teacher_mode, prior_model_mode = teacher.training, teacher.model.training
            result = attack.generate(
                AttackRequest(inputs=batch.images, labels=batch.labels, student=teacher, generator=generator)
            )
            if teacher.training != prior_teacher_mode or teacher.model.training != prior_model_mode:
                raise TeacherAuditError("PGD did not restore teacher eval mode")
            adversarial = result.adversarial
            _validate_pixels(adversarial)
            epsilon = config.attack.epsilon_value
            assert epsilon is not None
            if (adversarial - batch.images).detach().abs().amax() > epsilon + 1e-7:
                raise TeacherAuditError("PGD result violates the configured pixel-space Linf bound")
            with torch.no_grad():
                adversarial_logits = teacher.logits(adversarial, require_input_grad=False).float()
            _validate_logits(adversarial_logits, batch.labels)
            pgd_correct += int((adversarial_logits.argmax(dim=1) == batch.labels).sum().item())
            count += len(batch.labels)
            if any(parameter.grad is not None for parameter in teacher.parameters()):
                raise TeacherAuditError("teacher audit PGD populated a frozen teacher parameter gradient")
        peak_allocated = peak_reserved = 0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
    resolved = resolved_teacher_audit_config(config)
    config_sha256 = hashlib.sha256(
        json.dumps(resolved, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return TeacherAuditReport(
        schema_version=config.schema_version,
        config_sha256=config_sha256,
        registry_id=spec.registry_id,
        upstream_model_id=spec.upstream_model_id,
        checkpoint_sha256=cast(str, spec.checkpoint_sha256),
        preprocessing_owner=teacher.metadata.preprocessing_owner,
        preprocessing_profile=teacher.metadata.preprocessing_profile,
        threat_model=cast(dict[str, str], teacher.metadata.threat_model),
        teacher_metadata=cast(dict[str, object], teacher.metadata.model_dump(mode="json")),
        lineage=lineage,
        environment=_environment_identity(),
        backend_flags=backend_flags,
        selected_source_ids=selected_source_ids,
        selected_source_ids_sha256=_source_ids_sha256(selected_source_ids),
        clean=_accuracy(clean_correct, count),
        pgd=_accuracy(pgd_correct, count),
        attack_identity=config.attack.identity(),
        attack_identity_sha256=config.attack.identity_sha256(),
        device=_device_identity(device),
        cuda_peak_allocated_bytes=peak_allocated,
        cuda_peak_reserved_bytes=peak_reserved,
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish one directory without replacing a concurrent writer."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise TeacherAuditError("atomic no-clobber directory publication is unavailable on this platform")
    result = renameat2(
        ctypes.c_int(-100),
        os.fsencode(source),
        ctypes.c_int(-100),
        os.fsencode(destination),
        ctypes.c_uint(1),
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise TeacherAuditError(f"refusing to overwrite existing teacher audit output: {destination}")
    raise OSError(error_number, os.strerror(error_number), str(destination))


def write_teacher_audit_artifacts(
    config: TeacherAuditConfig,
    report: TeacherAuditReport,
    *,
    failure_injector: FailureInjector | None = None,
) -> tuple[Path, Path]:
    """Publish the resolved config/result as one no-clobber directory pair."""
    output_dir = config.run.output_dir
    resolved = resolved_teacher_audit_config(config)
    expected_config_sha256 = hashlib.sha256(
        json.dumps(resolved, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    if report.config_sha256 != expected_config_sha256:
        raise TeacherAuditError("teacher audit report/config digest mismatch")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise TeacherAuditError(f"refusing to overwrite existing teacher audit output: {output_dir}")
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.stage-", dir=output_dir.parent))
    try:
        staged_resolved = staging / "resolved_teacher_audit.yaml"
        staged_result = staging / "teacher_audit_result.json"
        save_resolved_teacher_audit_config(config, staged_resolved)
        staged_result.write_text(json.dumps(report.result_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with staged_result.open("rb") as handle:
            os.fsync(handle.fileno())
        _fsync_directory(staging)
        if failure_injector is not None:
            failure_injector()
        if output_dir.exists():
            raise TeacherAuditError(f"refusing to overwrite existing teacher audit output: {output_dir}")
        _rename_directory_no_replace(staging, output_dir)
        try:
            _fsync_directory(output_dir.parent)
        except Exception:
            shutil.rmtree(output_dir, ignore_errors=True)
            _fsync_directory(output_dir.parent)
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return output_dir / "resolved_teacher_audit.yaml", output_dir / "teacher_audit_result.json"
