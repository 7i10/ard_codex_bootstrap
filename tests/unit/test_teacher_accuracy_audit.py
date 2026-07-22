from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from ard.attacks import LinfPGD
from ard.cli.audit_teacher import build_parser
from ard.config.schema import AttackConfig, NormalizationConfig, TeacherConfig
from ard.config.teacher_audit import TeacherAuditConfig, load_teacher_audit_config
from ard.evaluation import teacher_audit as audit_module
from ard.evaluation.teacher_audit import (
    AuditLineage,
    BackendFlags,
    TeacherAuditError,
    collect_audit_lineage,
    run_teacher_audit,
    select_stratified_source_ids,
    write_teacher_audit_artifacts,
)
from ard.models.teacher import TeacherAdapter, TeacherMetadata
from ard.models.teacher_registry import TeacherRegistry, TeacherSpec, sha256_file
from ard.testing.impact import select

pytestmark = [pytest.mark.unit, pytest.mark.t1, pytest.mark.t2]


class TinyOfficialCIFAR(Dataset[tuple[torch.Tensor, int, int]]):
    """CPU-only indexed fixture with two source samples per CIFAR-10 class."""

    def __init__(self) -> None:
        self.targets = [index % 10 for index in range(20)]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        label = self.targets[index]
        image = torch.full((3, 4, 4), label / 10.0, dtype=torch.float32)
        return image, label, index


class TinyLogitModel(nn.Module):
    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        first = pixels.flatten(1).sum(dim=1)
        return torch.cat((first[:, None], torch.zeros((len(pixels), 9), device=pixels.device)), dim=1)


class FakeRegistry:
    def __init__(self, spec: TeacherSpec, checkpoint: Path, repository_commit: str) -> None:
        self._spec = spec
        self._checkpoint = checkpoint
        self.repository_commit = repository_commit
        self.external_validated = False
        self.config_validated = False
        self.teacher_builder: Any = None

    def spec(self, registry_id: str) -> TeacherSpec:
        assert registry_id == self._spec.registry_id
        return self._spec

    def validate_external(self) -> None:
        self.external_validated = True

    def validate_config(self, _config: object) -> TeacherSpec:
        self.config_validated = True
        return self._spec

    def checkpoint_path(self, _spec: TeacherSpec) -> Path:
        return self._checkpoint


def _raw_config(tmp_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "teacher": {"registry_id": "chen2021_ltd_wrn34_10"},
        "dataset": {"name": "cifar10", "root": str(tmp_path / "cifar10"), "split": "test", "download": False},
        "run": {
            "max_samples": 10,
            "batch_size": 4,
            "num_workers": 0,
            "seed": 17,
            "device": "cpu",
            "output_dir": str(tmp_path / "audit-output"),
        },
        "attack": {
            "norm": "linf",
            "input_domain": "pixel_0_1",
            "epsilon": "8/255",
            "step_size": "2/255",
            "steps": 20,
            "random_start": True,
            "loss": "ce",
            "kl_target": None,
            "temperature": 1.0,
            "temperature_squared": True,
            "student_mode": "eval",
            "teacher_mode": "eval",
            "trace_step_losses": False,
        },
    }


def _fixture_runtime(tmp_path: Path) -> tuple[TeacherAuditConfig, FakeRegistry, list[TeacherAdapter]]:
    config = TeacherAuditConfig.model_validate(_raw_config(tmp_path))
    locked_registry = TeacherRegistry.load(Path(__file__).resolve().parents[2])
    spec = locked_registry.spec(config.teacher.registry_id)
    registry = FakeRegistry(spec, tmp_path / "teacher.pt", locked_registry.repository_commit)
    constructed: list[TeacherAdapter] = []

    def teacher_builder(_config: TeacherConfig) -> TeacherAdapter:
        metadata = TeacherMetadata(
            architecture=spec.architecture,
            num_classes=10,
            normalization=spec.preprocessing.normalization(),
            checkpoint_sha256=spec.checkpoint_sha256 or "",
            registry_id=spec.registry_id,
            upstream_model_id=spec.upstream_model_id,
            external_commit=registry.repository_commit,
            preprocessing_owner=spec.preprocessing.owner,
            preprocessing_profile=spec.preprocessing.profile,
            threat_model={
                "norm": spec.threat.norm,
                "epsilon": spec.threat.epsilon,
                "input_domain": spec.threat.input_domain,
            },
        )
        teacher = TeacherAdapter(TinyLogitModel(), metadata)
        constructed.append(teacher)
        return teacher

    setattr(registry, "teacher_builder", teacher_builder)
    return config, registry, constructed


def _fake_lineage(tmp_path: Path, registry: FakeRegistry) -> AuditLineage:
    return AuditLineage(
        project_root=str(tmp_path.resolve()),
        project_git_sha="1" * 40,
        project_git_status=" M fixture.py",
        project_git_dirty=True,
        project_binary_diff_sha256="2" * 64,
        project_untracked_sha256={"fixture.py": "5" * 64},
        project_untracked_digest_sha256="6" * 64,
        external_lock_sha256="3" * 64,
        teachers_lock_sha256="4" * 64,
        robustbench_locked_commit=registry.repository_commit,
        robustbench_observed_commit=registry.repository_commit,
    )


def _run_fixture(
    config: TeacherAuditConfig,
    registry: FakeRegistry,
    teacher_builder: Any,
    *,
    project_root: Path,
):
    return run_teacher_audit(
        config,
        project_root=project_root,
        dataset_builder=lambda _dataset_config: TinyOfficialCIFAR(),
        teacher_builder=teacher_builder,
        registry_loader=lambda _root: registry,
        lineage_collector=lambda _root: _fake_lineage(project_root, registry),
    )


def test_stratified_bounded_source_ids_are_seed_fixed_and_keep_original_ids() -> None:
    dataset = TinyOfficialCIFAR()
    first = select_stratified_source_ids(dataset, max_samples=10, seed=17)
    second = select_stratified_source_ids(dataset, max_samples=10, seed=17)
    assert first == second
    assert len(first) == len(set(first)) == 10
    assert Counter(dataset.targets[source_id] for source_id in first) == Counter(range(10))
    assert set(first).issubset(set(range(len(dataset))))


def test_cli_requires_a_strict_audit_config_path() -> None:
    args = build_parser().parse_args(("--config", "configs/audit/chen.yaml", "run.device=cpu"))
    assert args.config == Path("configs/audit/chen.yaml")
    assert args.overrides == ["run.device=cpu"]


@pytest.mark.parametrize(
    "path",
    (
        "src/ard/config/teacher_audit.py",
        "src/ard/evaluation/teacher_audit.py",
        "src/ard/cli/audit_teacher.py",
        "configs/audit/cifar10_chen.yaml",
    ),
)
def test_impact_selection_targets_teacher_audit_contract(path: str) -> None:
    selection = select((path,), ("tests/unit/test_teacher_accuracy_audit.py",))
    assert selection.tests == ("tests/unit/test_teacher_accuracy_audit.py",)
    # Generic config/evaluation rules retain T3 as a conservative tier label,
    # but this dedicated path maps to the bounded CPU contract itself.
    assert {"T0", "T1", "T2"}.issubset(selection.tiers)


def test_run_reports_accuracy_count_attack_identity_freeze_and_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, registry, constructed = _fixture_runtime(tmp_path)
    captured: list[object] = []

    class SpyPGD(LinfPGD):
        def __init__(self, attack_config: AttackConfig) -> None:
            captured.append(attack_config)
            super().__init__(attack_config)

    monkeypatch.setattr(audit_module, "LinfPGD", SpyPGD)
    builder = registry.teacher_builder
    assert builder is not None
    before = BackendFlags(
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    report = _run_fixture(config, registry, builder, project_root=tmp_path)
    assert registry.external_validated and registry.config_validated
    assert len(captured) == 1 and captured[0] is config.attack
    assert report.clean.count == report.pgd.count == 10
    assert report.clean.correct == 1 and report.clean.accuracy == 0.1
    assert 0 <= report.pgd.correct <= report.pgd.count and 0 <= report.pgd.accuracy <= 1
    assert report.attack_identity == config.attack.identity()
    assert report.attack_identity_sha256 == config.attack.identity_sha256()
    assert report.preprocessing_owner == "teacher_adapter"
    assert report.preprocessing_profile == "cifar10_raw_identity"
    assert report.threat_model == {"norm": "linf", "epsilon": "8/255", "input_domain": "pixel_0_1"}
    assert report.teacher_metadata == constructed[0].metadata.model_dump(mode="json")
    assert report.teacher_metadata["architecture"] == registry.spec(config.teacher.registry_id).architecture
    assert report.teacher_metadata["num_classes"] == 10
    assert report.teacher_metadata["external_commit"] == registry.repository_commit
    assert report.lineage == _fake_lineage(tmp_path, registry)
    assert report.environment["torch"] == str(torch.__version__)
    assert {"cuda", "cudnn"} <= report.environment.keys()
    assert "cuda_visible_devices" in report.environment
    assert report.backend_flags == BackendFlags(True, False, True, False, False)
    assert report.device == {"type": "cpu"}
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in constructed[0].parameters())
    assert not constructed[0].training and not constructed[0].model.training
    assert (
        BackendFlags(
            torch.are_deterministic_algorithms_enabled(),
            torch.backends.cudnn.benchmark,
            torch.backends.cudnn.deterministic,
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
        )
        == before
    )


def test_atomic_resolved_config_and_result_artifacts(tmp_path: Path) -> None:
    config, registry, _ = _fixture_runtime(tmp_path)
    builder = registry.teacher_builder
    assert builder is not None
    report = _run_fixture(config, registry, builder, project_root=tmp_path)
    resolved_path, result_path = write_teacher_audit_artifacts(config, report)
    assert resolved_path.is_file() and result_path.is_file()
    resolved = resolved_path.read_text(encoding="utf-8")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert "epsilon_value:" in resolved and result["selected_source_ids"] == list(report.selected_source_ids)
    assert result["attack_identity_sha256"] == config.attack.identity_sha256()
    assert result["lineage"]["external_lock_sha256"] == "3" * 64
    assert not list(config.run.output_dir.parent.glob(f".{config.run.output_dir.name}.stage-*"))

    before = {path.name: path.read_bytes() for path in config.run.output_dir.iterdir()}
    with pytest.raises(TeacherAuditError, match="refusing to overwrite"):
        write_teacher_audit_artifacts(config, report)
    assert {path.name: path.read_bytes() for path in config.run.output_dir.iterdir()} == before


def test_artifact_pair_failure_leaves_no_partial_final_directory(tmp_path: Path) -> None:
    config, registry, _ = _fixture_runtime(tmp_path)
    builder = registry.teacher_builder
    assert builder is not None
    report = _run_fixture(config, registry, builder, project_root=tmp_path)

    def fail_before_publish() -> None:
        raise OSError("injected publication failure")

    with pytest.raises(OSError, match="injected"):
        write_teacher_audit_artifacts(config, report, failure_injector=fail_before_publish)
    assert not config.run.output_dir.exists()
    assert not list(config.run.output_dir.parent.glob(f".{config.run.output_dir.name}.stage-*"))


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda raw: raw["attack"].update({"steps": 10}), "exact PGD-20"),
        (lambda raw: raw["dataset"].update({"download": True}), "download"),
        (lambda raw: raw["dataset"].update({"split": "train"}), "test"),
        (lambda raw: raw.update({"unexpected": 1}), "Extra inputs"),
    ),
)
def test_rejects_bad_attack_download_non_test_and_unknown_config(tmp_path: Path, mutate: object, message: str) -> None:
    raw = _raw_config(tmp_path)
    mutate(raw)  # type: ignore[operator]
    with pytest.raises(Exception, match=message):
        TeacherAuditConfig.model_validate(raw)


def test_yaml_environment_expansion_and_dot_override_are_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _raw_config(tmp_path)
    raw["dataset"]["root"] = "${ARD_AUDIT_DATA_ROOT}"
    path = tmp_path / "audit.yaml"
    import yaml

    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.setenv("ARD_AUDIT_DATA_ROOT", str(tmp_path / "from-env"))
    config = load_teacher_audit_config(path, ["run.max_samples=20"])
    assert config.dataset.root == tmp_path / "from-env"
    assert config.run.max_samples == 20
    monkeypatch.delenv("ARD_AUDIT_DATA_ROOT")
    with pytest.raises(ValueError, match="missing environment"):
        load_teacher_audit_config(path)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("architecture", "wrong-architecture"),
        ("num_classes", 11),
        ("normalization", NormalizationConfig(profile="fixture_unit")),
        ("checkpoint_sha256", "9" * 64),
        ("registry_id", "wrong-registry"),
        ("upstream_model_id", "wrong-upstream"),
        ("external_commit", "8" * 40),
        ("preprocessing_owner", "wrong-owner"),
        ("preprocessing_profile", "wrong-profile"),
        ("threat_model", {"norm": "linf", "epsilon": "4/255", "input_domain": "pixel_0_1"}),
    ),
)
def test_every_teacher_metadata_field_mismatch_fails_before_measurement(
    tmp_path: Path, field: str, value: object
) -> None:
    config, registry, _ = _fixture_runtime(tmp_path)
    spec = registry.spec(config.teacher.registry_id)

    def bad_teacher(_config: TeacherConfig) -> TeacherAdapter:
        teacher = TeacherAdapter(
            TinyLogitModel(),
            TeacherMetadata(
                architecture=spec.architecture,
                num_classes=10,
                normalization=spec.preprocessing.normalization(),
                checkpoint_sha256=spec.checkpoint_sha256 or "",
                registry_id=spec.registry_id,
                upstream_model_id=spec.upstream_model_id,
                external_commit=registry.repository_commit,
                preprocessing_owner=spec.preprocessing.owner,
                preprocessing_profile=spec.preprocessing.profile,
                threat_model={"norm": "linf", "epsilon": "8/255", "input_domain": "pixel_0_1"},
            ),
        )
        teacher.metadata = teacher.metadata.model_copy(update={field: value})
        return teacher

    with pytest.raises(TeacherAuditError, match="metadata"):
        _run_fixture(config, registry, bad_teacher, project_root=tmp_path)


def test_backend_flags_restore_after_audit_failure(tmp_path: Path) -> None:
    config, registry, _ = _fixture_runtime(tmp_path)
    before = BackendFlags(
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )

    def fail_during_model_construction(_config: TeacherConfig) -> TeacherAdapter:
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
        raise RuntimeError("injected model failure")

    with pytest.raises(RuntimeError, match="injected model"):
        _run_fixture(config, registry, fail_during_model_construction, project_root=tmp_path)
    assert (
        BackendFlags(
            torch.are_deterministic_algorithms_enabled(),
            torch.backends.cudnn.benchmark,
            torch.backends.cudnn.deterministic,
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
        )
        == before
    )


def test_lineage_collection_fails_closed_without_project_git_and_locks(tmp_path: Path) -> None:
    with pytest.raises(TeacherAuditError, match="Git lineage"):
        collect_audit_lineage(tmp_path)


def test_lineage_collection_records_git_locks_and_observed_robustbench_commit(tmp_path: Path) -> None:
    checkout = tmp_path / ".external" / "robustbench"
    checkout.mkdir(parents=True)

    def git(root: Path, *arguments: str) -> str:
        completed = subprocess.run(["git", "-C", str(root), *arguments], capture_output=True, text=True, check=True)
        return completed.stdout.strip()

    git(checkout, "init")
    git(checkout, "config", "user.email", "audit@example.invalid")
    git(checkout, "config", "user.name", "Audit Fixture")
    (checkout / "source.py").write_text("value = 1\n", encoding="utf-8")
    git(checkout, "add", "source.py")
    git(checkout, "commit", "-m", "fixture")
    robustbench_commit = git(checkout, "rev-parse", "HEAD")

    git(tmp_path, "init")
    git(tmp_path, "config", "user.email", "audit@example.invalid")
    git(tmp_path, "config", "user.name", "Audit Fixture")
    (tmp_path / ".gitignore").write_text(".external/\n", encoding="utf-8")
    (tmp_path / "external.lock.yaml").write_text(
        "version: 1\nrepositories:\n  robustbench:\n    commit: " + robustbench_commit + "\n",
        encoding="utf-8",
    )
    (tmp_path / "teachers.lock.yaml").write_text("version: 1\n", encoding="utf-8")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("before\n", encoding="utf-8")
    git(tmp_path, "add", ".gitignore", "external.lock.yaml", "teachers.lock.yaml", "tracked.txt")
    git(tmp_path, "commit", "-m", "fixture")
    project_commit = git(tmp_path, "rev-parse", "HEAD")
    tracked.write_text("after\n", encoding="utf-8")
    untracked = tmp_path / "untracked.bin"
    untracked.write_bytes(b"first bytes")
    binary_diff = subprocess.run(
        ["git", "-C", str(tmp_path), "diff", "--binary", "HEAD"], capture_output=True, check=True
    ).stdout

    lineage = collect_audit_lineage(tmp_path)
    assert lineage.project_git_sha == project_commit
    assert lineage.project_git_dirty and "tracked.txt" in lineage.project_git_status
    assert lineage.project_binary_diff_sha256 == hashlib.sha256(binary_diff).hexdigest()
    assert lineage.project_untracked_sha256 == {"untracked.bin": hashlib.sha256(b"first bytes").hexdigest()}
    assert lineage.external_lock_sha256 == sha256_file(tmp_path / "external.lock.yaml")
    assert lineage.teachers_lock_sha256 == sha256_file(tmp_path / "teachers.lock.yaml")
    assert lineage.robustbench_locked_commit == lineage.robustbench_observed_commit == robustbench_commit

    untracked.write_bytes(b"different bytes")
    changed = collect_audit_lineage(tmp_path)
    assert changed.project_git_sha == lineage.project_git_sha
    assert changed.project_git_status == lineage.project_git_status
    assert changed.project_binary_diff_sha256 == lineage.project_binary_diff_sha256
    assert changed.project_untracked_sha256 != lineage.project_untracked_sha256
    assert changed.project_untracked_digest_sha256 != lineage.project_untracked_digest_sha256

    (tmp_path / "unsafe-link").symlink_to(untracked)
    with pytest.raises(TeacherAuditError, match="not a regular file"):
        collect_audit_lineage(tmp_path)
