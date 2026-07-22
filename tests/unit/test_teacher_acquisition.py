from __future__ import annotations

import sys
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

pytestmark = [pytest.mark.t0, pytest.mark.t1, pytest.mark.unit]

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import acquire_robustbench_teachers as acquire_script  # noqa: E402
import audit_robustbench_teacher as audit_script  # noqa: E402
from acquire_robustbench_teachers import (  # noqa: E402
    AcquisitionError,
    acquire,
    expected_checkpoint_path,
)
from audit_robustbench_teacher import BackendFlags, audit  # noqa: E402

from ard.config.schema import NormalizationConfig  # noqa: E402
from ard.models.teacher import TeacherAdapter, TeacherMetadata  # noqa: E402
from ard.models.teacher_registry import TeacherRegistry, sha256_file  # noqa: E402


class TinyTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(3 * 32 * 32, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(inputs.flatten(1))


def _spec(
    *,
    registry_id: str = "chen2021_ltd_wrn34_10",
    checkpoint_status: str = "missing",
    checkpoint_sha256: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        registry_id=registry_id,
        upstream_model_id="Chen2021LTD_WRN34_10",
        expected_parameter_count=sum(parameter.numel() for parameter in TinyTeacher().parameters()),
        checkpoint_status=checkpoint_status,
        checkpoint_sha256=checkpoint_sha256,
    )


class _Registry:
    def __init__(self, spec: SimpleNamespace) -> None:
        self._spec = spec
        self.external_validated = False

    def spec(self, registry_id: str) -> SimpleNamespace:
        assert registry_id == self._spec.registry_id
        return self._spec

    def validate_external(self) -> None:
        self.external_validated = True


def _downloading_tiny_loader(**kwargs: object) -> TinyTeacher:
    model_dir = Path(str(kwargs["model_dir"]))
    model_id = str(kwargs["model_name"])
    checkpoint = model_dir / "cifar10" / "Linf" / f"{model_id}.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"complete checkpoint")
    return TinyTeacher()


def _resolver(loader: object):
    def resolve(_root: Path, _local_only: bool):
        return loader

    return resolve


def test_acquisition_requires_explicit_opt_in_and_allowlist(tmp_path: Path) -> None:
    registry = _Registry(_spec())
    with pytest.raises(AcquisitionError, match="--allow-network"):
        acquire(
            root=tmp_path,
            registry_id="chen2021_ltd_wrn34_10",
            model_dir=tmp_path / "models",
            allow_network=False,
            registry_loader=lambda _root: registry,  # type: ignore[arg-type]
            load_model_resolver=_resolver(_downloading_tiny_loader),  # type: ignore[arg-type]
        )
    assert not registry.external_validated
    with pytest.raises(AcquisitionError, match="allowlisted"):
        acquire(
            root=tmp_path,
            registry_id="unapproved",
            model_dir=tmp_path / "models",
            allow_network=True,
            registry_loader=lambda _root: registry,  # type: ignore[arg-type]
            load_model_resolver=_resolver(_downloading_tiny_loader),  # type: ignore[arg-type]
        )


def test_acquisition_validates_staging_and_publishes_without_retaining_stage(tmp_path: Path) -> None:
    registry = _Registry(_spec())
    model_dir = tmp_path / "models"
    report = acquire(
        root=tmp_path,
        registry_id="chen2021_ltd_wrn34_10",
        model_dir=model_dir,
        allow_network=True,
        registry_loader=lambda _root: registry,  # type: ignore[arg-type]
        load_model_resolver=_resolver(_downloading_tiny_loader),  # type: ignore[arg-type]
    )
    published = expected_checkpoint_path(model_dir, registry._spec)
    assert registry.external_validated
    assert published.read_bytes() == b"complete checkpoint"
    assert report.checkpoint == str(published)
    assert report.checkpoint_sha256 == sha256_file(published)
    assert report.logits_shape == (1, 10)
    assert not list(tmp_path.glob(".robustbench-stage-*"))


def test_acquisition_staging_failure_never_publishes_partial_checkpoint(tmp_path: Path) -> None:
    registry = _Registry(_spec())
    model_dir = tmp_path / "models"

    def partial_then_fail(**kwargs: object) -> TinyTeacher:
        model_dir_arg = Path(str(kwargs["model_dir"]))
        checkpoint = model_dir_arg / "cifar10" / "Linf" / "Chen2021LTD_WRN34_10.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"partial")
        raise OSError("injected downloader failure")

    with pytest.raises(OSError, match="injected downloader failure"):
        acquire(
            root=tmp_path,
            registry_id="chen2021_ltd_wrn34_10",
            model_dir=model_dir,
            allow_network=True,
            registry_loader=lambda _root: registry,  # type: ignore[arg-type]
            load_model_resolver=_resolver(partial_then_fail),  # type: ignore[arg-type]
        )
    assert not expected_checkpoint_path(model_dir, registry._spec).exists()
    assert not list(tmp_path.glob(".robustbench-stage-*"))


def test_acquisition_never_clobbers_existing_final_checkpoint(tmp_path: Path) -> None:
    registry = _Registry(_spec())
    model_dir = tmp_path / "models"
    destination = expected_checkpoint_path(model_dir, registry._spec)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"existing")

    def must_not_download(**_kwargs: object) -> TinyTeacher:
        pytest.fail("downloader must not run when final checkpoint already exists")

    with pytest.raises(AcquisitionError, match="refusing to overwrite"):
        acquire(
            root=tmp_path,
            registry_id="chen2021_ltd_wrn34_10",
            model_dir=model_dir,
            allow_network=True,
            registry_loader=lambda _root: registry,  # type: ignore[arg-type]
            load_model_resolver=_resolver(must_not_download),  # type: ignore[arg-type]
        )
    assert destination.read_bytes() == b"existing"


def test_acquisition_removes_our_link_on_post_publish_fsync_failure_and_allows_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _Registry(_spec())
    model_dir = tmp_path / "models"
    original_fsync = acquire_script._fsync_directory
    monkeypatch.setattr(
        acquire_script,
        "_fsync_directory",
        lambda _path: (_ for _ in ()).throw(OSError("injected directory fsync failure")),
    )
    with pytest.raises(OSError, match="injected directory fsync failure"):
        acquire(
            root=tmp_path,
            registry_id="chen2021_ltd_wrn34_10",
            model_dir=model_dir,
            allow_network=True,
            registry_loader=lambda _root: registry,  # type: ignore[arg-type]
            load_model_resolver=_resolver(_downloading_tiny_loader),  # type: ignore[arg-type]
        )
    destination = expected_checkpoint_path(model_dir, registry._spec)
    assert not destination.exists()
    monkeypatch.setattr(acquire_script, "_fsync_directory", original_fsync)
    assert acquire(
        root=tmp_path,
        registry_id="chen2021_ltd_wrn34_10",
        model_dir=model_dir,
        allow_network=True,
        registry_loader=lambda _root: registry,  # type: ignore[arg-type]
        load_model_resolver=_resolver(_downloading_tiny_loader),  # type: ignore[arg-type]
    ).checkpoint == str(destination)


def test_verified_acquisition_rejects_mismatched_staged_bytes_without_publish_or_stage(tmp_path: Path) -> None:
    expected = sha256(b"expected verified bytes").hexdigest()
    registry = _Registry(_spec(checkpoint_status="verified", checkpoint_sha256=expected))
    model_dir = tmp_path / "models"

    with pytest.raises(AcquisitionError, match="hash mismatch: expected .* got"):
        acquire(
            root=tmp_path,
            registry_id="chen2021_ltd_wrn34_10",
            model_dir=model_dir,
            allow_network=True,
            registry_loader=lambda _root: registry,  # type: ignore[arg-type]
            load_model_resolver=_resolver(_downloading_tiny_loader),  # type: ignore[arg-type]
        )

    assert not expected_checkpoint_path(model_dir, registry._spec).exists()
    assert not list(tmp_path.glob(".robustbench-stage-*"))


def test_verified_acquisition_publishes_only_matching_staged_bytes(tmp_path: Path) -> None:
    expected = sha256(b"complete checkpoint").hexdigest()
    registry = _Registry(_spec(checkpoint_status="verified", checkpoint_sha256=expected))
    model_dir = tmp_path / "models"

    report = acquire(
        root=tmp_path,
        registry_id="chen2021_ltd_wrn34_10",
        model_dir=model_dir,
        allow_network=True,
        registry_loader=lambda _root: registry,  # type: ignore[arg-type]
        load_model_resolver=_resolver(_downloading_tiny_loader),  # type: ignore[arg-type]
    )

    published = expected_checkpoint_path(model_dir, registry._spec)
    assert published.read_bytes() == b"complete checkpoint"
    assert report.checkpoint_sha256 == expected


def test_pinned_resolver_rejects_preloaded_unverified_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = ModuleType("robustbench")
    fake.__file__ = str(tmp_path / "unverified" / "robustbench" / "__init__.py")
    monkeypatch.setitem(sys.modules, "robustbench", fake)
    with pytest.raises(AcquisitionError, match="preloaded RobustBench module"):
        acquire_script._pinned_load_model(tmp_path, False)


def test_local_only_loader_blocks_gdrive_and_timm_pretrained_routes() -> None:
    class FakeTimm:
        def create_model(self, *_args: object, **_kwargs: object) -> None:
            pytest.fail("unpatched timm route")

    module = ModuleType("robustbench.utils")
    module.timm = FakeTimm()  # type: ignore[attr-defined]
    acquire_script._disable_robustbench_downloads(module)
    with pytest.raises(AcquisitionError, match="local-only audit refuses"):
        module.download_gdrive_new("gdrive", "target")  # type: ignore[attr-defined]
    with pytest.raises(AcquisitionError, match="local-only audit refuses"):
        module.timm.create_model("pretend-timm-model", pretrained=True)  # type: ignore[attr-defined]


@pytest.mark.t2
@pytest.mark.regression
def test_local_audit_uses_registered_bytes_attack_contract_and_restores_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_registry = TeacherRegistry.load(ROOT)
    initial = original_registry.spec("chen2021_ltd_wrn34_10")
    torch.manual_seed(4)
    reference = TinyTeacher()
    checkpoint_bytes = b"registered teacher bytes"
    digest = __import__("hashlib").sha256(checkpoint_bytes).hexdigest()
    spec = replace(
        initial,
        expected_parameter_count=sum(parameter.numel() for parameter in reference.parameters()),
        checkpoint_path=Path("teacher_cache/test.pt"),
        checkpoint_sha256=digest,
        checkpoint_status="verified",
    )
    registry = replace(original_registry, root=tmp_path, specs={spec.registry_id: spec})
    cache_checkpoint = registry.checkpoint_path(spec)
    cache_checkpoint.parent.mkdir(parents=True)
    cache_checkpoint.write_bytes(checkpoint_bytes)
    model_dir = tmp_path / "external-models"
    source_checkpoint = expected_checkpoint_path(model_dir, spec)
    source_checkpoint.parent.mkdir(parents=True)
    source_checkpoint.write_bytes(checkpoint_bytes)
    monkeypatch.setattr(TeacherRegistry, "validate_external", lambda _self: None)
    local_only_values: list[bool] = []

    def pinned_resolver(_root: Path, local_only: bool):
        local_only_values.append(local_only)

        def load(**_kwargs: object) -> TinyTeacher:
            model = TinyTeacher()
            model.load_state_dict(reference.state_dict())
            return model

        return load

    def strict_loader(_config: object) -> TeacherAdapter:
        model = TinyTeacher()
        model.load_state_dict(reference.state_dict())
        return TeacherAdapter(
            model,
            TeacherMetadata(
                architecture="robustbench_wide_resnet",
                num_classes=10,
                normalization=NormalizationConfig(profile="cifar10_raw_identity"),
                checkpoint_sha256=digest,
                registry_id=spec.registry_id,
                upstream_model_id=spec.upstream_model_id,
                preprocessing_owner="teacher_adapter",
            ),
        )

    before = BackendFlags(
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
    )
    production_generate = audit_script.LinfPGD.generate
    observed_attack_configs: list[object] = []

    def record_production_generate(self: object, request: object):
        observed_attack_configs.append(getattr(self, "config"))
        return production_generate(self, request)  # type: ignore[arg-type]

    monkeypatch.setattr(audit_script.LinfPGD, "generate", record_production_generate)
    report = audit(
        root=tmp_path,
        registry_id=spec.registry_id,
        model_dir=model_dir,
        device=torch.device("cpu"),
        registry_loader=lambda _root: registry,
        load_model_resolver=pinned_resolver,  # type: ignore[arg-type]
        ard_loader=strict_loader,  # type: ignore[arg-type]
    )
    assert local_only_values == [True]
    assert report.checkpoint_sha256 == digest
    assert report.preprocessing_owner == "teacher_adapter"
    assert report.input_gradient_l1 > 0 and 0 < report.pgd_linf <= 2 / 255 + 1e-7
    assert len(observed_attack_configs) == 1
    config = observed_attack_configs[0]
    assert getattr(config, "epsilon") == "8/255"
    assert getattr(config, "step_size") == "2/255"
    assert getattr(config, "steps") == 1 and getattr(config, "random_start") is False
    assert getattr(config, "loss") == "ce" and getattr(config, "student_mode") == "eval"
    assert report.max_abs_diff == 0 and report.logits_dtype == "torch.float32"
    assert report.backend_flags == BackendFlags(True, False, True, False, False).report()
    assert (
        BackendFlags(
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        )
        == before
    )
