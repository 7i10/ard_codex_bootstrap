from __future__ import annotations

import importlib.util
import json
import random
import subprocess
from pathlib import Path
from types import ModuleType

import pytest
import torch
import yaml

from ard.analysis import summarize
from ard.config.schema import ExperimentConfig
from ard.tracking import (
    QUALITATIVE_COLUMNS,
    LocalTracker,
    NullTracker,
    TrackingError,
    create_tracker,
    stable_run_id,
    validate_tracking_guard,
)
from ard.tracking.adapter import _rng_preserving_rank_zero_phase, collect_git_state

pytestmark = [pytest.mark.t1, pytest.mark.wandb]


def test_collect_git_state_hashes_untracked_file_content(tmp_path: Path) -> None:
    import subprocess

    subprocess.run(["git", "init", str(tmp_path)], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "test"], check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("base", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "base"], check=True, stdout=subprocess.DEVNULL)
    untracked = tmp_path / "untracked.bin"
    spaced = tmp_path / "untracked file.bin"
    nested = tmp_path / "nested" / "untracked.bin"
    untracked.write_bytes(b"first")
    spaced.write_bytes(b"space")
    nested.parent.mkdir()
    nested.write_bytes(b"nested")
    first = collect_git_state(tmp_path)["untracked_sha256"]
    untracked.write_bytes(b"second")
    second = collect_git_state(tmp_path)["untracked_sha256"]
    assert first != second
    assert set(first) == set(second) == {"untracked.bin", "untracked file.bin", "nested/untracked.bin"}


@pytest.mark.parametrize("change", ("tracked", "lock"))
def test_resume_rejects_current_lineage_drift(tmp_path: Path, change: str) -> None:
    import subprocess

    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", str(root)], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "test"], check=True)
    (root / "tracked.txt").write_text("base", encoding="utf-8")
    (root / "external.lock.yaml").write_text("version: 1\nrepositories: {}\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-m", "base"], check=True, stdout=subprocess.DEVNULL)
    output = root / "output"
    output.mkdir()
    cfg = config(output, mode="disabled")
    tracker = create_tracker(config=cfg, output_dir=output, config_hash="abc", root=root)
    tracker.finish()
    target = root / ("tracked.txt" if change == "tracked" else "external.lock.yaml")
    target.write_text("changed", encoding="utf-8")
    with pytest.raises(TrackingError, match="lineage drift"):
        create_tracker(config=cfg, output_dir=output, config_hash="abc", root=root)


def test_resume_rejects_teacher_checkpoint_byte_drift(tmp_path: Path) -> None:
    checkpoint = tmp_path / "teacher.pt"
    checkpoint.write_bytes(b"first")
    digest = __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest()
    cfg = ExperimentConfig.model_validate(
        {
            "schema_version": 2,
            "protocol": {"id": "synthetic_smoke_v2"},
            "tier": "smoke",
            "seeds": {
                k: 0
                for k in (
                    "split",
                    "model_init",
                    "data_order",
                    "augmentation",
                    "train_attack",
                    "evaluation_attack",
                    "qualitative_panel",
                )
            },
            "dataset": {"name": "synthetic_cifar", "num_samples": 4, "num_classes": 2},
            "student": {"architecture": "fixture_cnn", "num_classes": 2},
            "teacher": {
                "source": "checkpoint",
                "architecture": "fixture_cnn",
                "num_classes": 2,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": digest,
            },
            "method": {"id": "rslad", "version": 1, "attack": {"loss": "kl", "kl_target": "teacher_clean", "steps": 1}},
            "optimizer": {"id": "sgd", "learning_rate": 0.01, "momentum": 0.9, "weight_decay": 0.0, "nesterov": False},
            "scheduler": {"id": "identity", "milestones": [], "gamma": 1.0, "step_at": "epoch_end"},
            "training": {"epochs": 1, "per_rank_batch_size": 2, "global_batch_size": 2},
            "output_dir": str(tmp_path / "run"),
        }
    )
    output = cfg.output_dir
    output.mkdir()
    create_tracker(config=cfg, output_dir=output, config_hash="abc", root=tmp_path).finish()
    checkpoint.write_bytes(b"second")
    with pytest.raises(TrackingError, match="declared SHA"):
        create_tracker(config=cfg, output_dir=output, config_hash="abc", root=tmp_path)


def test_resume_rejects_environment_snapshot_drift(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    tracker = create_tracker(
        config=config(output, mode="disabled"), output_dir=output, config_hash="abc", root=tmp_path
    )
    tracker.finish()
    environment = output / "run-bundle" / "environment.json"
    environment.write_text('{"drift": true}\n', encoding="utf-8")

    with pytest.raises(TrackingError, match="lineage drift: environment"):
        create_tracker(config=config(output, mode="disabled"), output_dir=output, config_hash="abc", root=tmp_path)


def test_manifest_preserves_complete_structured_training_seed_lineage(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    cfg = config(output, mode="disabled")
    create_tracker(config=cfg, output_dir=output, config_hash="abc", root=tmp_path).finish()
    manifest = json.loads((output / "run-bundle" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_seeds"] == cfg.seeds.model_dump(mode="json")
    assert manifest["training_seed"] == cfg.seeds.model_init
    assert manifest["protocol_id"] == cfg.protocol.id


def test_resume_rejects_untracked_content_drift(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", str(root)], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "test"], check=True)
    (root / ".gitignore").write_text("output/\n", encoding="utf-8")
    (root / "tracked.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-m", "base"], check=True, stdout=subprocess.DEVNULL)
    untracked = root / "nested path" / "sample file.bin"
    untracked.parent.mkdir()
    untracked.write_bytes(b"first")
    output = root / "output"
    output.mkdir()
    create_tracker(config=config(output, mode="disabled"), output_dir=output, config_hash="abc", root=root).finish()
    untracked.write_bytes(b"second")

    with pytest.raises(TrackingError, match="lineage drift: git"):
        create_tracker(config=config(output, mode="disabled"), output_dir=output, config_hash="abc", root=root)


def test_directory_digest_keeps_files_when_ancestor_is_named_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "artifacts" / "parent"
    bundle = root / "run-bundle"
    bundle.mkdir(parents=True)
    (bundle / "completion.json").write_text("ok", encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    tracker = create_tracker(
        config=config(output, mode="disabled"), output_dir=output, config_hash="abc", root=tmp_path
    )
    tracker.log_artifact(bundle, name="bundle", artifact_type="run-bundle")
    entry = json.loads((output / "run-bundle" / "manifest.json").read_text(encoding="utf-8"))["artifacts"][-1]
    assert any(item["path"] == "completion.json" for item in entry["files"])
    digest = __import__("hashlib").sha256()
    expected_files = []
    for candidate in sorted(bundle.rglob("*")):
        relative = candidate.relative_to(bundle)
        if not candidate.is_file() or relative.name == "manifest.json" or relative.parts[0] == "artifacts":
            continue
        name = relative.as_posix()
        file_hash = __import__("hashlib").sha256(candidate.read_bytes()).hexdigest()
        digest.update(name.encode() + b"\0" + file_hash.encode() + b"\n")
        expected_files.append({"path": name, "sha256": file_hash})
    assert entry["files"] == expected_files and entry["directory_digest"] == digest.hexdigest()


ROOT = Path(__file__).resolve().parents[2]


def config(output: Path, *, mode: str = "offline") -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "schema_version": 2,
            "protocol": {"id": "synthetic_smoke_v2"},
            "tier": "smoke",
            "seeds": {
                k: 9
                for k in (
                    "split",
                    "model_init",
                    "data_order",
                    "augmentation",
                    "train_attack",
                    "evaluation_attack",
                    "qualitative_panel",
                )
            },
            "dataset": {"name": "synthetic_cifar", "num_samples": 4, "num_classes": 2},
            "student": {"architecture": "fixture_cnn", "num_classes": 2},
            "method": {"id": "pgd_at", "version": 1, "attack": {"steps": 1}},
            "optimizer": {"id": "sgd", "learning_rate": 0.01, "momentum": 0.9, "weight_decay": 0.0, "nesterov": False},
            "scheduler": {"id": "identity", "milestones": [], "gamma": 1.0, "step_at": "epoch_end"},
            "training": {"epochs": 1, "per_rank_batch_size": 2, "global_batch_size": 2},
            "tracking": {"mode": mode, "project": "ard-test", "run_id": "stable-run"}
            if mode != "disabled"
            else {"mode": mode},
            "output_dir": str(output),
        }
    )


def production_config(output: Path, dataset_root: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "schema_version": 2,
            "protocol": {"id": "controlled_cifar10_r18_v1"},
            "tier": "production",
            "seeds": {
                "split": 20260722,
                "model_init": 9,
                "data_order": 9,
                "augmentation": 9,
                "train_attack": 9,
                "evaluation_attack": 0,
                "qualitative_panel": 9,
            },
            "dataset": {
                "name": "cifar10",
                "root": str(dataset_root),
                "split": "train",
                "download": False,
                "num_classes": 10,
                "image_size": 32,
            },
            "student": {
                "architecture": "saad_resnet18_cifar_v1",
                "num_classes": 10,
                "preprocessing_owner": "student_adapter",
                "normalization": {"profile": "cifar10_raw_identity"},
            },
            "teacher": {
                "source": "checkpoint",
                "architecture": "resnet18_cifar",
                "num_classes": 10,
                "normalization": {"profile": "cifar10_standard"},
                "checkpoint": str(dataset_root / "teacher.pt"),
                "checkpoint_sha256": "a" * 64,
            },
            "method": {
                "id": "rslad",
                "version": 1,
                "attack": {
                    "loss": "kl",
                    "kl_target": "teacher_clean",
                    "temperature": 1.0,
                    "temperature_squared": True,
                    "epsilon": "8/255",
                    "step_size": "2/255",
                    "steps": 10,
                    "random_start": True,
                    "student_mode": "eval",
                    "teacher_mode": "eval",
                },
                "selection_attack": {
                    "loss": "ce",
                    "temperature": 1.0,
                    "temperature_squared": True,
                    "epsilon": "8/255",
                    "step_size": "2/255",
                    "steps": 20,
                    "random_start": True,
                    "student_mode": "eval",
                    "teacher_mode": "eval",
                },
            },
            "optimizer": {"id": "sgd", "learning_rate": 0.1, "momentum": 0.9, "weight_decay": 5e-4, "nesterov": False},
            "scheduler": {"id": "multistep", "milestones": [100, 150], "gamma": 0.1, "step_at": "epoch_end"},
            "training": {
                "epochs": 200,
                "per_rank_batch_size": 128,
                "global_batch_size": 128,
                "deterministic": True,
                "validation_fraction": 0.1,
            },
            "tracking": {
                "mode": "offline_sync",
                "entity": "ard-test-entity",
                "project": "ard-test-project",
                "group": "ard-test-group",
            },
            "output_dir": str(output),
            "evaluation": {"seed": 0},
        }
    )


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def _lineage_repository(tmp_path: Path, *, main_commit: bool = True) -> tuple[Path, ExperimentConfig]:
    external = tmp_path / ".external" / "saad"
    external.mkdir(parents=True)
    _git(external, "init")
    _git(external, "config", "user.email", "test@example.invalid")
    _git(external, "config", "user.name", "Test")
    (external / "source.txt").write_text("pinned\n", encoding="utf-8")
    _git(external, "add", "source.txt")
    _git(external, "commit", "-m", "pinned fixture")
    _git(external, "remote", "add", "origin", "https://github.com/HongsinLee/saad.git")
    external_sha = _git(external, "rev-parse", "HEAD")

    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / ".gitignore").write_text(".external/\n", encoding="utf-8")
    (tmp_path / "tracked.txt").write_text("committed\n", encoding="utf-8")
    (tmp_path / "external.lock.yaml").write_text(
        yaml.safe_dump(
            {
                "repositories": {
                    "saad": {
                        "url": "https://github.com/HongsinLee/saad.git",
                        "commit": external_sha,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    if main_commit:
        _git(tmp_path, "add", ".gitignore", "tracked.txt", "external.lock.yaml")
        _git(tmp_path, "commit", "-m", "main fixture")
        (tmp_path / "tracked.txt").write_text("tracked dirty state\n", encoding="utf-8")
    return tmp_path, production_config(tmp_path / "output", tmp_path / "data")


@pytest.mark.parametrize("drift", ("head", "dirty"))
def test_resume_rejects_external_checkout_drift(tmp_path: Path, drift: str) -> None:
    root, _ = _lineage_repository(tmp_path)
    output = root / "output"
    output.mkdir()
    with (root / ".git" / "info" / "exclude").open("a", encoding="utf-8") as handle:
        handle.write("output/\n")
    tracker = create_tracker(config=config(output, mode="disabled"), output_dir=output, config_hash="abc", root=root)
    tracker.finish()
    checkout = root / ".external" / "saad"
    (checkout / "source.txt").write_text("changed\n", encoding="utf-8")
    if drift == "head":
        _git(checkout, "add", "source.txt")
        _git(checkout, "commit", "-m", "checkout drift")

    with pytest.raises(TrackingError, match="lineage drift: external"):
        create_tracker(config=config(output, mode="disabled"), output_dir=output, config_hash="abc", root=root)


def _load_sync_wandb() -> ModuleType:
    path = ROOT / "scripts" / "sync_wandb.py"
    spec = importlib.util.spec_from_file_location("ard_test_sync_wandb", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_sync_manifest(
    root: Path,
    *,
    tracking_mode: str = "offline_sync",
    sync_state: str | None = "sync_pending",
    status: str = "sync_pending",
    segments: list[dict[str, str]] | None = None,
) -> Path:
    manifest = root / "run" / "run-bundle" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "run_id": "ID",
                "tracking_mode": tracking_mode,
                "sync_state": sync_state,
                "status": status,
                "wandb_segments": [] if segments is None else segments,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _valid_sync_segments(root: Path) -> tuple[list[dict[str, str]], tuple[Path, Path]]:
    directories = (root / "segments" / "one", root / "segments" / "two")
    for directory in directories:
        directory.mkdir(parents=True)
        (directory / "run-ID.wandb").write_bytes(b"offline segment")
    return ([{"path": str(directory), "run_id": "ID"} for directory in directories], directories)


def test_offline_tracker_has_stable_id_and_complete_local_pending_bundle(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    resolved = output / "resolved_config.yaml"
    resolved.write_text("seed: 9\n", encoding="utf-8")
    tracker = create_tracker(config=config(output), output_dir=output, config_hash="abc", root=Path.cwd())
    assert isinstance(tracker, LocalTracker)
    assert tracker.run_id == "stable-run"
    tracker.attach_resolved_config(resolved)
    tracker.log({"epoch": 0, "val_pgd_accuracy": 0.5}, step=3)
    tracker.finish()
    bundle = output / "run-bundle"
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sync_state"] is None
    assert manifest["status"] == "completed"
    assert (bundle / "resolved_config.yaml").is_file()
    assert json.loads((bundle / "metrics.jsonl").read_text(encoding="utf-8"))["global_step"] == 3


def test_generated_and_resume_run_identity_are_stable(tmp_path: Path) -> None:
    current = config(tmp_path / "run", mode="disabled")
    generated = stable_run_id(current, config_hash="fixed")
    assert generated == stable_run_id(current, config_hash="fixed")
    assert stable_run_id(current, config_hash="fixed", resume_run_id=generated) == generated
    with pytest.raises(TrackingError, match="does not match"):
        stable_run_id(
            current.model_copy(update={"tracking": current.tracking.model_copy(update={"run_id": "other"})}),
            config_hash="fixed",
            resume_run_id=generated,
        )


def test_nonzero_rank_receives_null_tracker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ard.tracking.adapter.is_rank_zero", lambda: False)
    tracker = create_tracker(
        config=config(tmp_path / "run"), output_dir=tmp_path / "run", config_hash="abc", root=Path.cwd()
    )
    assert isinstance(tracker, NullTracker)
    assert tracker.run_id == "stable-run"


def test_tracking_phase_is_rng_observational() -> None:
    import numpy as np
    import torch

    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    expected = (random.random(), float(np.random.rand()), torch.rand(3))
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)

    def consume_rng() -> None:
        random.random()
        np.random.rand()
        torch.rand(20)

    _rng_preserving_rank_zero_phase(consume_rng, phase="rng unit")
    actual = (random.random(), float(np.random.rand()), torch.rand(3))
    assert expected[0] == actual[0] and expected[1] == actual[1]
    assert torch.equal(expected[2], actual[2])


def test_fixed_qualitative_contract_and_run_aggregation(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    tracker = create_tracker(config=config(output), output_dir=output, config_hash="abc", root=Path.cwd())
    assert isinstance(tracker, LocalTracker)
    row = {column: None for column in QUALITATIVE_COLUMNS}
    row.update({"sample_id": 2, "epoch": 1, "clean_correct": True, "robust_correct": False})
    tracker.log_table("fixed_panel", [row])
    with pytest.raises(TrackingError, match="missing required"):
        tracker.log_table("bad_panel", [{"sample_id": 2}])
    assert summarize([0.4, 0.6, 0.8]) == {
        "count": 3,
        "mean": pytest.approx(0.6),
        "std": pytest.approx(0.2),
        "worst": 0.4,
        "best": 0.8,
    }


def test_production_guard_accepts_auditable_tracked_dirty_state(tmp_path: Path) -> None:
    root, cfg = _lineage_repository(tmp_path)
    validate_tracking_guard(cfg, root=root)


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("unborn", "real Git HEAD"),
        ("untracked", "untracked repository files"),
        ("missing_lock", "valid external.lock.yaml saad entry"),
        ("malformed_lock", "valid external.lock.yaml saad entry"),
        ("remote_mismatch", "origin, commit, or clean state"),
        ("sha_mismatch", "origin, commit, or clean state"),
        ("dirty_external", "origin, commit, or clean state"),
    ),
)
def test_production_guard_rejects_incomplete_lineage(tmp_path: Path, case: str, message: str) -> None:
    root, cfg = _lineage_repository(tmp_path, main_commit=case != "unborn")
    external = root / ".external" / "saad"
    if case == "untracked":
        (root / "untracked.txt").write_text("not auditable\n", encoding="utf-8")
    elif case == "missing_lock":
        (root / "external.lock.yaml").unlink()
    elif case == "malformed_lock":
        (root / "external.lock.yaml").write_text("repositories: []\n", encoding="utf-8")
    elif case == "remote_mismatch":
        _git(external, "remote", "set-url", "origin", "https://example.invalid/not-saad.git")
    elif case == "sha_mismatch":
        (root / "external.lock.yaml").write_text(
            yaml.safe_dump(
                {
                    "repositories": {
                        "saad": {
                            "url": "https://github.com/HongsinLee/saad.git",
                            "commit": "0" * 40,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
    elif case == "dirty_external":
        (external / "source.txt").write_text("dirty external\n", encoding="utf-8")

    with pytest.raises(TrackingError, match=message):
        validate_tracking_guard(cfg, root=root)


def test_repro_offline_sync_fails_when_wandb_init_raises(tmp_path: Path) -> None:
    class FailingWandb:
        @staticmethod
        def init(**kwargs: object) -> None:
            del kwargs
            raise RuntimeError("offline init failed")

    cfg = production_config(tmp_path / "run", tmp_path / "data").model_copy(update={"tier": "repro"})
    with pytest.raises(TrackingError, match="requested W&B tracker could not initialize"):
        create_tracker(
            config=cfg,
            output_dir=tmp_path / "run",
            config_hash="abc",
            root=tmp_path,
            wandb_module=FailingWandb(),
        )
    manifest = json.loads((tmp_path / "run" / "run-bundle" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed" and manifest["wandb_initialized"] is False


def test_failed_wandb_initialization_retries_with_never_resume(tmp_path: Path) -> None:
    class FailingWandb:
        @staticmethod
        def init(**kwargs: object) -> None:
            del kwargs
            raise RuntimeError("first init failure")

    class Run:
        url = "mock://run"
        dir = str(tmp_path / "wandb")
        summary: dict[str, object] = {}

        def finish(self, *, exit_code: int) -> None:
            del exit_code

    class WorkingWandb:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] = {}

        def init(self, **kwargs: object) -> Run:
            self.kwargs = dict(kwargs)
            return Run()

    cfg = production_config(tmp_path / "run", tmp_path / "data").model_copy(update={"tier": "repro"})
    with pytest.raises(TrackingError):
        create_tracker(
            config=cfg, output_dir=tmp_path / "run", config_hash="abc", root=tmp_path, wandb_module=FailingWandb()
        )
    working = WorkingWandb()
    tracker = create_tracker(
        config=cfg, output_dir=tmp_path / "run", config_hash="abc", root=tmp_path, wandb_module=working
    )
    assert working.kwargs["resume"] == "never"
    tracker.finish(status="failed")


def test_failed_finish_uses_wandb_failure_exit_code(tmp_path: Path) -> None:
    class Run:
        url = "mock://run"
        dir = str(tmp_path / "wandb")
        summary: dict[str, object] = {}
        exit_code: int | None = None

        def finish(self, *, exit_code: int) -> None:
            self.exit_code = exit_code

    class Wandb:
        def __init__(self) -> None:
            self.run = Run()

        def init(self, **kwargs: object) -> Run:
            del kwargs
            return self.run

    output = tmp_path / "output"
    output.mkdir()
    wandb = Wandb()
    tracker = create_tracker(
        config=config(output), output_dir=output, config_hash="abc", root=tmp_path, wandb_module=wandb
    )
    tracker.finish(status="failed")
    assert wandb.run.exit_code == 1


def test_sync_wandb_pending_manifests_excludes_ordinary_offline_completed(tmp_path: Path) -> None:
    sync_wandb = _load_sync_wandb()
    _write_sync_manifest(
        tmp_path,
        tracking_mode="offline",
        sync_state=None,
        status="completed",
    )
    assert sync_wandb.pending_manifests(tmp_path) == ()


@pytest.mark.parametrize("case", ("no_segments", "mismatched_id", "missing_dir", "no_wandb"))
def test_sync_wandb_invalid_pending_segment_fails_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    sync_wandb = _load_sync_wandb()
    segment_dir = tmp_path / "segment"
    if case != "missing_dir":
        segment_dir.mkdir()
    if case not in {"missing_dir", "no_wandb"}:
        (segment_dir / "run-ID.wandb").write_bytes(b"offline segment")
    segments: list[dict[str, str]] = []
    if case != "no_segments":
        segments = [{"path": str(segment_dir), "run_id": "OTHER" if case == "mismatched_id" else "ID"}]
    manifest = _write_sync_manifest(tmp_path, segments=segments)

    def unexpected_run(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("invalid sync input must not invoke wandb")

    monkeypatch.setattr(sync_wandb.subprocess, "run", unexpected_run)
    assert sync_wandb.main(["--root", str(tmp_path)]) == 1
    state = json.loads(manifest.read_text(encoding="utf-8"))
    assert state["sync_state"] == "sync_pending" and "sync_cursor" not in state
    assert not (manifest.parent / "sync-complete.json").exists()


def test_sync_wandb_two_segments_append_and_commit_state_after_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync_wandb = _load_sync_wandb()
    segments, directories = _valid_sync_segments(tmp_path)
    manifest = _write_sync_manifest(tmp_path, segments=segments)
    commands: list[list[str]] = []

    def successful_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(sync_wandb.subprocess, "run", successful_run)
    assert sync_wandb.main(["--root", str(tmp_path)]) == 0
    assert commands == [
        ["wandb", "sync", "--id", "ID", str(directories[0])],
        ["wandb", "sync", "--id", "ID", "--append", str(directories[1])],
    ]
    marker = manifest.parent / "sync-complete.json"
    assert json.loads(marker.read_text(encoding="utf-8")) == {"run_id": "ID", "synced": True}
    updated = json.loads(manifest.read_text(encoding="utf-8"))
    assert updated["status"] == "completed"
    assert updated["sync_state"] == "synced"


def test_sync_wandb_keeps_application_failure_status_after_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sync_wandb = _load_sync_wandb()
    segments, _ = _valid_sync_segments(tmp_path)
    manifest = _write_sync_manifest(tmp_path, status="failed", segments=segments)
    monkeypatch.setattr(sync_wandb.subprocess, "run", lambda command: subprocess.CompletedProcess(command, 0))

    assert sync_wandb.main(["--root", str(tmp_path)]) == 0

    state = json.loads(manifest.read_text(encoding="utf-8"))
    assert state["status"] == "failed" and state["sync_state"] == "synced"


def test_sync_wandb_second_segment_failure_keeps_pending_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync_wandb = _load_sync_wandb()
    segments, directories = _valid_sync_segments(tmp_path)
    manifest = _write_sync_manifest(tmp_path, segments=segments)
    commands: list[list[str]] = []

    def fail_second_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0 if len(commands) == 1 else 1)

    monkeypatch.setattr(sync_wandb.subprocess, "run", fail_second_run)
    assert sync_wandb.main(["--root", str(tmp_path)]) == 1
    assert commands == [
        ["wandb", "sync", "--id", "ID", str(directories[0])],
        ["wandb", "sync", "--id", "ID", "--append", str(directories[1])],
    ]
    state = json.loads(manifest.read_text(encoding="utf-8"))
    assert state["sync_state"] == "sync_pending" and state["sync_cursor"] == 1
    assert not (manifest.parent / "sync-complete.json").exists()
    commands.clear()
    monkeypatch.setattr(
        sync_wandb.subprocess,
        "run",
        lambda command: commands.append(command) or subprocess.CompletedProcess(command, 0),
    )
    assert sync_wandb.main(["--root", str(tmp_path)]) == 0
    assert commands == [["wandb", "sync", "--id", "ID", "--append", str(directories[1])]]


def test_sync_wandb_dry_run_validates_and_prints_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sync_wandb = _load_sync_wandb()
    segments, directories = _valid_sync_segments(tmp_path)
    manifest = _write_sync_manifest(tmp_path, segments=segments)
    original = manifest.read_bytes()

    def unexpected_run(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("dry-run must not invoke wandb")

    monkeypatch.setattr(sync_wandb.subprocess, "run", unexpected_run)
    assert sync_wandb.main(["--root", str(tmp_path), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert output.splitlines() == [
        f"would sync: wandb sync --id ID {directories[0]}",
        f"would sync: wandb sync --id ID --append {directories[1]}",
    ]
    assert manifest.read_bytes() == original
    assert not (manifest.parent / "sync-complete.json").exists()


def test_offline_sync_uses_offline_wandb_mode_and_public_artifact_api(tmp_path: Path) -> None:
    class Artifact:
        def __init__(self, name: str, type: str) -> None:
            self.name, self.type, self.files, self.directories = name, type, [], []

        def add_file(self, path: str, name: str) -> None:
            self.files.append((path, name))

        def add_dir(self, path: str) -> None:
            self.directories.append(path)

    class Run:
        url = "mock://run"

        def __init__(self, run_dir: Path) -> None:
            self.dir = str(run_dir)
            self.summary: dict = {}
            self.artifacts: list[tuple[Artifact, list[str]]] = []
            self.logged: list[dict] = []

        def log(self, values: dict, step: int | None = None) -> None:
            del step
            self.logged.append(values)

        def log_artifact(self, artifact: Artifact, aliases: list[str]) -> None:
            self.artifacts.append((artifact, aliases))

        def finish(self, *, exit_code: int) -> None:
            assert exit_code == 0

    class Wandb:
        def __init__(self, run_dir: Path) -> None:
            self.kwargs: dict | None = None
            self.run = Run(run_dir)
            self.images: list[str] = []
            self.tables: list[dict[str, object]] = []
            self.created_artifacts: list[Artifact] = []

        def init(self, **kwargs: object) -> Run:
            self.kwargs = dict(kwargs)
            return self.run

        def Image(self, path: str) -> tuple[str, str]:
            self.images.append(path)
            return ("image", path)

        def Table(self, **kwargs: object) -> dict[str, object]:
            self.tables.append(dict(kwargs))
            return dict(kwargs)

        def Artifact(self, name: str, type: str) -> Artifact:
            artifact = Artifact(name, type)
            self.created_artifacts.append(artifact)
            return artifact

    output = tmp_path / "run"
    output.mkdir()
    wandb_parent = tmp_path / "offline-run-stable-run"
    wandb_files = wandb_parent / "files"
    wandb_files.mkdir(parents=True)
    (wandb_parent / "run-stable-run.wandb").write_bytes(b"offline segment")
    fake = Wandb(wandb_files)
    cfg = config(output).model_copy(
        update={"tracking": config(output).tracking.model_copy(update={"mode": "offline_sync"})}
    )
    tracker = create_tracker(config=cfg, output_dir=output, config_hash="abc", root=Path.cwd(), wandb_module=fake)
    assert isinstance(tracker, LocalTracker)
    assert fake.kwargs is not None and fake.kwargs["mode"] == "offline"
    manifest = json.loads((output / "run-bundle" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["wandb_segments"] == [{"path": str(wandb_parent.resolve()), "run_id": "stable-run"}]
    checkpoint = output / "last.pt"
    checkpoint.write_bytes(b"checkpoint")
    tracker.log_artifact(checkpoint, name="model-stable-run-last", artifact_type="model", aliases=("last",))
    assert fake.run.artifacts[0][0].name == "model-stable-run-last"

    tensor_row = {column: None for column in QUALITATIVE_COLUMNS}
    tensor_row.update(
        {
            "sample_id": 2,
            "epoch": 1,
            "clean_image": torch.zeros(3, 4, 4),
            "adversarial_image": torch.ones(3, 4, 4),
            "perturbation_visualization": torch.full((3, 4, 4), 0.5),
        }
    )
    nullable_row = {column: None for column in QUALITATIVE_COLUMNS}
    nullable_row.update({"sample_id": 3, "epoch": 1})
    tracker.log_table("fixed_panel", [tensor_row, nullable_row])
    assert len(fake.images) == 3
    assert all(Path(path).is_file() and Path(path).suffix == ".png" for path in fake.images)
    assert len(fake.tables) == 1
    image_columns = [
        QUALITATIVE_COLUMNS.index(name)
        for name in (
            "clean_image",
            "adversarial_image",
            "perturbation_visualization",
        )
    ]
    table_data = fake.tables[0]["data"]
    assert isinstance(table_data, list)
    assert all(table_data[1][index] is None for index in image_columns)

    directory = output / "directory-artifact"
    directory.mkdir()
    (directory / "payload.txt").write_text("payload\n", encoding="utf-8")
    tracker.log_artifact(directory, name="directory-stable-run", artifact_type="run-bundle")
    assert fake.created_artifacts[-1].directories == [str(directory)]


def test_wandb_artifact_failure_rolls_back_local_manifest_entry(tmp_path: Path) -> None:
    class Artifact:
        def __init__(self, name: str, type: str) -> None:
            del name, type

        def add_file(self, path: str, name: str) -> None:
            del path, name

    class Run:
        url = "mock://run"
        dir = str(tmp_path / "wandb")
        summary: dict[str, object] = {}

        def log_artifact(self, artifact: Artifact, aliases: list[str]) -> None:
            del artifact, aliases
            raise RuntimeError("W&B publication failed")

        def finish(self, *, exit_code: int) -> None:
            del exit_code

    class Wandb:
        def init(self, **kwargs: object) -> Run:
            del kwargs
            return Run()

        def Artifact(self, name: str, type: str) -> Artifact:
            return Artifact(name, type)

    output = tmp_path / "output"
    output.mkdir()
    tracker = create_tracker(
        config=config(output), output_dir=output, config_hash="abc", root=tmp_path, wandb_module=Wandb()
    )
    checkpoint = output / "last.pt"
    checkpoint.write_bytes(b"checkpoint")
    with pytest.raises(RuntimeError, match="publication failed"):
        tracker.log_artifact(checkpoint, name="model-stable-run-last", artifact_type="model")
    manifest = json.loads((output / "run-bundle" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"] == []
    assert not (output / "run-bundle" / "artifacts" / "model-stable-run-last").exists()


def test_wandb_failure_preserves_prior_version_of_same_named_artifact(tmp_path: Path) -> None:
    class Artifact:
        def __init__(self, name: str, type: str) -> None:
            del name, type

        def add_file(self, path: str, name: str) -> None:
            del path, name

    class Run:
        url = "mock://run"
        dir = str(tmp_path / "wandb")
        summary: dict[str, object] = {}
        fail = False

        def log_artifact(self, artifact: Artifact, aliases: list[str]) -> None:
            del artifact, aliases
            if self.fail:
                raise RuntimeError("second publication failed")

        def finish(self, *, exit_code: int) -> None:
            del exit_code

    class Wandb:
        def __init__(self) -> None:
            self.run = Run()

        def init(self, **kwargs: object) -> Run:
            del kwargs
            return self.run

        def Artifact(self, name: str, type: str) -> Artifact:
            return Artifact(name, type)

    output = tmp_path / "output"
    output.mkdir()
    wandb = Wandb()
    tracker = create_tracker(
        config=config(output), output_dir=output, config_hash="abc", root=tmp_path, wandb_module=wandb
    )
    checkpoint = output / "last.pt"
    checkpoint.write_bytes(b"first")
    tracker.log_artifact(checkpoint, name="model-stable-run-last", artifact_type="model")
    manifest_path = output / "run-bundle" / "manifest.json"
    prior = json.loads(manifest_path.read_text(encoding="utf-8"))["artifacts"][-1]
    prior_copy = (output / "run-bundle" / prior["local_path"] / "last.pt").read_bytes()
    checkpoint.write_bytes(b"second")
    wandb.run.fail = True
    with pytest.raises(RuntimeError, match="second publication failed"):
        tracker.log_artifact(checkpoint, name="model-stable-run-last", artifact_type="model")
    current = json.loads(manifest_path.read_text(encoding="utf-8"))["artifacts"]
    assert current == [prior] and (output / "run-bundle" / prior["local_path"] / "last.pt").read_bytes() == prior_copy


def test_failed_finish_removes_bundle_artifact_and_records_exact_snapshot(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    tracker = create_tracker(
        config=config(output, mode="disabled"), output_dir=output, config_hash="abc", root=tmp_path
    )
    bundle = output / "run-bundle"
    (bundle / "completion.json").write_text("complete\n", encoding="utf-8")
    tracker.log_artifact(bundle, name="run-bundle-stable-run", artifact_type="run-bundle")
    tracker.finish(status="failed")

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed" and manifest["artifacts"] == []
    assert not (bundle / "completion.json").exists()
    assert (bundle / "error-marker.txt").read_text(encoding="utf-8").strip() == "application failure recorded"
    digest = __import__("hashlib").sha256()
    expected_files = []
    for candidate in sorted(bundle.rglob("*")):
        relative = candidate.relative_to(bundle)
        if not candidate.is_file() or relative.name == "manifest.json" or relative.parts[0] == "artifacts":
            continue
        path = relative.as_posix()
        value = __import__("hashlib").sha256(candidate.read_bytes()).hexdigest()
        digest.update(path.encode() + b"\0" + value.encode() + b"\n")
        expected_files.append({"path": path, "sha256": value})
    assert manifest["failure_snapshot"] == {
        "directory_digest": digest.hexdigest(),
        "files": expected_files,
        "digest_excludes": ["manifest.json", "artifacts/"],
    }


def test_noop_resume_rejects_failed_prior_finalization(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    create_tracker(config=config(output, mode="disabled"), output_dir=output, config_hash="abc", root=tmp_path).finish(
        status="failed"
    )
    resumed = create_tracker(
        config=config(output, mode="disabled"), output_dir=output, config_hash="abc", root=tmp_path
    )
    assert isinstance(resumed, LocalTracker)

    with pytest.raises(TrackingError, match="completed prior manifest"):
        resumed.validate_terminal_resume()
