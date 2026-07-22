from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
import torch
import yaml

from ard.analysis import summarize_checkpoint_groups
from ard.cli import evaluate as evaluate_cli
from ard.cli import train as train_cli
from ard.config.schema import ExperimentConfig
from ard.data import build_dataset, stratified_train_validation_split
from ard.engine.trainer import Trainer
from ard.tracking import QUALITATIVE_COLUMNS, LocalTracker

pytestmark = [pytest.mark.t3, pytest.mark.wandb]


def _run(root: Path, module: str, *args: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
    )


def _training_config(output: Path) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "protocol": {"id": "synthetic_smoke_v2"},
        "tier": "smoke",
        "seeds": {
            k: 4
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
        "dataset": {"name": "synthetic_cifar", "num_samples": 8, "num_classes": 2, "image_size": 4},
        "student": {"architecture": "fixture_cnn", "num_classes": 2},
        "method": {
            "id": "pgd_at",
            "version": 1,
            "attack": {
                "epsilon": "1/255",
                "step_size": "1/255",
                "steps": 1,
                "random_start": False,
            },
        },
        "optimizer": {"id": "sgd", "learning_rate": 0.01, "momentum": 0.9, "weight_decay": 0.0, "nesterov": False},
        "scheduler": {"id": "identity", "milestones": [], "gamma": 1.0, "step_at": "epoch_end"},
        "training": {"epochs": 1, "per_rank_batch_size": 2, "global_batch_size": 2, "device": "cpu"},
        "tracking": {
            "mode": "offline",
            "project": "ard-test",
            "run_id": "offline-smoke",
            "panel_size": 2,
        },
        "evaluation": {
            "dataset": {
                "name": "synthetic_cifar",
                "split": "test",
                "num_samples": 8,
                "num_classes": 2,
                "image_size": 4,
                "seed": 4,
            },
            "checkpoints": "both",
            "panel_size": 2,
            "write_sample_stats": True,
        },
        "output_dir": str(output),
    }


@pytest.fixture(scope="module")
def offline_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    temporary = tmp_path_factory.mktemp("tracking-evaluation")
    output = temporary / "train"
    raw_config = _training_config(output)
    train_config = temporary / "train.yaml"
    train_config.write_text(yaml.safe_dump(raw_config), encoding="utf-8")
    trained = _run(root, "ard.cli.train", "--config", str(train_config))
    assert trained.returncode == 0, trained.stderr

    evaluation_config = temporary / "evaluation.yaml"
    evaluation_config.write_text(yaml.safe_dump(raw_config), encoding="utf-8")
    evaluated = _run(
        root,
        "ard.cli.evaluate",
        "--config",
        str(evaluation_config),
        "--checkpoint-dir",
        str(output),
    )
    assert evaluated.returncode == 0, evaluated.stderr
    return {
        "output": output,
        "config": ExperimentConfig.model_validate(raw_config),
        "trained": trained,
        "evaluated": evaluated,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_offline_training_bundle_and_checkpoint_publication(offline_run: dict[str, Any]) -> None:
    output: Path = offline_run["output"]
    bundle = output / "run-bundle"
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == "offline-smoke"
    assert manifest["sync_state"] is None
    assert manifest["external"]["repositories"]["saad"]["commit"]
    assert (output / "best.pt").is_file() and (output / "last.pt").is_file()

    # Publication copies the completed atomic checkpoint; it must never mutate
    # model tensors while logging an artifact.
    best_entry = next(entry for entry in manifest["artifacts"] if entry["name"] == "model-offline-smoke-best")
    published = bundle / best_entry["local_path"] / "best.pt"
    assert published.is_file()
    original = torch.load(output / "best.pt", map_location="cpu", weights_only=False)["model"]
    copied = torch.load(published, map_location="cpu", weights_only=False)["model"]
    assert original.keys() == copied.keys()
    assert all(torch.equal(original[key], copied[key]) for key in original)


def test_training_sample_stats_cover_exact_train_subset(offline_run: dict[str, Any]) -> None:
    output: Path = offline_run["output"]
    cfg: ExperimentConfig = offline_run["config"]
    stats_path = output / "sample-stats-train.parquet"
    assert stats_path.read_bytes()[:4] == b"PAR1"
    table = pq.read_table(stats_path)
    row_ids = table.column("sample_id").to_pylist()
    dataset = build_dataset(cfg.dataset)
    train_subset, _ = stratified_train_validation_split(
        dataset,
        validation_fraction=cfg.training.validation_fraction,
        seed=cfg.seed,
    )
    assert row_ids == sorted(train_subset.indices)
    assert len(row_ids) == 6


def test_training_panel_has_fixed_columns_and_real_images(offline_run: dict[str, Any]) -> None:
    output: Path = offline_run["output"]
    bundle = output / "run-bundle"
    panel = bundle / "panels" / "panel-epoch-0.jsonl"
    rows = [json.loads(line) for line in panel.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert all(set(row) == set(QUALITATIVE_COLUMNS) for row in rows)
    for row in rows:
        for field in ("clean_image", "adversarial_image", "perturbation_visualization"):
            assert row[field] is not None
            assert (bundle / row[field]).is_file()


def test_training_summary_logs_and_artifact_digests_are_complete(offline_run: dict[str, Any]) -> None:
    output: Path = offline_run["output"]
    bundle = output / "run-bundle"
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    summary = manifest["summary"]
    assert {
        "best_metric",
        "best_epoch",
        "best_clean_accuracy",
        "best_pgd_accuracy",
        "last_clean_accuracy",
        "last_pgd_accuracy",
        "robust_overfit_gap",
    } <= summary.keys()
    assert summary["robust_overfit_gap"] == pytest.approx(summary["best_pgd_accuracy"] - summary["last_pgd_accuracy"])
    assert (bundle / "completion.json").read_text(encoding="utf-8").strip()
    assert (bundle / "error-marker.txt").read_text(encoding="utf-8").strip()

    by_name = {entry["name"]: entry for entry in manifest["artifacts"]}
    expected_files = {
        "model-offline-smoke-best": output / "best.pt",
        "model-offline-smoke-last": output / "last.pt",
        "sample-stats-offline-smoke": output / "sample-stats-train.parquet",
    }
    assert expected_files.keys() <= by_name.keys()
    for name, source in expected_files.items():
        assert by_name[name]["sha256"] == _sha256(source)
    run_bundle = by_name["run-bundle-offline-smoke"]
    assert len(run_bundle["directory_digest"]) == 64
    assert run_bundle["files"]
    assert all(len(item["sha256"]) == 64 for item in run_bundle["files"])


def test_saved_checkpoint_evaluation_is_canonical_and_aggregates(offline_run: dict[str, Any]) -> None:
    output: Path = offline_run["output"]
    evaluation_output = output / "evaluation"
    results = json.loads((evaluation_output / "evaluation-results.json").read_text(encoding="utf-8"))
    assert {item["checkpoint_alias"] for item in results} == {"best", "last"}
    assert {item["checkpoint"] for item in results} == {"best.pt", "last.pt"}
    assert len({item["threat_hash"] for item in results}) == 1
    required_identity_keys = {
        "dataset_identity",
        "dataset_provenance",
        "student_identity",
        "method_identity",
        "training_protocol_identity",
        "teacher_identity",
        "training_seed",
        "training_seeds",
        "evaluation_seed",
    }
    assert all(required_identity_keys <= item.keys() for item in results)
    assert all(item["training_seeds"] == offline_run["config"].seeds.model_dump(mode="json") for item in results)
    assert all(item["training_protocol_identity"]["id"] == offline_run["config"].protocol.id for item in results)
    assert all(
        {"name", "split", "classes", "image_size", "version"} <= item["dataset_identity"].keys()
        and "root" not in item["dataset_identity"]
        and "root" in item["dataset_provenance"]
        for item in results
    )
    assert all(item["sample_stats"] is not None for item in results)
    assert all(Path(item["sample_stats"]).read_bytes()[:4] == b"PAR1" for item in results)

    grouped = summarize_checkpoint_groups(results, metric="pgd_accuracy")
    assert set(grouped) == {"best", "last"}
    by_alias = {item["checkpoint_alias"]: item for item in results}
    for alias in ("best", "last"):
        assert grouped[alias]["count"] == 1
        assert grouped[alias]["mean"] == pytest.approx(by_alias[alias]["pgd_accuracy"])


def test_evaluation_run_bundle_contains_complete_artifact_lineage(offline_run: dict[str, Any]) -> None:
    output: Path = offline_run["output"]
    evaluation_output = output / "evaluation"
    bundle = evaluation_output / "run-bundle"
    assert (bundle / "resolved_config.yaml").is_file()

    expected_outputs = {
        "evaluation-results.json",
        "resolved_evaluation_config.yaml",
        "evaluation-lineage.json",
        "panel-best.jsonl",
        "panel-last.jsonl",
        "sample-stats-best.parquet",
        "sample-stats-last.parquet",
    }
    assert all((evaluation_output / name).is_file() for name in expected_outputs)

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    entries_by_basename = {Path(entry["path"]).name: entry for entry in manifest["artifacts"]}
    required_manifest_paths = expected_outputs | {"run-bundle"}
    assert required_manifest_paths <= entries_by_basename.keys(), (
        f"missing evaluation artifact manifest entries: {sorted(required_manifest_paths - entries_by_basename.keys())}"
    )
    for basename in expected_outputs:
        entry = entries_by_basename[basename]
        assert entry["sha256"] == _sha256(evaluation_output / basename)
        assert (bundle / entry["local_path"] / basename).is_file()
    run_bundle = entries_by_basename["run-bundle"]
    assert len(run_bundle["directory_digest"]) == 64
    assert run_bundle["files"]


def _failed_bundle(output: Path) -> dict[str, Any]:
    bundle = output / "run-bundle"
    assert not (bundle / "completion.json").exists()
    assert (bundle / "error-marker.txt").read_text(encoding="utf-8").strip() == "application failure recorded"
    return json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))


def test_evaluation_setup_failure_marks_tracker_failed(
    offline_run: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    output: Path = offline_run["output"]
    failed_output = output / "evaluation-build-dataset-failure"

    def fail_dataset(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("injected dataset construction failure")

    monkeypatch.setattr(evaluate_cli, "build_dataset", fail_dataset)
    with pytest.raises(RuntimeError, match="injected dataset construction failure"):
        evaluate_cli.main(
            [
                "--config",
                str(output / "resolved_config.yaml"),
                "--checkpoint-dir",
                str(output),
                "--output",
                str(failed_output),
            ]
        )
    assert _failed_bundle(failed_output)["status"] == "failed"


@pytest.mark.parametrize("entrypoint", ("train", "evaluate"))
def test_failure_finalization_does_not_mask_application_error(
    tmp_path: Path, offline_run: dict[str, Any], monkeypatch: pytest.MonkeyPatch, entrypoint: str
) -> None:
    def fail_finish(self: LocalTracker, *, status: str = "completed") -> None:
        del self, status
        raise RuntimeError("injected tracker finalization failure")

    monkeypatch.setattr(LocalTracker, "finish", fail_finish)
    if entrypoint == "train":
        output = tmp_path / "train-original-error"
        config_path = tmp_path / "train-original-error.yaml"
        config_path.write_text(yaml.safe_dump(_training_config(output)), encoding="utf-8")
        monkeypatch.setattr(
            train_cli,
            "build_train_validation_views",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("train original")),
        )
        with pytest.raises(ValueError, match="train original"):
            train_cli.main(["--config", str(config_path)])
    else:
        output: Path = offline_run["output"]
        monkeypatch.setattr(
            evaluate_cli, "build_dataset", lambda _: (_ for _ in ()).throw(ValueError("evaluation original"))
        )
        with pytest.raises(ValueError, match="evaluation original"):
            evaluate_cli.main(
                [
                    "--config",
                    str(output / "resolved_config.yaml"),
                    "--checkpoint-dir",
                    str(output),
                    "--output",
                    str(tmp_path / "evaluation-original-error"),
                ]
            )


def test_evaluation_rejects_weakened_selection_attack(offline_run: dict[str, Any], tmp_path: Path) -> None:
    training_output: Path = offline_run["output"]
    raw = _training_config(tmp_path / "unused")
    raw["evaluation"]["attack"] = {
        "epsilon": "1/510",
        "step_size": "1/510",
        "steps": 1,
        "random_start": False,
        "loss": "ce",
        "student_mode": "eval",
        "teacher_mode": "eval",
    }
    config_path = tmp_path / "weakened-evaluation.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly match"):
        evaluate_cli.main(
            [
                "--config",
                str(config_path),
                "--checkpoint-dir",
                str(training_output),
                "--output",
                str(tmp_path / "weakened-evaluation"),
            ]
        )


def test_evaluation_rejects_temperature_squared_drift_before_output_creation(
    offline_run: dict[str, Any], tmp_path: Path
) -> None:
    training_output: Path = offline_run["output"]
    raw = _training_config(tmp_path / "unused")
    raw["evaluation"]["attack"] = {
        "epsilon": "1/255",
        "step_size": "1/255",
        "steps": 1,
        "random_start": False,
        "loss": "ce",
        "student_mode": "eval",
        "teacher_mode": "eval",
        "temperature_squared": False,
    }
    config_path = tmp_path / "temperature-squared-drift.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    output = tmp_path / "temperature-squared-drift-output"

    with pytest.raises(ValueError, match="exactly match"):
        evaluate_cli.main(
            [
                "--config",
                str(config_path),
                "--checkpoint-dir",
                str(training_output),
                "--output",
                str(output),
            ]
        )
    assert not output.exists()


@pytest.mark.parametrize("world_size", (1, 2))
def test_evaluation_preflight_rejects_mixed_checkpoint_world_size_before_output_creation(
    offline_run: dict[str, Any], tmp_path: Path, world_size: int
) -> None:
    training_output: Path = offline_run["output"]
    checkpoint_dir = tmp_path / f"checkpoints-{world_size}"
    checkpoint_dir.mkdir()
    for name in ("best.pt", "last.pt", "resolved_config.yaml"):
        shutil.copy2(training_output / name, checkpoint_dir / name)
    if world_size != 1:
        payload = torch.load(checkpoint_dir / "last.pt", map_location="cpu", weights_only=False)
        payload["world_size"] = world_size
        torch.save(payload, checkpoint_dir / "last.pt")
    output = tmp_path / f"evaluation-{world_size}"
    args = [
        "--config",
        str(checkpoint_dir / "resolved_config.yaml"),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--output",
        str(output),
    ]
    if world_size == 1:
        assert evaluate_cli.main(args) == 0
        assert output.is_dir()
    else:
        with pytest.raises(ValueError, match="world-size identity"):
            evaluate_cli.main(args)
        assert not output.exists()


@pytest.mark.parametrize("entrypoint", ("train", "evaluate"))
def test_post_prepare_run_bundle_publish_failure_marks_manifest_failed(
    tmp_path: Path,
    offline_run: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
) -> None:
    original = LocalTracker.log_artifact

    def fail_run_bundle(
        self: LocalTracker, path: Path, *, name: str, artifact_type: str, aliases: tuple[str, ...] = ()
    ) -> None:
        original(self, path, name=name, artifact_type=artifact_type, aliases=aliases)
        if artifact_type == "run-bundle":
            raise RuntimeError("injected run-bundle publication failure")

    monkeypatch.setattr(LocalTracker, "log_artifact", fail_run_bundle)
    if entrypoint == "train":
        output = tmp_path / "train-failure"
        config_path = tmp_path / "train-failure.yaml"
        config_path.write_text(yaml.safe_dump(_training_config(output)), encoding="utf-8")
        with pytest.raises(RuntimeError, match="injected run-bundle publication failure"):
            train_cli.main(["--config", str(config_path)])
    else:
        training_output: Path = offline_run["output"]
        output = training_output / "evaluation-publish-failure"
        with pytest.raises(RuntimeError, match="injected run-bundle publication failure"):
            evaluate_cli.main(
                [
                    "--config",
                    str(training_output / "resolved_config.yaml"),
                    "--checkpoint-dir",
                    str(training_output),
                    "--output",
                    str(output),
                ]
            )
    assert _failed_bundle(output)["status"] == "failed"


def test_resume_with_no_remaining_epochs_uses_checkpoint_last_metrics(offline_run: dict[str, Any]) -> None:
    output: Path = offline_run["output"]
    manifest_path = output / "run-bundle" / "manifest.json"
    before_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    before = before_manifest["summary"]
    stats_before = (output / "sample-stats-train.parquet").read_bytes()
    artifacts_before = before_manifest["artifacts"]

    assert train_cli.main(["--config", str(output / "resolved_config.yaml"), "--resume", str(output / "last.pt")]) == 0

    after_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    after = after_manifest["summary"]
    assert after == before
    assert (output / "sample-stats-train.parquet").read_bytes() == stats_before
    assert after_manifest["artifacts"] == artifacts_before


def test_resume_nonimprovement_keeps_best_and_updates_last_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "resumed-summary"
    config = _training_config(output)
    config["training"]["epochs"] = 2
    config_path = tmp_path / "resumed-summary.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    original_fit = Trainer.fit

    def fixed_train_epoch(self: Trainer, loader: object) -> dict[str, float]:
        del self, loader
        return {"loss": 0.0, "clean_accuracy": 0.0, "robust_accuracy": 0.0}

    def fixed_validate_epoch(self: Trainer, loader: object) -> dict[str, float]:
        del loader
        return (
            {"clean_accuracy": 0.5, "pgd_accuracy": 0.8}
            if self.current_epoch == 0
            else {"clean_accuracy": 0.4, "pgd_accuracy": 0.7}
        )

    def first_leg_only(self: Trainer, *args: object, **kwargs: object) -> list[dict[str, float]]:
        if kwargs.get("start_epoch", 0) == 0:
            kwargs["epochs"] = 1
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(Trainer, "train_epoch", fixed_train_epoch)
    monkeypatch.setattr(Trainer, "validate_epoch", fixed_validate_epoch)
    monkeypatch.setattr(Trainer, "fit", first_leg_only)
    assert train_cli.main(["--config", str(config_path)]) == 0
    assert train_cli.main(["--config", str(config_path), "--resume", str(output / "last.pt")]) == 0

    summary = json.loads((output / "run-bundle" / "manifest.json").read_text(encoding="utf-8"))["summary"]
    assert summary == {
        "best_metric": 0.8,
        "best_epoch": 0,
        "best_clean_accuracy": 0.5,
        "best_pgd_accuracy": 0.8,
        "last_clean_accuracy": 0.4,
        "last_pgd_accuracy": 0.7,
        "robust_overfit_gap": pytest.approx(0.1),
    }
