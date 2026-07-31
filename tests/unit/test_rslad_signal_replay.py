from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from ard.analysis import rslad_signal_replay
from ard.analysis.rslad_signal_replay import (
    FEATURE_EPOCHS,
    OUTCOME_EPOCHS,
    PANEL_EMA_BETA,
    PERIODIC_EPOCHS,
    RSLADSignalReplayError,
    TemporalPanelInventory,
    _cache_paths,
    build_feature_panel,
    build_outcome_panel,
    checkpoint_cache_identity,
    domain_seed,
    join_feature_outcome_panels,
    load_cached_checkpoint,
    portable_cifar10_train_identity,
    predictive_audit,
    replay_lineage,
    semantic_source_paths,
    source_hashes,
    validate_exact_epoch_schedule,
    validate_rslad_replay_attack,
    verify_semantic_sources_tracked,
    write_checkpoint_cache,
    write_replay_outputs,
)
from ard.analysis.signal_audit import CheckpointInventory
from ard.config import load_config
from ard.config.loader import resolved_config_dict
from ard.data import IndexedBatch
from ard.engine.checkpoint import config_digest


def _checkpoint(epoch: int) -> CheckpointInventory:
    return CheckpointInventory(
        run_id="rslad-s0",
        artifact_name=f"model-rslad-last-{epoch}",
        aliases=("last",),
        publication_order=epoch,
        path=f"/immutable/checkpoint-{epoch}.pt",
        sha256=f"{epoch:064x}",
        epoch=epoch,
        sample_state_present=False,
        sample_state_count=0,
        config_hash="a" * 64,
        scientific_git_sha="b" * 40,
    )


def _replay_rows(epochs: tuple[int, ...], *, samples: int = 2) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for epoch in epochs:
        for sample_id in range(samples):
            margin = (0.2 if sample_id == 0 else -0.4) + 0.001 * epoch
            rows.append(
                {
                    "namespace": "train",
                    "sample_id": sample_id,
                    "class_id": sample_id,
                    "epoch": epoch,
                    "teacher_entropy_normalized": 0.2 + 0.001 * epoch,
                    "student_probability_margin": margin,
                    "student_margin_risk": (1.0 - margin) / 2.0,
                    "robust_correct": not (sample_id == 0 and epoch in {104, 114}),
                }
            )
    return rows


def _cache_identity(
    *, batch_size: int = 2, source: str = "a" * 64, raw_digest: str = "c" * 64, teacher_sha: str = "d" * 64
) -> dict[str, object]:
    config = load_config(Path(__file__).parents[2] / "configs" / "experiments" / "synthetic_rslad.yaml")
    return checkpoint_cache_identity(
        checkpoint=_checkpoint(4),
        training_config=config,
        seed_domain="feature",
        base_seed=7,
        expected_count=45000,
        device=torch.device("cpu"),
        replay_batch_size=batch_size,
        saved_resolved_config_mapping_sha256=raw_digest,
        saved_resolved_config_file_sha256="9" * 64,
        teacher_metadata={"checkpoint_sha256": teacher_sha, "registry_id": "teacher"},
        dataset_identity={"dataset": {"name": "cifar10", "split": "train"}},
        analysis_provenance={"git": {"sha": "e" * 40, "dirty": False}, "source_files": {"pgd": source}},
    )


def test_exact_periodic_epoch_schedule_rejects_missing_extra_and_duplicate_epochs() -> None:
    scheduled = tuple(_checkpoint(epoch) for epoch in PERIODIC_EPOCHS)
    assert tuple(item.epoch for item in validate_exact_epoch_schedule(scheduled)) == PERIODIC_EPOCHS

    with pytest.raises(RSLADSignalReplayError, match="expected exactly"):
        validate_exact_epoch_schedule(scheduled[:-1])
    with pytest.raises(RSLADSignalReplayError, match="exactly one"):
        validate_exact_epoch_schedule((*scheduled, _checkpoint(PERIODIC_EPOCHS[-1])))
    with pytest.raises(RSLADSignalReplayError, match="expected exactly"):
        validate_exact_epoch_schedule((*scheduled, _checkpoint(200)))


def test_feature_ema_and_checkpoint_panel_outcome_use_stable_id_joins() -> None:
    features = build_feature_panel(_replay_rows(FEATURE_EPOCHS), expected_count=2)
    outcomes = build_outcome_panel(_replay_rows(OUTCOME_EPOCHS), expected_count=2)

    first = next(row for row in features if row["sample_id"] == 0)
    expected = 0.2 + 0.001 * FEATURE_EPOCHS[0]
    for epoch in FEATURE_EPOCHS[1:]:
        expected = PANEL_EMA_BETA * expected + (1.0 - PANEL_EMA_BETA) * (0.2 + 0.001 * epoch)
    assert first["student_margin_panel_ema"] == pytest.approx(expected, abs=1e-12)
    assert next(row for row in outcomes if row["sample_id"] == 0)["checkpoint_panel_forgetting"] == 1
    assert next(row for row in outcomes if row["sample_id"] == 1)["checkpoint_panel_forgetting"] == 0

    bad = [{**row} for row in outcomes]
    bad[0]["class_id"] = 9
    with pytest.raises(RSLADSignalReplayError, match="class_id mismatch"):
        join_feature_outcome_panels(features, bad, expected_count=2)


def test_feature_and_outcome_seed_domains_are_deterministic_and_independent() -> None:
    feature = domain_seed(base_seed=7, domain="feature")
    outcome = domain_seed(base_seed=7, domain="outcome")
    assert feature == domain_seed(base_seed=7, domain="feature")
    assert feature != outcome
    with pytest.raises(RSLADSignalReplayError, match="domain"):
        domain_seed(base_seed=7, domain="shared")


def test_fixed_domain_seed_and_cache_identity_include_batch_partition() -> None:
    first = _cache_identity()
    later = {**_cache_identity(), "checkpoint": {**_cache_identity()["checkpoint"], "epoch": 99}}
    different_batch = _cache_identity(batch_size=4)
    different_source = _cache_identity(source="f" * 64)
    different_raw = _cache_identity(raw_digest="1" * 64)
    different_teacher = _cache_identity(teacher_sha="2" * 64)
    assert first["attack_seed_base"] == later["attack_seed_base"]
    assert first["protocol"]["batch_size"] == 2
    assert first != different_batch
    assert first != different_source
    assert first != different_raw
    assert first != different_teacher


def test_final_lineage_binds_raw_config_teacher_dataset_runtime_and_analysis_provenance() -> None:
    config = load_config(Path(__file__).parents[2] / "configs" / "experiments" / "synthetic_rslad.yaml")
    panel = TemporalPanelInventory("run", "a" * 64, "b" * 40, 1, ())
    lineage = replay_lineage(
        panel=panel,
        training_config=config,
        expected_count=45000,
        replay_batch_size=128,
        device_type="cpu",
        runtime={"selected_device": "cpu", "torch": "fixture"},
        feature_seed=3,
        outcome_seed=5,
        saved_resolved_config_mapping_sha256="c" * 64,
        saved_resolved_config_file_sha256="9" * 64,
        teacher_metadata={"checkpoint_sha256": "d" * 64, "registry_id": "teacher"},
        dataset_identity={"dataset": {"name": "cifar10", "split": "train"}},
        analysis_provenance={"git": {"sha": "e" * 40, "dirty": False}, "source_files": {"pgd": "f" * 64}},
        feature_results=(),
        outcome_results=(),
    )
    assert lineage["saved_resolved_config_mapping_sha256"] == "c" * 64
    assert lineage["saved_resolved_config_file_sha256"] == "9" * 64
    assert lineage["teacher"]["checkpoint_sha256"] == "d" * 64
    assert lineage["dataset_identity"]["dataset"]["name"] == "cifar10"
    assert lineage["runtime"]["selected_device"] == "cpu"
    assert lineage["analysis_provenance"]["git"]["dirty"] is False
    assert set(lineage["attack_identity"]) == {
        "norm",
        "input_domain",
        "epsilon",
        "epsilon_value",
        "step_size",
        "step_size_value",
        "steps",
        "random_start",
        "loss",
        "kl_target",
        "temperature",
        "temperature_squared",
        "student_mode",
        "teacher_mode",
    }


def test_exact_rslad_threat_accepts_temperature_squared_canonical_config() -> None:
    config = load_config(Path(__file__).parents[2] / "configs" / "experiments" / "synthetic_rslad.yaml")
    assert config.method.attack.temperature_squared is True
    validate_rslad_replay_attack(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("norm", "l2"),
        ("input_domain", "normalized"),
        ("epsilon", "0.03137254901960784"),
        ("epsilon_value", 0.02),
        ("step_size", "0.00784313725490196"),
        ("step_size_value", 0.01),
        ("steps", 9),
        ("random_start", False),
        ("loss", "ce"),
        ("kl_target", "student_clean"),
        ("temperature", 2.0),
        ("temperature_squared", False),
        ("student_mode", "train"),
        ("teacher_mode", "train"),
    ],
)
def test_exact_rslad_threat_rejects_each_canonical_identity_field_drift(field: str, value: object) -> None:
    config = load_config(Path(__file__).parents[2] / "configs" / "experiments" / "synthetic_rslad.yaml")
    attack = config.method.attack.model_copy(update={field: value})
    drifted = SimpleNamespace(method=SimpleNamespace(id="rslad", attack=attack))
    with pytest.raises(RSLADSignalReplayError, match="exact RSLAD"):
        validate_rslad_replay_attack(drifted)


def test_cifar10_population_contract_rejects_wrong_dataset_or_count() -> None:
    resolved = {
        "dataset": {"name": "cifar10", "split": "train"},
        "seeds": {"split": 3},
        "training": {"validation_fraction": 0.1},
    }
    assert portable_cifar10_train_identity(resolved, expected_count=45000)["dataset"]["name"] == "cifar10"
    with pytest.raises(RSLADSignalReplayError, match="expected_count"):
        portable_cifar10_train_identity(resolved, expected_count=44999)
    with pytest.raises(RSLADSignalReplayError, match="CIFAR-10"):
        portable_cifar10_train_identity(
            {**resolved, "dataset": {"name": "cifar100", "split": "train"}}, expected_count=45000
        )


def test_execution_source_hashes_cover_replay_dependencies() -> None:
    assert set(source_hashes()) >= {
        "analysis_module",
        "cli_module",
        "pgd",
        "teacher_risk_replay",
        "signal_audit",
        "robust_margin",
        "model_registry",
        "model_teacher",
        "teacher_registry",
    }


def test_semantic_sources_must_be_git_tracked(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).parents[2]
    calls: list[list[str]] = []

    def tracked(command: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(rslad_signal_replay.subprocess, "run", tracked)
    verify_semantic_sources_tracked(root=root, paths=semantic_source_paths())
    assert "ls-files" in calls[0]
    assert "src/ard/attacks/pgd.py" in calls[0]

    def untracked(*_args: object, **_kwargs: object) -> SimpleNamespace:
        raise rslad_signal_replay.subprocess.CalledProcessError(1, "git")

    monkeypatch.setattr(rslad_signal_replay.subprocess, "run", untracked)
    with pytest.raises(RSLADSignalReplayError, match="not Git-tracked"):
        verify_semantic_sources_tracked(root=root, paths=semantic_source_paths())


def test_cli_rejects_distributed_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    from ard.cli.rslad_signal_replay import _require_single_process

    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(RSLADSignalReplayError, match="one non-distributed"):
        _require_single_process()
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    with pytest.raises(RSLADSignalReplayError, match="one non-distributed"):
        _require_single_process()


def test_saved_resolved_config_mapping_digest_is_distinct_from_file_byte_digest(tmp_path: Path) -> None:
    from ard.cli.rslad_signal_replay import saved_resolved_config_digests

    source = Path(__file__).parents[2] / "configs" / "experiments" / "synthetic_rslad.yaml"
    raw = resolved_config_dict(load_config(source))
    raw["method"]["attack"].pop("trace_step_losses")
    saved = tmp_path / "resolved_config.yaml"
    saved.write_text(yaml.safe_dump(raw), encoding="utf-8")
    digests = saved_resolved_config_digests(saved)
    assert digests["mapping_sha256"] == config_digest(raw)
    assert config_digest(resolved_config_dict(load_config(saved))) != config_digest(raw)
    reformatted = tmp_path / "reformatted.yaml"
    reformatted.write_text("\n" + yaml.safe_dump(raw, indent=4, sort_keys=False), encoding="utf-8")
    reformatted_digests = saved_resolved_config_digests(reformatted)
    assert reformatted_digests["mapping_sha256"] == digests["mapping_sha256"]
    assert reformatted_digests["file_sha256"] != digests["file_sha256"]


def test_common_trajectory_replay_passes_saved_checkpoint_hash_to_strict_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(Path(__file__).parents[2] / "configs" / "experiments" / "synthetic_rslad.yaml")
    checkpoint = _checkpoint(4)
    captured: dict[str, object] = {}
    student = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 10))
    teacher = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 10))
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    def load_selected(*_args: object, **kwargs: object) -> tuple[nn.Module, dict[str, object]]:
        captured["expected"] = kwargs["expected_config_hash"]
        return student, {}

    monkeypatch.setattr(rslad_signal_replay, "load_historical_student", load_selected)
    result = rslad_signal_replay.replay_checkpoint_rows(
        checkpoint=checkpoint,
        training_config=config,
        teacher=teacher,
        loader=(
            IndexedBatch(images=torch.rand(2, 3, 4, 4), labels=torch.tensor([1, 2]), sample_ids=torch.tensor([7, 3])),
        ),
        device=torch.device("cpu"),
        seed_domain="feature",
        base_seed=7,
    )
    assert captured["expected"] == checkpoint.config_hash
    assert len(result.rows) == 2


def test_univariate_models_report_paired_student_minus_entropy_metric_signs() -> None:
    rows = []
    for sample_id in range(200):
        outcome = int(sample_id % 4 in {0, 1})
        rows.append(
            {
                "namespace": "train",
                "sample_id": sample_id,
                "class_id": sample_id % 2,
                "teacher_entropy_normalized": 0.5,
                "student_margin_panel_risk": 0.9 if outcome else 0.1,
                "student_margin_risk_epoch99": 0.9 if outcome else 0.1,
                "checkpoint_panel_forgetting": outcome,
            }
        )
    report = predictive_audit(rows, split_seed=13, bootstrap_seed=17, bootstrap_replicates=40)
    delta = report["student_minus_entropy"]["point_estimates"]
    assert delta["auroc"] > 0.0
    assert delta["auprc"] > 0.0
    assert delta["log_loss"] < 0.0
    bounds = report["student_minus_entropy"]["bootstrap_95"]["metrics"]
    assert bounds["auroc"]["lower"] > 0.0
    assert bounds["log_loss"]["upper"] < 0.0
    assert report["proxy_decision"]["gating_metrics"] == ["auroc", "log_loss"]
    assert report["proxy_decision"]["secondary_metrics"] == ["auprc"]


def test_canonical_json_output_is_byte_deterministic_and_parquet_is_real(tmp_path: Path) -> None:
    observations = _replay_rows((4,), samples=2)
    panel = [
        {
            "namespace": "train",
            "sample_id": row["sample_id"],
            "class_id": row["class_id"],
            "feature_epoch": 99,
            "teacher_entropy_normalized": row["teacher_entropy_normalized"],
            "student_margin_panel_ema": row["student_probability_margin"],
            "student_margin_panel_risk": row["student_margin_risk"],
        }
        for row in observations
    ]
    outcomes = [
        {
            "namespace": "train",
            "sample_id": row["sample_id"],
            "class_id": row["class_id"],
            "outcome_start_epoch": 99,
            "outcome_end_epoch": 199,
            "checkpoint_panel_forgetting": 0,
            "checkpoint_panel_transition_count": 0,
            "final_robust_error": 0,
        }
        for row in observations
    ]
    lineage = {"schema_version": 1, "source": "fixture"}
    report = {"models": {"entropy": {"auroc": 0.5}}, "z": [1, 2]}
    first = write_replay_outputs(
        output_dir=tmp_path / "first",
        feature_observations=observations,
        outcome_observations=observations,
        feature_panel=panel,
        outcome_panel=outcomes,
        lineage=lineage,
        report=report,
    )
    second = write_replay_outputs(
        output_dir=tmp_path / "second",
        feature_observations=list(reversed(observations)),
        outcome_observations=list(reversed(observations)),
        feature_panel=list(reversed(panel)),
        outcome_panel=list(reversed(outcomes)),
        lineage=lineage,
        report=report,
    )
    assert first["lineage"].read_bytes() == second["lineage"].read_bytes()
    assert first["report"].read_bytes() == second["report"].read_bytes()
    assert first["feature_panel"].read_bytes()[:4] == b"PAR1"
    saved_lineage = json.loads(first["lineage"].read_text())
    assert saved_lineage["output_parquet_sha256"]
    assert saved_lineage["predictive_audit_sha256"] == hashlib.sha256(first["report"].read_bytes()).hexdigest()


def test_checkpoint_cache_reuses_only_exact_hash_bound_identity(tmp_path: Path) -> None:
    identity = {
        "checkpoint": {"epoch": 4, "sha256": "a" * 64, "config_hash": "b" * 64, "run_id": "rslad-s0"},
        "protocol": {"seed_domain": "feature"},
        "expected_count": 2,
        "attack_seed_base": 17,
    }
    from ard.analysis.rslad_signal_replay import ReplayCheckpointResult

    result = ReplayCheckpointResult(
        epoch=4,
        seed_domain="feature",
        attack_seed_base=17,
        max_abs_delta=8 / 255,
        rows=tuple(_replay_rows((4,), samples=2)),
    )
    written = write_checkpoint_cache(cache_dir=tmp_path, identity=identity, result=result)
    assert load_cached_checkpoint(cache_dir=tmp_path, identity=identity) == written

    metadata = next(tmp_path.glob("*.json"))
    metadata.unlink()
    assert load_cached_checkpoint(cache_dir=tmp_path, identity=identity) is None
    assert not next(tmp_path.glob("*.parquet"), None)
    parquet_path, _ = _cache_paths(tmp_path, identity)
    temporary = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
    temporary.write_bytes(b"partial")
    assert load_cached_checkpoint(cache_dir=tmp_path, identity=identity) is None
    assert not temporary.exists()
    write_checkpoint_cache(cache_dir=tmp_path, identity=identity, result=result)

    metadata = next(tmp_path.glob("*.json"))
    document = json.loads(metadata.read_text(encoding="utf-8"))
    document["identity"]["attack_seed_base"] = 18
    metadata.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RSLADSignalReplayError, match="identity"):
        load_cached_checkpoint(cache_dir=tmp_path, identity=identity)
