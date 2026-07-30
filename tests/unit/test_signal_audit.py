from __future__ import annotations

import hashlib
import json
from base64 import b64encode
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from ard.analysis.signal_audit import (
    CheckpointInventory,
    SignalAuditError,
    _bootstrap_indices,
    _state_outcomes,
    associate_wandb_versions,
    audit_report,
    binary_metrics,
    bootstrap_metric_delta,
    deterministic_hash_split,
    final_state_association,
    inventory_run_bundle,
    join_samples,
    load_final_sample_stats,
    logical_dataset_fingerprint,
    logical_dataset_identity,
    namespaced_samples,
    prospective_prediction,
    select_prospective_checkpoints,
    validate_sample_partitions,
    validate_teacher_risk_replay,
    write_audit_report,
)
from ard.cli.signal_audit import main as signal_audit_main
from ard.engine.checkpoint import REQUIRED_KEYS, config_digest


def _checkpoint(
    *, run_id: str, config_hash: str, epoch: int, records: dict[str, dict[str, object]]
) -> dict[str, object]:
    payload: dict[str, object] = {key: None for key in REQUIRED_KEYS}
    payload.update(
        {
            "format_version": 1,
            "epoch": epoch,
            "epoch_boundary": "end",
            "model": {},
            "optimizer": {},
            "scheduler": None,
            "scaler": None,
            "rng": [],
            "sampler_epoch": [],
            "sampler_state": [],
            "sample_state": {
                "format_version": 1,
                "ema_decay": 0.9,
                "records": records,
                "pending": [],
                "next_order": 0,
            },
            "global_step": 1,
            "best_metric": 0.0,
            "selection_metadata": {},
            "tracker_run_id": run_id,
            "config_hash": config_hash,
            "world_size": 1,
        }
    )
    assert set(payload) == REQUIRED_KEYS
    return payload


def _manifest_with_checkpoint(tmp_path: Path, *, payload: dict[str, object], alias: str = "last") -> Path:
    bundle = tmp_path / "run-bundle"
    source = tmp_path / "last.pt"
    torch.save(payload, source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    artifact = bundle / "artifacts" / "model-run-last" / digest
    artifact.mkdir(parents=True)
    copied = artifact / source.name
    copied.write_bytes(source.read_bytes())
    manifest = {
        "run_id": "run-0",
        "config_hash": payload["config_hash"],
        "git": {"sha": "a" * 40},
        "artifacts": [
            {
                "name": "model-run-last",
                "type": "model",
                "aliases": [alias],
                "path": str(source),
                "local_path": str(artifact.relative_to(bundle)),
                "sha256": digest,
            }
        ],
    }
    path = bundle / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _add_formal_lineage(manifest_path: Path, *, final_parquet: Path, resolved: dict[str, object]) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["teacher"] = {
        "checkpoint_sha256": resolved["teacher"]["checkpoint_sha256"],
        "registry_id": resolved["teacher"]["registry_id"],
    }
    manifest["training_seed"] = resolved["seeds"]["model_init"]
    manifest["artifacts"].append(
        {
            "name": "sample-stats-run-0",
            "type": "sample-stats",
            "sha256": hashlib.sha256(final_parquet.read_bytes()).hexdigest(),
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (manifest_path.parent / "resolved_config.yaml").write_text(json.dumps(resolved), encoding="utf-8")


def _append_checkpoint_artifact(
    manifest_path: Path, *, payload: dict[str, object], name: str = "model-run-last"
) -> None:
    bundle = manifest_path.parent
    source = bundle.parent / f"checkpoint-{payload['epoch']}" / "last.pt"
    source.parent.mkdir()
    torch.save(payload, source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    artifact = bundle / "artifacts" / name / digest
    artifact.mkdir(parents=True)
    (artifact / source.name).write_bytes(source.read_bytes())
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"].append(
        {
            "name": name,
            "type": "model",
            "aliases": ["last"],
            "path": str(source),
            "local_path": str(artifact.relative_to(bundle)),
            "sha256": digest,
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _records() -> dict[str, dict[str, object]]:
    return {
        "0": {
            "margin_ema": 0.1,
            "seen": 2,
            "robust_correct_count": 1,
            "previous_robust_correct": False,
            "forgetting_count": 1,
            "last_update": 2,
        },
        "1": {
            "margin_ema": 0.2,
            "seen": 2,
            "robust_correct_count": 2,
            "previous_robust_correct": True,
            "forgetting_count": 0,
            "last_update": 2,
        },
    }


def test_inventory_extracts_epoch_state_and_content_addressed_identity(tmp_path: Path) -> None:
    manifest = _manifest_with_checkpoint(
        tmp_path, payload=_checkpoint(run_id="run-0", config_hash="b" * 64, epoch=5, records=_records())
    )

    inventory = inventory_run_bundle(manifest)

    assert len(inventory) == 1
    assert inventory[0].epoch == 5
    assert inventory[0].sample_state_present is True
    assert inventory[0].sample_state_count == 2
    assert inventory[0].publication_order == 0
    assert inventory[0].scientific_git_sha == "a" * 40


def test_inventory_refuses_checkpoint_manifest_identity_mismatch(tmp_path: Path) -> None:
    manifest = _manifest_with_checkpoint(
        tmp_path, payload=_checkpoint(run_id="other-run", config_hash="b" * 64, epoch=5, records=_records())
    )

    with pytest.raises(SignalAuditError, match="identity"):
        inventory_run_bundle(manifest)


def test_inventory_accepts_exact_empty_sample_state_for_stateless_methods(tmp_path: Path) -> None:
    payload = _checkpoint(run_id="run-0", config_hash="b" * 64, epoch=5, records=_records())
    payload["sample_state"] = {}
    manifest = _manifest_with_checkpoint(tmp_path, payload=payload)

    inventory = inventory_run_bundle(manifest)

    assert inventory[0].sample_state_present is False
    assert inventory[0].sample_state_count == 0


def test_namespaces_are_explicit_and_cross_namespace_join_is_rejected() -> None:
    train = namespaced_samples([{"namespace": "train", "sample_id": 0}], namespace="train", expected_count=1)
    test = namespaced_samples([{"namespace": "test", "sample_id": 0}], namespace="test", expected_count=1)
    assert validate_sample_partitions(
        [{"namespace": "train", "sample_id": 0}],
        [{"namespace": "test", "sample_id": 0}],
        train_expected_count=1,
        test_expected_count=1,
    ) == (train, test)

    with pytest.raises(SignalAuditError, match="forbidden"):
        join_samples(train, test)


def test_hash_split_and_clustered_bootstrap_are_deterministic() -> None:
    rows = [{"namespace": "train", "sample_id": sample_id, "class_id": sample_id % 2} for sample_id in range(8)]
    assert deterministic_hash_split(rows, seed=17) == deterministic_hash_split(list(reversed(rows)), seed=17)
    repeated = [
        {"namespace": "train", "sample_id": sample_id, "class_id": sample_id % 2, "outcome": sample_id % 2}
        for sample_id in range(4)
        for _ in range(2)
    ]
    baseline = [0.25 if row["outcome"] else 0.75 for row in repeated]
    candidate = [0.9 if row["outcome"] else 0.1 for row in repeated]
    first = bootstrap_metric_delta(repeated, baseline=baseline, candidate=candidate, seed=7, replicates=25)
    assert first == bootstrap_metric_delta(repeated, baseline=baseline, candidate=candidate, seed=7, replicates=25)
    assert first["clustered_by_sample_id"] is True


def test_cluster_bootstrap_keeps_each_repeated_sample_whole_and_stratifies_only_class() -> None:
    rows = [
        {"namespace": "train", "sample_id": 0, "class_id": 0, "outcome": 0},
        {"namespace": "train", "sample_id": 0, "class_id": 0, "outcome": 1},
        {"namespace": "train", "sample_id": 1, "class_id": 0, "outcome": 0},
        {"namespace": "train", "sample_id": 1, "class_id": 0, "outcome": 0},
        {"namespace": "train", "sample_id": 2, "class_id": 1, "outcome": 1},
        {"namespace": "train", "sample_id": 2, "class_id": 1, "outcome": 0},
        {"namespace": "train", "sample_id": 3, "class_id": 1, "outcome": 1},
        {"namespace": "train", "sample_id": 3, "class_id": 1, "outcome": 1},
    ]
    selected = _bootstrap_indices(rows, seed=2, replicate=1, cluster=True)
    for sample_id in range(4):
        selected_rows = [index for index in selected if rows[index]["sample_id"] == sample_id]
        assert len(selected_rows) in {0, 2, 4}
        if selected_rows:
            assert {index % 2 for index in selected_rows} == {0, 1}


def test_metrics_report_positive_prevalence() -> None:
    metrics = binary_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert metrics == {"auroc": 1.0, "auprc": 1.0, "prevalence": 0.5, "log_loss": pytest.approx(metrics["log_loss"])}


def test_tied_score_average_precision_is_order_invariant() -> None:
    first = binary_metrics([1, 0], [0.5, 0.5])
    second = binary_metrics([0, 1], [0.5, 0.5])
    assert first["auprc"] == pytest.approx(0.5)
    assert second["auprc"] == pytest.approx(0.5)


def test_logical_dataset_fingerprint_excludes_host_root_and_binds_split_identity() -> None:
    base = {
        "dataset": {"name": "cifar10", "root": "/home/shunsukenaito/data", "download": False, "content_sha256": None},
        "seeds": {"split": 11},
        "training": {"validation_fraction": 0.1},
    }
    ferret = {
        **base,
        "dataset": {**base["dataset"], "root": "/home/islab/data", "download": True},
    }
    assert logical_dataset_fingerprint(base, train_expected_count=45000) == logical_dataset_fingerprint(
        ferret, train_expected_count=45000
    )
    changed_split = {**base, "seeds": {"split": 12}}
    assert logical_dataset_fingerprint(base, train_expected_count=45000) != logical_dataset_fingerprint(
        changed_split, train_expected_count=45000
    )
    identity = logical_dataset_identity(base, train_expected_count=45000)
    assert identity["dataset"] == {
        "name": "cifar10",
        "version": "torchvision-cifar10",
        "content_fingerprint": "c58f30108f718f92721af3b95e74349a",
    }
    cifar100 = {**base, "dataset": {**base["dataset"], "name": "cifar100"}}
    assert logical_dataset_identity(cifar100, train_expected_count=45000)["dataset"] == {
        "name": "cifar100",
        "version": "torchvision-cifar100",
        "content_fingerprint": "eb9058c3a382ffc7106e4002c42a8d85",
    }


def test_prospective_rejects_missing_periodic_teacher_risk_without_final_inference() -> None:
    checkpoint = CheckpointInventory(
        run_id="run-0",
        artifact_name="model-run-last",
        aliases=("last",),
        publication_order=0,
        path="unused.pt",
        sha256="c" * 64,
        epoch=5,
        sample_state_present=True,
        sample_state_count=1,
        config_hash="b" * 64,
        scientific_git_sha="a" * 40,
    )
    result = prospective_prediction(
        [{"namespace": "train", "sample_id": 0, "class_id": 0, "student_risk": 0.5}],
        historical=checkpoint,
        final=checkpoint,
        split_seed=0,
        bootstrap_seed=0,
        bootstrap_replicates=2,
    )
    assert result == {
        "analysis_type": "prospective_prediction",
        "decision": "insufficient_data",
        "reason": "periodic_teacher_risk_absent",
    }


def test_teacher_replay_requires_complete_provenance_envelope() -> None:
    historical = CheckpointInventory(
        "run-0", "model", ("last",), 0, "unused.pt", "a" * 64, 1, True, 1, "b" * 64, "c" * 40, "v1"
    )
    with pytest.raises(SignalAuditError, match="complete provenance"):
        validate_teacher_risk_replay(
            {},
            historical=historical,
            teacher_checkpoint_sha256="d" * 64,
            dataset_fingerprint="dataset",
            threat_or_attack_identity={"attack": "pgd"},
        )


def test_explicit_historical_epoch_selects_only_the_requested_stateful_run() -> None:
    stateful = [
        CheckpointInventory(
            "stateful", "model-stateful-last", ("last",), epoch, "x", "c" * 64, epoch, True, 2, "b" * 64, "a" * 40
        )
        for epoch in (1, 2, 3)
    ]
    stateless = CheckpointInventory(
        "stateless", "model-stateless-last", ("last",), 0, "x", "c" * 64, 9, False, 0, "b" * 64, "a" * 40
    )
    historical, final = select_prospective_checkpoints((*stateful, stateless), run_id="stateful", historical_epoch=2)
    assert (historical.epoch, final.epoch) == (2, 3)
    with pytest.raises(SignalAuditError, match="exactly one explicit"):
        select_prospective_checkpoints((*stateful, stateless), run_id="stateful", historical_epoch=0)


def test_state_outcome_uses_final_previous_correctness_not_cumulative_frequency(tmp_path: Path) -> None:
    previous_path, final_path = tmp_path / "historical.pt", tmp_path / "final.pt"
    prior = _records()
    current = _records()
    current["0"] = {**current["0"], "robust_correct_count": 2, "previous_robust_correct": False, "forgetting_count": 2}
    torch.save(_checkpoint(run_id="run-0", config_hash="b" * 64, epoch=1, records=prior), previous_path)
    torch.save(_checkpoint(run_id="run-0", config_hash="b" * 64, epoch=2, records=current), final_path)
    historical = CheckpointInventory(
        "run-0", "model", ("last",), 0, str(previous_path), "c" * 64, 1, True, 2, "b" * 64, "a" * 40
    )
    final = CheckpointInventory(
        "run-0", "model", ("last",), 1, str(final_path), "d" * 64, 2, True, 2, "b" * 64, "a" * 40
    )
    result = _state_outcomes(historical, final, [{"namespace": "train", "sample_id": 0}])[0]
    assert result["final_robust_error"] == 1
    assert result["final_robust_correct_frequency"] == 1.0


def test_prepared_student_risk_cannot_override_historical_state_feature(tmp_path: Path) -> None:
    previous_path, final_path = tmp_path / "historical.pt", tmp_path / "final.pt"
    prior, current = _records(), _records()
    torch.save(_checkpoint(run_id="run-0", config_hash="b" * 64, epoch=1, records=prior), previous_path)
    torch.save(_checkpoint(run_id="run-0", config_hash="b" * 64, epoch=2, records=current), final_path)
    historical = CheckpointInventory(
        "run-0", "model", ("last",), 0, str(previous_path), "c" * 64, 1, True, 2, "b" * 64, "a" * 40
    )
    final = CheckpointInventory(
        "run-0", "model", ("last",), 1, str(final_path), "d" * 64, 2, True, 2, "b" * 64, "a" * 40
    )
    base = {"namespace": "train", "sample_id": 0, "student_risk": 0.0}
    contradictory = {**base, "student_risk": 1.0}
    first = _state_outcomes(historical, final, [base])[0]
    second = _state_outcomes(historical, final, [contradictory])[0]
    assert first == second
    assert first["historical_student_risk"] == pytest.approx(0.45)
    assert "student_risk" not in first


def test_final_parquet_loader_adds_namespace_and_validates_signal_formulas(tmp_path: Path) -> None:
    path = tmp_path / "sample-stats-train.parquet"
    pq.write_table(
        pa.table(
            {
                "sample_id": [0, 1],
                "true_label": [0, 1],
                "teacher_prediction": [0, 0],
                "teacher_entropy": [0.0, 0.0],
                "student_unlearnability": [0.25, 0.75],
                "joint_risk": [0.25, 0.75],
                "robust_correct": [True, False],
            }
        ),
        path,
    )
    rows = load_final_sample_stats(path, expected_count=2, num_classes=10)
    assert rows[0]["namespace"] == "train"
    assert rows[0]["teacher_correct"] is True
    assert rows[1]["teacher_correct"] is False
    assert rows[1]["final_robust_error"] == 1
    assert rows[1]["implied_rho"] == pytest.approx(0.375)


def test_final_state_association_reports_exploratory_descriptives_with_tie_aware_ranks() -> None:
    rows = [
        {
            "sample_id": index,
            "teacher_risk": risk,
            "student_risk": risk,
            "stored_applied_risk": risk * risk,
            "implied_rho": 0.5 * risk * risk,
            "teacher_entropy": 1.0 - risk,
            "teacher_correct": index != 4,
            "final_robust_error": index % 2,
        }
        for index, risk in enumerate((0.0, 0.25, 0.5, 0.75, 1.0))
    ]
    report = final_state_association(rows, rho_zero_threshold=1e-6)

    assert report["exploratory_only"] is True
    assert set(report["metrics"]) == {"teacher_risk", "student_risk", "teacher_student_product"}
    assert report["spearman_teacher_student_risk"] == {"value": pytest.approx(1.0), "tie_aware": True}
    assert report["risk_quantiles"]["teacher_risk"]["p50"] == pytest.approx(0.5)
    assert report["rho_distribution"]["near_zero_fraction"] == pytest.approx(0.2)
    assert report["rho_distribution"]["near_rho_max_fraction"] == pytest.approx(0.2)
    assert report["teacher_correctness"]["teacher_wrong"]["count"] == 1
    assert report["teacher_correctness"]["teacher_wrong_fraction_in_top_teacher_risk_decile"] == 1.0


def test_wandb_periodic_last_mapping_requires_matching_bytes_and_associates_version(tmp_path: Path) -> None:
    manifest = _manifest_with_checkpoint(
        tmp_path, payload=_checkpoint(run_id="run-0", config_hash="b" * 64, epoch=5, records=_records())
    )
    local = inventory_run_bundle(manifest)[0]
    local_path = Path(local.path)
    remote = {
        "run_id": "run-0",
        "artifact_name": local.artifact_name,
        "version": "v7",
        "file_name": local_path.name,
        "file_md5": b64encode(hashlib.md5(local_path.read_bytes(), usedforsecurity=False).digest()).decode(),
        "size": local_path.stat().st_size,
    }
    assert associate_wandb_versions((local,), (remote,))[0].wandb_version == "v7"
    with pytest.raises(SignalAuditError, match="size or MD5"):
        associate_wandb_versions((local,), ({**remote, "size": remote["size"] + 1},))


def test_wandb_periodic_last_mapping_orders_explicit_numeric_versions(tmp_path: Path) -> None:
    manifest = _manifest_with_checkpoint(
        tmp_path, payload=_checkpoint(run_id="run-0", config_hash="b" * 64, epoch=1, records=_records())
    )
    _append_checkpoint_artifact(
        manifest, payload=_checkpoint(run_id="run-0", config_hash="b" * 64, epoch=2, records=_records())
    )
    local = inventory_run_bundle(manifest)
    remote = [
        {
            "run_id": item.run_id,
            "artifact_name": item.artifact_name,
            "version": version,
            "file_name": Path(item.path).name,
            "file_md5": b64encode(hashlib.md5(Path(item.path).read_bytes(), usedforsecurity=False).digest()).decode(),
            "size": Path(item.path).stat().st_size,
        }
        for item, version in ((local[1], "v9"), (local[0], "v8"))
    ]
    assert [item.wandb_version for item in associate_wandb_versions(local, remote)] == ["v8", "v9"]


def test_valid_replay_envelope_fits_prospective_models_and_writes_byte_stable_report(tmp_path: Path) -> None:
    previous_path, final_path = tmp_path / "historical.pt", tmp_path / "final.pt"
    prior, current, final_rows, replay_rows = {}, {}, [], []
    for sample_id in range(20):
        class_id = sample_id % 2
        target = class_id
        margin = -0.8 + 0.08 * sample_id
        prior[str(sample_id)] = {
            "margin_ema": margin,
            "seen": 1,
            "robust_correct_count": int(not target),
            "previous_robust_correct": not target,
            "forgetting_count": 0,
            "last_update": 1,
        }
        current[str(sample_id)] = {
            **prior[str(sample_id)],
            "seen": 2,
            "robust_correct_count": int(not target),
            "previous_robust_correct": not target,
            "forgetting_count": target,
            "last_update": 2,
        }
        teacher_risk = 0.2 + 0.6 * target
        replay_rows.append(
            {"namespace": "train", "sample_id": sample_id, "class_id": class_id, "teacher_risk": teacher_risk}
        )
        final_rows.append(
            {
                "namespace": "train",
                "sample_id": sample_id,
                "class_id": class_id,
                "teacher_risk": teacher_risk,
                "student_risk": (1.0 - margin) / 2.0,
                "stored_applied_risk": teacher_risk * ((1.0 - margin) / 2.0),
                "implied_rho": 0.5 * teacher_risk * ((1.0 - margin) / 2.0),
                "teacher_entropy": 1.0,
                "teacher_correct": not target,
                "final_robust_error": target,
            }
        )
    torch.save(_checkpoint(run_id="run-0", config_hash="b" * 64, epoch=1, records=prior), previous_path)
    torch.save(_checkpoint(run_id="run-0", config_hash="b" * 64, epoch=2, records=current), final_path)
    historical = CheckpointInventory(
        "run-0", "model", ("last",), 0, str(previous_path), "a" * 64, 1, True, 20, "b" * 64, "c" * 40, "v1"
    )
    final = CheckpointInventory(
        "run-0", "model", ("last",), 1, str(final_path), "d" * 64, 2, True, 20, "b" * 64, "c" * 40, "v2"
    )
    envelope = {
        "run_id": "run-0",
        "historical_epoch": 1,
        "historical_checkpoint_sha256": "a" * 64,
        "teacher_checkpoint_sha256": "e" * 64,
        "dataset_fingerprint": "synthetic-dataset",
        "threat_identity": {"name": "synthetic"},
        "replay_output_sha256": hashlib.sha256(
            json.dumps(replay_rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "rows": replay_rows,
    }
    report = audit_report(
        config={
            "train_expected_count": 20,
            "prospective_run_id": "run-0",
            "historical_epoch": 1,
            "split_seed": 3,
            "bootstrap_seed": 5,
            "bootstrap_replicates": 5,
        },
        inventories=(historical, final),
        final_rows=final_rows,
        input_hashes={"fixture": "f" * 64},
        teacher_risk_replay=envelope,
        lineage={
            "teacher_checkpoint_sha256": "e" * 64,
            "dataset_fingerprint": "synthetic-dataset",
            "threat_or_attack_identity": {"name": "synthetic"},
        },
    )
    assert "models" in report["prospective_prediction"]["outcomes"]["final_robust_error"]
    assert report["prospective_prediction"]["decision"] in {"go", "no_go", "inconclusive"}
    assert len(report["analysis_source_sha256"]) == 64
    assert set(report["analysis_source_files"]) == {"analysis_module", "cli_module"}
    assert all(len(digest) == 64 for digest in report["analysis_source_files"].values())
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    write_audit_report(first, report)
    write_audit_report(second, report)
    assert first.read_bytes() == second.read_bytes()

    mismatched_rows = [{**row} for row in replay_rows]
    mismatched_rows[0]["class_id"] = 1 - int(mismatched_rows[0]["class_id"])
    mismatched_envelope = {
        **envelope,
        "rows": mismatched_rows,
        "replay_output_sha256": hashlib.sha256(
            json.dumps(mismatched_rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    with pytest.raises(SignalAuditError, match="class_id mismatch"):
        audit_report(
            config={
                "train_expected_count": 20,
                "prospective_run_id": "run-0",
                "historical_epoch": 1,
                "split_seed": 3,
                "bootstrap_seed": 5,
                "bootstrap_replicates": 5,
            },
            inventories=(historical, final),
            final_rows=final_rows,
            input_hashes={"fixture": "f" * 64},
            teacher_risk_replay=mismatched_envelope,
            lineage={
                "teacher_checkpoint_sha256": "e" * 64,
                "dataset_fingerprint": "synthetic-dataset",
                "threat_or_attack_identity": {"name": "synthetic"},
            },
        )


def test_cli_accepts_real_parquet_with_manifest_list_and_keeps_formal_decision_insufficient(tmp_path: Path) -> None:
    historical_records = _records()
    final_records = _records()
    final_records["0"] = {**final_records["0"], "previous_robust_correct": False, "forgetting_count": 2}
    resolved = {
        "teacher": {"checkpoint_sha256": "c" * 64, "registry_id": "teacher-0"},
        "method": {"id": "rslad_joint", "attack": {"name": "synthetic"}},
        "seeds": {"model_init": 7, "split": 3},
        "training": {"validation_fraction": 0.25},
        "dataset": {"name": "synthetic", "split": "train", "root": "/host-a/data", "download": False},
    }
    resolved_hash = config_digest(resolved)
    manifest = _manifest_with_checkpoint(
        tmp_path, payload=_checkpoint(run_id="run-0", config_hash=resolved_hash, epoch=1, records=historical_records)
    )
    _append_checkpoint_artifact(
        manifest, payload=_checkpoint(run_id="run-0", config_hash=resolved_hash, epoch=2, records=final_records)
    )
    parquet = tmp_path / "sample-stats-train.parquet"
    pq.write_table(
        pa.table(
            {
                "sample_id": [0, 1],
                "true_label": [0, 1],
                "teacher_prediction": [0, 0],
                "teacher_entropy": [0.0, 0.0],
                "student_unlearnability": [0.25, 0.75],
                "joint_risk": [0.25, 0.75],
                "robust_correct": [False, True],
            }
        ),
        parquet,
    )
    _add_formal_lineage(manifest, final_parquet=parquet, resolved=resolved)
    inventory = inventory_run_bundle(manifest)
    target_rows = [
        {
            "run_id": item.run_id,
            "artifact_name": item.artifact_name,
            "version": version,
            "file_name": Path(item.path).name,
            "file_md5": b64encode(hashlib.md5(Path(item.path).read_bytes(), usedforsecurity=False).digest()).decode(),
            "size": Path(item.path).stat().st_size,
        }
        for item, version in ((inventory[0], "v19"), (inventory[1], "v39"))
    ]
    wandb_inventory = tmp_path / "wandb-inventory.json"
    wandb_inventory.write_text(
        json.dumps(
            {
                "artifacts": [
                    *target_rows,
                    *[
                        {
                            "run_id": f"other-{index}",
                            "artifact_name": "model-other-last",
                            "version": f"v{index}",
                            "file_name": "last.pt",
                            "file_md5": "AAAAAAAAAAAAAAAAAAAAAA==",
                            "size": 1,
                        }
                        for index in range(6)
                    ],
                ]
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "audit.json"
    config.write_text(
        json.dumps(
            {
                "manifests": [str(manifest)],
                "final_parquet": "sample-stats-train.parquet",
                "prospective_run_id": "run-0",
                "historical_epoch": 1,
                "train_expected_count": 2,
                "method_id": "rslad_joint",
                "training_seed": 7,
                "teacher": {"registry_id": "teacher-0"},
                "wandb_inventory": "wandb-inventory.json",
                "dataset_fingerprint": logical_dataset_fingerprint(resolved, train_expected_count=2),
                "threat_identity": {"name": "synthetic"},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "audit-output.json"
    assert signal_audit_main(["--config", str(config), "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["prospective_prediction"]["decision"] == "insufficient_data"
    assert report["artifact_only_temporal_diagnostics"]["status"] == "available"
    assert len(report["artifact_only_temporal_diagnostics"]["historical_student_risk_deciles"]) == 10
    mismatch = json.loads(config.read_text(encoding="utf-8"))
    mismatch["teacher"] = {"registry_id": "wrong-teacher"}
    config.write_text(json.dumps(mismatch), encoding="utf-8")
    with pytest.raises(SignalAuditError, match="registry_id"):
        signal_audit_main(["--config", str(config), "--output", str(output)])
