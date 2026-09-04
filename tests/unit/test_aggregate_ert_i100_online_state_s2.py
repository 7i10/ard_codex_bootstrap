"""Lineage-identity, report-field, and provenance contracts for the I100 Online-State S2 aggregator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPT = Path(__file__).parents[2] / "scripts/aggregate_ert_i100_online_state_s2.py"
SPEC = importlib.util.spec_from_file_location("aggregate_i100_online_state_s2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

SOURCE_SHA = "bcb09a73814e7788026b287309d148b7791ccf75"


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_45k_rows(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"sample_id": list(range(45_000))}), path)
    return path


def _canonical_tree(tmp_path: Path, *, attack_sha: str) -> tuple[Path, dict[str, Any]]:
    """Build a minimal canonical e114 replay root plus its parent training payload."""

    campaign = tmp_path / "campaign"
    root = campaign / "canonical" / "dev-1" / "control"
    rows = _write_45k_rows(root / "state-rows.parquet")
    checkpoint_sha = "a" * 64
    _write_json(
        root / "state-replay.json",
        {
            "contract": MODULE.CANONICAL_CONTRACT,
            "checkpoint_epoch": 114,
            "checkpoint_sha256": checkpoint_sha,
            "observation": {"attack_identity_sha256": attack_sha},
            "row_count": 45_000,
            "rows_sha256": MODULE.sha256(rows),
            "teacher_checkpoint_sha256": MODULE.TEACHER_SHA256,
        },
    )
    training = {"checkpoints": {114: {"path": "unused", "sha256": checkpoint_sha}}}
    return campaign, training


# --- B1: the canonical replay identity is sample-keyed, not the batch-keyed endpoint identity ---


def test_registered_canonical_identity_is_distinct_from_the_endpoint_identity() -> None:
    assert MODULE.CANONICAL_CE20_ATTACK_SHA256 == "675a8d4e3cd16d345acd7fe9e5d1e721f834fcf63c7225e9035e1736ed0c07b6"
    assert MODULE.CANONICAL_CE20_ATTACK_SHA256 != MODULE.ENDPOINT_ATTACK_SHA256


def test_canonical_payload_accepts_the_sample_keyed_canonical_ce20_identity(tmp_path: Path) -> None:
    campaign, training = _canonical_tree(tmp_path, attack_sha=MODULE.CANONICAL_CE20_ATTACK_SHA256)
    payload = MODULE._canonical_payload(campaign, seed="dev-1", arm="control", training=training)
    assert payload["metadata"]["observation"]["attack_identity_sha256"] == MODULE.CANONICAL_CE20_ATTACK_SHA256
    assert len(payload["rows"]) == 45_000


def test_canonical_payload_rejects_the_batch_keyed_endpoint_identity(tmp_path: Path) -> None:
    campaign, training = _canonical_tree(tmp_path, attack_sha=MODULE.ENDPOINT_ATTACK_SHA256)
    with pytest.raises(MODULE.AggregationError, match="canonical e114 replay contract differs"):
        MODULE._canonical_payload(campaign, seed="dev-1", arm="control", training=training)


# --- B2: the frozen lineage record carries the registered e99 parent the report renders ---


def _prefix_tree(tmp_path: Path, seed: str) -> Path:
    campaign = tmp_path / "campaign"
    checkpoint = campaign / "prefix" / seed / "training/checkpoints/epoch-100.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(f"frozen e100 prefix for {seed}".encode())
    _write_json(
        campaign / "prefix" / seed / "prefix-summary.json",
        {
            "contract": MODULE.SOURCE_PREFIX_CONTRACT,
            "seed": seed,
            "source_git_sha": SOURCE_SHA,
            "result": {
                "parent_checkpoint_sha256": MODULE.PARENT_SHA256[seed],
                "horizon_checkpoints": {"100": {"sha256": MODULE.sha256(checkpoint)}},
            },
        },
    )
    threshold_path = _write_json(
        campaign / "thresholds" / seed / "frozen-thresholds.json",
        {
            "contract": MODULE.THRESHOLD_CONTRACT,
            "kind": "frozen_thresholds",
            "source_git_sha": SOURCE_SHA,
            "original_parent_checkpoint_sha256": MODULE.PARENT_SHA256[seed],
            "training_attack_identity_sha256": MODULE.TRAIN_ATTACK_SHA256,
            "thresholds": {"student_global_logit_q10": 0.048, "teacher_global_logit_q10": 0.065},
        },
    )
    threshold_path.with_name(threshold_path.name + ".sha256").write_text(
        MODULE.sha256(threshold_path) + "\n", encoding="utf-8"
    )
    return campaign


def test_prefix_and_threshold_emits_the_registered_parent_checkpoint_sha256(tmp_path: Path) -> None:
    campaign = _prefix_tree(tmp_path, "dev-1")
    frozen = MODULE._prefix_and_threshold(campaign, seed="dev-1", source_sha=SOURCE_SHA)
    assert frozen["parent_checkpoint_sha256"] == MODULE.PARENT_SHA256["dev-1"]
    assert frozen["thresholds"] == {"student_global_logit_q10": 0.048, "teacher_global_logit_q10": 0.065}


def _markdown_fixture(frozen: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The smallest result the report formatter fully renders."""

    effect = {
        "clean": {"accuracy_delta": 0.001},
        "robust": {"accuracy_delta": 0.001, "rescue_count": 12, "harm_count": 5},
    }
    mechanics = {
        "total_s2_t1_state_active_epochs": 10,
        "mean_s2_t1_state_fraction": 0.01,
        "total_action_exposure": 0,
        "total_pair_gated_boundary_count": 0.0,
        "mean_active_fraction": 0.0,
        "total_entries": 0,
        "total_reentries": 0,
        "total_action_switches": 0,
        "total_s2_t1_state_entries": 3,
        "total_s2_t1_state_exits": 2,
        "total_s2_t1_state_reentries": 1,
        "transition_totals": {},
        "last_current_q10_minus_frozen": {"student": 0.001, "teacher": -0.002},
    }
    agreement = {
        "online_s2_t1_count": 5,
        "canonical_s2_t1_count": 6,
        "true_positive": 4,
        "false_positive": 1,
        "false_negative": 2,
        "true_negative": 10,
        "precision": 0.8,
        "recall": 0.6,
        "jaccard": 0.5,
    }
    arm = {
        "endpoints": {
            str(epoch): {
                "clean_accuracy": 0.8,
                "robust_accuracy": 0.5,
                "clean_delta_vs_control": 0.0,
                "robust_delta_vs_control": 0.0,
            }
            for epoch in MODULE.HORIZONS
        },
        "route_mechanics": mechanics,
        "online_vs_canonical_e114": agreement,
        "canonical_e99_s2_t1_e114_outcomes": {"reference_n": 100, "e114_branch_counts": {"S1": 60}},
        "runtime": {"mean_train_seconds": 120.0, "mean_train_images_per_second": 375.0},
    }
    return {
        "frozen_contract": frozen,
        "seeds": {
            seed: {
                "arms": {name: json.loads(json.dumps(arm)) for name in MODULE.ARMS},
                "comparisons": {
                    str(epoch): {name: effect for name in ("pmp_vs_control", "dbdp_vs_control", "dbdp_vs_pmp")}
                    for epoch in MODULE.HORIZONS
                },
            }
            for seed in MODULE.SEEDS
        },
        "decision": {
            "pmp_vs_control": "SUPPORTED",
            "dbdp_vs_control": "SUPPORTED",
            "dbdp_vs_pmp": "NOT_SUPPORTED",
            "machine_readable": {
                "overall": "ONLINE_S2_SUPPORTED_FOR_NEXT_STAGE",
                "pmp": "OS_PMP_SUPPORTED_FOR_NEXT_STAGE",
                "dbdp": "OS_DBDP_SUPPORTED_FOR_NEXT_STAGE",
                "dbdp_specific": "DBDP_SPECIFIC_SUPERIORITY_NOT_SUPPORTED",
            },
        },
    }


def test_markdown_renders_the_frozen_lineage_from_the_aggregated_record(tmp_path: Path) -> None:
    frozen = {
        seed: MODULE._prefix_and_threshold(_prefix_tree(tmp_path / seed, seed), seed=seed, source_sha=SOURCE_SHA)
        for seed in MODULE.SEEDS
    }
    report = MODULE._markdown(_markdown_fixture(frozen))
    for seed in MODULE.SEEDS:
        assert MODULE.PARENT_SHA256[seed] in report
    assert "## Provenance" not in report


# --- D1: declared path, realpath, and producing campaign are recorded separately ---


def _completion(path: Path, *, campaign_id: str, job_id: str) -> Path:
    return _write_json(
        path,
        {
            "campaign_id": campaign_id,
            "job_id": job_id,
            "completed_at": "2026-09-04T06:19:24.095786+00:00",
            "source_sha": SOURCE_SHA,
            "status": "completed",
            "output_dir": str(path.parent),
        },
    )


def test_collection_provenance_separates_the_collected_path_from_the_produced_bytes(tmp_path: Path) -> None:
    produced = tmp_path / "recovery14/arms/dev-1/control"
    (produced / "training").mkdir(parents=True)
    _completion(produced / "completion.json", campaign_id="campaign-recovery14", job_id="retry3-arm-dev-1-control")
    collected = tmp_path / "recovery17/arms"
    collected.parent.mkdir(parents=True, exist_ok=True)
    collected.symlink_to(tmp_path / "recovery14/arms")

    record = MODULE._collection_provenance(collected / "dev-1/control/training")
    assert record["declared_path"] == str(collected / "dev-1/control/training")
    assert record["realpath"] == str(produced / "training")
    assert record["declared_path"] != record["realpath"]
    assert record["producer"] == {
        "campaign_id": "campaign-recovery14",
        "job_id": "retry3-arm-dev-1-control",
        "completed_at": "2026-09-04T06:19:24.095786+00:00",
        "source_sha": SOURCE_SHA,
    }


def test_collection_provenance_reads_a_completion_record_inside_a_non_training_root(tmp_path: Path) -> None:
    root = tmp_path / "recovery15/endpoints/dev-1/control"
    root.mkdir(parents=True)
    _completion(root / "completion.json", campaign_id="campaign-recovery15", job_id="repair15-endpoint-dev-1-control")
    record = MODULE._collection_provenance(root)
    assert record["realpath"] == str(root)
    assert record["producer"]["campaign_id"] == "campaign-recovery15"


def test_collection_provenance_reports_an_absent_completion_record_as_unknown(tmp_path: Path) -> None:
    root = tmp_path / "recovery16/canonical/dev-1/control"
    root.mkdir(parents=True)
    # A sibling record must not be borrowed from an unrelated directory level.
    _completion(tmp_path / "recovery16/completion.json", campaign_id="campaign-recovery16", job_id="repair16")
    assert MODULE._collection_provenance(root)["producer"] is None


def test_provenance_summary_lists_the_distinct_producing_campaigns_per_artifact_class(tmp_path: Path) -> None:
    def record(campaign_id: str | None, realpath: str) -> dict[str, Any]:
        producer = None if campaign_id is None else {"campaign_id": campaign_id, "job_id": "j"}
        return {"declared_path": "declared", "realpath": realpath, "producer": producer}

    summary = MODULE._provenance_summary(
        tmp_path / "recovery17",
        {
            "training": [
                record("c-recovery14", "/r14/a"),
                record("c-base", "/base/b"),
                record("c-recovery14", "/r14/c"),
            ],
            "endpoints": [record(None, "/r15/e")],
        },
    )
    assert summary["campaign_root"] == str(tmp_path / "recovery17")
    assert summary["artifact_classes"]["training"] == {
        "campaign_ids": ["c-base", "c-recovery14"],
        "roots": 3,
        "roots_without_completion_record": 0,
        "realpath_roots": ["/base/b", "/r14/a", "/r14/c"],
    }
    assert summary["artifact_classes"]["endpoints"]["campaign_ids"] == []
    assert summary["artifact_classes"]["endpoints"]["roots_without_completion_record"] == 1


def test_markdown_appends_a_provenance_section_without_touching_the_scientific_body(tmp_path: Path) -> None:
    frozen = {
        seed: MODULE._prefix_and_threshold(_prefix_tree(tmp_path / seed, seed), seed=seed, source_sha=SOURCE_SHA)
        for seed in MODULE.SEEDS
    }
    base = _markdown_fixture(frozen)
    without = MODULE._markdown(base)
    with_provenance = MODULE._markdown(
        base
        | {
            "provenance": MODULE._provenance_summary(
                tmp_path,
                {
                    "training": [
                        {
                            "declared_path": "declared",
                            "realpath": "/r15/arms/dev-1/control/training",
                            "producer": {"campaign_id": "c-recovery14", "job_id": "j"},
                        }
                    ],
                    "canonical": [{"declared_path": "declared", "realpath": "/r16/canonical", "producer": None}],
                },
            )
        }
    )
    assert with_provenance.startswith(without.rstrip("\n"))
    assert "## Provenance" in with_provenance
    assert "| training | c-recovery14 | 1 | 0 |" in with_provenance
    assert "| canonical | — | 1 | 1 |" in with_provenance


def test_aggregator_source_defaults_to_the_artifact_source_sha() -> None:
    assert MODULE._aggregator_source("a" * 40, None) == "a" * 40
    assert MODULE._aggregator_source("a" * 40, "b" * 40) == "b" * 40


def test_clean_source_guard_checks_the_aggregator_sha_not_the_artifact_sha(monkeypatch, tmp_path: Path) -> None:
    seen: list[str] = []
    monkeypatch.setattr(MODULE, "_require_clean_source", lambda expected: seen.append(expected))
    with pytest.raises(MODULE.AggregationError, match="campaign root is missing"):
        MODULE.aggregate(campaign=tmp_path / "absent", expected_source_sha="a" * 40, aggregator_source_sha="b" * 40)
    assert seen == ["b" * 40]
