from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from ard.analysis.pre39_online_state import Pre39OnlineStateError, export_pre39_online_state, write_pre39_online_state
from ard.analysis.rslad_signal_replay import FEATURE_EPOCHS
from ard.analysis.signal_audit import sha256_file
from ard.engine.checkpoint import REQUIRED_KEYS

pytestmark = pytest.mark.t1


def _teacher(prefix: str, class_id: int) -> dict[str, object]:
    return {
        f"{prefix}_prediction": class_id,
        f"{prefix}_correct": True,
        f"{prefix}_true_probability": 0.8,
        f"{prefix}_max_wrong_probability": 0.1,
        f"{prefix}_wrong_confidence": -0.7,
        f"{prefix}_probability_margin": 0.7,
        f"{prefix}_entropy_normalized": 0.4,
    }


def _row(sample_id: int, epoch: int) -> dict[str, object]:
    class_id = sample_id % 10
    clean, adversarial = _teacher("teacher_clean", class_id), _teacher("teacher_adversarial", class_id)
    return {
        "namespace": "train",
        "sample_id": sample_id,
        "class_id": class_id,
        "epoch": epoch,
        "observation_schema_version": 2,
        "teacher_entropy_normalized": 0.4,
        "student_probability_margin": 0.5,
        "student_margin_risk": 0.25,
        "robust_correct": True,
        **clean,
        **adversarial,
        "teacher_clean_to_adversarial_prediction_flip": False,
        "teacher_clean_to_adversarial_true_probability_delta": 0.0,
        "teacher_clean_to_adversarial_margin_delta": 0.0,
        "student_clean_prediction": class_id,
        "student_clean_correct": True,
        "student_clean_probability_margin": 0.5,
    }


def _checkpoint(path: Path, anchor: int, ids: tuple[int, ...], *, epoch: int | None = None) -> str:
    payload = {key: {} for key in REQUIRED_KEYS}
    payload.update(
        {
            "epoch": anchor if epoch is None else epoch,
            "epoch_boundary": "end",
            "tracker_run_id": "run",
            "config_hash": "a" * 64,
            "world_size": 1,
            "sample_state": {
                "format_version": 3,
                "pending": [],
                "records": {
                    str(sample_id): {
                        "true_label": sample_id % 10,
                        "seen": anchor + 1,
                        "robust_correct_count": anchor,
                        "previous_robust_correct": True,
                        "margin_ema": 0.2,
                        "last_margin": 0.3,
                    }
                    for sample_id in ids
                },
            },
        }
    )
    torch.save(payload, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path: Path, *, anchor: int = 14) -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    ids = (7, 43, 101, 900)
    observations, lineage, checkpoint = tmp_path / "feature.parquet", tmp_path / "lineage.json", tmp_path / "anchor.pt"
    pq.write_table(
        pa.Table.from_pylist([_row(sample_id, epoch) for epoch in FEATURE_EPOCHS for sample_id in ids]), observations
    )
    digest = _checkpoint(checkpoint, anchor, ids)
    lineage.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "observation_schema_version": 2,
                "run_id": "run",
                "config_hash": "a" * 64,
                "scientific_git_sha": "b" * 40,
                "seed": 1,
                "train_expected_count": len(ids),
                "feature_observations_sha256": sha256_file(observations),
                "attack_identity": {"steps": 10},
                "dataset_identity": {"name": "cifar10"},
                "teacher": {"registry_id": "teacher"},
                "feature_protocol": {"domain": "feature"},
                "checkpoints": [{"epoch": anchor, "sha256": digest}],
            }
        ),
        encoding="utf-8",
    )
    return observations, lineage, checkpoint


def test_pre39_online_candidate_sparse_state_and_nonoverwrite(tmp_path: Path) -> None:
    observations, lineage, checkpoint = _inputs(tmp_path)
    export = export_pre39_online_state(
        checkpoint=checkpoint,
        feature_observations=observations,
        feature_lineage=lineage,
        anchor=14,
        expected_count=4,
        analysis_provenance={"test": True},
    )
    assert [row["sample_id"] for row in export.rows] == [7, 43, 101, 900]
    assert export.rows[0]["robust_correct_frequency_inclusive"] == pytest.approx(14 / 15)
    paths = write_pre39_online_state(output_dir=tmp_path / "out", export=export)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_pre39_online_state(output_dir=tmp_path / "out", export=export)
    assert paths["lineage"].is_file()


def test_pre39_online_candidate_rejects_hash_epoch_schedule_and_sparse_join_drift(tmp_path: Path) -> None:
    observations, lineage, checkpoint = _inputs(tmp_path)
    _checkpoint(checkpoint, 14, (7, 43, 101, 900), epoch=19)
    with pytest.raises(Pre39OnlineStateError, match="SHA"):
        export_pre39_online_state(
            checkpoint=checkpoint,
            feature_observations=observations,
            feature_lineage=lineage,
            anchor=14,
            expected_count=4,
            analysis_provenance={},
        )
    observations, lineage, checkpoint = _inputs(tmp_path / "epoch")
    digest = _checkpoint(checkpoint, 14, (7, 43, 101, 900), epoch=19)
    value = json.loads(lineage.read_text(encoding="utf-8"))
    value["checkpoints"][0]["sha256"] = digest
    lineage.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(Pre39OnlineStateError, match="format/epoch boundary"):
        export_pre39_online_state(
            checkpoint=checkpoint,
            feature_observations=observations,
            feature_lineage=lineage,
            anchor=14,
            expected_count=4,
            analysis_provenance={},
        )
    observations, lineage, checkpoint = _inputs(tmp_path / "schedule")
    with pytest.raises(Pre39OnlineStateError, match="candidate anchor/count"):
        export_pre39_online_state(
            checkpoint=checkpoint,
            feature_observations=observations,
            feature_lineage=lineage,
            anchor=4,
            expected_count=4,
            analysis_provenance={},
        )
    observations, lineage, checkpoint = _inputs(tmp_path / "join")
    digest = _checkpoint(checkpoint, 14, (7, 43, 101, 999))
    value = json.loads(lineage.read_text(encoding="utf-8"))
    value["checkpoints"][0]["sha256"] = digest
    lineage.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(Pre39OnlineStateError, match="sparse ID/class join"):
        export_pre39_online_state(
            checkpoint=checkpoint,
            feature_observations=observations,
            feature_lineage=lineage,
            anchor=14,
            expected_count=4,
            analysis_provenance={},
        )


def test_pre39_online_candidate_provenance_fail_fast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    observations, lineage, checkpoint = _inputs(tmp_path)
    monkeypatch.setattr(
        "ard.analysis.pre39_online_state._provenance",
        lambda: (_ for _ in ()).throw(Pre39OnlineStateError("tracked-clean")),
    )
    with pytest.raises(Pre39OnlineStateError, match="tracked-clean"):
        export_pre39_online_state(
            checkpoint=checkpoint,
            feature_observations=observations,
            feature_lineage=lineage,
            anchor=14,
            expected_count=4,
        )
