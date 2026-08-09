from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path
from subprocess import CompletedProcess

import pytest

import ard.analysis.ffnr_strong_diagnostics as diagnostics
from ard.analysis.ffnr_strong_diagnostics import (
    StrongDiagnosticsError,
    _apply_empirical_rank,
    _blinded_candidate_rows,
    _class_stratified_folds,
    _cross_seed_tables,
    _dense_non_recovery,
    _dense_nr_subtype_tables,
    _endpoint,
    _fit_empirical_rank,
    _fold_rank_vectors,
    _merge_dense_chunks,
    _oof_scores,
    _render_blinded_cifar10_panel,
    _snapshot_taxonomy,
    _teacher,
    _tracked_clean_provenance,
    _validate_online_strong_class_join,
    _validated_replay_identity,
    _vectorized_logistic_predict,
    _wilson,
)
from ard.analysis.signal_audit import _fit_logistic, _predict_logistic

pytestmark = pytest.mark.unit


def _teacher_row() -> dict[str, object]:
    return {
        "class_id": 1,
        "teacher_adversarial_probabilities": [0.05, 0.2, 0.7, 0.05, 0, 0, 0, 0, 0, 0],
        "teacher_clean_probabilities": [0.1, 0.8, 0.1, 0, 0, 0, 0, 0, 0, 0],
    }


def test_teacher_probability_algebra_and_explicit_argmax_tie_rejection() -> None:
    teacher = _teacher(_teacher_row())
    assert teacher["correct"] is False and teacher["dominance"] == pytest.approx(0.5)
    different_wrong_prediction = _teacher_row()
    different_wrong_prediction["teacher_clean_probabilities"] = [0.05, 0.2, 0.05, 0.7, 0, 0, 0, 0, 0, 0]
    clean_teacher = _teacher(different_wrong_prediction, clean=True)
    assert teacher["correct"] is clean_teacher["correct"] is False
    assert teacher["prediction"] != clean_teacher["prediction"]
    tied = _teacher_row()
    tied["teacher_adversarial_probabilities"] = [0.4, 0.4, 0.2, 0, 0, 0, 0, 0, 0, 0]
    with pytest.raises(StrongDiagnosticsError, match="argmax tie"):
        _teacher(tied)
    assert _snapshot_taxonomy([True, False, True]) == "oscillating"
    assert _dense_non_recovery([False, False]) == "persistent_wrong"
    assert _dense_non_recovery([False, True, False]) == "recovered_relapsed"


def test_oof_folds_are_exactly_class_stratified_and_heldout_values_do_not_fit_transform() -> None:
    ids = tuple(range(100))
    classes = {item: item % 10 for item in ids}
    folds = _class_stratified_folds(ids, classes)
    for class_id in range(10):
        counts = [sum(classes[item] == class_id and folds[item] == fold for item in ids) for fold in range(5)]
        assert max(counts) - min(counts) <= 1
    transform = _fit_empirical_rank((0.0, 1.0, 1.0, 3.0))
    assert _apply_empirical_rank(transform, -1.0) == 0.0
    assert _apply_empirical_rank(transform, 1.0) == pytest.approx(0.5)
    assert _apply_empirical_rank(transform, 2.0) == pytest.approx(0.75)
    assert _apply_empirical_rank(transform, 1000.0) == 1.0
    labels = {item: int(item % 2 == 0) for item in ids}
    features = {
        "M": {item: float(labels[item]) for item in ids},
        "H": {item: float((item // 2) % 2) for item in ids},
        "D": {item: float(item % 3) for item in ids},
    }
    train, test = list(ids[:80]), list(ids[80:])
    train_vectors, _ = _fold_rank_vectors(train=train, test=test, columns=("M", "H"), features=features)
    heldout_mutated = {name: dict(values) for name, values in features.items()}
    heldout_mutated["M"][test[0]] = 1_000_000.0
    mutated_train_vectors, _ = _fold_rank_vectors(
        train=train,
        test=test,
        columns=("M", "H"),
        features=heldout_mutated,
    )
    assert train_vectors == mutated_train_vectors
    first = _oof_scores(ids, labels, features, classes)
    labels_with_extra = {**labels, 99_999: 1}
    assert _oof_scores(ids, labels_with_extra, features, classes) == first
    assert set(first) == {"M", "M+D", "H", "H+D", "M+H", "M+H+D"}
    assert first["M+D"]["folds"] == 5


def test_dense_merge_rejects_overlap_and_blind_rows_do_not_leak_selection_causes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = {
        "run_id": "r",
        "teacher": {},
        "dataset_identity": {},
        "attack_identity": {},
        "saved_resolved_config_mapping_sha256": "c",
        "manifest_sha256": "m",
    }
    panel = {39: {10: {"class_id": 2}, 13: {"class_id": 2}}}
    monkeypatch.setattr(diagnostics, "_strong_lineage", lambda **_: {**reference, "requested_epochs": [39]})
    monkeypatch.setattr(diagnostics, "_validated_replay_identity", lambda _: {"same": True})
    monkeypatch.setattr(diagnostics, "_read_parquet", lambda _: [])
    monkeypatch.setattr(diagnostics, "_strong_panel", lambda *_args, **_kwargs: panel)
    chunks = (
        {"observations": Path("first.parquet"), "lineage": Path("first.json")},
        {"observations": Path("second.parquet"), "lineage": Path("second.json")},
    )
    with pytest.raises(StrongDiagnosticsError, match="overlap"):
        _merge_dense_chunks(chunks, reference=reference, expected_count=2, expected_universe_sha256="x")
    rows = _blinded_candidate_rows(
        "L2",
        taxonomy={
            **{item: "persistent_wrong" for item in range(10, 18)},
            21: "recovered_stable",
            22: "recovered_relapsed",
            **{item: "persistent_wrong" for item in range(30, 38)},
            **{item: "recovered_stable" for item in range(40, 48)},
        },
        classes={**{item: 2 for item in range(10, 23)}, **{item: 3 for item in range(30, 48)}},
        teacher={
            **{item: {"correct": False} for item in range(10, 18)},
            21: {"correct": True},
            22: {"correct": True},
            **{item: {"correct": False} for item in range(30, 38)},
            **{item: {"correct": True} for item in range(40, 48)},
        },
        student_clean_correct={
            **{item: False for item in range(10, 18)},
            **{item: True for item in range(21, 23)},
            **{item: False for item in range(30, 38)},
            **{item: True for item in range(40, 48)},
        },
    )
    assert rows and all(set(row) == {"sample_id", "class_id"} for row in rows)
    assert rows == sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"ffnr-blind-order-v2:L2:{row['class_id']}:{row['sample_id']}".encode()
        ).digest(),
    )
    assert len(rows) == 14  # class 2: 2 pairs; class 3: capped at 5 pairs.
    assert {row["class_id"] for row in rows} == {2, 3}


def test_dense_replay_identity_rejects_cache_source_drift() -> None:
    checkpoint = {
        "epoch": 39,
        "path": "/bundle/39.pt",
        "sha256": "a" * 64,
        "run_id": "run",
        "config_hash": "c" * 64,
        "scientific_git_sha": "d" * 40,
    }
    meta = {
        "contract": "ffnr_strong_replay_ce_pgd20_v1",
        "requested_epochs": [39],
        "checkpoints": [{key: checkpoint[key] for key in ("epoch", "path", "sha256")}],
        "checkpoint_cache_identities": [
            {
                "checkpoint": checkpoint,
                "contract": "ffnr_strong_replay_ce_pgd20_v1",
                "analysis_provenance": {"source_sha256": "e" * 64},
                "attack_identity": {"loss": "ce"},
                "attack_identity_sha256": "f" * 64,
                "dataset": {"name": "cifar10"},
                "teacher": {"registry_id": "teacher"},
                "runtime": {"deterministic_backend": {}},
                "replay_batch_size": 128,
                "attack_seed_base": 7,
                "seed_formula": "formula",
            }
        ],
        "results": [{"epoch": 39, "checkpoint_sha256": "a" * 64}],
        "analysis_provenance": {"source_sha256": "e" * 64},
        "checkpoint_inventory": "/cache/inventory.json",
        "checkpoint_inventory_sha256": "0" * 64,
        "run_id": "run",
        "saved_resolved_config_mapping_sha256": "c" * 64,
        "attack_identity": {"loss": "ce"},
        "attack_identity_sha256": "f" * 64,
        "dataset_identity": {"name": "cifar10"},
        "teacher": {"registry_id": "teacher"},
        "runtime": {"deterministic_backend": {}},
        "replay_batch_size": 128,
        "attack_seed_base": 7,
        "seed_formula": "formula",
        "manifest_sha256": "1" * 64,
    }
    assert _validated_replay_identity(meta)["scientific_git_sha"] == "d" * 40
    meta["checkpoint_cache_identities"][0]["analysis_provenance"] = {"source_sha256": "2" * 64}
    with pytest.raises(StrongDiagnosticsError, match="cache identity"):
        _validated_replay_identity(meta)


def test_cross_seed_overlap_only_uses_shared_taxonomy_keys() -> None:
    def report(d4: dict[str, set[int]], d5: dict[str, set[int]]) -> dict[str, object]:
        return {
            "input_identity": {"stable_id_class_universe_sha256": "same"},
            "_d4_taxonomies": {str(anchor): {"online": d4, "strong": d4} for anchor in (39, 59, 79)},
            "_d5_taxonomies": {str(anchor): d5 for anchor in (39, 59, 79)},
        }

    tables = _cross_seed_tables(
        {
            "L2": report({"oscillating": {1}, "other": {2}}, {"persistent_wrong": {3}}),
            "L4": report(
                {"oscillating": {1, 4}},
                {"persistent_wrong": {3}, "recovered_stable": {5}},
            ),
        }
    )
    assert tables["available"] is True
    assert "e39:online:oscillating" in tables["D4"]
    assert tables["D4"]["e39:online:other"]["right_count"] == 0
    assert tables["D5"]["e39:persistent_wrong"]["jaccard"] == 1.0


def test_dense_nr_subtype_tables_are_a_complete_strong_current_wrong_partition() -> None:
    taxonomy = {10: "persistent_wrong", 11: "recovered_relapsed", 12: "recovered_stable"}
    raw = {
        item: {
            "student_clean_probability_margin": 0.4,
            "student_adversarial_probability_margin": -0.2,
            "student_clean_logit_margin": 1.0,
            "student_adversarial_logit_margin": -0.5,
            "student_adversarial_ce": 2.0,
            "student_clean_to_adversarial_probability_margin_delta": -0.6,
            "student_clean_to_adversarial_logit_margin_delta": -1.5,
            "student_clean_correct": True,
            "student_robust_correct": False,
            "student_clean_to_adversarial_prediction_flip": True,
        }
        for item in taxonomy
    }
    strong = {item: {"class_id": item % 10, "teacher_js": 0.1} for item in taxonomy}
    teacher = {
        item: {
            "prediction": 2,
            "correct": False,
            "true_probability": 0.1,
            "max_wrong_probability": 0.8,
            "dominance": 0.7,
            "entropy": 0.6,
        }
        for item in taxonomy
    }
    clean_teacher = {
        item: {
            "prediction": 1,
            "correct": True,
            "true_probability": 0.8,
            "max_wrong_probability": 0.1,
            "dominance": -0.7,
            "entropy": 0.5,
        }
        for item in taxonomy
    }
    outcome = {epoch: {item: {"correct": item != 10} for item in taxonomy} for epoch in (189, 194, 199)}
    tables = _dense_nr_subtype_tables(
        taxonomy,
        online_current_wrong={10, 12},
        raw_rows=raw,
        strong_rows=strong,
        outcome=outcome,
        teacher=teacher,
        clean_teacher=clean_teacher,
    )
    assert tuple(tables) == ("persistent_wrong", "recovered_relapsed", "recovered_stable")
    assert sum(table["count"] for table in tables.values()) == len(taxonomy)
    assert tables["persistent_wrong"]["online_current_wrong_overlap"] == {"count": 1, "fraction": 1.0}


def test_blinded_renderer_emits_only_stable_id_label_and_image_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Image:
        def save(self, path: Path, *, format: str) -> None:  # noqa: A002
            assert format == "PNG"
            path.write_bytes(b"png")

    class Dataset:
        def __init__(self, **_: object) -> None:
            pass

        def __len__(self) -> int:
            return 20

        def __getitem__(self, item: int) -> tuple[Image, int]:
            return Image(), 3

    torchvision = types.ModuleType("torchvision")
    datasets = types.ModuleType("torchvision.datasets")
    datasets.CIFAR10 = Dataset  # type: ignore[attr-defined]
    torchvision.datasets = datasets  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torchvision", torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.datasets", datasets)
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    rows = _render_blinded_cifar10_panel(
        output_dir=tmp_path / "output",
        rows=[{"run_label": "L2", "sample_id": 7, "class_id": 3}],
        dataset_root=dataset_root,
    )
    assert rows == [
        {
            "run_label": "L2",
            "panel_index": 1,
            "sample_id": 7,
            "class_id": 3,
            "image_path": "images/0001-L2-id7.png",
        }
    ]


def test_wilson_contract() -> None:
    assert _wilson(5, 10)["rate"] == 0.5


def test_common_terminal_endpoint_requires_exact_ce_pgd20_window() -> None:
    panel = {epoch: {7: {"correct": epoch != 194}} for epoch in (189, 194, 199)}
    assert _endpoint(panel, "majority") == {7: 0}
    assert _endpoint(panel, "all") == {7: 0}
    with pytest.raises(StrongDiagnosticsError, match="common terminal"):
        _endpoint({189: {7: {"correct": True}}, 194: {7: {"correct": False}}}, "majority")


def test_online_strong_join_rejects_class_drift_with_same_stable_ids() -> None:
    feature = {39: {7: {"class_id": 1}}}
    outcome = {189: {7: {"class_id": 1}}}
    online = {39: {7: {"class_id": 1}}}
    assert _validate_online_strong_class_join(feature, outcome, online) == {7}
    online[39][7]["class_id"] = 2
    with pytest.raises(StrongDiagnosticsError, match="class join"):
        _validate_online_strong_class_join(feature, outcome, online)


def test_vectorized_oof_logistic_matches_repository_fixed_fit() -> None:
    train = [(0.1, 0.9), (0.2, 0.8), (0.8, 0.2), (0.9, 0.1)]
    targets = [0, 0, 1, 1]
    test = [(0.3, 0.7), (0.7, 0.3)]
    expected = _predict_logistic(_fit_logistic(train, targets), test)
    assert _vectorized_logistic_predict(train, targets, test) == pytest.approx(expected, abs=1e-12)


def test_tracked_clean_provenance_rejects_dirty_source(monkeypatch: pytest.MonkeyPatch) -> None:
    def clean(command: list[str], **_: object) -> CompletedProcess[bytes | str]:
        return CompletedProcess(command, 0, stdout="a" * 40 if "rev-parse" in command else "")

    monkeypatch.setattr(diagnostics.subprocess, "run", clean)
    assert _tracked_clean_provenance()["git"]["dirty"] is False

    def dirty(command: list[str], **_: object) -> CompletedProcess[bytes | str]:
        stdout = ""
        if "status" in command:
            stdout = "M src/ard/analysis/ffnr_strong_diagnostics.py\n"
        elif "rev-parse" in command:
            stdout = "a" * 40
        return CompletedProcess(command, 0, stdout=stdout)

    monkeypatch.setattr(diagnostics.subprocess, "run", dirty)
    with pytest.raises(StrongDiagnosticsError, match="tracked-clean"):
        _tracked_clean_provenance()
