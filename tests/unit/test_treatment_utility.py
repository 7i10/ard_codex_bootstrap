from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest
import torch

from ard.analysis.rslad_signal_replay import canonical_json
from ard.analysis.signal_audit import _fit_logistic as base_fit_logistic
from ard.analysis.signal_audit import sha256_file
from ard.analysis.treatment_utility import (
    EXPECTED_ATTACK,
    EXPECTED_TRAIN_COUNT,
    TreatmentUtilityError,
    _analysis_source,
    _features,
    _metrics,
    _panel,
    _rank,
    _state,
    run_treatment_utility,
)
from ard.engine.checkpoint import REQUIRED_KEYS

pytestmark = pytest.mark.t1


def test_treatment_utility_point_audit_is_deterministic_and_non_overwriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = {
        f"{seed}:{route}:{arm}": tuple(tmp_path / f"{seed}-{route}-{arm}-{kind}" for kind in range(3))
        for seed in ("L1", "L3")
        for route in ("PF", "NR")
        for arm in ("C", "H", "R")
    }
    for trio in paths.values():
        for path in trio:
            path.write_text("x", encoding="utf-8")
    teacher = {
        "teacher_clean_correct": True,
        "teacher_clean_true_probability": 0.7,
        "teacher_clean_probability_margin": 0.4,
        "teacher_clean_entropy_normalized": 0.3,
        "teacher_adversarial_correct": True,
        "teacher_clean_to_adversarial_prediction_flip": False,
        "teacher_clean_to_adversarial_true_probability_delta": -0.1,
        "teacher_clean_to_adversarial_margin_delta": -0.1,
        "teacher_clean_to_adversarial_kl": 0.02,
    }

    def fake_panel(_obs: Path, _lineage: Path, *, arm: str):
        meta = {
            "seed": 1,
            "teacher": {"t": 1},
            "dataset_identity": {"name": "cifar10"},
            "attack_identity": {"steps": 20},
            "analysis_seed": 7,
            "analysis_provenance": {"git": "x"},
            "parent": {"checkpoint_sha256": "a" * 64, "sample_state_sha256": "b" * 64},
        }
        control = {i: {"sample_id": i, "class_id": i % 2, "robust_correct": bool(i % 2), **teacher} for i in range(40)}
        endpoint = {
            i: {
                **row,
                "mask_selected": arm != "C",
                "robust_correct": not bool(i % 2) if arm != "C" else bool(i % 2),
                "class_id": row["class_id"],
                # A child outcome must never become a post-treatment feature.
                "teacher_clean_true_probability": float("inf") if arm != "C" else 0.7,
            }
            for i, row in control.items()
        }
        return meta, {79: control, 99: endpoint, 119: endpoint}

    monkeypatch.setattr("ard.analysis.treatment_utility._panel", fake_panel)
    monkeypatch.setattr(
        "ard.analysis.treatment_utility._state",
        lambda *_args, **_kwargs: ({i: (i % 2, 0.5, -0.1) for i in range(40)}, "a" * 64, "b" * 64),
    )
    monkeypatch.setattr(
        "ard.analysis.treatment_utility._analysis_source",
        lambda: {
            "git": {"sha": "0" * 40, "dirty": False},
            "files": {"analysis_module": "a", "cli_module": "b"},
            "sha256": "c",
        },
    )
    fit_widths: list[int] = []

    def record_route_fit(features: list[tuple[float, ...]], targets: list[int]):
        fit_widths.append(len(features[0]))
        return base_fit_logistic(features, targets)

    monkeypatch.setattr("ard.analysis.treatment_utility._fit_logistic", record_route_fit)
    report = run_treatment_utility(panels=paths, output=tmp_path / "report.json")
    assert report["exploratory_only"] is True
    assert report["contract"] == "prescriptive_v3_treatment_utility_m2a_point_v1"
    assert report["split"]["train_ids_sha256"] != report["split"]["held_out_ids_sha256"]
    assert len(report["input_hashes"]) == 36
    assert set(report["parent_identities"]) == {"L1:PF", "L1:NR", "L3:PF", "L3:NR"}
    assert set(report["analysis_source"]["files"]) == {"analysis_module", "cli_module"}
    assert set(report["reports"]["L1:PF:H"]["models"]) == {"S", "T", "S+T"}
    assert set(report["reports"]["L1:PF:H"]["top10_jaccard_pairwise"]) == {"S_T", "S_S+T", "T_S+T"}
    assert set(report["point_prerequisites"]) == {"PF", "NR"}
    assert sorted(fit_widths) == [2, 2, 9, 9, 11, 11]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_treatment_utility(panels=paths, output=tmp_path / "report.json")

    def drifted_panel(obs: Path, lineage: Path, *, arm: str):
        meta, panel = fake_panel(obs, lineage, arm=arm)
        if arm == "PF-H":
            meta = {**meta, "teacher": {"t": 2}}
        return meta, panel

    monkeypatch.setattr("ard.analysis.treatment_utility._panel", drifted_panel)
    with pytest.raises(TreatmentUtilityError, match="source/teacher/attack drifted"):
        run_treatment_utility(panels=paths, output=tmp_path / "lineage-drift.json")

    def seed_drifted_panel(obs: Path, lineage: Path, *, arm: str):
        meta, panel = fake_panel(obs, lineage, arm=arm)
        if arm == "PF-H":
            meta = {**meta, "analysis_seed": 8}
        return meta, panel

    monkeypatch.setattr("ard.analysis.treatment_utility._panel", seed_drifted_panel)
    with pytest.raises(TreatmentUtilityError, match="source/teacher/attack drifted"):
        run_treatment_utility(panels=paths, output=tmp_path / "analysis-seed-drift.json")

    def anchor_drifted_panel(obs: Path, lineage: Path, *, arm: str):
        meta, panel = fake_panel(obs, lineage, arm=arm)
        if arm == "C" and "-NR-C-" in obs.name:
            panel = {**panel, 79: {**panel[79], 0: {**panel[79][0], "teacher_clean_entropy_normalized": 0.2}}}
        return meta, panel

    monkeypatch.setattr("ard.analysis.treatment_utility._panel", anchor_drifted_panel)
    with pytest.raises(TreatmentUtilityError, match="one exact epoch-79 observation panel"):
        run_treatment_utility(panels=paths, output=tmp_path / "parent-observation-drift.json")

    monkeypatch.setattr("ard.analysis.treatment_utility._panel", fake_panel)
    monkeypatch.setattr(
        "ard.analysis.treatment_utility.deterministic_hash_split", lambda *_args, **_kwargs: ((1,), (1,))
    )
    with pytest.raises(TreatmentUtilityError, match="split leakage"):
        run_treatment_utility(panels=paths, output=tmp_path / "split-leak.json")


def test_treatment_utility_ties_are_stable_and_official_input_is_rejected(tmp_path: Path) -> None:
    ranked = _rank(
        [{"sample_id": 7, "rescue": 1, "utility": 1}, {"sample_id": 2, "rescue": 0, "utility": 0}],
        [0.5, 0.5],
        model="S",
    )
    assert ranked["top_ids"] == [2]
    with pytest.raises(TreatmentUtilityError, match="exact L1/L3"):
        run_treatment_utility(panels={}, output=tmp_path / "x.json")


def test_treatment_utility_feature_order_outcome_metrics_and_v3_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    teacher = {
        "teacher_clean_correct": True,
        "teacher_clean_true_probability": 0.7,
        "teacher_clean_probability_margin": 0.4,
        "teacher_clean_entropy_normalized": 0.3,
        "teacher_adversarial_correct": False,
        "teacher_clean_to_adversarial_prediction_flip": True,
        "teacher_clean_to_adversarial_true_probability_delta": -0.1,
        "teacher_clean_to_adversarial_margin_delta": -0.2,
        "teacher_clean_to_adversarial_kl": 0.02,
    }
    features = _features(teacher, (3, 0.25, -0.5))
    assert features[:2] == (0.25, -0.5)
    assert features[2:7] == (1.0, 0.7, 0.4, 0.3, 0.0)
    assert features[-1] == pytest.approx(0.019802627)
    with pytest.raises(TreatmentUtilityError, match="unavailable/non-finite"):
        _features({**teacher, "teacher_clean_entropy_normalized": float("inf")}, (3, 0.25, -0.5))

    rows = [
        {"discordant": True, "rescue": 1, "utility": 1},
        {"discordant": True, "rescue": 0, "utility": -1},
        {"discordant": False, "rescue": 0, "utility": 0},
    ]
    metrics = _metrics(rows, [0.9, 0.1, 0.5])
    assert metrics == pytest.approx(
        {"auroc": 1.0, "auprc": 1.0, "prevalence": 0.5, "log_loss": 0.10536051565782628, "brier": 0.01, "count": 2}
    )

    calls: list[dict[str, object]] = []
    panel = {99: {sample_id: {"sample_id": sample_id} for sample_id in range(EXPECTED_TRAIN_COUNT)}}
    meta = {
        "dataset_identity": {
            "dataset": {"name": "cifar10", "split": "train"},
            "train_expected_count": EXPECTED_TRAIN_COUNT,
        },
        "attack_identity": dict(EXPECTED_ATTACK),
        "row_count": EXPECTED_TRAIN_COUNT,
    }

    def fake_v3_reader(*_args: object, **kwargs: object):
        calls.append(kwargs)
        return meta, panel

    monkeypatch.setattr("ard.analysis.treatment_utility._read_v3_observations", fake_v3_reader)
    assert len(_panel(Path("observations.parquet"), Path("lineage.json"), arm="PF-H")[1][99]) == EXPECTED_TRAIN_COUNT
    assert calls[-1]["require_parent_epoch"] is False
    _panel(Path("observations.parquet"), Path("lineage.json"), arm="C")
    assert calls[-1]["require_parent_epoch"] is True

    drifts = [
        ("dataset_identity", {"dataset": {"name": "cifar100", "split": "train"}, "train_expected_count": 45_000}),
        ("dataset_identity", {"dataset": {"name": "cifar10", "split": "test"}, "train_expected_count": 45_000}),
        ("dataset_identity", {"dataset": {"name": "cifar10", "split": "train"}, "train_expected_count": 44_999}),
        ("attack_identity", {**EXPECTED_ATTACK, "loss": "kl"}),
        ("attack_identity", {**EXPECTED_ATTACK, "norm": "l2"}),
        ("attack_identity", {**EXPECTED_ATTACK, "epsilon": "4/255"}),
        ("attack_identity", {**EXPECTED_ATTACK, "epsilon_value": 4 / 255}),
        ("attack_identity", {**EXPECTED_ATTACK, "step_size": "1/255"}),
        ("attack_identity", {**EXPECTED_ATTACK, "step_size_value": 1 / 255}),
        ("attack_identity", {**EXPECTED_ATTACK, "steps": 10}),
        ("attack_identity", {**EXPECTED_ATTACK, "random_start": False}),
        ("attack_identity", {**EXPECTED_ATTACK, "input_domain": "normalized"}),
        ("attack_identity", {**EXPECTED_ATTACK, "student_mode": "train"}),
        ("attack_identity", {**EXPECTED_ATTACK, "teacher_mode": "train"}),
        ("attack_identity", {**EXPECTED_ATTACK, "name": "AutoAttack"}),
    ]
    for key, value in drifts:
        bad_meta = {**meta, key: value}
        monkeypatch.setattr(
            "ard.analysis.treatment_utility._read_v3_observations",
            lambda *_args, _meta=bad_meta, **_kwargs: (_meta, panel),
        )
        with pytest.raises(TreatmentUtilityError, match="lineage/dataset/attack contract"):
            _panel(Path("observations.parquet"), Path("lineage.json"), arm="PF-H")


@pytest.mark.parametrize(
    "records",
    [
        {True: {"true_label": 1, "seen": 2, "robust_correct_count": 1, "margin_ema": 0.1}},
        {"01": {"true_label": 1, "seen": 2, "robust_correct_count": 1, "margin_ema": 0.1}},
        {"1": {"true_label": True, "seen": 2, "robust_correct_count": 1, "margin_ema": 0.1}},
        {"1": {"true_label": 1, "seen": True, "robust_correct_count": 1, "margin_ema": 0.1}},
        {"1": {"true_label": 1, "seen": 2, "robust_correct_count": True, "margin_ema": 0.1}},
        {"1": {"true_label": 1, "seen": 2, "robust_correct_count": -1, "margin_ema": 0.1}},
        {"1": {"true_label": 1, "seen": 2, "robust_correct_count": 3, "margin_ema": 0.1}},
        {"1": {"true_label": 1, "seen": 2, "robust_correct_count": 1, "margin_ema": 1.01}},
    ],
)
def test_treatment_utility_state_rejects_noncanonical_or_invalid_records(
    tmp_path: Path, records: dict[object, dict[str, object]]
) -> None:
    state = {"records": records}
    payload = {key: {} for key in REQUIRED_KEYS}
    payload.update({"epoch": 79, "sample_state": state})
    checkpoint = tmp_path / "parent.pt"
    torch.save(payload, checkpoint)
    parent = {
        "checkpoint_sha256": sha256_file(checkpoint),
        "sample_state_sha256": hashlib.sha256(canonical_json(state)).hexdigest(),
    }
    with pytest.raises(TreatmentUtilityError, match="parent stable-ID record|parent state feature"):
        _state(checkpoint, parent=parent)


def test_treatment_utility_source_provenance_requires_a_clean_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command: tuple[str, ...], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        stdout = (
            "a" * 40
            if "rev-parse" in command
            else " M src/ard/analysis/treatment_utility.py\n"
            if "status" in command
            else ""
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout)

    monkeypatch.setattr("ard.analysis.treatment_utility.subprocess.run", fake_run)
    with pytest.raises(TreatmentUtilityError, match="tracked-clean"):
        _analysis_source()
