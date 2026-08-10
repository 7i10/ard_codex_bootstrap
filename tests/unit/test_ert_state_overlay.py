from __future__ import annotations

from pathlib import Path

import pytest

import ard.analysis.ert_state_overlay as overlay

# ruff: noqa: E501

pytestmark = pytest.mark.unit


def _anchor_row(*, student_adv: bool, teacher_adv: bool, m_teacher_adv: float) -> dict[str, object]:
    return {
        "class_id": 1,
        "student_clean_correct": True,
        "student_robust_correct": student_adv,
        "teacher_clean_correct": True,
        "teacher_adv_correct": teacher_adv,
        "mS_clean": 0.4,
        "mS_adv": 0.2 if student_adv else -0.2,
        "mT_clean": 0.7,
        "mT_adv": m_teacher_adv,
        "DeltaS": 0.2 if student_adv else 0.6,
        "DeltaT": 0.7 - m_teacher_adv,
    }


def test_anchor_state_masks_are_stable_and_contain_no_future_outcomes(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = {
        0: _anchor_row(student_adv=False, teacher_adv=True, m_teacher_adv=0.1),
        1: _anchor_row(student_adv=False, teacher_adv=True, m_teacher_adv=0.2),
        2: _anchor_row(student_adv=False, teacher_adv=True, m_teacher_adv=0.3),
        3: _anchor_row(student_adv=False, teacher_adv=False, m_teacher_adv=-0.4),
        4: {**_anchor_row(student_adv=True, teacher_adv=True, m_teacher_adv=0.4), "student_clean_correct": False},
    }
    online = {
        79: {
            item: {
                "class_id": 1,
                "current_correct": bool(item % 2),
                "frequency_risk": 0.1,
                "margin_risk": 0.2,
                "last_margin_risk": 0.3,
            }
            for item in rows
        }
    }
    meta = {
        "run_id": "run",
        "teacher": {"name": "Chen"},
        "dataset_identity": {"name": "cifar10"},
        "saved_resolved_config_mapping_sha256": "cfg",
        "requested_epochs": [79],
        "checkpoints": [{"epoch": 79, "sha256": "parent"}],
    }
    monkeypatch.setattr(overlay, "_strong_lineage", lambda **_: meta)
    monkeypatch.setattr(
        overlay,
        "_online_panel",
        lambda *_: (
            online,
            {
                "run_id": "run",
                "teacher": {"name": "Chen"},
                "dataset_identity": {"name": "cifar10"},
                "config_hash": "cfg",
                "attack_identity": overlay.EXPECTED_ONLINE_ATTACK,
            },
        ),
    )
    monkeypatch.setattr(overlay, "_validate_online_attack", lambda _: None)
    monkeypatch.setattr(overlay, "_read_compact_observations", lambda *_args, **_kwargs: {79: {}})
    monkeypatch.setattr(overlay, "_margin_rows", lambda *_args, **_kwargs: rows)
    monkeypatch.setattr(
        overlay,
        "_json",
        lambda *_args, **_kwargs: {
            "parent_epoch": 79,
            "parent_checkpoint_sha256": "parent",
            "parent_raw_config_sha256": "cfg",
            "parent_sample_state_records": 5,
            "parent_sample_state_sha256": "state",
        },
    )
    monkeypatch.setattr(overlay, "sha256_file", lambda _path: "a" * 64)

    table, manifest = overlay.build_state_bundle(
        label="L2",
        feature_observations=Path("feature.parquet"),
        feature_lineage=Path("feature.json"),
        online_states=Path("online.parquet"),
        online_lineage=Path("online.json"),
        parent_fork_lineage=Path("parent.json"),
        expected_count=5,
    )

    assert set(table[0]) == set(overlay.STATE_COLUMNS)
    assert not any("future" in field or "outcome" in field for field in overlay.STATE_COLUMNS)
    assert manifest["definitions"]["student_state_definition"]["S3"].startswith("student_clean_correct")
    assert [row["teacher_state_q10"] for row in table] == ["T2", "T1", "T1", "T3", "T1"]
    assert [row["teacher_state_q20"] for row in table] == ["T2", "T1", "T1", "T3", "T1"]
    assert [row["signed_teacher_dominance"] for row in table] == [-0.1, -0.2, -0.3, 0.4, -0.4]
    assert manifest["masks"]["s3_t2_q10"]["selected_ids"] == [0]
    assert manifest["masks"]["s3_t3_q20"]["selected_ids"] == [3]
    assert manifest["masks"]["student_clean_wrong_teacher_clean_correct"]["selected_ids"] == [4]


def _endpoint_row(
    sample_id: int, *, robust: bool, clean: bool, selected: bool = False, random: bool = False
) -> dict[str, object]:
    return {
        "class_id": 1,
        "student_robust_correct": robust,
        "student_clean_correct": clean,
        "student_clean_probability_margin": 0.2 if clean else -0.1,
        "student_adversarial_probability_margin": 0.1 if robust else -0.2,
        "route_b_selected": selected,
        "route_b_random": random,
    }


def test_overlay_effect_reports_rescue_harm_margin_and_clean_harm() -> None:
    state_rows = [
        {
            "sample_id": 0,
            "class_id": 1,
            "teacher_clean_correct": True,
            "teacher_adv_correct": True,
            "mT_adv": 0.2,
            "DeltaT": 0.4,
        },
        {
            "sample_id": 1,
            "class_id": 1,
            "teacher_clean_correct": False,
            "teacher_adv_correct": False,
            "mT_adv": -0.3,
            "DeltaT": 1.0,
        },
    ]
    control = {
        0: _endpoint_row(0, robust=False, clean=True, selected=True),
        1: _endpoint_row(1, robust=True, clean=True, random=True),
    }
    treated = {
        0: _endpoint_row(0, robust=True, clean=True, selected=True),
        1: _endpoint_row(1, robust=False, clean=False, random=True),
    }
    by_arm = {"C79": control, "RA": treated, "RAR": treated, "RB": treated, "RBR": treated}
    masks = {"s3_t1_q10": {"selected_ids": [0]}, "s3_t2_q10": {"selected_ids": [0]}, "s3_t3_q10": {"selected_ids": [1]}}
    report = overlay.summarize_overlay(state_rows=state_rows, masks=masks, endpoints={84: by_arm})

    effect = report["horizons"]["84"]["old_route_b_selected"]["effects"]["RB"]
    assert effect["rescue_count"] == 1
    assert effect["harm_count"] == 0
    assert effect["clean_harm_count"] == 0
    assert effect["clean_harm_rate_over_cohort"] == 0
    assert effect["control_clean_correct_count"] == 1
    assert effect["clean_harm_rate_given_control_clean_correct"] == 0
    random_effect = report["horizons"]["84"]["old_route_b_random"]["effects"]["RB"]
    assert random_effect["harm_count"] == 1
    assert random_effect["clean_harm_count"] == 1
    assert random_effect["clean_harm_rate_over_cohort"] == 1.0
    assert random_effect["clean_harm_rate_given_control_clean_correct"] == 1.0
    assert "mean_mT_adv" in report["horizons"]["84"]["s3_t1_q10"]["anchor"]


def test_non_overwrite_is_checked_before_source_provenance(tmp_path: Path) -> None:
    config = tmp_path / "overlay.yaml"
    config.write_text("schema_version: 1\n", encoding="utf-8")
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(overlay.ERTStateOverlayError, match="overwrite"):
        overlay.run_overlay(config_path=config, output_dir=output)


def test_overlay_partial_output_is_removed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = {name: tmp_path / f"missing-{name}" for name in (
        "feature_observations",
        "feature_lineage",
        "online_states",
        "online_lineage",
        "parent_fork_lineage",
    )}
    monkeypatch.setattr(overlay, "_tracked_clean_provenance", lambda: {})
    monkeypatch.setattr(
        overlay,
        "load_config",
        lambda _path: {
            "expected_count": 1,
            "runs": {"L2": {"anchor_state": source}, "L4": {"anchor_state": source}},
        },
    )
    monkeypatch.setattr(overlay, "build_state_bundle", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    output = tmp_path / "overlay-output"
    with pytest.raises(RuntimeError, match="boom"):
        overlay.run_overlay(config_path=tmp_path / "config.yaml", output_dir=output)
    assert not output.exists()
    assert not list(tmp_path.glob(".overlay-output.tmp-*"))


def test_endpoint_fork_identity_rejects_child_or_epoch_drift() -> None:
    fork = {
        "child_tracker_run_id": "child-run",
        "child_config_sha256": "c" * 64,
        "parent_git_sha": "a" * 40,
    }

    def meta() -> dict[str, object]:
        return {
            "endpoint_epoch": 84,
            "arms": {
                arm: {
                    "checkpoint": {
                        "run_id": "child-run" if arm == "C79" else f"{arm}-run",
                        "config_hash": "c" * 64 if arm == "C79" else "d" * 64,
                        "epoch": 84,
                        "scientific_git_sha": "a" * 40,
                        "sha256": "e" * 64,
                    }
                }
                for arm in overlay.ARMS
            },
        }

    overlay._validate_endpoint_fork_identity(meta=meta(), parent_fork=fork, horizon=84)
    changed_run = meta()
    changed_run["arms"]["C79"]["checkpoint"]["run_id"] = "other"  # type: ignore[index]
    with pytest.raises(overlay.ERTStateOverlayError, match="child run/config"):
        overlay._validate_endpoint_fork_identity(meta=changed_run, parent_fork=fork, horizon=84)
    changed_config = meta()
    changed_config["arms"]["C79"]["checkpoint"]["config_hash"] = "f" * 64  # type: ignore[index]
    with pytest.raises(overlay.ERTStateOverlayError, match="child run/config"):
        overlay._validate_endpoint_fork_identity(meta=changed_config, parent_fork=fork, horizon=84)
    changed_epoch = meta()
    changed_epoch["endpoint_epoch"] = 83
    with pytest.raises(overlay.ERTStateOverlayError, match="registered horizon"):
        overlay._validate_endpoint_fork_identity(meta=changed_epoch, parent_fork=fork, horizon=84)
