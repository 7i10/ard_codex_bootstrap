from __future__ import annotations

import json
from pathlib import Path

import pytest

from ard.analysis.history_bootstrap import HistoryBootstrapError, _final_gate, run_bootstrap

pytestmark = pytest.mark.t1


def _point(path: Path, *, passed: bool = True) -> Path:
    rows = [
        {
            "sample_id": i,
            "class_id": i % 2,
            "outcome": int(i % 3 == 0),
            "baseline": i / 30,
            "candidate": min(1.0, i / 25),
        }
        for i in range(30)
    ]
    tasks = (
        [
            {
                "task_id": f"{label}-epoch39-peak_failure",
                "run": label,
                "anchor": 39,
                "outcome": "peak_failure",
                "stratum": "online_anchor_correct",
                "point_gate_pass": True,
                "joint_primary_gate": "epoch39-peak_failure",
                "rows": rows,
            }
            for label in ("L1", "L3")
        ]
        if passed
        else []
    )
    path.write_text(
        json.dumps(
            {
                "contract": "h5_early_online_collection_v1",
                "cohort_inventory_sha256": "a" * 64,
                "primary_bootstrap_gate": {"epoch39-peak_failure": {"pass": passed}},
                "bootstrap_tasks": tasks,
                "status": "point_gate_pass_bootstrap_pending" if passed else "no_go_point_gate",
            }
        )
    )
    return path


def test_point_fail_creates_no_task_result(tmp_path: Path) -> None:
    result = run_bootstrap(
        point_report=_point(tmp_path / "point.json", passed=False),
        output=tmp_path / "out.json",
        progress_dir=tmp_path / "p",
        max_replicates=5,
    )
    assert result["results"] == []


def test_resume_worker_independence_fingerprint_and_nonoverwrite(tmp_path: Path) -> None:
    point = _point(tmp_path / "point.json")
    partial = run_bootstrap(
        point_report=point, output=tmp_path / "partial.json", progress_dir=tmp_path / "p", workers=1, max_replicates=10
    )
    assert partial["partial"]
    full1 = run_bootstrap(
        point_report=point, output=tmp_path / "full1.json", progress_dir=tmp_path / "p", workers=1, max_replicates=20
    )
    full2 = run_bootstrap(
        point_report=point, output=tmp_path / "full2.json", progress_dir=tmp_path / "p2", workers=2, max_replicates=20
    )
    assert full1["results"] == full2["results"] == []
    assert (
        json.loads((tmp_path / "p" / "L1-epoch39-peak_failure.json").read_text())["completed"]
        == json.loads((tmp_path / "p2" / "L1-epoch39-peak_failure.json").read_text())["completed"]
    )
    with pytest.raises(FileExistsError):
        run_bootstrap(
            point_report=point, output=tmp_path / "full1.json", progress_dir=tmp_path / "p", max_replicates=20
        )
    changed = json.loads(point.read_text())
    changed["bootstrap_tasks"][0]["rows"][0]["candidate"] = 0.99
    point.write_text(json.dumps(changed))
    with pytest.raises(HistoryBootstrapError, match="fingerprint"):
        run_bootstrap(
            point_report=point, output=tmp_path / "changed.json", progress_dir=tmp_path / "p", max_replicates=20
        )


def test_interrupted_bootstrap_persists_completed_replicates_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    point = _point(tmp_path / "point.json")
    from ard.analysis import history_bootstrap

    real_delta = history_bootstrap._delta

    def interrupt_at_three(replicate: int) -> tuple[int, float | None]:
        if replicate == 3:
            raise KeyboardInterrupt
        return real_delta(replicate)

    monkeypatch.setattr(history_bootstrap, "_delta", interrupt_at_three)
    with pytest.raises(KeyboardInterrupt):
        run_bootstrap(
            point_report=point,
            output=tmp_path / "interrupted.json",
            progress_dir=tmp_path / "p",
            max_replicates=10,
        )
    state = json.loads((tmp_path / "p" / "L1-epoch39-peak_failure.json").read_text())
    assert set(state["completed"]) == {"0", "1", "2"}

    monkeypatch.setattr(history_bootstrap, "_delta", real_delta)
    result = run_bootstrap(
        point_report=point,
        output=tmp_path / "resumed.json",
        progress_dir=tmp_path / "p",
        max_replicates=10,
    )
    assert result["partial"]
    assert set(json.loads((tmp_path / "p" / "L1-epoch39-peak_failure.json").read_text())["completed"]) == {
        str(index) for index in range(10)
    }


def test_bootstrap_rejects_forged_gate_duplicate_ids_and_estimator_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    point = _point(tmp_path / "point.json")
    forged = json.loads(point.read_text())
    forged["bootstrap_tasks"][0]["joint_primary_gate"] = "not-a-gate"
    point.write_text(json.dumps(forged))
    with pytest.raises(HistoryBootstrapError, match="point gate"):
        run_bootstrap(
            point_report=point, output=tmp_path / "forged.json", progress_dir=tmp_path / "p", max_replicates=2
        )

    point = _point(tmp_path / "point2.json")
    duplicate = json.loads(point.read_text())
    duplicate["bootstrap_tasks"][0]["rows"][1]["sample_id"] = 0
    point.write_text(json.dumps(duplicate))
    with pytest.raises(HistoryBootstrapError, match="sample/class"):
        run_bootstrap(
            point_report=point, output=tmp_path / "duplicate.json", progress_dir=tmp_path / "p2", max_replicates=2
        )

    point = _point(tmp_path / "point3.json")
    run_bootstrap(point_report=point, output=tmp_path / "partial.json", progress_dir=tmp_path / "p3", max_replicates=2)
    from ard.analysis import history_bootstrap

    monkeypatch.setattr(history_bootstrap, "_source_fingerprint", lambda: {"estimator": "f" * 64})
    with pytest.raises(HistoryBootstrapError, match="fingerprint"):
        run_bootstrap(
            point_report=point, output=tmp_path / "changed.json", progress_dir=tmp_path / "p3", max_replicates=3
        )


def test_bootstrap_rejects_duplicate_gate_run_even_with_distinct_task_ids(tmp_path: Path) -> None:
    point = _point(tmp_path / "point.json")
    duplicate = json.loads(point.read_text())
    repeated = dict(duplicate["bootstrap_tasks"][0])
    repeated["task_id"] = "distinct-but-duplicate-gate-run"
    duplicate["bootstrap_tasks"].append(repeated)
    point.write_text(json.dumps(duplicate))
    with pytest.raises(HistoryBootstrapError, match="duplicate bootstrap gate/run"):
        run_bootstrap(
            point_report=point,
            output=tmp_path / "duplicate-gate-run.json",
            progress_dir=tmp_path / "progress",
            max_replicates=2,
        )


def test_final_gate_requires_both_l1_l3_lowers_and_early_routes_are_independent() -> None:
    rows = [
        {"task_id": "L1-a", "run": "L1", "joint_primary_gate": "epoch39-peak_failure", "lower": 0.01},
        {"task_id": "L3-a", "run": "L3", "joint_primary_gate": "epoch39-peak_failure", "lower": 0.01},
        {"task_id": "L1-b", "run": "L1", "joint_primary_gate": "epoch39-non_recovery", "lower": 0.01},
        {"task_id": "L3-b", "run": "L3", "joint_primary_gate": "epoch39-non_recovery", "lower": -0.01},
    ]
    early = _final_gate("h5_early_online_collection_v1", rows, partial=False)
    assert early["status"] == "any_route_go"
    assert early["routes"]["peak_failure"]["status"] == "go"
    assert early["routes"]["non_recovery"]["status"] == "no_go"
    late = _final_gate("h5_late_history_screen_collection_v2", rows[:2], partial=False)
    assert late["status"] == "go"
    rows[1]["lower"] = 0.0
    assert _final_gate("h5_late_history_screen_collection_v2", rows[:2], partial=False)["status"] == "no_go"
