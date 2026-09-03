from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from ard.analysis.ert_i100_s2_longitudinal import (
    LongitudinalStateError,
    canonical_action_states,
    prepare_output_dir,
    replay_canonical_state,
)
from ard.analysis.ert_i100_s2_secant_forensic import (
    central_difference,
    dynamic_pair_margin,
    scalar_secant_loss,
    secant_components,
)
from scripts.aggregate_ert_i100_s2_forensic_audit import (
    endpoint_checkpoint,
    entrant_summary,
    execution_config_lineage,
    fixed_cohort_trajectory,
    fixed_mask_ids,
    merged_runtime_proxy_payloads,
    runtime_proxy_payloads,
)


def _row(sample_id: int, *, clean: bool, adv: bool, margin: float, teacher_adv: bool = True) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "student_clean_correct": clean,
        "student_ce20_adv_correct": adv,
        "student_ce20_adv_margin": margin,
        "teacher_ce20_adv_correct": teacher_adv,
        "teacher_ce20_adv_margin": margin + 0.1,
    }


def test_action_state_partition_gives_clean_wrong_precedence() -> None:
    rows = [
        _row(1, clean=False, adv=False, margin=-0.4),
        _row(2, clean=True, adv=False, margin=-0.2),
        _row(3, clean=True, adv=True, margin=0.001),
        _row(4, clean=True, adv=True, margin=0.8),
    ]
    states = canonical_action_states(rows)["state_by_id"]
    assert states[1]["branch"] == "Clean-Wrong"
    assert states[2]["branch"] == "S3-non-CW"
    assert states[3]["branch"] == "S2"
    assert states[4]["branch"] == "S1"


def test_fixed_cohort_trajectory_counts_adjacent_observed_transitions() -> None:
    state_by_epoch = {
        99: {7: {"branch": "S2", "teacher": "T1", "joint": "S2xT1"}},
        104: {7: {"branch": "S1", "teacher": "T1", "joint": "S1xT1"}},
        109: {7: {"branch": "S2", "teacher": "T1", "joint": "S2xT1"}},
        114: {7: {"branch": "S2", "teacher": "T3", "joint": "S2xT3"}},
    }

    summary = fixed_cohort_trajectory(selected={7}, state_by_epoch=state_by_epoch)

    assert summary["membership_patterns"] == {"1010": 1}
    assert summary["overlapping_observed_indicators"]["P5_leave_then_reenter_S2xT1"] == 1
    assert summary["explicit_observed_reentry_routes"]["S2_to_S1_to_S2"] == 1
    assert summary["teacher_transitions_when_student_S2_at_either_endpoint"] == {
        "T1_to_T1": 2,
        "T1_to_T3": 1,
    }
    assert summary["observability"]["P6_multiple_exit_reentry"]["observable"] is False


def test_entrant_persistence_uses_active_runs_not_first_entry() -> None:
    patterns = ("000", "001", "010", "011", "100", "101", "110", "111")
    initial = {
        index: {
            "branch": "S2" if index == 1 else "S1",
            "teacher": "T2" if index == 1 else "T1",
            "joint": "S2xT2" if index == 1 else "S1xT1",
        }
        for index in range(len(patterns))
    }

    def state(active: bool) -> dict[str, str]:
        return {"branch": "S2" if active else "S1", "teacher": "T1", "joint": "S2xT1" if active else "S1xT1"}

    state_by_epoch = {
        epoch: {index: state(pattern[offset] == "1") for index, pattern in enumerate(patterns)}
        for offset, epoch in enumerate((104, 109, 114))
    }
    summary = entrant_summary(initial=initial, state_by_epoch=state_by_epoch)

    assert summary["n"] == 7
    assert summary["e99_origin"] == {"e99_S1": 6, "e99_S2xT2T3": 1}
    assert summary["persistence"] == {"one-endpoint-only": 3, "re-entry": 1, "repeated": 3}


def test_runtime_proxy_uses_same_seed_canonical_checkpoint() -> None:
    result = {
        "seeds": {
            "dev-1": {"arms": {"dpm": {"endpoint_metadata": {"104": {"checkpoint_sha256": "dev1"}}}}},
            "dev-2": {"arms": {"dpm": {"endpoint_metadata": {"104": {"checkpoint_sha256": "dev2"}}}}},
        }
    }
    assert endpoint_checkpoint(result, seed="dev-1", arm="dpm", epoch=104) == "dev1"
    assert endpoint_checkpoint(result, seed="dev-2", arm="dpm", epoch=104) == "dev2"


def test_fixed_mask_ids_are_loaded_from_the_requested_seed_file(tmp_path: Path) -> None:
    dev1 = tmp_path / "dev1.json"
    dev2 = tmp_path / "dev2.json"
    dev1.write_text(json.dumps({"masks": {"s2_t1": {"selected_ids": [1, 3]}}}), encoding="utf-8")
    dev2.write_text(json.dumps({"masks": {"s2_t1": {"selected_ids": [2, 4]}}}), encoding="utf-8")
    assert fixed_mask_ids(dev1) == {1, 3}
    assert fixed_mask_ids(dev2) == {2, 4}


def test_execution_config_lineage_allows_only_teacher_path_rebase(tmp_path: Path) -> None:
    teacher_sha = "fc398a4890e6856b5dd80856076000ec9e2debdd12d9f78a66171b9ffc383983"
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text(
        f"teacher:\n  checkpoint: /host-a/teacher.pt\n  checkpoint_sha256: {teacher_sha}\nmethod:\n  id: rslad\n",
        encoding="utf-8",
    )
    second.write_text(
        f"teacher:\n  checkpoint: /host-b/teacher.pt\n  checkpoint_sha256: {teacher_sha}\nmethod:\n  id: rslad\n",
        encoding="utf-8",
    )
    lineage = execution_config_lineage((first, second), seed="dev-2")
    assert len(lineage["configs"]) == 2
    changed = tmp_path / "changed.yaml"
    changed.write_text(
        f"teacher:\n  checkpoint: /host-b/teacher.pt\n  checkpoint_sha256: {teacher_sha}\nmethod:\n  id: changed\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="permitted Teacher checkpoint path rebase"):
        execution_config_lineage((first, changed), seed="dev-2")


def test_runtime_proxy_payloads_discovers_nested_arm_epoch_artifacts(tmp_path: Path) -> None:
    payload = {
        "contract": "ert_rslad_i100_s2_checkpoint_no_update_runtime_activity_proxy_v1",
        "seed": "dev-1",
        "arm": "dpm",
        "checkpoint_epoch": 104,
    }
    nested = tmp_path / "dev1" / "dpm" / "e104.json"
    nested.parent.mkdir(parents=True)
    nested.write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "orchestration.json").write_text("{}", encoding="utf-8")

    loaded = runtime_proxy_payloads(tmp_path)

    assert loaded == {("dev-1", "dpm", 104): payload}


def test_explicit_runtime_proxy_override_replaces_only_the_matching_key(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    override = tmp_path / "override"
    base = {
        "contract": "ert_rslad_i100_s2_checkpoint_no_update_runtime_activity_proxy_v1",
        "seed": "dev-1",
        "arm": "dpm",
        "checkpoint_epoch": 104,
        "source": "old",
    }
    replacement = {**base, "source": "new"}
    primary.mkdir()
    override.mkdir()
    (primary / "base.json").write_text(json.dumps(base), encoding="utf-8")
    (override / "replacement.json").write_text(json.dumps(replacement), encoding="utf-8")

    merged = merged_runtime_proxy_payloads(primary_dirs=(primary,), override_dirs=(override,))

    assert merged[("dev-1", "dpm", 104)]["source"] == "new"


def test_secant_scalar_autograd_matches_central_difference_away_from_kinks() -> None:
    adv = torch.tensor([-0.40], dtype=torch.float64, requires_grad=True)
    clean = torch.tensor([-0.10], dtype=torch.float64, requires_grad=True)
    rho = torch.tensor([0.03125], dtype=torch.float64)
    teacher_distance = torch.tensor([0.6], dtype=torch.float64)
    loss = scalar_secant_loss(adv, clean, rho=rho, d_teacher=teacher_distance, epsilon=1e-12).sum()
    grad_adv, grad_clean = torch.autograd.grad(loss, (adv, clean))
    fd_adv = central_difference(
        lambda value: scalar_secant_loss(
            value, clean.detach(), rho=rho, d_teacher=teacher_distance, epsilon=1e-12
        ).sum(),
        adv.detach(),
        step=1e-6,
    )
    fd_clean = central_difference(
        lambda value: scalar_secant_loss(adv.detach(), value, rho=rho, d_teacher=teacher_distance, epsilon=1e-12).sum(),
        clean.detach(),
        step=1e-6,
    )
    assert torch.allclose(grad_adv, fd_adv, atol=1e-6, rtol=1e-5)
    assert torch.allclose(grad_clean, fd_clean, atol=1e-6, rtol=1e-5)


def test_secant_student_q_retains_graph_and_teacher_pair_gate_is_not_teacher_argmax() -> None:
    # Student rival is class 1.  Teacher is globally wrong to class 2 but is
    # still pair-positive relative to that Student-selected rival.
    student_logits = torch.tensor([[0.2, 0.8, 0.7]], requires_grad=True)
    teacher_logits = torch.tensor([[0.7, 0.4, 0.9]])
    labels = torch.tensor([0])
    student_adv, rival = dynamic_pair_margin(student_logits, labels)
    teacher_adv, _ = dynamic_pair_margin(teacher_logits, labels, rival)
    assert rival.tolist() == [1]
    assert teacher_logits.argmax(dim=1).tolist() == [2]
    assert teacher_adv.item() > 0.0
    values = secant_components(
        student_adv_margin=student_adv,
        student_clean_margin=torch.tensor([0.1], requires_grad=True),
        teacher_adv_margin=teacher_adv,
        teacher_clean_margin=torch.tensor([0.5]),
        rho=torch.tensor([0.03]),
        selected=torch.ones(1),
        epsilon=1e-12,
    )
    assert values["teacher_pair_gate"].item() == 1.0
    gradient = torch.autograd.grad(values["raw_loss"].sum(), student_logits, allow_unused=False)[0]
    assert torch.isfinite(gradient).all()


def test_replay_allows_orchestrator_metadata_but_not_scientific_overwrite(tmp_path: Path) -> None:
    # Exercise the early output guard without loading a checkpoint: an
    # orchestrator-side log directory is execution metadata, while a prior
    # scientific result must fail closed.
    output = tmp_path / "output"
    (output / "orchestration").mkdir(parents=True)
    prepare_output_dir(output)
    assert (output / "orchestration").is_dir()
    with pytest.raises(FileNotFoundError):
        # Passing invalid typed values reaches the checkpoint loader only after
        # the output guard has accepted the metadata-only directory.
        replay_canonical_state(  # type: ignore[arg-type]
            config_path=Path("missing.yaml"),
            checkpoint=Path("missing.pt"),
            expected_checkpoint_sha256="0" * 64,
            expected_epoch=104,
            output_dir=output,
            device=torch.device("cpu"),
        )
    (output / "state-replay.json").write_text("{}", encoding="utf-8")
    with pytest.raises(LongitudinalStateError):
        replay_canonical_state(  # type: ignore[arg-type]
            config_path=Path("missing.yaml"),
            checkpoint=Path("missing.pt"),
            expected_checkpoint_sha256="0" * 64,
            expected_epoch=104,
            output_dir=output,
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize(
    "script_name",
    (
        "replay_ert_i100_s2_runtime_activity_proxy.py",
        "forensic_ert_i100_s2_secant_boundary_distance.py",
    ),
)
def test_forensic_cli_modules_import_before_gpu_launch(script_name: str) -> None:
    repo = Path(__file__).parents[2]
    environment = {**os.environ, "PYTHONPATH": str(repo / "src")}
    completed = subprocess.run(
        [sys.executable, str(repo / "scripts" / script_name), "--help"],
        cwd=repo,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
