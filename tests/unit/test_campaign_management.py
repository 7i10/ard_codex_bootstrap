from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _management_module() -> ModuleType:
    path = Path("scripts/campaign/manage.py").resolve()
    spec = importlib.util.spec_from_file_location("ard_campaign_management", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _acceptance_module() -> ModuleType:
    path = Path("scripts/campaign/accept_pilots.py").resolve()
    spec = importlib.util.spec_from_file_location("ard_campaign_acceptance", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _evidence() -> dict[str, object]:
    checks = {
        "finite_train_metrics": True,
        "best_last_pgd_10000": True,
        "terminal_lineage": True,
        "wandb_completed": True,
        "process_adoption": True,
        "execution_profile_match": True,
    }
    sha = "a" * 40

    def pilot(job_id: str, peak: int, *, joint: bool = False) -> dict[str, object]:
        pilot_checks = dict(checks)
        if joint:
            pilot_checks["joint_post_warmup_signal_active"] = True
        return {
            "job_id": job_id,
            "state": "completed",
            "output": f"pilot/{job_id}",
            "wandb_run_id": f"{job_id}-{sha[:7]}",
            "peak_reserved_mib": peak,
            "training_manifest_sha256": "1" * 64,
            "evaluation_results_sha256": "2" * 64,
            "job_state_sha256": "3" * 64,
            "training_config_hash": "4" * 64,
            "threat_hash": "5" * 64,
            "checks": pilot_checks,
        }

    return {
        "version": 1,
        "status": "accepted",
        "git_sha": sha,
        "execution_profile": "ws1_prb128_gb128_localbn_v1",
        "pilots": {
            "pilot-h-chen-rslad-s0": pilot("pilot-h-chen-rslad-s0", 2048),
            "pilot-h-chen-joint-s0": pilot("pilot-h-chen-joint-s0", 2304, joint=True),
            "pilot-f-bart-rslad-s0": pilot("pilot-f-bart-rslad-s0", 6144),
        },
    }


@pytest.mark.unit
@pytest.mark.t1
def test_production_pilot_acceptance_is_exact_sha_complete_and_finite(tmp_path: Path) -> None:
    management = _management_module()
    path = tmp_path / "pilot-acceptance.json"
    path.write_text(json.dumps(_evidence()), encoding="utf-8")
    assert management._validate_pilot_evidence(path, sha="a" * 40)["status"] == "accepted"

    stale = _evidence()
    stale["git_sha"] = "b" * 40
    path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(management.ManagementError, match="stale"):
        management._validate_pilot_evidence(path, sha="a" * 40)

    nonfinite = _evidence()
    nonfinite["pilots"]["pilot-f-bart-rslad-s0"]["peak_reserved_mib"] = float("nan")  # type: ignore[index]
    path.write_text(json.dumps(nonfinite), encoding="utf-8")
    with pytest.raises(management.ManagementError, match="incomplete"):
        management._validate_pilot_evidence(path, sha="a" * 40)

    failed_check = _evidence()
    failed_check["pilots"]["pilot-h-chen-joint-s0"]["checks"]["finite_train_metrics"] = False  # type: ignore[index]
    path.write_text(json.dumps(failed_check), encoding="utf-8")
    with pytest.raises(management.ManagementError, match="incomplete"):
        management._validate_pilot_evidence(path, sha="a" * 40)

    self_attested = _evidence()
    del self_attested["pilots"]["pilot-h-chen-rslad-s0"]["training_manifest_sha256"]  # type: ignore[index]
    path.write_text(json.dumps(self_attested), encoding="utf-8")
    with pytest.raises(management.ManagementError, match="incomplete"):
        management._validate_pilot_evidence(path, sha="a" * 40)

    wrong_run = _evidence()
    wrong_run["pilots"]["pilot-f-bart-rslad-s0"]["wandb_run_id"] = "arbitrary-run"  # type: ignore[index]
    path.write_text(json.dumps(wrong_run), encoding="utf-8")
    with pytest.raises(management.ManagementError, match="incomplete"):
        management._validate_pilot_evidence(path, sha="a" * 40)

    extra_check = _evidence()
    extra_check["pilots"]["pilot-h-chen-rslad-s0"]["checks"]["claimed_without_observation"] = True  # type: ignore[index]
    path.write_text(json.dumps(extra_check), encoding="utf-8")
    with pytest.raises(management.ManagementError, match="incomplete"):
        management._validate_pilot_evidence(path, sha="a" * 40)


@pytest.mark.unit
@pytest.mark.t1
def test_joint_pilot_acceptance_requires_uniform_mix_policy_and_active_risk(tmp_path: Path) -> None:
    acceptance = _acceptance_module()
    expected_policy = {
        "id": "teacher_target_uniform_mix",
        "version": 1,
        "risk_transform": "identity",
        "mixing": "uniform",
        "apply_to": "adversarial_student_kd",
        "rho_max": 0.5,
    }
    acceptance._validate_joint_target_policy(expected_policy)
    with pytest.raises(acceptance.PilotAcceptanceError, match="teacher_target_uniform_mix"):
        acceptance._validate_joint_target_policy(None)
    with pytest.raises(acceptance.PilotAcceptanceError, match="teacher_target_uniform_mix"):
        acceptance._validate_joint_target_policy({**expected_policy, "id": "wrong"})

    path = tmp_path / "joint-sample-stats.parquet"

    def write_signals(*, risks: list[float], weights: list[float]) -> None:
        pq.write_table(
            pa.table(
                {
                    "epoch": [0, *([1] * len(risks))],
                    "joint_risk": [0.0, *risks],
                    "kd_weight": [1.0, *weights],
                }
            ),
            path,
        )

    write_signals(risks=[0.0024, 0.135, 0.609], weights=[1.0, 1.0, 1.0])
    acceptance._validate_joint_signal(path, warmup_epochs=1)

    write_signals(risks=[0.0, 0.0, 0.0], weights=[1.0, 1.0, 1.0])
    with pytest.raises(acceptance.PilotAcceptanceError, match="positive and nonconstant"):
        acceptance._validate_joint_signal(path, warmup_epochs=1)

    write_signals(risks=[0.0024, 0.135, 0.609], weights=[1.0, 0.75, 1.0])
    with pytest.raises(acceptance.PilotAcceptanceError, match="KD weight 1.0"):
        acceptance._validate_joint_signal(path, warmup_epochs=1)
