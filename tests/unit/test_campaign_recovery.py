from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from ard.campaign.gpu import GPUSnapshot
from ard.campaign.schema import CampaignSpec
from ard.campaign.state import NVIDIA_SMI_ADMISSION_ERROR, CampaignStateStore, JobState, StateError


def _script(name: str) -> ModuleType:
    path = Path("scripts/campaign") / name
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _raw_campaign(
    *, two_jobs: bool = False, reservation: bool = False, marker: Path | None = None
) -> dict[str, object]:
    job = {
        "id": "job-1",
        "host": "hamster",
        "gpu": 0,
        "teacher": "teacher",
        "method": "rslad",
        "seed": 0,
        "config": "configs/job.yaml",
        "output": "outputs/job-1",
        "wandb": {"entity": "entity", "project": "project", "group": "group", "run_id": "job-1"},
        "phases": {"train": ["python", "-c", "pass"], "pgd_evaluate": ["python", "-c", "pass"]},
    }
    jobs: list[dict[str, object]] = [job]
    if two_jobs:
        jobs.append(
            {
                **job,
                "id": "job-2",
                "output": "outputs/job-2",
                "wandb": {"entity": "entity", "project": "project", "group": "group", "run_id": "job-2"},
            }
        )
    raw: dict[str, object] = {
        "campaign_id": "recovery-campaign",
        "git_sha": "a" * 40,
        "execution_profile": {"id": "ws1_prb128_gb128_localbn_v1"},
        "external_process_policy": "allow_with_memory_gate",
        "hosts": {"hamster": {"gpus": [0]}, "ferret": {"gpus": [0]}},
        "jobs": jobs,
    }
    if reservation:
        assert marker is not None
        raw["reservations"] = [
            {
                "host": "hamster",
                "gpu": 0,
                "run_id": "protected",
                "execution_profile": "ws2_prb64_gb128_localbn",
                "protected_git_sha": "b" * 40,
                "release_marker": str(marker),
            }
        ]
    return raw


def _blocked(store: CampaignStateStore, job_id: str, **updates: object) -> None:
    store.transition_job(job_id, JobState.PREFLIGHT)
    store.transition_job(
        job_id,
        JobState.BLOCKED,
        failure="GPU inventory unavailable",
        inventory_error=NVIDIA_SMI_ADMISSION_ERROR,
        **updates,
    )


@pytest.mark.unit
@pytest.mark.t1
def test_transient_recovery_refuses_atomically_when_any_target_has_launch_evidence(tmp_path: Path) -> None:
    store = CampaignStateStore(tmp_path / "state")
    store.initialize(CampaignSpec.model_validate(_raw_campaign(two_jobs=True)))
    _blocked(store, "job-1")
    _blocked(store, "job-2", phase={"name": "train"})
    before = {job_id: store.job(job_id) for job_id in ("job-1", "job-2")}

    with pytest.raises(StateError, match="phase or launch evidence"):
        store.recover_transient_gpu_blocks(["job-1", "job-2"])

    assert {job_id: store.job(job_id) for job_id in before} == before


@pytest.mark.unit
@pytest.mark.t1
def test_transient_recovery_preserves_failure_history_and_rearms_only_from_review(tmp_path: Path) -> None:
    store = CampaignStateStore(tmp_path / "state")
    store.initialize(CampaignSpec.model_validate(_raw_campaign()))
    _blocked(store, "job-1")
    recovered = store.recover_transient_gpu_blocks(["job-1"])["job-1"]

    assert recovered["state"] == "preflight"
    assert "failure" not in recovered and "inventory_error" not in recovered
    assert recovered["recovery_history"][-1]["inventory_error"] == NVIDIA_SMI_ADMISSION_ERROR
    with pytest.raises(StateError, match="awaiting_scientific_review"):
        store.rearm_after_transient_gpu_recovery()
    store.set_campaign_state("armed")
    store.set_campaign_state("awaiting_scientific_review")
    assert store.rearm_after_transient_gpu_recovery()["state"] == "armed"
    assert "transient_gpu_block_recovered" in (tmp_path / "state" / "events.jsonl").read_text(encoding="utf-8")


@pytest.mark.unit
@pytest.mark.t1
def test_recovery_cli_refuses_missing_marker_existing_output_and_live_controller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recovery = _script("recover_transient_block.py")
    repository = tmp_path / "repo"
    repository.mkdir()
    marker = tmp_path / "release.json"
    raw = _raw_campaign(reservation=True, marker=marker)
    campaign = repository / "campaign.yaml"
    campaign.write_text(yaml.safe_dump(raw), encoding="utf-8")
    state = CampaignStateStore(tmp_path / "run" / "state")
    state.initialize(CampaignSpec.model_validate(raw))
    _blocked(state, "job-1")
    state.set_campaign_state("armed")
    state.set_campaign_state("awaiting_scientific_review")
    args = argparse.Namespace(
        campaign=campaign,
        sha="a" * 40,
        repository=repository,
        state_root=state.root,
        output_root=tmp_path / "run",
        host="hamster",
        job=["job-1"],
        controller_record=None,
        failure_kind="gpu_inventory",
        apply=False,
    )
    monkeypatch.setattr(recovery, "_fixed_sha", lambda *_args: None)
    monkeypatch.setattr(
        recovery,
        "inventory",
        lambda: (GPUSnapshot(0, "GPU-test", 100, 0, 100, 0, 30, ()),),
    )
    with pytest.raises(recovery.RecoveryError, match="release marker"):
        recovery.validate(args)

    marker.write_text(
        json.dumps(
            {
                "status": "completed",
                "run_id": "protected",
                "training_git_sha": "b" * 40,
                "execution_profile": "ws2_prb64_gb128_localbn",
                "training_sync": "completed",
                "saved_checkpoint_pgd": "completed",
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "run" / "outputs" / "job-1"
    output.mkdir(parents=True)
    with pytest.raises(recovery.RecoveryError, match="output already exists"):
        recovery.validate(args)

    output.rmdir()
    controller = tmp_path / "run" / "control" / "controller.json"
    controller.parent.mkdir()
    controller.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(recovery, "_controller_live", lambda _record: True)
    with pytest.raises(recovery.RecoveryError, match="controller is live"):
        recovery.validate(args)


@pytest.mark.unit
@pytest.mark.t1
def test_watchdog_starts_no_duplicate_controller_after_one_successful_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    watchdog = _script("watch_controller.py")
    run_dir = tmp_path / "run"
    repository = tmp_path / "repo"
    (repository / "scripts" / "campaign").mkdir(parents=True)
    (repository / "scripts" / "campaign" / "manage.py").write_text("", encoding="utf-8")
    campaign = repository / "campaign.yaml"
    raw = _raw_campaign()
    campaign.write_text(yaml.safe_dump(raw), encoding="utf-8")
    spec = CampaignSpec.model_validate(raw)
    state = CampaignStateStore(run_dir / "state")
    state.initialize(spec)
    state.set_campaign_state("armed")
    (run_dir / "control").mkdir()
    monkeypatch.setattr(watchdog, "_fixed_campaign", lambda _args: (spec, repository, campaign))
    monkeypatch.setattr(watchdog, "_control_identity", lambda: (tmp_path / "control-repo", "c" * 40))
    monkeypatch.setattr(watchdog, "controller_live", lambda *_args, **_kwargs: False)
    starts: list[dict[str, object]] = []

    def fake_launch(_args: argparse.Namespace, **kwargs: object) -> dict[str, object]:
        starts.append(kwargs)
        state.set_campaign_state("awaiting_scientific_review")
        return {}

    monkeypatch.setattr(watchdog, "_launch_controller", fake_launch)
    monkeypatch.setattr(watchdog.time, "sleep", lambda _seconds: None)
    args = argparse.Namespace(
        run_dir=run_dir,
        campaign="campaign.yaml",
        host="hamster",
        gpu_lock_root=None,
        interval_seconds=1.0,
        max_backoff_seconds=4.0,
    )
    assert watchdog.watch(args) == 0
    assert len(starts) == 1
    assert starts[0]["scientific_repository"] == repository
    assert starts[0]["control_sha"] == "c" * 40


@pytest.mark.unit
@pytest.mark.t1
def test_missing_runtime_environment_failure_is_archived_before_exact_requeue(tmp_path: Path) -> None:
    store = CampaignStateStore(tmp_path / "state")
    store.initialize(CampaignSpec.model_validate(_raw_campaign()))
    phase_dir = store.root / "phases" / "job-1" / "train"
    phase_dir.mkdir(parents=True)
    launch = phase_dir / "launch.json"
    exit_record = phase_dir / "exit.json"
    launch.write_text("{}\n", encoding="utf-8")
    exit_record.write_text("{}\n", encoding="utf-8")
    (phase_dir / "stderr.log").write_text(
        "traceback\nValueError: missing environment variables: ARD_CIFAR10_ROOT\n",
        encoding="utf-8",
    )
    store.transition_job("job-1", JobState.PREFLIGHT)
    store.transition_job("job-1", JobState.LAUNCHING)
    store.transition_job(
        "job-1",
        JobState.TRAINING,
        phase={
            "name": "train",
            "exit_record": str(exit_record),
            "launch_record": str(launch),
        },
    )
    store.transition_job(
        "job-1",
        JobState.FAILED,
        failure="phase returned nonzero",
        phase_exit={"exit_code": 1, "error": None},
    )

    recovered = store.recover_missing_runtime_environment_failures(["job-1"])["job-1"]

    archive = store.root / "recovery-archive" / "job-1" / "train-missing-runtime-environment"
    assert recovered["state"] == "preflight"
    assert archive.is_dir() and not phase_dir.exists()
    assert recovered["recovery_history"][-1]["phase_archive"] == str(archive)
    assert "phase" not in recovered and "phase_exit" not in recovered


@pytest.mark.unit
@pytest.mark.t1
def test_watchdog_controller_environment_contains_fixed_runtime_inputs(tmp_path: Path) -> None:
    watchdog = _script("watch_controller.py")
    scientific = tmp_path / "scientific"
    control = tmp_path / "control"

    environment = watchdog._controller_environment(scientific, control)

    assert environment["PYTHONPATH"] == str(control / "src")
    assert environment["ARD_CIFAR10_ROOT"]
    assert environment["ARD_TEACHER_CHEN2021_LTD_WRN34_10_CHECKPOINT"].startswith(str(scientific))
    assert environment["ARD_TEACHER_BARTOLDSON2024_ADVERSARIAL_WRN94_16_CHECKPOINT"].startswith(str(scientific))
