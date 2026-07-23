from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from ard.campaign.gpu import GPUProcess, GPUSnapshot, admit, required_free_memory_mib
from ard.campaign.launcher import argv_digest, launch_phase, process_identity, read_exit_record
from ard.campaign.schema import (
    CampaignError,
    CampaignSpec,
    bind_git_sha,
    effective_wandb_run_id,
    require_aggregation_compatible,
)
from ard.campaign.state import CampaignStateStore, JobState, StateError
from ard.campaign.worker import CampaignWorker, _validate_evaluation_results, default_phase_success_validator


def _raw_campaign(*, autoattack: bool = True) -> dict[str, object]:
    return {
        "campaign_id": "campaign-v1",
        "git_sha": "a" * 40,
        "execution_profile": {"id": "ws1_prb128_gb128_localbn_v1"},
        "external_process_policy": "allow_with_memory_gate",
        "hosts": {"hamster": {"gpus": [0]}, "ferret": {"gpus": [0]}},
        "jobs": [
            {
                "id": "job-1",
                "host": "hamster",
                "gpu": 0,
                "teacher": "teacher",
                "method": "rslad",
                "seed": 0,
                "config": "configs/production/job.yaml",
                "output": "outputs/job-1",
                "wandb": {"entity": "entity", "project": "project", "group": "group", "run_id": "wandb-job-1"},
                "phases": {
                    "train": ["python", "-c", "pass"],
                    "pgd_evaluate": ["python", "-c", "pass"],
                    **({"autoattack": ["python", "-c", "pass"]} if autoattack else {}),
                },
                "pilot_peak_reserved_mib": 100,
            }
        ],
    }


def _snapshot(*, processes: tuple[GPUProcess, ...] = ()) -> GPUSnapshot:
    return GPUSnapshot(
        index=0,
        uuid="GPU-abc123",
        memory_free_mib=1000,
        memory_used_mib=0,
        memory_total_mib=1000,
        utilization_percent=0,
        temperature_c=30,
        processes=processes,
    )


def _evaluation_results(*, count: int = 10000) -> list[dict[str, object]]:
    return [
        {
            "checkpoint_alias": alias,
            "checkpoint_filename": f"{alias}.pt",
            "train_run_id": "train-run",
            "config_hash": "config-hash",
            "threat_hash": "threat-hash",
            "method": "rslad",
            "teacher_identity": {"registry_id": "teacher"},
            "count": count,
            "clean_accuracy": 0.8,
            "pgd_accuracy": 0.4,
            "autoattack": None,
        }
        for alias in ("best", "last")
    ]


@pytest.mark.unit
@pytest.mark.t1
def test_campaign_schema_rejects_identity_collisions_and_unsafe_paths() -> None:
    raw = _raw_campaign()
    duplicate = dict(raw["jobs"][0])  # type: ignore[index]
    duplicate["id"] = "job-2"
    duplicate["output"] = "outputs/job-2"
    duplicate["wandb"] = dict(duplicate["wandb"])  # type: ignore[arg-type]
    duplicate["wandb"]["run_id"] = "wandb-job-1"  # type: ignore[index]
    raw["jobs"] = [raw["jobs"][0], duplicate]  # type: ignore[index]
    with pytest.raises(ValidationError, match="W&B run IDs"):
        CampaignSpec.model_validate(raw)

    raw = _raw_campaign()
    raw["jobs"][0]["output"] = "../escape"  # type: ignore[index]
    with pytest.raises(ValidationError, match="safe relative"):
        CampaignSpec.model_validate(raw)


@pytest.mark.unit
@pytest.mark.t1
def test_campaign_sha_binding_and_execution_profile_aggregation_are_fail_closed() -> None:
    raw = _raw_campaign()
    raw["git_sha"] = None
    template = CampaignSpec.model_validate(raw)
    fixed = bind_git_sha(template, "b" * 40)
    with pytest.raises(CampaignError, match="drift"):
        bind_git_sha(fixed, "c" * 40)
    with pytest.raises(CampaignError, match="different execution profiles"):
        require_aggregation_compatible(
            [{"execution_profile": "ws1_prb128_gb128_localbn_v1"}, {"execution_profile": "ws2_prb64_gb128_localbn"}]
        )
    assert effective_wandb_run_id(fixed, fixed.jobs[0]) == "wandb-job-1-bbbbbbb"


@pytest.mark.unit
@pytest.mark.t1
def test_state_is_atomic_and_transitions_are_finite(tmp_path: Path) -> None:
    spec = CampaignSpec.model_validate(_raw_campaign(autoattack=False))
    store = CampaignStateStore(tmp_path / "state")
    store.initialize(spec)
    assert store.campaign()["state"] == "unarmed"
    assert store.job("job-1")["state"] == "pending"
    assert store.job("job-1")["identity"]["effective_wandb_run_id"] == "wandb-job-1-aaaaaaa"
    with pytest.raises(StateError, match="invalid job transition"):
        store.transition_job("job-1", JobState.COMPLETED)
    store.transition_job("job-1", JobState.PREFLIGHT)
    contents = json.loads((tmp_path / "state" / "jobs" / "job-1.json").read_text(encoding="utf-8"))
    assert contents["state"] == "preflight"
    assert list((tmp_path / "state" / "jobs").glob("*.tmp")) == []

    changed = _raw_campaign(autoattack=False)
    changed["jobs"][0]["phases"]["train"] = ["python", "-c", "changed"]  # type: ignore[index]
    with pytest.raises(StateError, match="identity"):
        store.initialize(CampaignSpec.model_validate(changed))


@pytest.mark.unit
@pytest.mark.t1
def test_memory_gate_allows_external_process_only_with_measured_headroom() -> None:
    external = _snapshot(processes=(GPUProcess(pid=10, memory_mib=20, name="other", user="someone"),))
    blocked = admit(
        external,
        external_process_policy="allow_with_memory_gate",
        pilot_peak_reserved_mib=100,
        campaign_claimed=False,
        reserved_by_current_run=False,
        external_processes_enabled=True,
    )
    assert blocked.allowed
    assert blocked.shared_gpu_at_launch
    assert blocked.required_free_memory_mib == 125
    assert required_free_memory_mib(100) == 125
    protected = admit(
        _snapshot(),
        external_process_policy="allow_with_memory_gate",
        pilot_peak_reserved_mib=100,
        campaign_claimed=False,
        reserved_by_current_run=True,
        external_processes_enabled=True,
    )
    assert protected.state == "waiting_gpu"


@pytest.mark.unit
@pytest.mark.t1
def test_unarmed_worker_does_not_launch_then_reconciles_train_pgd_and_autoattack(tmp_path: Path) -> None:
    spec = CampaignSpec.model_validate(_raw_campaign())
    state = CampaignStateStore(tmp_path / "state")
    launches: list[str] = []
    launch_environments: list[dict[str, str]] = []

    def fake_launcher(argv: tuple[str, ...], **kwargs: object) -> dict[str, object]:
        phase = Path(kwargs["exit_record"]).parent.name  # type: ignore[index,arg-type]
        launches.append(phase)
        launch_environments.append(kwargs["environment"])  # type: ignore[index,arg-type]
        exit_record = Path(kwargs["exit_record"])  # type: ignore[index,arg-type]
        exit_record.parent.mkdir(parents=True, exist_ok=True)
        exit_record.write_text(
            json.dumps(
                {
                    "exit_code": 0,
                    "run_id": "job-1",
                    "git_sha": "a" * 40,
                    "phase_argv_digest": argv_digest(argv),
                }
            ),
            encoding="utf-8",
        )
        return {
            "wrapper": {"pid": 999999, "start_time_ticks": 1, "cwd": str(tmp_path), "argv_digest": "not-live"},
            "phase_argv_digest": argv_digest(argv),
            "exit_record": str(exit_record),
        }

    worker = CampaignWorker(
        spec,
        state,
        host="hamster",
        repository=tmp_path,
        output_root=tmp_path,
        inventory_provider=lambda: (_snapshot(),),
        launcher=fake_launcher,
        phase_success_validator=lambda _job, _phase, _output: None,
        external_processes_enabled=True,
    )
    assert worker.run_once()["job-1"] == "pending"
    assert launches == []
    worker.arm()
    assert worker.run_once()["job-1"] == "training"
    assert worker.run_once()["job-1"] == "training_completed"
    assert worker.run_once()["job-1"] == "pgd_evaluation"
    assert worker.run_once()["job-1"] == "pgd_completed"
    assert worker.run_once()["job-1"] == "autoattack"
    assert worker.run_once()["job-1"] == "completed"
    assert launches == ["train", "pgd", "autoattack"]
    assert launch_environments[0] == {
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONPATH": str(tmp_path / "src"),
        "ARD_JOB_OUTPUT_DIR": str(tmp_path / "outputs" / "job-1"),
        "ARD_RUN_ID": "wandb-job-1-aaaaaaa",
        "ARD_SEED": "0",
        "WANDB_ENTITY": "entity",
        "WANDB_PROJECT": "project",
        "WANDB_GROUP": "group",
        "WANDB_GROUP_CHEN": "group",
        "WANDB_GROUP_BARTOLDSON": "group",
    }
    assert state.campaign()["state"] == "awaiting_scientific_review"


@pytest.mark.unit
@pytest.mark.t1
def test_exit_zero_without_terminal_artifacts_blocks_before_pgd(tmp_path: Path) -> None:
    spec = CampaignSpec.model_validate(_raw_campaign(autoattack=False))

    def fake_launcher(argv: tuple[str, ...], **kwargs: object) -> dict[str, object]:
        exit_record = Path(kwargs["exit_record"])  # type: ignore[index,arg-type]
        exit_record.parent.mkdir(parents=True, exist_ok=True)
        exit_record.write_text(
            json.dumps(
                {
                    "exit_code": 0,
                    "run_id": "job-1",
                    "git_sha": "a" * 40,
                    "phase_argv_digest": argv_digest(argv),
                }
            ),
            encoding="utf-8",
        )
        return {
            "wrapper": {"pid": 999999, "start_time_ticks": 1, "cwd": str(tmp_path), "argv_digest": "not-live"},
            "phase_argv_digest": argv_digest(argv),
            "exit_record": str(exit_record),
        }

    worker = CampaignWorker(
        spec,
        CampaignStateStore(tmp_path / "state"),
        host="hamster",
        repository=tmp_path,
        output_root=tmp_path,
        inventory_provider=lambda: (_snapshot(),),
        launcher=fake_launcher,
        external_processes_enabled=True,
    )
    worker.arm()
    assert worker.run_once()["job-1"] == "training"
    assert worker.run_once()["job-1"] == "blocked"


@pytest.mark.unit
@pytest.mark.t1
def test_phase_tokens_and_evaluation_terminal_contract_match_real_cli_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _raw_campaign(autoattack=False)
    raw["jobs"][0]["phases"] = {  # type: ignore[index]
        "train": ["{PYTHON}", "--config", "{CONFIG_PATH}", "--output", "{JOB_OUTPUT_DIR}"],
        "pgd_evaluate": ["{PYTHON}", "--output", "{JOB_OUTPUT_DIR}/evaluation-pgd"],
    }
    spec = CampaignSpec.model_validate(raw)
    worker = CampaignWorker(
        spec,
        CampaignStateStore(tmp_path / "state"),
        host="hamster",
        repository=tmp_path,
        output_root=tmp_path,
        inventory_provider=lambda: (_snapshot(),),
    )
    argv = worker._phase_argv(spec.jobs[0], "pgd")  # noqa: SLF001 - exact command contract
    assert argv[-1] == str(tmp_path / "outputs" / "job-1" / "evaluation-pgd")

    evaluation = tmp_path / "outputs" / "job-1" / "evaluation-pgd"
    training = tmp_path / "outputs" / "job-1"
    bundle = evaluation / "run-bundle"
    bundle.mkdir(parents=True)
    (training / "run-bundle").mkdir(exist_ok=True)
    (training / "resolved_config.yaml").write_text("resolved\n", encoding="utf-8")
    (evaluation / "resolved_evaluation_config.yaml").write_text("resolved\n", encoding="utf-8")
    (training / "run-bundle" / "manifest.json").write_text(
        json.dumps({"run_id": "train-run", "config_hash": "config-hash"}), encoding="utf-8"
    )
    selection_attack = SimpleNamespace(identity_sha256=lambda: "threat-hash", epsilon_value=8 / 255)
    training_config = SimpleNamespace(method=SimpleNamespace(selection_attack=selection_attack))
    evaluation_config = SimpleNamespace(
        evaluation=SimpleNamespace(autoattack=False, seed=0, autoattack_batch_size=128)
    )

    def fake_load_config(path: Path) -> object:
        return evaluation_config if path.name == "resolved_evaluation_config.yaml" else training_config

    monkeypatch.setattr("ard.config.load_config", fake_load_config)
    monkeypatch.setattr("ard.config.loader.resolved_config_dict", lambda _config: {})
    monkeypatch.setattr("ard.engine.config_digest", lambda _config: "config-hash")
    (evaluation / "evaluation-results.json").write_text(json.dumps(_evaluation_results()), encoding="utf-8")
    (bundle / "manifest.json").write_text(
        json.dumps({"status": "completed", "tracking_mode": "online"}), encoding="utf-8"
    )
    (bundle / "completion.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    (bundle / "error-marker.txt").write_text("no application error recorded\n", encoding="utf-8")
    assert default_phase_success_validator(spec.jobs[0], "pgd", tmp_path) is None
    (evaluation / "evaluation-results.json").write_text(json.dumps(_evaluation_results(count=9999)), encoding="utf-8")
    assert "10000" in str(default_phase_success_validator(spec.jobs[0], "pgd", tmp_path))
    (evaluation / "evaluation-results.json").write_text(json.dumps(_evaluation_results()), encoding="utf-8")
    (bundle / "manifest.json").write_text(
        json.dumps({"status": "sync_pending", "sync_state": "sync_pending", "tracking_mode": "offline_sync"}),
        encoding="utf-8",
    )
    assert "not completed" in str(default_phase_success_validator(spec.jobs[0], "pgd", tmp_path))
    assert "AutoAttack" in str(
        _validate_evaluation_results(
            evaluation / "evaluation-results.json",
            training_output=training,
            phase="autoattack",
            job=spec.jobs[0],
        )
    )


@pytest.mark.unit
@pytest.mark.t1
def test_host_worker_stops_for_review_without_waiting_for_remote_state(tmp_path: Path) -> None:
    raw = _raw_campaign(autoattack=False)
    remote = dict(raw["jobs"][0])  # type: ignore[index]
    remote.update({"id": "job-remote", "host": "ferret", "output": "outputs/job-remote"})
    remote["wandb"] = {"entity": "entity", "project": "project", "group": "group", "run_id": "wandb-remote"}
    raw["jobs"] = [raw["jobs"][0], remote]  # type: ignore[index]
    spec = CampaignSpec.model_validate(raw)
    state = CampaignStateStore(tmp_path / "state")
    worker = CampaignWorker(
        spec,
        state,
        host="hamster",
        repository=tmp_path,
        output_root=tmp_path,
        inventory_provider=lambda: (_snapshot(),),
        phase_success_validator=lambda _job, _phase, _output: None,
    )
    worker.arm()
    for target in (JobState.PREFLIGHT, JobState.LAUNCHING, JobState.TRAINING, JobState.TRAINING_COMPLETED):
        state.transition_job("job-1", target)
    for target in (JobState.LAUNCHING, JobState.PGD_EVALUATION, JobState.PGD_COMPLETED, JobState.COMPLETED):
        state.transition_job("job-1", target)
    worker.run_once()
    assert state.campaign()["state"] == "awaiting_scientific_review"
    assert state.job("job-remote")["state"] == "pending"


@pytest.mark.unit
@pytest.mark.t1
def test_detached_argv_phase_writes_an_exit_record(tmp_path: Path) -> None:
    exit_record = tmp_path / "control" / "exit.json"
    lease = tmp_path / "host-locks" / "gpu-GPU-abc123.lock"
    handshake = tmp_path / "control" / "lease-handshake.json"
    launch_phase(
        [sys.executable, "-c", "pass"],
        cwd=tmp_path,
        stdout_path=tmp_path / "control" / "stdout.log",
        stderr_path=tmp_path / "control" / "stderr.log",
        exit_record=exit_record,
        gpu_lease_path=lease,
        lease_handshake=handshake,
        run_id="job-1",
        git_sha="a" * 40,
    )
    for _ in range(100):
        result = read_exit_record(exit_record)
        if result is not None:
            break
        time.sleep(0.01)
    assert result["exit_code"] == 0
    assert result["phase_argv_digest"] == argv_digest([sys.executable, "-c", "pass"])
    assert json.loads(handshake.read_text(encoding="utf-8"))["run_id"] == "job-1"


@pytest.mark.unit
@pytest.mark.t1
def test_live_phase_adoption_evidence_is_bounded_to_once_per_phase(tmp_path: Path) -> None:
    spec = CampaignSpec.model_validate(_raw_campaign(autoattack=False))
    state = CampaignStateStore(tmp_path / "state")
    worker = CampaignWorker(
        spec,
        state,
        host="hamster",
        repository=tmp_path,
        output_root=tmp_path,
        inventory_provider=lambda: (_snapshot(),),
        launcher=lambda *_args, **_kwargs: {
            "wrapper": {},
            "phase_argv_digest": "digest",
            "exit_record": str(tmp_path / "missing"),
        },
    )
    worker.arm()
    for target in (JobState.PREFLIGHT, JobState.LAUNCHING, JobState.TRAINING):
        state.transition_job("job-1", target)
    state.transition_job(
        "job-1",
        JobState.TRAINING,
        phase={"name": "train", "phase_argv_digest": "digest", "wrapper": {}},
    )
    # Exercise the bounded-evidence branch without requiring a real long-lived
    # process; process matching itself has a separate detached test.
    import ard.campaign.worker as worker_module

    original = worker_module.phase_is_live
    worker_module.phase_is_live = lambda _phase: True
    try:
        worker.run_once()
        worker.run_once()
    finally:
        worker_module.phase_is_live = original
    evidence = state.job("job-1").get("evidence", [])
    assert [item["kind"] for item in evidence] == ["process_adopted"]


@pytest.mark.unit
@pytest.mark.t1
def test_host_global_gpu_lease_blocks_a_second_state_root(tmp_path: Path) -> None:
    spec = CampaignSpec.model_validate(_raw_campaign(autoattack=False))
    lock_root = tmp_path / "host-locks"
    launches: list[str] = []

    def first_launcher(argv: tuple[str, ...], **kwargs: object) -> dict[str, object]:
        launches.append("first")
        return {
            "wrapper": {"pid": 999999, "start_time_ticks": 1, "cwd": str(tmp_path), "argv_digest": "not-live"},
            "phase_argv_digest": argv_digest(argv),
            "run_id": "job-1",
            "git_sha": "a" * 40,
            "exit_record": str(kwargs["exit_record"]),
            "launch_record": str(kwargs["launch_record"]),
        }

    first = CampaignWorker(
        spec,
        CampaignStateStore(tmp_path / "state-a"),
        host="hamster",
        repository=tmp_path,
        output_root=tmp_path,
        inventory_provider=lambda: (_snapshot(),),
        launcher=first_launcher,
        phase_success_validator=lambda _job, _phase, _output: None,
        gpu_lock_root=lock_root,
    )
    first.arm()
    assert first.run_once()["job-1"] == "training"

    second = CampaignWorker(
        spec,
        CampaignStateStore(tmp_path / "state-b"),
        host="hamster",
        repository=tmp_path,
        output_root=tmp_path,
        inventory_provider=lambda: (_snapshot(),),
        launcher=lambda *_args, **_kwargs: pytest.fail("second root must not launch"),
        gpu_lock_root=lock_root,
    )
    second.arm()
    assert second.run_once()["job-1"] == "waiting_gpu"
    assert launches == ["first"]


@pytest.mark.unit
@pytest.mark.t1
def test_launch_record_adopts_after_return_crash_without_duplicate_launch(tmp_path: Path) -> None:
    spec = CampaignSpec.model_validate(_raw_campaign(autoattack=False))
    lock_root = tmp_path / "host-locks"
    sleeper: subprocess.Popen[bytes] | None = None

    def crash_after_popen(argv: tuple[str, ...], **kwargs: object) -> dict[str, object]:
        nonlocal sleeper
        sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], cwd=tmp_path)
        wrapper_argv = [sys.executable, "-c", "import time; time.sleep(30)"]
        record = {
            "wrapper": process_identity(sleeper.pid, cwd=tmp_path, argv=wrapper_argv),
            "phase_argv_digest": argv_digest(argv),
            "run_id": "job-1",
            "git_sha": "a" * 40,
            "exit_record": str(kwargs["exit_record"]),
            "launch_record": str(kwargs["launch_record"]),
        }
        Path(kwargs["launch_record"]).parent.mkdir(parents=True, exist_ok=True)  # type: ignore[index,arg-type]
        Path(kwargs["launch_record"]).write_text(json.dumps(record), encoding="utf-8")  # type: ignore[index,arg-type]
        raise RuntimeError("controller crashed after Popen")

    state = CampaignStateStore(tmp_path / "state")
    first = CampaignWorker(
        spec,
        state,
        host="hamster",
        repository=tmp_path,
        output_root=tmp_path,
        inventory_provider=lambda: (_snapshot(),),
        launcher=crash_after_popen,
        gpu_lock_root=lock_root,
    )
    first.arm()
    assert first.run_once()["job-1"] == "launching"
    replacement = CampaignWorker(
        spec,
        state,
        host="hamster",
        repository=tmp_path,
        output_root=tmp_path,
        inventory_provider=lambda: (_snapshot(),),
        launcher=lambda *_args, **_kwargs: pytest.fail("launch record must be adopted, not relaunched"),
        gpu_lock_root=lock_root,
    )
    try:
        assert replacement.run_once()["job-1"] == "training"
    finally:
        assert sleeper is not None
        sleeper.terminate()
        sleeper.wait(timeout=5)


@pytest.mark.unit
@pytest.mark.t1
def test_reservation_release_marker_is_strict(tmp_path: Path) -> None:
    marker = tmp_path / "current-run-release.json"
    raw = _raw_campaign(autoattack=False)
    raw["reservations"] = [
        {
            "host": "hamster",
            "gpu": 0,
            "run_id": "protected-run",
            "execution_profile": "ws2_prb64_gb128_localbn",
            "protected_git_sha": "b" * 40,
            "release_marker": str(marker),
        }
    ]
    spec = CampaignSpec.model_validate(raw)
    worker = CampaignWorker(
        spec,
        CampaignStateStore(tmp_path / "state"),
        host="hamster",
        repository=tmp_path,
        output_root=tmp_path,
        inventory_provider=lambda: (_snapshot(),),
        gpu_lock_root=tmp_path / "locks",
    )
    assert worker._reserved(spec.jobs[0])  # noqa: SLF001
    marker.write_text(json.dumps({"status": "failed", "run_id": "protected-run"}), encoding="utf-8")
    assert worker._reserved(spec.jobs[0])  # noqa: SLF001
    marker.write_text(json.dumps({"status": "completed", "run_id": "other"}), encoding="utf-8")
    assert worker._reserved(spec.jobs[0])  # noqa: SLF001
    marker.write_text(json.dumps({"status": "completed", "run_id": "protected-run"}), encoding="utf-8")
    assert worker._reserved(spec.jobs[0])  # noqa: SLF001
    marker.write_text(
        json.dumps(
            {
                "status": "completed",
                "run_id": "protected-run",
                "training_git_sha": "b" * 40,
                "execution_profile": "ws2_prb64_gb128_localbn",
                "training_sync": "completed",
                "saved_checkpoint_pgd": "completed",
            }
        ),
        encoding="utf-8",
    )
    assert not worker._reserved(spec.jobs[0])  # noqa: SLF001


@pytest.mark.unit
@pytest.mark.t1
def test_waiting_reserved_gpu_does_not_starve_a_free_gpu(tmp_path: Path) -> None:
    raw = _raw_campaign(autoattack=False)
    raw["hosts"]["hamster"] = {"gpus": [0, 1]}  # type: ignore[index]
    first = raw["jobs"][0]  # type: ignore[index]
    second = dict(first)
    second.update({"id": "job-2", "gpu": 1, "output": "outputs/job-2"})
    second["wandb"] = {"entity": "entity", "project": "project", "group": "group", "run_id": "wandb-job-2"}
    raw["jobs"] = [first, second]
    raw["reservations"] = [
        {"host": "hamster", "gpu": 0, "run_id": "protected", "execution_profile": "ws2", "active": True}
    ]
    spec = CampaignSpec.model_validate(raw)
    launched: list[str] = []

    def launcher(argv: tuple[str, ...], **kwargs: object) -> dict[str, object]:
        launched.append(str(kwargs["run_id"]))
        return {
            "wrapper": {"pid": 999999, "start_time_ticks": 1, "cwd": str(tmp_path), "argv_digest": "not-live"},
            "phase_argv_digest": argv_digest(argv),
            "run_id": str(kwargs["run_id"]),
            "git_sha": "a" * 40,
            "exit_record": str(kwargs["exit_record"]),
            "launch_record": str(kwargs["launch_record"]),
        }

    gpu_one = GPUSnapshot(
        index=1,
        uuid="GPU-def456",
        memory_free_mib=1000,
        memory_used_mib=0,
        memory_total_mib=1000,
        utilization_percent=0,
        temperature_c=30,
        processes=(),
    )
    worker = CampaignWorker(
        spec,
        CampaignStateStore(tmp_path / "state"),
        host="hamster",
        repository=tmp_path,
        output_root=tmp_path,
        inventory_provider=lambda: (_snapshot(), gpu_one),
        launcher=launcher,
        gpu_lock_root=tmp_path / "locks",
    )
    worker.arm()
    assert worker.run_once() == {"job-1": "waiting_gpu", "job-2": "training"}
    assert launched == ["job-2"]


@pytest.mark.unit
@pytest.mark.t1
def test_pgd_precedes_next_train_on_the_same_gpu(tmp_path: Path) -> None:
    raw = _raw_campaign(autoattack=False)
    first = raw["jobs"][0]  # type: ignore[index]
    first["phases"] = {"train": ["train-1"], "pgd_evaluate": ["pgd-1"]}
    second = dict(first)
    second.update({"id": "job-2", "output": "outputs/job-2"})
    second["wandb"] = {"entity": "entity", "project": "project", "group": "group", "run_id": "wandb-job-2"}
    second["phases"] = {"train": ["train-2"], "pgd_evaluate": ["pgd-2"]}
    raw["jobs"] = [first, second]
    spec = CampaignSpec.model_validate(raw)
    launched: list[str] = []

    def launcher(argv: tuple[str, ...], **kwargs: object) -> dict[str, object]:
        launched.append(argv[0])
        return {
            "wrapper": {"pid": 999999, "start_time_ticks": 1, "cwd": str(tmp_path), "argv_digest": "not-live"},
            "phase_argv_digest": argv_digest(argv),
            "run_id": "job-1",
            "git_sha": "a" * 40,
            "exit_record": str(kwargs["exit_record"]),
            "launch_record": str(kwargs["launch_record"]),
        }

    state = CampaignStateStore(tmp_path / "state")
    worker = CampaignWorker(
        spec,
        state,
        host="hamster",
        repository=tmp_path,
        output_root=tmp_path,
        inventory_provider=lambda: (_snapshot(),),
        launcher=launcher,
        gpu_lock_root=tmp_path / "locks",
    )
    worker.arm()
    for target in (JobState.PREFLIGHT, JobState.LAUNCHING, JobState.TRAINING, JobState.TRAINING_COMPLETED):
        state.transition_job("job-1", target)
    assert worker.run_once()["job-1"] == "pgd_evaluation"
    assert launched == ["pgd-1"]
