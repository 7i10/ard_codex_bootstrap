from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / ".agents/skills/multi-gpu-experiment-orchestrator/scripts/orchestrate.py"
SHA = "a" * 40


def write_manifest(tmp_path: Path, jobs: list[dict], *, hosts: dict | None = None) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "campaign_id": "dummy-campaign",
        "source": {"git_sha": SHA},
        "state_path": str(tmp_path / "state.json"),
        "hosts": hosts
        or {"local": {"backend": "local", "gpus": [{"index": 0, "throughput": 10}, {"index": 1, "throughput": 1}]}},
        "jobs": jobs,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def job(
    tmp_path: Path,
    job_id: str,
    code: str,
    *,
    dependencies: list[str] | None = None,
    estimated_work: float = 1,
    gpu: int | None = None,
    retries: int = 1,
) -> dict:
    value = {
        "job_id": job_id,
        "run_id": job_id,
        "command": [sys.executable, "-c", code],
        "output_dir": str(tmp_path / "outputs" / job_id),
        "dependencies": dependencies or [],
        "estimated_work": estimated_work,
        "scientific_identity": {"method_id": job_id, "seed_bundle": "dummy"},
        "retry_policy": {"max_attempts": retries},
    }
    if gpu is not None:
        value["gpu"] = gpu
    return value


def invoke(command: str, manifest: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), command, "--manifest", str(manifest), *extra], text=True, capture_output=True
    )


def test_validate_and_dry_plan_are_read_only(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, [job(tmp_path, "root", "pass")])
    assert invoke("validate", manifest).returncode == 0
    planned = invoke("plan", manifest, "--dry-run")
    assert planned.returncode == 0
    assert json.loads(planned.stdout)["jobs"][0]["host"] == "local"
    assert not (tmp_path / "state.json").exists()


def test_cycle_and_missing_dependency_fail_closed(tmp_path: Path) -> None:
    missing = write_manifest(tmp_path / "missing", [job(tmp_path / "missing", "a", "pass", dependencies=["nope"])])
    result = invoke("validate", missing)
    assert result.returncode != 0
    assert "missing dependencies" in result.stderr
    cycletmp = tmp_path / "cycle"
    cycletmp.mkdir()
    cyclic = write_manifest(
        cycletmp, [job(cycletmp, "a", "pass", dependencies=["b"]), job(cycletmp, "b", "pass", dependencies=["a"])]
    )
    result = invoke("validate", cyclic)
    assert result.returncode != 0
    assert "cycle" in result.stderr


def test_success_dag_chains_and_is_idempotent(tmp_path: Path) -> None:
    jobs = [
        job(tmp_path, "root-a", "from pathlib import Path; Path('root-a.txt').write_text('ok')", estimated_work=10),
        job(tmp_path, "root-b", "from pathlib import Path; Path('root-b.txt').write_text('ok')", estimated_work=9),
        job(
            tmp_path, "prefix", "from pathlib import Path; Path('prefix.txt').write_text('ok')", dependencies=["root-b"]
        ),
        job(
            tmp_path,
            "control",
            "from pathlib import Path; Path('control.txt').write_text('ok')",
            dependencies=["prefix"],
        ),
        job(
            tmp_path,
            "treatment",
            "from pathlib import Path; Path('treatment.txt').write_text('ok')",
            dependencies=["prefix"],
        ),
        job(
            tmp_path,
            "endpoint",
            "from pathlib import Path; Path('endpoint.txt').write_text('ok')",
            dependencies=["treatment"],
        ),
        job(
            tmp_path,
            "aggregate",
            "from pathlib import Path; Path('aggregate.txt').write_text('ok')",
            dependencies=["control", "endpoint"],
        ),
        job(
            tmp_path,
            "report",
            "from pathlib import Path; Path('report.txt').write_text('ok')",
            dependencies=["aggregate"],
            gpu=1,
        ),
    ]
    manifest = write_manifest(tmp_path, jobs)
    result = invoke("run", manifest, "--foreground", "--poll-interval", "0.02")
    assert result.returncode == 0, result.stderr
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["status"] == "completed"
    assert all(record["status"] == "completed" for record in state["jobs"].values())
    assert any(e["event_type"] == "stable_confirmed" for e in state["events"])
    attempts_before = {key: len(value["attempts"]) for key, value in state["jobs"].items()}
    result = invoke("run", manifest, "--foreground", "--poll-interval", "0.02")
    assert result.returncode == 0
    state_after = json.loads((tmp_path / "state.json").read_text())
    assert {key: len(value["attempts"]) for key, value in state_after["jobs"].items()} == attempts_before


def test_valid_marker_recovers_without_relaunch(tmp_path: Path) -> None:
    manifest = write_manifest(
        tmp_path, [job(tmp_path, "root", "from pathlib import Path; Path('ran.txt').write_text('ok')")]
    )
    assert invoke("run", manifest, "--foreground", "--poll-interval", "0.02").returncode == 0
    (tmp_path / "state.json").unlink()
    assert invoke("run", manifest, "--foreground", "--poll-interval", "0.02").returncode == 0
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["jobs"]["root"]["status"] == "completed"
    assert state["jobs"]["root"]["attempts"] == []
    assert any(event["event_type"] == "completion_marker_recovered" for event in state["events"])


def test_legacy_output_marker_recovers_after_sidecar_upgrade(tmp_path: Path) -> None:
    value = job(tmp_path, "root", "from pathlib import Path; Path('ran.txt').write_text('ok')")
    manifest = write_manifest(tmp_path, [value])
    assert invoke("run", manifest, "--foreground", "--poll-interval", "0.02").returncode == 0

    sidecar_marker = next((tmp_path / "orchestration").rglob("completion.json"))
    legacy_marker = Path(value["output_dir"]) / "completion.json"
    legacy_marker.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sidecar_marker, legacy_marker)
    shutil.rmtree(tmp_path / "orchestration")
    (tmp_path / "state.json").unlink()

    assert invoke("run", manifest, "--foreground", "--poll-interval", "0.02").returncode == 0
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["jobs"]["root"]["status"] == "completed"
    assert state["jobs"]["root"]["attempts"] == []


def test_preflight_checks_job_paths_and_environment(tmp_path: Path) -> None:
    value = job(tmp_path, "root", "pass")
    value["required_paths"] = [str(tmp_path / "missing")]
    value["required_env"] = ["MISSING_ORCHESTRATOR_TEST_ENV"]
    manifest = write_manifest(tmp_path, [value])
    result = invoke("preflight", manifest)
    assert result.returncode == 2
    assert "required path missing" in result.stdout
    assert "environment variable missing" in result.stdout


def test_validate_rejects_direct_non_executable_shell_wrapper(tmp_path: Path) -> None:
    wrapper = tmp_path / "launcher.sh"
    wrapper.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    wrapper.chmod(0o644)
    value = job(tmp_path, "root", "pass")
    value["command"] = [str(wrapper)]
    manifest = write_manifest(tmp_path, [value])
    rejected = invoke("validate", manifest)
    assert rejected.returncode != 0
    assert "non-executable shell wrapper" in rejected.stderr

    value["command"] = ["bash", str(wrapper)]
    manifest = write_manifest(tmp_path, [value])
    assert invoke("validate", manifest).returncode == 0


def test_external_probe_requires_host_confirmation_before_completion(tmp_path: Path) -> None:
    remote_command = ["remote-python", "train.py", "--epochs", "3"]
    confirmation = (
        "import json,os; print(json.dumps({'schema_version':1,'status':'running','process_present':True,"
        "'campaign_id':os.environ['ARD_ORCH_CAMPAIGN_ID'],'job_id':os.environ['ARD_ORCH_JOB_ID'],"
        "'identity_hash':os.environ['ARD_ORCH_IDENTITY_HASH'],'source_sha':os.environ['ARD_ORCH_SOURCE_SHA'],"
        "'host':os.environ['ARD_ORCH_HOST'],'gpu_index':int(os.environ['ARD_ORCH_GPU_INDEX']),"
        "'gpu_uuid':os.environ['ARD_ORCH_GPU_UUID'],'pid':123,'command_argv':"
        + repr(remote_command)
        + ",'remote_manifest':'/remote/run/manifest.json'}))"
    )
    value = job(tmp_path, "remote", "pass")
    value.update(
        {
            "host": "remote",
            "command": [sys.executable, "-c", "pass"],
            "executor": {"type": "external_probe"},
            "completion_probe": [sys.executable, "-c", "pass"],
            "host_confirm_probe": [sys.executable, "-c", confirmation],
            "host_confirm_timeout_seconds": 1,
            "host_confirm_interval_seconds": 0.01,
            "remote_command": remote_command,
        }
    )
    hosts = {"remote": {"backend": "external", "gpus": [{"index": 0, "uuid": "GPU-remote", "throughput": 10}]}}
    manifest = write_manifest(tmp_path, [value], hosts=hosts)
    result = invoke("run", manifest, "--foreground", "--poll-interval", "0.02")
    assert result.returncode == 0, result.stderr
    state = json.loads((tmp_path / "state.json").read_text())
    events = [event["event_type"] for event in state["events"]]
    assert "controller_spawned" in events
    assert "host_confirmed_started" in events
    assert "stable_confirmed" not in events
    attempt = state["jobs"]["remote"]["attempts"][0]
    assert attempt["host_confirmation"]["remote_manifest"] == "/remote/run/manifest.json"


def test_technical_retry_preserves_identity_and_unblocks_endpoint(tmp_path: Path) -> None:
    output = tmp_path / "outputs" / "flaky"
    code = (
        "import json, os; from pathlib import Path; out=Path(os.environ['TEST_OUT']); "
        "\nif os.environ['ARD_ORCH_ATTEMPT']=='1':\n"
        " out.mkdir(parents=True, exist_ok=True); "
        "(out/'technical-failure.json').write_text(json.dumps({'failure_class': 'technical', 'retryable': True})); "
        "raise SystemExit(7)\n"
        "else:\n out.mkdir(parents=True, exist_ok=True); (out/'ok.txt').write_text('ok')"
    )
    flaky = job(tmp_path, "flaky", code, retries=2)
    flaky["env"] = {"TEST_OUT": str(output)}
    endpoint = job(
        tmp_path, "endpoint", "from pathlib import Path; Path('endpoint.txt').write_text('ok')", dependencies=["flaky"]
    )
    manifest = write_manifest(tmp_path, [flaky, endpoint])
    result = invoke("run", manifest, "--foreground", "--poll-interval", "0.02")
    assert result.returncode == 0, result.stderr
    state = json.loads((tmp_path / "state.json").read_text())
    flaky_state = state["jobs"]["flaky"]
    assert flaky_state["status"] == "completed"
    assert len(flaky_state["attempts"]) == 2
    assert flaky_state["attempts"][0]["status"] == "technical_failed"
    assert flaky_state["attempts"][1]["status"] == "completed"
    assert flaky_state["attempts"][0]["identity_hash"] == flaky_state["attempts"][1]["identity_hash"]
    assert any(e["event_type"] == "technical_retry" for e in state["events"])
    assert state["jobs"]["endpoint"]["status"] == "completed"


def test_stale_result_from_prior_campaign_cannot_release_gpu_slot(tmp_path: Path) -> None:
    value = job(tmp_path, "root", "import time; time.sleep(0.08)", gpu=0)
    manifest = write_manifest(tmp_path, [value])
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    job_key = hashlib.sha256(
        (json.dumps({"job_id": "root"}, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    stale = (
        tmp_path
        / "orchestration"
        / "dummy-campaign"
        / manifest_sha
        / job_key
        / "root.attempt-1.result.json"
    )
    stale.parent.mkdir(parents=True)
    stale.write_text(
        json.dumps(
            {
                "campaign_id": "prior-campaign",
                "job_id": "root",
                "attempt": 1,
                "attempt_id": "root-attempt-1",
                "identity_hash": "stale",
                "exit_code": 1,
                "status": "failed",
            }
        ),
        encoding="utf-8",
    )

    result = invoke("run", manifest, "--foreground", "--poll-interval", "0.01")
    assert result.returncode == 0, result.stderr
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["jobs"]["root"]["status"] == "completed"
    assert [event["event_type"] for event in state["events"]].count("stale_result_ignored") == 1


def test_controller_never_precreates_or_pollutes_scientific_output(tmp_path: Path) -> None:
    output = tmp_path / "outputs" / "strict-public-cli"
    code = (
        "from pathlib import Path; import sys; output=Path(" + repr(str(output)) + "); "
        "output.exists() and sys.exit('output must be absent before public CLI starts'); "
        "output.mkdir(parents=True); (output/'science.json').write_text('ok')"
    )
    value = job(tmp_path, "strict-public-cli", code)
    manifest = write_manifest(tmp_path, [value])

    result = invoke("run", manifest, "--foreground", "--poll-interval", "0.02")

    assert result.returncode == 0, result.stderr
    assert (output / "science.json").read_text() == "ok"
    assert not (output / "orchestration").exists()
    state = json.loads((tmp_path / "state.json").read_text())
    attempt = state["jobs"]["strict-public-cli"]["attempts"][0]
    assert str(output) not in attempt["log"]
    assert (tmp_path / "orchestration").is_dir()


def test_detached_controller_finishes_without_caller_lifetime(tmp_path: Path) -> None:
    manifest = write_manifest(
        tmp_path,
        [job(tmp_path, "root", "from pathlib import Path; Path('detached.txt').write_text('ok')")],
    )
    result = invoke("run", manifest)
    assert result.returncode == 0, result.stderr
    launch = json.loads(result.stdout)
    assert launch["controller_pid"] > 0
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state_path = tmp_path / "state.json"
        if state_path.exists() and json.loads(state_path.read_text())["status"] == "completed":
            break
        time.sleep(0.02)
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["status"] == "completed"


@pytest.mark.parametrize("gpu_case", ["same_sequential", "missing"])
def test_gpu_reservation_constraints_are_visible(tmp_path: Path, gpu_case: str) -> None:
    if gpu_case == "same_sequential":
        jobs = [job(tmp_path, "a", "pass", gpu=0), job(tmp_path, "b", "pass", gpu=0)]
        planned = json.loads(invoke("plan", write_manifest(tmp_path, jobs)).stdout)
        assert all(row.get("status") != "resource_conflict" for row in planned["jobs"])
        assert all(row["gpu"] == 0 for row in planned["jobs"])
    else:
        jobs = [job(tmp_path, "a", "pass", gpu=4)]
        result = invoke("validate", write_manifest(tmp_path, jobs))
        assert result.returncode == 0
        planned = json.loads(invoke("plan", write_manifest(tmp_path, jobs)).stdout)
        assert planned["jobs"][0]["status"] == "resource_conflict"
