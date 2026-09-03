from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / ".agents/skills/production-launch-gate/scripts/launch_gate.py"


def gate_module():
    spec = importlib.util.spec_from_file_location("launch_gate_fixture", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Regression provenance: historical launch failures are summarized in
# docs/ERT_RSLAD_UNSEEN_CONFIRMATION_ORCHESTRATION_AUDIT.md (source identifier
# recorded there as 48edebc).  Keep the mapping close to the executable tests.
REGRESSION_CASES = {
    "R1": "test_explicit_inclusive_runtime_bound_is_rejected",
    "R2": "test_parent_alias_mismatch_is_fail_closed",
    "R3": "test_dependency_output_is_deferred_but_producer_path_is_bound",
    "R4": "test_explicit_wandb_ids_must_not_collide",
    "R5": "test_host_logical_dataset_path_mismatch_is_rejected",
    "R6": "test_mask_metadata_hash_mismatch_is_rejected",
    "R7": "test_dependency_input_must_bind_to_producer_output",
    "R8": "test_output_collision_and_retry_scientific_mutation_fail",
    "R9": "test_source_config_drift_is_detected_after_freeze",
    "R10": "test_dependency_output_is_not_required_before_producer_completion",
    "R11": "test_technical_retry_keeps_identity_and_gets_new_execution_id",
    "R12": "test_validate_run_rejects_exit_zero_without_required_outputs",
    "R13": "test_remote_wrapper_metadata_rejects_non_executable_and_accepts_bash",
    "R14": "test_external_host_preflight_requires_every_declared_binding",
    "R15": "test_external_probe_requires_host_confirmation_before_completion",
    "R16": "test_remote_source_drift_is_rechecked_before_launch",
    "R17": "test_collection_stages_remote_metadata_to_canonical_local_path",
    "R18": "test_inventory_rejects_missing_required_endpoint_cell",
    "R19": "test_prepare_lock_serializes_concurrent_mutations",
    "R20": "test_remote_lifecycle_canary_requires_status_and_hash_roundtrip",
    "R21": "test_inventory_rejects_collected_hash_mismatch",
    "R22": "test_inventory_rejects_foreign_campaign_or_source_identity",
    "R23": (
        "tests/remote/test_ferret_scripts.py::"
        "test_remote_confirmation_uses_remote_manifest_identity_and_origin_not_local_reinjection"
    ),
    "R24": (
        "tests/skills/test_multi_gpu_orchestrator.py::"
        "test_host_confirmation_requires_observed_remote_origin_when_registered"
    ),
    "R25": (
        "tests/skills/test_artifact_inventory.py::"
        "test_collection_stages_remote_metadata_to_canonical_local_path"
    ),
    "R26": (
        "tests/skills/test_multi_gpu_orchestrator.py::"
        "test_launch_ledger_requires_full_prelaunch_evidence_and_measures_target"
    ),
    "R27": (
        "tests/skills/test_multi_gpu_orchestrator.py::"
        "test_strict_critical_path_ledger_records_automatic_slo_breach"
    ),
    "R28": (
        "tests/skills/test_multi_gpu_orchestrator.py::"
        "test_workspace_contract_rejects_future_output_outside_registered_runtime"
    ),
    "R29": (
        "tests/skills/test_multi_gpu_orchestrator.py::"
        "test_controller_never_precreates_or_pollutes_scientific_output"
    ),
    "R30": "test_static_cli_failure_blocks_before_manifest_freeze",
    "R31": "test_exact_public_cli_smoke_binds_source_command_config_parent_and_execution_class",
    "R32": (
        "tests/skills/test_multi_gpu_orchestrator.py::"
        "test_stale_result_from_prior_campaign_cannot_release_gpu_slot"
    ),
    "R33": (
        "docs/ARD_OPERATIONAL_FOUNDATION.md#layer-ownership-and-first-remediation "
        "(policy; R29 regression proves the boundary)"
    ),
    "R34": (
        "tests/skills/test_multi_gpu_orchestrator.py::"
        "test_attempt_scoped_retry_keeps_partial_output_out_of_canonical_namespace"
    ),
    "R35": "test_equivalent_fanout_requires_exact_public_cli_smoke",
}


def test_regression_case_registry_covers_r1_to_r35() -> None:
    assert set(REGRESSION_CASES) == {f"R{number}" for number in range(1, 36)}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Launch Gate Test"], check=True)
    (repo / "train.py").write_text("print('ok')\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "train.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    return repo, commit


def base_spec(tmp_path: Path, *, jobs: list[dict] | None = None) -> tuple[dict, Path, Path, Path]:
    repo, commit = git_repo(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.index").write_text("ids\n", encoding="utf-8")
    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps({"epoch": 1, "seed": "s1", "source_sha": commit}), encoding="utf-8")
    parent_desc = {
        "path": str(parent),
        "sha256": sha(parent),
        "epoch": 1,
        "seed": "s1",
        "source_sha": commit,
    }
    spec = {
        "schema_version": 1,
        "campaign_id": "gate-fixture",
        "source": {"git_sha": commit, "repo_path": str(repo)},
        "dataset": {
            "identity": "toy-v1",
            "split_identity": "split-v1",
            "host_paths": {"local": str(data)},
            "required_files": ["train.index"],
        },
        "teacher": {"identity": "teacher-v1"},
        "training": {"scientific_start_epoch": 0, "scientific_final_epoch": 2},
        "attacks": {
            "train": {
                "loss": "kl",
                "epsilon": "8/255",
                "step_size": "2/255",
                "steps": 10,
                "random_start": True,
                "target": "teacher_clean",
            }
        },
        "augmentation_identity": "none",
        "rng_contract": {"attack": "fixture"},
        "hosts": {
            "local": {
                "backend": "local",
                "repo_path": str(repo),
                "python": sys.executable,
                "dataset_paths": {"toy-v1": str(data)},
                "gpus": [{"index": 0, "uuid": "GPU-fixture", "throughput": 10}],
            }
        },
        "canary": {"jobs": [{"job_id": "train", "command": [sys.executable, "-c", "pass"]}]},
        "jobs": jobs
        or [
            {
                "job_id": "train",
                "arm": "BASE",
                "seed": "s1",
                "host": "local",
                "command": ["${PYTHON}", "train.py", "--epochs", "999"],
                "cwd": str(repo),
                "output_dir": str(tmp_path / "outputs" / "train"),
                "attack": "train",
                "parent": parent_desc,
                "expected_outputs": [{"path": "last.json", "epoch": 2}],
            }
        ],
    }
    return spec, repo, parent, data


def write_spec(tmp_path: Path, spec: dict) -> Path:
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(spec, sort_keys=True), encoding="utf-8")
    return path


def remote_preflight_command(*, source_override: Path | None = None, omit_artifacts: bool = False) -> list[str]:
    source = (
        "source=Path(" + repr(str(source_override)) + ").read_text().strip()"
        if source_override is not None
        else "source=e['source_sha']"
    )
    artifacts = "[]" if omit_artifacts else "e['artifacts']"
    code = (
        "import json,os; from pathlib import Path; e=json.loads(os.environ['ARD_LAUNCH_GATE_REMOTE_EXPECTED']); "
        + source
        + "; print(json.dumps({'schema_version':1,'host':e['host'],'source_sha':source,"
        "'python':{'path':e['python'],'available':True},'gpus':e['gpus'],'artifacts':"
        + artifacts
        + ",'output':{'path':e['output_root'],'writable':True},'disk_free_bytes':999999999,"
        "'launcher':{'argv':e['launcher']['argv'],'valid':True},"
        "'completion_probe':{'argv':e['completion_probe']['argv'],'valid':True}}))"
    )
    return [sys.executable, "-c", code]


def artifact_inventory_manifest(tmp_path: Path, *, campaign_id: str = "gate-fixture", source_sha: str) -> Path:
    identity = {
        "campaign_id": campaign_id,
        "source_sha": source_sha,
        "job_id": "remote-train",
        "seed": "s1",
        "arm": "BASE",
        "epoch": 2,
        "split": "held-out",
        "attack_identity": "train",
    }
    payload = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "source_sha": source_sha,
        "artifacts": [
            {
                "origin": {"host": "remote", "path": "/remote/rows.json"},
                "staging_path": str(tmp_path / "staging" / "rows.json"),
                "collected_path": str(tmp_path / "canonical" / "rows.json"),
                "sha256": "b" * 64,
                "identity": identity,
            }
        ],
        "required_cells": [identity],
    }
    path = tmp_path / "artifact-inventory.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def add_external_host(
    spec: dict, tmp_path: Path, *, launcher: dict | None = None, omit_artifacts: bool = False
) -> None:
    source_sha = spec["source"]["git_sha"]
    inventory = artifact_inventory_manifest(tmp_path, source_sha=source_sha)
    launcher = launcher or {"argv": ["bash", "launch.sh"], "executable": False}
    profile = {
        "backend": "external",
        "repo_path": "/remote/repo",
        "python": sys.executable,
        "dataset_paths": {"toy-v1": "/remote/data"},
        "teacher_paths": {},
        "gpus": [{"index": 0, "uuid": "GPU-remote", "throughput": 10}],
        "remote_preflight": {
            "command": remote_preflight_command(omit_artifacts=omit_artifacts),
            "timeout_seconds": 10,
            "minimum_disk_free_bytes": 1,
            "output_root": "/remote/outputs",
            "launcher": launcher,
            "completion_probe": {"argv": ["bash", "probe.sh"], "executable": False},
        },
    }
    spec["hosts"]["remote"] = profile
    spec["jobs"][0].update(
        {
            "job_id": "remote-train",
            "host": "remote",
            "command": [sys.executable, "-c", "pass", "--epochs", "999"],
            "executor": {"type": "external_probe"},
            "completion_probe": [sys.executable, "-c", "pass"],
            "host_confirm_probe": [sys.executable, "-c", "pass"],
            "remote_command": ["remote-python", "train.py", "--epochs", "3"],
        }
    )
    spec["canary"]["jobs"] = [{"job_id": "remote-train", "command": [sys.executable, "-c", "pass"]}]
    collection = {
        "job_id": "collect",
        "job_type": "collection",
        "arm": "COLLECT",
        "seed": "s1",
        "host": "local",
        "gpu_count": 0,
        "command": [sys.executable, "-c", "pass"],
        "cwd": spec["source"]["repo_path"],
        "output_dir": str(tmp_path / "outputs" / "collect"),
        "dependencies": ["remote-train"],
    }
    inventory_job = {
        "job_id": "inventory",
        "job_type": "inventory",
        "arm": "INVENTORY",
        "seed": "s1",
        "host": "local",
        "gpu_count": 0,
        "command": [sys.executable, "-c", "pass"],
        "cwd": spec["source"]["repo_path"],
        "output_dir": str(tmp_path / "outputs" / "inventory"),
        "dependencies": ["collect"],
    }
    spec["jobs"].extend([collection, inventory_job])
    spec["artifact_collection"] = {
        "manifest_path": str(inventory),
        "collection_job_id": "collect",
        "inventory_job_id": "inventory",
    }


def invoke(spec: Path, *flags: str, output: Path | None = None) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(SCRIPT), "--campaign-spec", str(spec)]
    if output is not None:
        command += ["--output-dir", str(output)]
    command += list(flags)
    return subprocess.run(command, text=True, capture_output=True)


def test_good_spec_freezes_and_binds_inclusive_final_to_exclusive_runtime(tmp_path: Path) -> None:
    spec, _, _, _ = base_spec(tmp_path)
    spec_path = write_spec(tmp_path, spec)
    gate_dir = tmp_path / "gate"
    result = invoke(spec_path, "--dry-run", output=gate_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((gate_dir / "resolved-manifest.json").read_text())
    assert manifest["jobs"][0]["command"][-1] == "3"
    assert manifest["jobs"][0]["epoch_binding"] == {
        "scientific_final_epoch": 2,
        "runtime_exclusive_epochs": 3,
    }
    assert (gate_dir / "freeze.json").is_file()
    assert json.loads(result.stdout)["jobs"][0]["scientific_identity_hash"]


def test_external_probe_fields_survive_resolved_manifest_freeze(tmp_path: Path) -> None:
    """Regression for the gate dropping probe argv before orchestrator validation."""
    spec, _, _, _ = base_spec(tmp_path)
    spec["jobs"][0]["executor"] = {"type": "external_probe"}
    spec["jobs"][0]["completion_probe"] = [sys.executable, "probe.py", "--output", "last.json"]
    spec["jobs"][0]["host_confirm_probe"] = [sys.executable, "host-confirm.py", "--output", "status.json"]
    spec["jobs"][0]["remote_command"] = ["remote-python", "train.py", "--epochs", "3"]
    spec["jobs"][0]["probe_interval_seconds"] = 7
    spec["jobs"][0]["probe_timeout_seconds"] = 31
    gate_dir = tmp_path / "gate"
    result = invoke(write_spec(tmp_path, spec), "--dry-run", output=gate_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    job = json.loads((gate_dir / "resolved-manifest.json").read_text())["jobs"][0]
    assert job["completion_probe"] == [sys.executable, "probe.py", "--output", "last.json"]
    assert job["host_confirm_probe"] == [sys.executable, "host-confirm.py", "--output", "status.json"]
    assert job["remote_command"] == ["remote-python", "train.py", "--epochs", "3"]
    assert job["probe_interval_seconds"] == 7
    assert job["probe_timeout_seconds"] == 31


def test_remote_wrapper_metadata_rejects_non_executable_and_accepts_bash(tmp_path: Path) -> None:
    spec, _, _, _ = base_spec(tmp_path)
    add_external_host(spec, tmp_path, launcher={"argv": ["launch.sh"], "executable": False})
    rejected = invoke(write_spec(tmp_path, spec), "--preflight-only", output=tmp_path / "bad-gate")
    assert rejected.returncode == 2
    assert "remote_preflight.launcher.executable" in rejected.stdout

    spec["hosts"]["remote"]["remote_preflight"]["launcher"] = {"argv": ["bash", "launch.sh"], "executable": False}
    accepted = invoke(write_spec(tmp_path, spec), "--dry-run", output=tmp_path / "good-gate")
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
    assert (tmp_path / "good-gate" / "remote-preflight.json").is_file()


def test_external_host_preflight_requires_every_declared_binding(tmp_path: Path) -> None:
    spec, _, _, _ = base_spec(tmp_path)
    add_external_host(spec, tmp_path, omit_artifacts=True)
    result = invoke(write_spec(tmp_path, spec), "--preflight-only", output=tmp_path / "gate")
    assert result.returncode == 2
    assert "remote_preflight.artifacts" in result.stdout


def test_remote_source_drift_is_rechecked_before_launch(tmp_path: Path) -> None:
    spec, _, _, _ = base_spec(tmp_path)
    remote_source = tmp_path / "remote-source.txt"
    remote_source.write_text(spec["source"]["git_sha"], encoding="utf-8")
    add_external_host(spec, tmp_path)
    spec["hosts"]["remote"]["remote_preflight"]["command"] = remote_preflight_command(source_override=remote_source)
    spec_path = write_spec(tmp_path, spec)
    gate_dir = tmp_path / "gate"
    assert invoke(spec_path, "--dry-run", output=gate_dir).returncode == 0
    remote_source.write_text("b" * 40, encoding="utf-8")
    result = invoke(spec_path, "--launch", output=gate_dir)
    assert result.returncode == 2
    assert "remote_preflight.source_sha" in result.stdout


def test_remote_lifecycle_canary_requires_status_and_hash_roundtrip(tmp_path: Path) -> None:
    spec, _, _, _ = base_spec(tmp_path)
    add_external_host(spec, tmp_path)
    staging = tmp_path / "remote-canary-staging.bin"
    collected = tmp_path / "remote-canary-canonical.bin"
    canary_code = (
        "import hashlib,json,os; from pathlib import Path; "
        f"staging=Path({str(staging)!r}); staging.write_bytes(b'canary-bytes'); "
        "identity={'campaign_id':'gate-fixture','source_sha':os.environ['ARD_LAUNCH_GATE_EXPECTED_SOURCE_SHA'],"
        "'job_id':'remote-train','seed':'s1','arm':'BASE','epoch':2,'split':'held-out','attack_identity':'train'}; "
        f"artifact={{'origin':{{'host':'remote','path':'/remote/canary.bin'}},'staging_path':str(staging),'collected_path':{str(collected)!r},"
        "'sha256':hashlib.sha256(staging.read_bytes()).hexdigest(),'identity':identity}; "
        "payload={'schema_version':1,'host':os.environ['ARD_LAUNCH_GATE_EXPECTED_HOST'],"
        "'source_sha':os.environ['ARD_LAUNCH_GATE_EXPECTED_SOURCE_SHA'],'process_confirmed':True,"
        "'remote_manifest':'/remote/manifest.json','completion_marker':True,'artifact':artifact}; "
        "print(json.dumps(payload))"
    )
    spec["canary"]["remote_lifecycle"] = [
        {"host": "remote", "command": [sys.executable, "-c", canary_code], "timeout_seconds": 10}
    ]
    result = invoke(write_spec(tmp_path, spec), "--canary-only", output=tmp_path / "gate")
    assert result.returncode == 0, result.stdout + result.stderr
    assert collected.read_bytes() == b"canary-bytes"


def test_preflight_requires_full_source_binding_and_clean_tree(tmp_path: Path) -> None:
    spec, repo, _, _ = base_spec(tmp_path)
    (repo / "uncommitted.txt").write_text("dirty", encoding="utf-8")
    result = invoke(write_spec(tmp_path, spec), "--preflight-only")
    assert result.returncode == 2
    assert "source.dirty" in result.stdout


def test_parent_alias_mismatch_and_mask_hash_are_fail_closed(tmp_path: Path) -> None:
    spec, _, parent, _ = base_spec(tmp_path)
    alias = tmp_path / "alias.json"
    alias.write_text("different", encoding="utf-8")
    spec["jobs"][0]["parent"]["alias_path"] = str(alias)
    mask = tmp_path / "mask.json"
    mask.write_text(json.dumps({"schema_version": 1, "stable_id_hash": "ids"}), encoding="utf-8")
    spec["jobs"][0]["masks"] = [{"path": str(mask), "sha256": "0" * 64, "schema_version": 1}]
    result = invoke(write_spec(tmp_path, spec), "--preflight-only")
    assert result.returncode == 2
    assert "parent.alias_path" in result.stdout
    assert "mask.sha256" in result.stdout
    assert parent.is_file()


def test_parent_alias_mismatch_is_fail_closed(tmp_path: Path) -> None:
    spec, _, _, _ = base_spec(tmp_path)
    alias = tmp_path / "alias.json"
    alias.write_text("different", encoding="utf-8")
    spec["jobs"][0]["parent"]["alias_path"] = str(alias)
    result = invoke(write_spec(tmp_path, spec), "--preflight-only")
    assert result.returncode == 2
    assert "parent.alias_path" in result.stdout


def test_mask_metadata_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    spec, _, _, _ = base_spec(tmp_path)
    mask = tmp_path / "mask.json"
    mask.write_text(json.dumps({"schema_version": 1, "stable_id_hash": "ids"}), encoding="utf-8")
    spec["jobs"][0]["masks"] = [{"path": str(mask), "sha256": "0" * 64, "schema_version": 1}]
    result = invoke(write_spec(tmp_path, spec), "--preflight-only")
    assert result.returncode == 2
    assert "mask.sha256" in result.stdout


def test_host_logical_dataset_path_mismatch_is_rejected(tmp_path: Path) -> None:
    spec, _, _, _ = base_spec(tmp_path)
    wrong = tmp_path / "wrong-data"
    wrong.mkdir()
    spec["jobs"][0]["dataset_path"] = str(wrong)
    result = invoke(write_spec(tmp_path, spec), "--preflight-only")
    assert result.returncode == 2
    assert "dataset_path" in result.stdout


def test_dependency_output_is_deferred_but_producer_path_is_bound(tmp_path: Path) -> None:
    spec, repo, _, _ = base_spec(tmp_path)
    producer = spec["jobs"][0]
    producer["job_id"] = "prefix"
    producer["output_dir"] = str(tmp_path / "outputs" / "prefix")
    producer["expected_outputs"] = [{"path": "checkpoint.json", "epoch": 2}]
    consumer = {
        "job_id": "endpoint",
        "arm": "EVAL",
        "seed": "s1",
        "host": "local",
        "command": ["${PYTHON}", "eval.py", "--epochs", "999"],
        "cwd": str(repo),
        "output_dir": str(tmp_path / "outputs" / "endpoint"),
        "dependencies": ["prefix"],
        "inputs": [
            {
                "kind": "dependency_output",
                "producer_job_id": "prefix",
                "path": str(tmp_path / "outputs" / "prefix" / "checkpoint.json"),
            }
        ],
    }
    spec["jobs"] = [producer, consumer]
    result = invoke(write_spec(tmp_path, spec), "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    bad = json.loads(json.dumps(spec))
    bad["jobs"][1]["inputs"][0]["path"] = str(tmp_path / "other.json")
    result = invoke(write_spec(tmp_path, bad), "--preflight-only")
    assert result.returncode == 2
    assert "input.path" in result.stdout


def test_dependency_input_must_bind_to_producer_output(tmp_path: Path) -> None:
    spec, repo, _, _ = base_spec(tmp_path)
    producer = spec["jobs"][0]
    producer["job_id"] = "prefix"
    producer["output_dir"] = str(tmp_path / "outputs" / "prefix")
    producer["expected_outputs"] = [{"path": "checkpoint.json", "epoch": 2}]
    spec["jobs"] = [
        producer,
        {
            "job_id": "endpoint",
            "arm": "EVAL",
            "seed": "s1",
            "host": "local",
            "command": ["${PYTHON}", "eval.py", "--epochs", "999"],
            "cwd": str(repo),
            "output_dir": str(tmp_path / "outputs" / "endpoint"),
            "dependencies": ["prefix"],
            "inputs": [
                {
                    "kind": "dependency_output",
                    "producer_job_id": "prefix",
                    "path": str(tmp_path / "not-produced.json"),
                }
            ],
        },
    ]
    result = invoke(write_spec(tmp_path, spec), "--preflight-only")
    assert result.returncode == 2
    assert "input.path" in result.stdout


def test_dependency_output_is_not_required_before_producer_completion(tmp_path: Path) -> None:
    spec, repo, _, _ = base_spec(tmp_path)
    producer = spec["jobs"][0]
    producer["job_id"] = "prefix"
    producer["output_dir"] = str(tmp_path / "outputs" / "prefix")
    producer["expected_outputs"] = [{"path": "checkpoint.json", "epoch": 2}]
    spec["jobs"] = [
        producer,
        {
            "job_id": "endpoint",
            "arm": "EVAL",
            "seed": "s1",
            "host": "local",
            "command": ["${PYTHON}", "eval.py", "--epochs", "999"],
            "cwd": str(repo),
            "output_dir": str(tmp_path / "outputs" / "endpoint"),
            "dependencies": ["prefix"],
            "inputs": [
                {
                    "kind": "dependency_output",
                    "producer_job_id": "prefix",
                    "path": str(tmp_path / "outputs" / "prefix" / "checkpoint.json"),
                }
            ],
        },
    ]
    result = invoke(write_spec(tmp_path, spec), "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr


def test_output_collision_and_retry_scientific_mutation_fail(tmp_path: Path) -> None:
    spec, repo, _, _ = base_spec(tmp_path)
    second = json.loads(json.dumps(spec["jobs"][0]))
    second["job_id"] = "second"
    second["cwd"] = str(repo)
    spec["jobs"].append(second)
    spec["jobs"][0]["retry_mutations"] = {"seed": "changed"}
    result = invoke(write_spec(tmp_path, spec), "--preflight-only")
    assert result.returncode == 2
    assert "output_dir" in result.stdout
    assert "retry_mutations" in result.stdout


def test_explicit_wandb_ids_must_not_collide(tmp_path: Path) -> None:
    spec, _, _, _ = base_spec(tmp_path)
    second = json.loads(json.dumps(spec["jobs"][0]))
    second["job_id"] = "second"
    second["output_dir"] = str(tmp_path / "outputs" / "second")
    second["wandb_run_id"] = "same-execution-id"
    spec["jobs"][0]["wandb_run_id"] = "same-execution-id"
    spec["jobs"].append(second)
    result = invoke(write_spec(tmp_path, spec), "--preflight-only")
    assert result.returncode == 2
    assert "wandb_run_id" in result.stdout


def test_canary_is_bounded_and_uses_resolved_job_context(tmp_path: Path) -> None:
    spec, _, _, _ = base_spec(tmp_path)
    spec_path = write_spec(tmp_path, spec)
    gate_dir = tmp_path / "gate"
    result = invoke(spec_path, "--canary-only", output=gate_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    canary = json.loads((gate_dir / "canary.json").read_text())
    assert canary["status"] == "pass"
    assert canary["results"][0]["job_id"] == "train"


def test_static_cli_failure_blocks_before_manifest_freeze(tmp_path: Path) -> None:
    spec, _, _, _ = base_spec(tmp_path)
    spec["canary"] = {
        "require_exact_smoke": True,
        "static_cli": [{"job_id": "train", "commands": [[sys.executable, "-c", "raise SystemExit(2)"]]}],
        "jobs": [
            {
                "job_id": "train",
                "kind": "exact_public_cli",
                "execution_class": "local",
                "command": [sys.executable, "-c", "pass"],
            }
        ],
    }

    gate = tmp_path / "gate"
    result = invoke(write_spec(tmp_path, spec), "--canary-only", output=gate)

    assert result.returncode == 2
    assert not (gate / "resolved-manifest.json").exists()
    assert json.loads((gate / "static-cli.json").read_text())["status"] == "fail"


def test_exact_public_cli_smoke_binds_source_command_config_parent_and_execution_class(tmp_path: Path) -> None:
    spec, repo, _, _ = base_spec(tmp_path)
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    spec["jobs"][0]["scientific_config_path"] = str(config)
    spec["jobs"][0]["config_sha256"] = sha(config)
    spec["canary"] = {
        "require_exact_smoke": True,
        "static_cli": [{"job_id": "train", "commands": [[sys.executable, "-m", "py_compile", "train.py"]]}],
        "jobs": [
            {
                "job_id": "train",
                "kind": "exact_public_cli",
                "execution_class": "local",
                "command": [sys.executable, "-c", "pass"],
            }
        ],
    }
    gate = tmp_path / "gate"

    result = invoke(write_spec(tmp_path, spec), "--canary-only", output=gate)

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((gate / "canary.json").read_text())
    binding = report["results"][0]["binding"]
    assert binding["source_sha"] == spec["source"]["git_sha"]
    assert binding["config_sha256"] == sha(config)
    assert binding["parent_checkpoint_sha256"] == spec["jobs"][0]["parent"]["sha256"]
    assert binding["execution_class"] == "local"
    assert binding["production_argv_sha256"]
    assert binding["orchestrator_source_sha256"]
    assert (repo / "train.py").is_file()


def test_source_or_orchestrator_change_invalidates_exact_smoke_binding(tmp_path: Path) -> None:
    spec, _, _, _ = base_spec(tmp_path)
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    spec["jobs"][0]["scientific_config_path"] = str(config)
    spec["jobs"][0]["config_sha256"] = sha(config)
    spec["canary"] = {
        "require_exact_smoke": True,
        "static_cli": [{"job_id": "train", "commands": [[sys.executable, "-c", "pass"]]}],
        "jobs": [
            {
                "job_id": "train",
                "kind": "exact_public_cli",
                "execution_class": "local",
                "command": [sys.executable, "-c", "pass"],
            }
        ],
    }
    gate = tmp_path / "gate"
    spec_path = write_spec(tmp_path, spec)
    assert invoke(spec_path, "--canary-only", output=gate).returncode == 0
    manifest = json.loads((gate / "resolved-manifest.json").read_text())
    report = json.loads((gate / "canary.json").read_text())
    report["results"][0]["binding"]["orchestrator_source_sha256"] = "0" * 64

    errors = gate_module().validate_exact_smoke_report(manifest, report)

    assert errors == ["exact smoke binding is stale for train"]


def test_equivalent_fanout_requires_exact_public_cli_smoke(tmp_path: Path) -> None:
    spec, _, _, _ = base_spec(tmp_path)
    second = json.loads(json.dumps(spec["jobs"][0]))
    second["job_id"] = "train-two"
    second["output_dir"] = str(tmp_path / "outputs" / "train-two")
    spec["jobs"].append(second)
    spec["canary"] = {
        "require_exact_smoke": True,
        "static_cli": [{"job_id": "train", "commands": [[sys.executable, "-c", "pass"]]}],
        "jobs": [{"job_id": "train", "kind": "bounded_canary", "command": [sys.executable, "-c", "pass"]}],
    }

    result = invoke(write_spec(tmp_path, spec), "--canary-only", output=tmp_path / "gate")

    assert result.returncode == 2
    assert "exact smoke" in result.stdout


def test_launch_gate_rejects_unknown_schema_and_missing_epoch_bound(tmp_path: Path) -> None:
    spec, _, _, _ = base_spec(tmp_path)
    spec["schema_version"] = 99
    spec["jobs"][0]["command"] = ["${PYTHON}", "train.py"]
    result = invoke(write_spec(tmp_path, spec), "--preflight-only")
    assert result.returncode == 2
    assert "schema_version" in result.stdout
    assert "command.--epochs" in result.stdout


def test_explicit_inclusive_runtime_bound_is_rejected(tmp_path: Path) -> None:
    spec, _, _, _ = base_spec(tmp_path)
    spec["jobs"][0]["runtime_epochs"] = 2
    result = invoke(write_spec(tmp_path, spec), "--preflight-only")
    assert result.returncode == 2
    assert "runtime_epochs" in result.stdout


def test_resource_plan_conflict_is_fail_closed_before_freeze(tmp_path: Path) -> None:
    spec, _, _, _ = base_spec(tmp_path)
    spec["hosts"]["local"]["gpus"] = []
    result = invoke(write_spec(tmp_path, spec), "--preflight-only")
    assert result.returncode == 2
    assert "resource_plan" in result.stdout


def test_source_config_drift_is_detected_after_freeze(tmp_path: Path) -> None:
    spec, repo, _, _ = base_spec(tmp_path)
    spec_path = write_spec(tmp_path, spec)
    gate_dir = tmp_path / "gate"
    assert invoke(spec_path, "--dry-run", output=gate_dir).returncode == 0
    (repo / "train.py").write_text("drift\n", encoding="utf-8")
    result = invoke(spec_path, "--launch", output=gate_dir)
    assert result.returncode == 2
    assert "source.dirty" in result.stdout or "source.git_sha" in result.stdout


def test_validate_run_rejects_exit_zero_without_required_outputs(tmp_path: Path) -> None:
    spec, _, _, _ = base_spec(tmp_path)
    spec_path = write_spec(tmp_path, spec)
    gate_dir = tmp_path / "gate"
    assert invoke(spec_path, "--dry-run", output=gate_dir).returncode == 0
    manifest = json.loads((gate_dir / "resolved-manifest.json").read_text())
    state_path = Path(manifest["state_path"])
    state_path.parent.mkdir(parents=True, exist_ok=True)
    job = manifest["jobs"][0]
    state_path.write_text(
        json.dumps(
            {
                "campaign_id": manifest["campaign_id"],
                "manifest_sha256": sha(gate_dir / "resolved-manifest.json"),
                "jobs": {job["job_id"]: {"status": "completed", "attempts": []}},
            }
        ),
        encoding="utf-8",
    )
    result = invoke(
        gate_dir / "resolved-manifest.json",
        "--validate-run",
        "--resolved-manifest",
        str(gate_dir / "resolved-manifest.json"),
    )
    assert result.returncode == 2
    assert "expected_output" in result.stdout


def test_wandb_template_is_attempt_specific_but_identity_hash_is_stable(tmp_path: Path) -> None:
    spec, _, _, _ = base_spec(tmp_path)
    spec_path = write_spec(tmp_path, spec)
    gate_dir = tmp_path / "gate"
    assert invoke(spec_path, "--dry-run", output=gate_dir).returncode == 0
    job = json.loads((gate_dir / "resolved-manifest.json").read_text())["jobs"][0]
    assert job["wandb_run_id_template"].format(attempt=1) != job["wandb_run_id_template"].format(attempt=2)
    assert (
        job["identity_hash"]
        == json.loads((gate_dir / "freeze.json").read_text())["scientific_identity_hashes"]["train"]
    )


def test_scientific_identity_excludes_host_local_teacher_path(tmp_path: Path) -> None:
    spec, _, _, _ = base_spec(tmp_path)
    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"teacher")
    spec["teacher"] = {
        "identity": "teacher-v1",
        "checkpoint": {"path": str(teacher), "sha256": sha(teacher)},
    }
    spec["hosts"]["local"]["teacher_paths"] = {"teacher-v1": str(teacher)}
    gate_dir = tmp_path / "gate"
    assert invoke(write_spec(tmp_path, spec), "--dry-run", output=gate_dir).returncode == 0
    manifest = json.loads((gate_dir / "resolved-manifest.json").read_text())
    identity = json.dumps(manifest["jobs"][0]["scientific_identity"], sort_keys=True)
    assert str(teacher) not in identity
    assert sha(teacher) in identity


def test_bounded_launch_chains_to_post_run_validation(tmp_path: Path) -> None:
    spec, repo, _, _ = base_spec(tmp_path)
    output = tmp_path / "outputs" / "train"
    code = (
        "import json,sys; from pathlib import Path; "
        f"Path({str(output)!r}).mkdir(parents=True, exist_ok=True); "
        f"Path({str(output / 'last.json')!r}).write_text(json.dumps({{'epoch': 2}}))"
    )
    spec["jobs"][0]["command"] = [sys.executable, "-c", code, "--epochs", "999"]
    spec["jobs"][0]["cwd"] = str(repo)
    spec_path = write_spec(tmp_path, spec)
    gate_dir = tmp_path / "gate"
    result = invoke(spec_path, "--launch", output=gate_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest_path = gate_dir / "resolved-manifest.json"
    state_path = Path(json.loads(manifest_path.read_text())["state_path"])
    for _ in range(200):
        if state_path.exists() and json.loads(state_path.read_text()).get("status") == "completed":
            break
        import time

        time.sleep(0.01)
    checked = invoke(manifest_path, "--validate-run", "--resolved-manifest", str(manifest_path))
    assert checked.returncode == 0, checked.stdout + checked.stderr


def test_technical_retry_keeps_identity_and_gets_new_execution_id(tmp_path: Path) -> None:
    spec, _, _, _ = base_spec(tmp_path)
    output = tmp_path / "outputs" / "train"
    code = (
        "import json,os; from pathlib import Path; "
        f"out=Path({str(output)!r}); out.mkdir(parents=True, exist_ok=True); "
        "attempt=os.environ['ARD_ORCH_ATTEMPT']; "
        "(out/'attempts.txt').open('a').write(os.environ.get('WANDB_RUN_ID','missing')+'\\n'); "
        "\nif attempt == '1':\n "
        "(out/'technical-failure.json').write_text(json.dumps({'failure_class':'technical','retryable':True})); "
        "raise SystemExit(7)\n"
        "(out/'last.json').write_text(json.dumps({'epoch':2}))"
    )
    spec["jobs"][0]["command"] = [sys.executable, "-c", code, "--epochs", "999"]
    spec["jobs"][0]["retry_policy"] = {"max_attempts": 2}
    spec_path = write_spec(tmp_path, spec)
    gate_dir = tmp_path / "gate"
    result = invoke(spec_path, "--launch", output=gate_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((gate_dir / "resolved-manifest.json").read_text())
    state_path = Path(manifest["state_path"])
    import time

    for _ in range(200):
        if state_path.exists() and json.loads(state_path.read_text()).get("status") == "completed":
            break
        time.sleep(0.01)
    state = json.loads(state_path.read_text())
    attempts = state["jobs"]["train"]["attempts"]
    assert len(attempts) == 2
    assert attempts[0]["identity_hash"] == attempts[1]["identity_hash"]
    ids = []
    for _ in range(100):
        ids = (output / "attempts.txt").read_text().splitlines()
        if len(ids) == 2:
            break
        time.sleep(0.01)
    assert len(ids) == 2 and ids[0] != ids[1]
