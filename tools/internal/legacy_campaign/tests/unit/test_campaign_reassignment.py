from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from ard.campaign.reassignment import (
    AUTOATTACK_AMENDMENT_PATH,
    AUTOATTACK_AMENDMENT_SHA256,
    canonical_json_sha256,
)
from ard.campaign.schema import CampaignSpec
from ard.campaign.state import CampaignStateStore, JobState, StateError

_PRIMARY_AA_RESULT = "5c0224c9bab73ec57f1c779cf2a9f0fb76fd1fa49564d40fc05b2cf55b4fcf79"
_AUXILIARY_AA_RESULT = "fa8e5b87b1f6acf21cc52b4b0709eafefba89d1e0f172cecfc42793d22f6b44e"


def _importer_module() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts" / "campaign" / "import_reassigned_sequence.py"
    module_spec = importlib.util.spec_from_file_location("test_import_reassigned_sequence", path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _spec(*, autoattack: bool) -> CampaignSpec:
    phases: dict[str, object] = {"train": ["python", "train"], "pgd_evaluate": ["python", "pgd"]}
    if autoattack:
        phases["autoattack"] = ["python", "autoattack"]
    return CampaignSpec.model_validate(
        {
            "campaign_id": "campaign-v1",
            "git_sha": "a" * 40,
            "execution_profile": {"id": "ws1_prb128_gb128_localbn_v1"},
            "hosts": {"hamster": {"gpus": [0]}},
            "jobs": [
                {
                    "id": "job-1",
                    "host": "hamster",
                    "gpu": 0,
                    "teacher": "teacher",
                    "method": "rslad",
                    "seed": 0,
                    "config": "configs/job.yaml",
                    "output": "outputs/job-1",
                    "wandb": {"entity": "entity", "project": "project", "group": "group", "run_id": "run-1"},
                    "phases": phases,
                }
            ],
        }
    )


def _advance_to_pgd_completed(store: CampaignStateStore, job_id: str = "job-1") -> None:
    for state in (
        JobState.PREFLIGHT,
        JobState.LAUNCHING,
        JobState.TRAINING,
        JobState.TRAINING_COMPLETED,
        JobState.PGD_EVALUATION,
        JobState.PGD_COMPLETED,
    ):
        store.transition_job(job_id, state)


def _evidence(store: CampaignStateStore, spec: CampaignSpec, *, auxiliary: bool = False) -> dict[str, object]:
    job = store.job("job-1")
    phases = ["train", "pgd", *(["autoattack"] if spec.jobs[0].phases.autoattack is not None else [])]
    reassigned = phases[1:]
    phase_events = [
        event
        for phase in reassigned
        for event in (
            {"at": f"{phase}-start", "event": "started", "phase": phase},
            {"at": f"{phase}-finish", "event": "finished", "phase": phase, "exit_code": 0},
        )
    ]
    entry: dict[str, object] = {
        "job_id": "job-1",
        "sequence_source_job_id": "job-1",
        "expected_state": job["state"],
        "expected_revision": job["revision"],
        "required_phases": phases,
        "outer_exit": {
            "version": 1,
            "exit_code": 0,
            "run_id": "reassigned-job-1",
            "git_sha": "a" * 40,
            "wrapper_pid": 123,
            "phase_argv_digest": "a" * 64,
            "finished_at": "finished",
            "error": None,
        },
        "phase_events": phase_events,
        "sequence_digests": {
            "sequence_spec": "1" * 64,
            "sequence_completion": "2" * 64,
            "outer_exit": "3" * 64,
            "phase_events": "4" * 64,
        },
        "prior_phase_digests": {"train": "5" * 64},
        "evidence_digests": {
            "best_checkpoint": "b" * 64,
            "last_checkpoint": "c" * 64,
            "evaluation_results": _PRIMARY_AA_RESULT if spec.jobs[0].phases.autoattack is not None else "d" * 64,
        },
    }
    if spec.jobs[0].phases.autoattack is not None:
        entry["posthoc_autoattack_attestation"] = {
            "posthoc_attested": True,
            "evaluation_results_sha256": _PRIMARY_AA_RESULT,
            "amendment": {
                "path": AUTOATTACK_AMENDMENT_PATH,
                "sha256": AUTOATTACK_AMENDMENT_SHA256,
            },
        }
    if auxiliary:
        entry["auxiliary_autoattack"] = {
            "sequence_source_job_id": "job-1-extra-autoattack",
            "execution_host": "hamster",
            "execution_gpu": 0,
            "execution_gpu_uuid": "GPU-fixture",
            "runtime_git_sha": None,
            "outer_exit": {
                "version": 1,
                "exit_code": 0,
                "run_id": "reassigned-job-1-extra-autoattack",
                "git_sha": "a" * 40,
                "wrapper_pid": 124,
                "phase_argv_digest": "e" * 64,
                "finished_at": "aux-finished",
                "error": None,
            },
            "phase_events": [
                {"at": "aux-start", "event": "started", "phase": "autoattack"},
                {"at": "aux-finish", "event": "finished", "phase": "autoattack", "exit_code": 0},
            ],
            "sequence_digests": {
                "sequence_spec": "6" * 64,
                "sequence_completion": "7" * 64,
                "outer_exit": "8" * 64,
                "phase_events": "9" * 64,
            },
            "evidence_digests": {
                "best_checkpoint": "b" * 64,
                "last_checkpoint": "c" * 64,
                "evaluation_results": _AUXILIARY_AA_RESULT,
            },
            "posthoc_autoattack_attestation": {
                "posthoc_attested": True,
                "evaluation_results_sha256": _AUXILIARY_AA_RESULT,
                "amendment": {
                    "path": AUTOATTACK_AMENDMENT_PATH,
                    "sha256": AUTOATTACK_AMENDMENT_SHA256,
                },
            },
        }
    document: dict[str, object] = {
        "version": 1,
        "campaign_id": spec.campaign_id,
        "campaign_identity_sha256": store.campaign()["identity_sha256"],
        "source_host": "hamster",
        "execution_host": "hamster",
        "execution_gpu": 0,
        "execution_gpu_uuid": "GPU-fixture",
        "scientific_git_sha": "a" * 40,
        "runtime_git_sha": None,
        "job": entry,
    }
    document["evidence_sha256"] = canonical_json_sha256(document)
    return document


@pytest.mark.unit
@pytest.mark.t1
def test_terminal_reassignment_is_dry_run_cas_archived_and_idempotent(tmp_path: Path) -> None:
    spec = _spec(autoattack=True)
    store = CampaignStateStore(tmp_path / "state")
    store.initialize(spec)
    _advance_to_pgd_completed(store)
    evidence = _evidence(store, spec)

    dry_run = store.import_terminal_reassignment(evidence, spec=spec)
    assert dry_run["status"] == "dry-run"
    assert dry_run["would_import"] == ["job-1"]
    assert dry_run["already_imported"] == []
    assert isinstance(dry_run["transaction_id"], str)
    assert store.job("job-1")["state"] == JobState.PGD_COMPLETED.value
    assert store.import_terminal_reassignment(evidence, spec=spec, dry_run=False)["imported"] == ["job-1"]
    imported = store.job("job-1")
    assert imported["state"] == JobState.COMPLETED.value
    assert imported["terminal_reassignment"]["posthoc_autoattack_attestation"]["posthoc_attested"] is True
    evidence_sha256 = imported["terminal_reassignment"]["evidence_sha256"]
    archive = tmp_path / "state" / "reassignment-archive" / "job-1" / f"{evidence_sha256}.json"
    assert archive.is_file()
    repeated = store.import_terminal_reassignment(evidence, spec=spec, dry_run=False)
    assert repeated["status"] == "imported"
    assert repeated["imported"] == []
    assert repeated["already_imported"] == ["job-1"]
    conflicting = copy.deepcopy(evidence)
    conflicting["job"]["evidence_digests"]["best_checkpoint"] = "f" * 64  # type: ignore[index]
    conflicting["evidence_sha256"] = canonical_json_sha256(
        {key: value for key, value in conflicting.items() if key != "evidence_sha256"}
    )
    with pytest.raises(StateError, match="conflicts|modified"):
        store.import_terminal_reassignment(conflicting, spec=spec, dry_run=False)
@pytest.mark.unit
@pytest.mark.t1
def test_terminal_reassignment_rejects_bad_digest_without_mutation_and_keeps_auxiliary_aa_auxiliary(
    tmp_path: Path,
) -> None:
    spec = _spec(autoattack=False)
    store = CampaignStateStore(tmp_path / "state")
    store.initialize(spec)
    _advance_to_pgd_completed(store)
    evidence = _evidence(store, spec, auxiliary=True)
    invalid = copy.deepcopy(evidence)
    invalid["evidence_sha256"] = "0" * 64
    with pytest.raises(StateError, match="digest"):
        store.import_terminal_reassignment(invalid, spec=spec, dry_run=False)
    assert "terminal_reassignment" not in store.job("job-1")
    store.import_terminal_reassignment(evidence, spec=spec, dry_run=False)
    imported = store.job("job-1")
    assert imported["state"] == JobState.COMPLETED.value
    assert imported["autoattack_status"] == "not_requested"


@pytest.mark.unit
@pytest.mark.t1
def test_terminal_reassignment_batch_cas_failure_mutates_no_job(tmp_path: Path) -> None:
    first = _spec(autoattack=True)
    raw = first.model_dump(mode="json")
    second = copy.deepcopy(raw["jobs"][0])
    second["id"] = "job-2"
    second["output"] = "outputs/job-2"
    second["wandb"]["run_id"] = "run-2"
    raw["jobs"].append(second)
    spec = CampaignSpec.model_validate(raw)
    store = CampaignStateStore(tmp_path / "state")
    store.initialize(spec)
    _advance_to_pgd_completed(store, "job-1")
    _advance_to_pgd_completed(store, "job-2")
    first_evidence = _evidence(store, spec)
    second_evidence = copy.deepcopy(first_evidence)
    second_entry = second_evidence["job"]
    assert isinstance(second_entry, dict)
    second_entry["job_id"] = "job-2"
    second_entry["sequence_source_job_id"] = "job-2"
    second_entry["expected_state"] = store.job("job-2")["state"]
    second_entry["expected_revision"] = store.job("job-2")["revision"] + 1
    outer_exit = second_entry["outer_exit"]
    assert isinstance(outer_exit, dict)
    outer_exit["run_id"] = "reassigned-job-2"
    second_evidence["evidence_sha256"] = canonical_json_sha256(
        {key: value for key, value in second_evidence.items() if key != "evidence_sha256"}
    )

    with pytest.raises(StateError, match="compare-and-swap"):
        store.import_terminal_reassignments([first_evidence, second_evidence], spec=spec, dry_run=False)
    assert "terminal_reassignment" not in store.job("job-1")
    assert "terminal_reassignment" not in store.job("job-2")
    assert not (tmp_path / "state" / "reassignment-transactions").exists()


@pytest.mark.unit
@pytest.mark.t1
def test_terminal_reassignment_reimport_rejects_stored_evidence_corruption(tmp_path: Path) -> None:
    spec = _spec(autoattack=True)
    store = CampaignStateStore(tmp_path / "state")
    store.initialize(spec)
    _advance_to_pgd_completed(store)
    evidence = _evidence(store, spec)
    store.import_terminal_reassignment(evidence, spec=spec, dry_run=False)
    job_path = tmp_path / "state" / "jobs" / "job-1.json"
    corrupted = store.job("job-1")
    corrupted["terminal_reassignment"]["execution_gpu_uuid"] = "GPU-corrupted"
    job_path.write_text(json.dumps(corrupted, sort_keys=True), encoding="utf-8")

    with pytest.raises(StateError, match="transaction|modified|drift"):
        store.import_terminal_reassignment(evidence, spec=spec, dry_run=False)


@pytest.mark.unit
@pytest.mark.t1
def test_terminal_reassignment_rejects_active_phase_lease_without_mutation(tmp_path: Path) -> None:
    spec = _spec(autoattack=True)
    store = CampaignStateStore(tmp_path / "state")
    store.initialize(spec)
    store.transition_job("job-1", JobState.PREFLIGHT)
    store.transition_job("job-1", JobState.LAUNCHING)
    exit_path = tmp_path / "train-exit.json"
    exit_path.write_text(
        json.dumps(
            {
                "exit_code": 0,
                "run_id": "job-1",
                "git_sha": "a" * 40,
                "phase_argv_digest": "1" * 64,
            }
        ),
        encoding="utf-8",
    )
    lease_path = tmp_path / "gpu.lease.json"
    lease_path.write_text("{}\n", encoding="utf-8")
    store.transition_job(
        "job-1",
        JobState.TRAINING,
        phase={
            "name": "train",
            "exit_record": str(exit_path),
            "gpu_lease_path": str(lease_path),
            "phase_argv_digest": "1" * 64,
            "wrapper": {},
        },
        launch_intent=None,
    )
    evidence = _evidence(store, spec)

    with pytest.raises(StateError, match="live or unresolved"):
        store.import_terminal_reassignment(evidence, spec=spec, dry_run=False)
    assert store.job("job-1")["state"] == JobState.TRAINING.value
    assert "terminal_reassignment" not in store.job("job-1")


@pytest.mark.unit
@pytest.mark.t1
def test_terminal_reassignment_rejects_unpinned_amendment_reference(tmp_path: Path) -> None:
    spec = _spec(autoattack=True)
    store = CampaignStateStore(tmp_path / "state")
    store.initialize(spec)
    _advance_to_pgd_completed(store)
    evidence = _evidence(store, spec)
    attestation = evidence["job"]["posthoc_autoattack_attestation"]  # type: ignore[index]
    attestation["amendment"] = {"path": "docs/fake.json", "sha256": "f" * 64}  # type: ignore[index]
    evidence["evidence_sha256"] = canonical_json_sha256(
        {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    )

    with pytest.raises(StateError, match="pinned amendment"):
        store.import_terminal_reassignment(evidence, spec=spec, dry_run=False)
    assert "terminal_reassignment" not in store.job("job-1")


@pytest.mark.unit
@pytest.mark.t1
def test_terminal_reassignment_rejects_result_absent_from_pinned_amendment(tmp_path: Path) -> None:
    spec = _spec(autoattack=True)
    store = CampaignStateStore(tmp_path / "state")
    store.initialize(spec)
    _advance_to_pgd_completed(store)
    evidence = _evidence(store, spec)
    entry = evidence["job"]
    forged_result = "f" * 64
    entry["evidence_digests"]["evaluation_results"] = forged_result  # type: ignore[index]
    entry["posthoc_autoattack_attestation"]["evaluation_results_sha256"] = forged_result  # type: ignore[index]
    evidence["evidence_sha256"] = canonical_json_sha256(
        {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    )

    with pytest.raises(StateError, match="does not attest this result"):
        store.import_terminal_reassignment(evidence, spec=spec, dry_run=False)
    assert "terminal_reassignment" not in store.job("job-1")


@pytest.mark.unit
@pytest.mark.t1
def test_sequence_input_contract_is_exact_and_binds_checkpoints(tmp_path: Path) -> None:
    importer = _importer_module()
    output = tmp_path / "output"
    files = [
        output / "resolved_config.yaml",
        output / "best.pt",
        output / "last.pt",
        output / "run-bundle" / "manifest.json",
    ]
    for index, path in enumerate(files):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture-{index}", encoding="utf-8")
    inputs = {str(path.resolve()): _file_sha256(path) for path in files}

    assert importer._validated_sequence_inputs(  # noqa: SLF001
        {"input_sha256": inputs}, output_path=output, starts_with_train=False
    ) == inputs
    assert importer._validated_sequence_inputs(  # noqa: SLF001
        {"input_sha256": {}}, output_path=output, starts_with_train=True
    ) == {}
    with pytest.raises(StateError, match="exact phase-dependent"):
        importer._validated_sequence_inputs(  # noqa: SLF001
            {"input_sha256": {}}, output_path=output, starts_with_train=False
        )
    missing = dict(inputs)
    missing.pop(str(files[0].resolve()))
    with pytest.raises(StateError, match="exact phase-dependent"):
        importer._validated_sequence_inputs(  # noqa: SLF001
            {"input_sha256": missing}, output_path=output, starts_with_train=False
        )
    swapped = dict(inputs)
    best_key, last_key = str(files[1].resolve()), str(files[2].resolve())
    swapped[best_key], swapped[last_key] = swapped[last_key], swapped[best_key]
    with pytest.raises(StateError, match="digest"):
        importer._validated_sequence_inputs(  # noqa: SLF001
            {"input_sha256": swapped}, output_path=output, starts_with_train=False
        )
