"""Portable validation for one completed cross-host reassigned sequence.

The importer deliberately consumes hashes and structured observations, never
remote paths.  It is terminal-only and is not a replacement scheduler.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from .schema import CampaignSpec, JobSpec
from .state import JobState, StateError

EVIDENCE_VERSION = 1
AUTOATTACK_AMENDMENT_PATH = "docs/experiments/0002-autoattack-provenance-amendment.json"
AUTOATTACK_AMENDMENT_SHA256 = "ef4ae85acc05c50c68bcd891be9acdb43d9a391830c37f3bf590f030779dc014"
AUTOATTACK_AMENDMENT_RESULTS = frozenset(
    {
        ("hamster", "92b48f7842201e85051d5a61e6febd4b7acce62a0fbc3855b053b7896360ead7"),
        ("hamster", "5c0224c9bab73ec57f1c779cf2a9f0fb76fd1fa49564d40fc05b2cf55b4fcf79"),
        ("hamster", "fa8e5b87b1f6acf21cc52b4b0709eafefba89d1e0f172cecfc42793d22f6b44e"),
        ("ferret", "a5868f1f5ff0d4f97115b9769a02ce2829adf3662e0c927a9effcfa2b6d1ef0f"),
        ("ferret", "a0d2d4b1009c053426b30fe035081c9afa3d520886aac04c9bae3c7c2ef5eebe"),
        ("ferret", "7311fe69b4206281f0f0d438d1b1ec51a0ea95e05a940dcf32b6b9c43a7b4eab"),
        ("ferret", "7b1e328727bf980f1ce3148b57534ad6b8da51bd63a093fd376e6856ba687eac"),
        ("ferret", "56f996236280322668512d02985db6ff15f0d8079c61aa35b89aacdc32c42094"),
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_PHASES = frozenset({"train", "pgd", "autoattack"})


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def required_phases(job: JobSpec) -> tuple[str, ...]:
    return ("train", "pgd", "autoattack") if job.phases.autoattack is not None else ("train", "pgd")


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise StateError(f"terminal reassignment {label} must be an object")
    return value


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise StateError(f"terminal reassignment {label} must be a lowercase SHA-256")
    return value


def _digest_mapping(value: object, label: str, *, allow_empty: bool = False) -> dict[str, str]:
    mapping = _object(value, label)
    valid = all(isinstance(key, str) and key and _SHA256.fullmatch(item) for key, item in mapping.items())
    if (not mapping and not allow_empty) or not valid:
        raise StateError(f"terminal reassignment {label} must contain named SHA-256 digests")
    return dict(mapping)


def _outer_exit(value: object, *, sequence_source_job_id: str, git_sha: str) -> dict[str, Any]:
    record = _object(value, "outer exit")
    if set(record) != {
        "version",
        "run_id",
        "git_sha",
        "wrapper_pid",
        "phase_argv_digest",
        "exit_code",
        "finished_at",
        "error",
    }:
        raise StateError("terminal reassignment outer exit fields are not exact")
    if (
        record["version"] != 1
        or record["exit_code"] != 0
        or record["error"] is not None
        or record["run_id"] != f"reassigned-{sequence_source_job_id}"
        or record["git_sha"] != git_sha
        or not isinstance(record["wrapper_pid"], int)
        or isinstance(record["wrapper_pid"], bool)
        or record["wrapper_pid"] <= 0
        or not isinstance(record["finished_at"], str)
        or not record["finished_at"]
    ):
        raise StateError("terminal reassignment outer exit identity or success is invalid")
    _sha(record["phase_argv_digest"], "outer exit argv digest")
    return record


def _sequence_events(value: object, *, allowed_phases: tuple[str, ...]) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    if not isinstance(value, list) or not value:
        raise StateError("terminal reassignment phase events must be a non-empty ordered list")
    completed: list[str] = []
    expected_start: str | None = None
    normalized: list[dict[str, Any]] = []
    for event in value:
        item = _object(event, "phase event")
        phase = item.get("phase")
        if not isinstance(phase, str) or phase not in allowed_phases:
            raise StateError("terminal reassignment phase event is not a required phase")
        if item.get("event") == "started":
            if set(item) != {"at", "event", "phase"} or expected_start is not None:
                raise StateError("terminal reassignment phase start sequence is invalid")
            expected_start = phase
        elif item.get("event") == "finished":
            if (
                set(item) != {"at", "event", "phase", "exit_code"}
                or expected_start != phase
                or item.get("exit_code") != 0
            ):
                raise StateError("terminal reassignment phase finish sequence is invalid")
            if phase in completed:
                raise StateError("terminal reassignment phase was completed more than once")
            completed.append(phase)
            expected_start = None
        else:
            raise StateError("terminal reassignment phase event kind is invalid")
        if not isinstance(item.get("at"), str) or not item["at"]:
            raise StateError("terminal reassignment phase event timestamp is invalid")
        normalized.append(item)
    if expected_start is not None:
        raise StateError("terminal reassignment phase event sequence is incomplete")
    return normalized, tuple(completed)


def _document_digest(document: Mapping[str, Any]) -> str:
    return canonical_json_sha256({key: value for key, value in document.items() if key != "evidence_sha256"})


def _validated_amendment(
    value: object,
    label: str,
    *,
    result_sha256: str,
    execution_host: str,
) -> dict[str, str]:
    amendment = _object(value, label)
    if amendment != {
        "path": AUTOATTACK_AMENDMENT_PATH,
        "sha256": AUTOATTACK_AMENDMENT_SHA256,
    }:
        raise StateError(f"terminal reassignment {label} is not the pinned amendment")
    if (execution_host, result_sha256) not in AUTOATTACK_AMENDMENT_RESULTS:
        raise StateError(f"terminal reassignment {label} does not attest this result on this host")
    return amendment


def parse_terminal_evidence(value: object, *, spec: CampaignSpec, campaign: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one evidence document without resolving any source-host path."""
    document = _object(value, "document")
    required = {
        "version",
        "campaign_id",
        "campaign_identity_sha256",
        "source_host",
        "execution_host",
        "execution_gpu",
        "execution_gpu_uuid",
        "scientific_git_sha",
        "runtime_git_sha",
        "evidence_sha256",
        "job",
    }
    if set(document) != required or document["version"] != EVIDENCE_VERSION:
        raise StateError("terminal reassignment document fields or version are invalid")
    if document["campaign_id"] != spec.campaign_id or document["campaign_identity_sha256"] != campaign.get(
        "identity_sha256"
    ):
        raise StateError("terminal reassignment campaign identity does not match the state store")
    source_host = document["source_host"]
    execution_host = document["execution_host"]
    if source_host not in spec.hosts or execution_host not in spec.hosts:
        raise StateError("terminal reassignment source or execution host is not in the campaign")
    if (
        not isinstance(document["execution_gpu"], int)
        or document["execution_gpu"] not in spec.hosts[execution_host].gpus
    ):
        raise StateError("terminal reassignment execution GPU is not owned by the execution host")
    execution_gpu_uuid = document["execution_gpu_uuid"]
    if not isinstance(execution_gpu_uuid, str) or not execution_gpu_uuid.startswith("GPU-"):
        raise StateError("terminal reassignment execution GPU UUID is invalid")
    if document["scientific_git_sha"] != spec.git_sha:
        raise StateError("terminal reassignment scientific Git SHA does not match the campaign")
    runtime_git_sha = document["runtime_git_sha"]
    if runtime_git_sha is not None and (not isinstance(runtime_git_sha, str) or not _SHA1.fullmatch(runtime_git_sha)):
        raise StateError("terminal reassignment runtime Git SHA must be a full SHA or null for legacy evidence")

    entry = _object(document["job"], "job entry")
    allowed = {
        "job_id",
        "sequence_source_job_id",
        "expected_state",
        "expected_revision",
        "required_phases",
        "outer_exit",
        "phase_events",
        "sequence_digests",
        "prior_phase_digests",
        "evidence_digests",
        "auxiliary_autoattack",
        "posthoc_autoattack_attestation",
    }
    minimum = allowed - {"auxiliary_autoattack", "posthoc_autoattack_attestation"}
    if set(entry) - allowed or not minimum.issubset(entry):
        raise StateError("terminal reassignment job entry fields are invalid")
    job_id = entry["job_id"]
    by_id = {job.id: job for job in spec.jobs}
    if not isinstance(job_id, str) or job_id not in by_id:
        raise StateError("terminal reassignment job identity is invalid")
    job = by_id[job_id]
    if job.host != source_host:
        raise StateError("terminal reassignment source host does not own the job")
    if not isinstance(entry["expected_state"], str) or entry["expected_state"] not in {
        state.value for state in JobState
    }:
        raise StateError("terminal reassignment expected state is invalid")
    if (
        not isinstance(entry["expected_revision"], int)
        or isinstance(entry["expected_revision"], bool)
        or entry["expected_revision"] < 0
    ):
        raise StateError("terminal reassignment expected revision is invalid")
    phases = required_phases(job)
    if not isinstance(entry["required_phases"], list) or tuple(entry["required_phases"]) != phases:
        raise StateError("terminal reassignment required phases do not match the campaign")
    sequence_source_job_id = entry["sequence_source_job_id"]
    if not isinstance(sequence_source_job_id, str) or (
        sequence_source_job_id != job_id
        and not re.fullmatch(
            re.escape(job_id) + r"-(?:extra-eval|successor-eval|retry-[A-Za-z0-9._-]+)", sequence_source_job_id
        )
    ):
        raise StateError("terminal reassignment sequence source job identity is invalid")
    outer_exit = _outer_exit(
        entry["outer_exit"], sequence_source_job_id=sequence_source_job_id, git_sha=str(spec.git_sha)
    )
    events, completed = _sequence_events(entry["phase_events"], allowed_phases=phases)
    sequence_digests = _digest_mapping(entry["sequence_digests"], "sequence digests")
    if {"sequence_spec", "sequence_completion", "outer_exit", "phase_events"} - set(sequence_digests):
        raise StateError("terminal reassignment sequence evidence is incomplete")
    prior = _digest_mapping(entry["prior_phase_digests"], "prior phase digests", allow_empty=True)
    if set(prior) != set(phases) - set(completed):
        raise StateError("terminal reassignment prior phase evidence does not cover exactly the non-reassigned phases")
    evidence_digests = _digest_mapping(entry["evidence_digests"], "evidence digests")
    if {"best_checkpoint", "last_checkpoint", "evaluation_results"} - set(evidence_digests):
        raise StateError("terminal reassignment evidence lacks checkpoint/result digests")
    posthoc = entry.get("posthoc_autoattack_attestation")
    if posthoc is not None:
        if job.phases.autoattack is None:
            raise StateError("legacy AutoAttack attestation cannot be attached to a PGD-only job")
        attestation = _object(posthoc, "post-hoc AutoAttack attestation")
        if (
            set(attestation) != {"posthoc_attested", "evaluation_results_sha256", "amendment"}
            or attestation.get("posthoc_attested") is not True
        ):
            raise StateError("terminal reassignment post-hoc AutoAttack attestation is invalid")
        if attestation.get("evaluation_results_sha256") != evidence_digests["evaluation_results"]:
            raise StateError("terminal reassignment legacy AutoAttack attestation is not bound to the result digest")
        _validated_amendment(
            attestation.get("amendment"),
            "legacy AutoAttack amendment",
            result_sha256=evidence_digests["evaluation_results"],
            execution_host=execution_host,
        )
    auxiliary = entry.get("auxiliary_autoattack")
    if auxiliary is not None:
        if job.phases.autoattack is not None:
            raise StateError("auxiliary AutoAttack is only valid for a PGD-only job")
        auxiliary_object = _object(auxiliary, "auxiliary AutoAttack")
        if set(auxiliary_object) != {
            "sequence_source_job_id",
            "execution_host",
            "execution_gpu",
            "execution_gpu_uuid",
            "runtime_git_sha",
            "outer_exit",
            "phase_events",
            "sequence_digests",
            "evidence_digests",
            "posthoc_autoattack_attestation",
        }:
            raise StateError("terminal reassignment auxiliary AutoAttack fields are not exact")
        auxiliary_source_job_id = auxiliary_object["sequence_source_job_id"]
        if auxiliary_source_job_id != f"{job_id}-extra-autoattack":
            raise StateError("terminal reassignment auxiliary AutoAttack source job identity is invalid")
        auxiliary_host = auxiliary_object["execution_host"]
        auxiliary_gpu = auxiliary_object["execution_gpu"]
        if (
            auxiliary_host not in spec.hosts
            or not isinstance(auxiliary_gpu, int)
            or isinstance(auxiliary_gpu, bool)
            or auxiliary_gpu not in spec.hosts[auxiliary_host].gpus
            or not isinstance(auxiliary_object["execution_gpu_uuid"], str)
            or not auxiliary_object["execution_gpu_uuid"].startswith("GPU-")
        ):
            raise StateError("terminal reassignment auxiliary AutoAttack execution identity is invalid")
        auxiliary_runtime = auxiliary_object["runtime_git_sha"]
        if auxiliary_runtime is not None and (
            not isinstance(auxiliary_runtime, str) or not _SHA1.fullmatch(auxiliary_runtime)
        ):
            raise StateError("terminal reassignment auxiliary AutoAttack runtime Git SHA is invalid")
        _outer_exit(
            auxiliary_object["outer_exit"],
            sequence_source_job_id=auxiliary_source_job_id,
            git_sha=str(spec.git_sha),
        )
        _, auxiliary_completed = _sequence_events(auxiliary_object["phase_events"], allowed_phases=("autoattack",))
        if auxiliary_completed != ("autoattack",):
            raise StateError("terminal reassignment auxiliary AutoAttack was not successfully completed")
        auxiliary_sequence_digests = _digest_mapping(
            auxiliary_object["sequence_digests"], "auxiliary AutoAttack sequence digests"
        )
        if {"sequence_spec", "sequence_completion", "outer_exit", "phase_events"} - set(auxiliary_sequence_digests):
            raise StateError("terminal reassignment auxiliary AutoAttack sequence evidence is incomplete")
        auxiliary_digests = _digest_mapping(
            auxiliary_object["evidence_digests"], "auxiliary AutoAttack evidence digests"
        )
        if set(auxiliary_digests) != {"best_checkpoint", "last_checkpoint", "evaluation_results"}:
            raise StateError("terminal reassignment auxiliary AutoAttack result digest is invalid")
        auxiliary_attestation = _object(
            auxiliary_object["posthoc_autoattack_attestation"], "auxiliary AutoAttack attestation"
        )
        if (
            set(auxiliary_attestation) != {
                "posthoc_attested",
                "evaluation_results_sha256",
                "amendment",
            }
            or auxiliary_attestation.get("posthoc_attested") is not True
            or auxiliary_attestation.get("evaluation_results_sha256") != auxiliary_digests["evaluation_results"]
        ):
            raise StateError("terminal reassignment auxiliary AutoAttack attestation is invalid")
        _validated_amendment(
            auxiliary_attestation["amendment"],
            "auxiliary AutoAttack amendment",
            result_sha256=auxiliary_digests["evaluation_results"],
            execution_host=auxiliary_host,
        )
    evidence_sha256 = _sha(document["evidence_sha256"], "document evidence digest")
    if evidence_sha256 != _document_digest(document):
        raise StateError("terminal reassignment evidence digest does not match document")
    return {
        "job_id": job_id,
        "sequence_source_job_id": sequence_source_job_id,
        "expected_state": entry["expected_state"],
        "expected_revision": entry["expected_revision"],
        "source_host": source_host,
        "execution_host": execution_host,
        "execution_gpu": document["execution_gpu"],
        "execution_gpu_uuid": execution_gpu_uuid,
        "scientific_git_sha": document["scientific_git_sha"],
        "runtime_git_sha": runtime_git_sha,
        "required_phases": list(phases),
        "outer_exit": outer_exit,
        "phase_events": events,
        "sequence_digests": sequence_digests,
        "prior_phase_digests": prior,
        "evidence_digests": evidence_digests,
        "evidence_sha256": evidence_sha256,
        "auxiliary_autoattack": auxiliary,
        "posthoc_autoattack_attestation": posthoc,
        "autoattack_status": "completed" if job.phases.autoattack is not None else "not_requested",
    }
