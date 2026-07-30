#!/usr/bin/env python3
"""Collect or import one portable terminal reassignment evidence document.

Collection reads only immutable terminal records and prints a normalized JSON
document. Import is dry-run by default; ``--apply`` is explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ard.campaign.reassignment import (
    AUTOATTACK_AMENDMENT_PATH,
    AUTOATTACK_AMENDMENT_SHA256,
    canonical_json_sha256,
    parse_terminal_evidence,
    required_phases,
)
from ard.campaign.schema import CampaignError, CampaignSpec, JobSpec, bind_git_sha, load_campaign
from ard.campaign.state import CampaignStateStore, JobState, StateError, _atomic_json


def _amendment(path: Path, *, result_sha256: str, execution_host: str) -> dict[str, str]:
    path = path.resolve()
    expected_path = Path(__file__).resolve().parents[2] / AUTOATTACK_AMENDMENT_PATH
    if path != expected_path or _sha256(path, "AutoAttack amendment") != AUTOATTACK_AMENDMENT_SHA256:
        raise StateError("AutoAttack amendment is not the checked-in pinned amendment")
    value = _read_object(path, "AutoAttack amendment")
    installed = value.get("observed_installation")
    rows = value.get("immutable_results")
    if not isinstance(installed, dict) or not isinstance(rows, list):
        raise StateError("AutoAttack amendment has no installed identity or immutable results")
    if installed.get("vcs_commit") != "a39220048b3c9f2cca9a4d3a54604793c68eca7e" or installed.get(
        "python_source_sha256"
    ) != "e74d6dab0e34faf840f1bdfe0f77e9ddcc5f753a7426cbaa54b11bf17f896487":
        raise StateError("AutoAttack amendment source identity is not the pinned installation")
    if not any(
        isinstance(row, dict)
        and row.get("evaluation_results_sha256") == result_sha256
        and row.get("execution_host") == execution_host
        for row in rows
    ):
        raise StateError("AutoAttack amendment does not attest this immutable result on this execution host")
    return {"path": AUTOATTACK_AMENDMENT_PATH, "sha256": AUTOATTACK_AMENDMENT_SHA256}


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StateError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise StateError(f"{label} must be a JSON object: {path}")
    return value


def _sha256(path: Path, label: str) -> str:
    if not path.is_file():
        raise StateError(f"{label} is absent: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _events(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise StateError(f"sequence events are unreadable: {path}") from exc
    if not all(isinstance(row, dict) for row in rows):
        raise StateError("sequence events must be JSON objects")
    return rows


def _parse_phase_exit(value: str, *, spec: CampaignSpec, job_id: str) -> tuple[str, Path]:
    phase, separator, raw_path = value.partition("=")
    if separator != "=" or phase not in {"train", "pgd", "autoattack"} or not raw_path:
        raise StateError("--prior-phase-exit must be phase=/absolute/or/relative/path")
    path = Path(raw_path).resolve()
    exit_record = _read_object(path, f"{phase} controller exit")
    if (
        exit_record.get("exit_code") != 0
        or exit_record.get("run_id") != job_id
        or exit_record.get("git_sha") != spec.git_sha
    ):
        raise StateError(f"{phase} controller exit does not prove the canonical successful phase")
    return phase, path


def _sequence_evidence(
    sequence_dir: Path, *, spec: CampaignSpec, job_id: str
) -> tuple[dict[str, Any], dict[str, str], str | None]:
    sequence_dir = sequence_dir.resolve()
    spec_path = sequence_dir / "sequence-spec.json"
    completion_path = sequence_dir / "sequence-completion.json"
    exit_path = sequence_dir / "exit.json"
    events_path = sequence_dir / "sequence-events.jsonl"
    sequence = _read_object(spec_path, "sequence spec")
    completion = _read_object(completion_path, "sequence completion")
    outer_exit = _read_object(exit_path, "reassigned outer exit")
    if completion.get("status") != "completed":
        raise StateError("reassigned sequence completion is not completed")
    source_job_id = sequence.get("source_job_id")
    if not isinstance(source_job_id, str):
        raise StateError("reassigned sequence spec lacks source_job_id")
    if sequence.get("git_sha") != spec.git_sha or outer_exit.get("git_sha") != spec.git_sha:
        raise StateError("reassigned sequence Git SHA does not match campaign")
    return (
        {
            "sequence_source_job_id": source_job_id,
            "source_host": sequence.get("source_host"),
            "execution_host": sequence.get("destination_host"),
            "execution_gpu": sequence.get("destination_gpu"),
            "execution_gpu_uuid": sequence.get("destination_gpu_uuid"),
            "outer_exit": outer_exit,
            "phase_events": _events(events_path),
        },
        {
            "sequence_spec": _sha256(spec_path, "sequence spec"),
            "sequence_completion": _sha256(completion_path, "sequence completion"),
            "outer_exit": _sha256(exit_path, "reassigned outer exit"),
            "phase_events": _sha256(events_path, "sequence events"),
        },
        str(sequence.get("runtime_git_sha")) if sequence.get("runtime_git_sha") is not None else None,
    )


def _expected_argv(raw: tuple[str, ...], *, repository: str, config: str, output: str, python: str) -> list[str]:
    substitutions = {"{PYTHON}": python, "{CONFIG_PATH}": str(Path(repository) / config), "{JOB_OUTPUT_DIR}": output}
    result: list[str] = []
    for token in raw:
        if token == "{CONFIG_PATH}":
            result.append(substitutions[token])
        elif token == "{JOB_OUTPUT_DIR}":
            result.append(output)
        elif token.startswith("{JOB_OUTPUT_DIR}/"):
            result.append(str(Path(output) / token.removeprefix("{JOB_OUTPUT_DIR}/")))
        elif token == "{PYTHON}":
            result.append(python)
        else:
            result.append(token)
    return result


def _validated_sequence_inputs(
    sequence_spec: dict[str, Any], *, output_path: Path, starts_with_train: bool
) -> dict[str, str]:
    value = sequence_spec.get("input_sha256")
    if not isinstance(value, dict) or not all(
        isinstance(path, str) and isinstance(digest, str) for path, digest in value.items()
    ):
        raise StateError("sequence input hash mapping is invalid")
    expected_paths = (
        ()
        if starts_with_train
        else (
            output_path / "resolved_config.yaml",
            output_path / "best.pt",
            output_path / "last.pt",
            output_path / "run-bundle" / "manifest.json",
        )
    )
    expected = {str(path.resolve()) for path in expected_paths}
    normalized = {str(Path(path).resolve()): digest for path, digest in value.items()}
    if set(normalized) != expected:
        raise StateError("sequence input hash mapping does not contain the exact phase-dependent input set")
    for path, expected_digest in normalized.items():
        if _sha256(Path(path), "sequence input") != expected_digest:
            raise StateError("sequence input digest does not match the supplied input")
    return normalized


def _auxiliary_autoattack(
    args: argparse.Namespace, *, spec: CampaignSpec, job: JobSpec, output_path: Path
) -> dict[str, Any]:
    if args.auxiliary_sequence_dir is None:
        raise StateError("PGD-only Student evidence requires --auxiliary-sequence-dir")
    if args.autoattack_amendment is None:
        raise StateError("auxiliary legacy AutoAttack requires --autoattack-amendment")
    sequence, digests, runtime_git_sha = _sequence_evidence(
        args.auxiliary_sequence_dir, spec=spec, job_id=job.id
    )
    sequence_spec = _read_object(
        args.auxiliary_sequence_dir.resolve() / "sequence-spec.json", "auxiliary sequence spec"
    )
    if sequence["source_host"] != job.host or sequence_spec.get("output") != str(output_path):
        raise StateError("auxiliary AutoAttack source host or output does not match the campaign job")
    commands = sequence_spec.get("commands")
    repository = sequence_spec.get("repository")
    if (
        not isinstance(repository, str)
        or not isinstance(commands, list)
        or len(commands) != 1
        or not isinstance(commands[0], dict)
        or commands[0].get("phase") != "autoattack"
        or not isinstance(commands[0].get("argv"), list)
    ):
        raise StateError("auxiliary AutoAttack command is malformed")
    argv = commands[0]["argv"]
    expected = [
        argv[0],
        "-m",
        "ard.cli.evaluate",
        "--config",
        str(Path(repository) / job.config),
        "--checkpoint-dir",
        str(output_path),
        "--output",
        str(output_path / "evaluation-autoattack"),
        "--allow-autoattack",
        "evaluation.autoattack=true",
    ]
    if argv != expected:
        raise StateError("auxiliary AutoAttack command does not match the saved-checkpoint evaluation contract")
    inputs = _validated_sequence_inputs(sequence_spec, output_path=output_path, starts_with_train=False)
    result_path = output_path / "evaluation-autoattack" / "evaluation-results.json"
    result_sha256 = _sha256(result_path, "auxiliary AutoAttack result")
    best_sha256 = _sha256(output_path / "best.pt", "auxiliary AutoAttack best checkpoint")
    last_sha256 = _sha256(output_path / "last.pt", "auxiliary AutoAttack last checkpoint")
    if (
        inputs[str((output_path / "best.pt").resolve())] != best_sha256
        or inputs[str((output_path / "last.pt").resolve())] != last_sha256
    ):
        raise StateError("auxiliary AutoAttack checkpoint input hashes do not match collected checkpoints")
    return {
        "sequence_source_job_id": sequence["sequence_source_job_id"],
        "execution_host": sequence["execution_host"],
        "execution_gpu": sequence["execution_gpu"],
        "execution_gpu_uuid": sequence["execution_gpu_uuid"],
        "runtime_git_sha": runtime_git_sha,
        "outer_exit": sequence["outer_exit"],
        "phase_events": sequence["phase_events"],
        "sequence_digests": digests,
        "evidence_digests": {
            "best_checkpoint": best_sha256,
            "last_checkpoint": last_sha256,
            "evaluation_results": result_sha256,
        },
        "posthoc_autoattack_attestation": {
            "posthoc_attested": True,
            "evaluation_results_sha256": result_sha256,
            "amendment": _amendment(
                args.autoattack_amendment,
                result_sha256=result_sha256,
                execution_host=str(sequence["execution_host"]),
            ),
        },
    }


def collect(args: argparse.Namespace, *, spec: CampaignSpec, store: CampaignStateStore) -> dict[str, Any]:
    if args.job_id is None or args.output is None or args.execution_host is None or args.execution_gpu is None:
        raise StateError("collection requires --job-id, --output, --execution-host, and --execution-gpu")
    by_id = {job.id: job for job in spec.jobs}
    job = by_id.get(args.job_id)
    if job is None:
        raise StateError("collection job is absent from campaign")
    sequence, sequence_digests, runtime_git_sha = _sequence_evidence(args.sequence_dir, spec=spec, job_id=job.id)
    if (
        sequence["source_host"] != job.host
        or sequence["execution_host"] != args.execution_host
        or sequence["execution_gpu"] != args.execution_gpu
        or not isinstance(sequence["execution_gpu_uuid"], str)
        or not sequence["execution_gpu_uuid"].startswith("GPU-")
    ):
        raise StateError("sequence source/destination host or GPU identity does not match collection arguments")
    sequence_spec = _read_object(args.sequence_dir.resolve() / "sequence-spec.json", "sequence spec")
    output = str(args.output.resolve())
    if sequence_spec.get("output") != output:
        raise StateError("sequence output does not match the supplied canonical output path")
    repository = sequence_spec.get("repository")
    commands = sequence_spec.get("commands")
    if not isinstance(repository, str) or not isinstance(commands, list):
        raise StateError("sequence spec lacks repository or commands")
    configured = {"train": job.phases.train, "pgd": job.phases.pgd_evaluate, "autoattack": job.phases.autoattack}
    for observed in commands:
        if not isinstance(observed, dict) or set(observed) != {"phase", "argv"}:
            raise StateError("sequence command is malformed")
        phase, argv = observed["phase"], observed["argv"]
        if phase not in configured or configured[phase] is None or not isinstance(argv, list) or not argv:
            raise StateError("sequence command phase is absent from the campaign job")
        if not all(isinstance(token, str) for token in argv):
            raise StateError("sequence command argv is invalid")
        expected = _expected_argv(
            configured[phase], repository=repository, config=job.config, output=output, python=argv[0]
        )
        if argv != expected:
            raise StateError("sequence command does not exactly match the campaign job after substitution")
    phase_events = sequence["phase_events"]
    completed = {str(event.get("phase")) for event in phase_events if event.get("event") == "finished"}
    prior: dict[str, str] = {}
    for raw in args.prior_phase_exit:
        phase, path = _parse_phase_exit(raw, spec=spec, job_id=job.id)
        if phase in prior:
            raise StateError("a prior phase exit was provided more than once")
        prior[phase] = _sha256(path, f"{phase} controller exit")
    phases = list(required_phases(job))
    inputs = _validated_sequence_inputs(
        sequence_spec,
        output_path=args.output.resolve(),
        starts_with_train=bool(commands and isinstance(commands[0], dict) and commands[0].get("phase") == "train"),
    )
    if set(prior) != set(phases) - completed:
        raise StateError("provide exactly one controller exit digest for each phase absent from reassigned events")
    output_path = args.output.resolve()
    evaluation = output_path / ("evaluation-autoattack" if job.phases.autoattack is not None else "evaluation-pgd")
    local_job = store.job(job.id)
    expected_state = args.expected_state if args.expected_state is not None else local_job["state"]
    expected_revision = args.expected_revision if args.expected_revision is not None else local_job["revision"]
    entry: dict[str, Any] = {
        "job_id": job.id,
        "sequence_source_job_id": sequence["sequence_source_job_id"],
        "expected_state": expected_state,
        "expected_revision": expected_revision,
        "required_phases": phases,
        "outer_exit": sequence["outer_exit"],
        "phase_events": phase_events,
        "sequence_digests": sequence_digests,
        "prior_phase_digests": prior,
        "evidence_digests": {
            "best_checkpoint": _sha256(output_path / "best.pt", "best checkpoint"),
            "last_checkpoint": _sha256(output_path / "last.pt", "last checkpoint"),
            "evaluation_results": _sha256(evaluation / "evaluation-results.json", "evaluation results"),
        },
    }
    if inputs and (
        inputs[str((output_path / "best.pt").resolve())] != entry["evidence_digests"]["best_checkpoint"]
        or inputs[str((output_path / "last.pt").resolve())] != entry["evidence_digests"]["last_checkpoint"]
    ):
        raise StateError("sequence checkpoint input hashes do not match collected checkpoint evidence")
    if job.phases.autoattack is None and args.auxiliary_sequence_dir is not None:
        entry["auxiliary_autoattack"] = _auxiliary_autoattack(
            args, spec=spec, job=job, output_path=output_path
        )
    if job.phases.autoattack is not None:
        results_path = evaluation / "evaluation-results.json"
        try:
            raw_results = json.loads(results_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StateError(f"evaluation results are unreadable: {results_path}") from exc
        if (
            isinstance(raw_results, list)
            and raw_results
            and all(
                isinstance(item, dict)
                and isinstance(item.get("autoattack"), dict)
                and item["autoattack"].get("version") == "unknown"
                for item in raw_results
            )
        ):
            if args.autoattack_amendment is None:
                raise StateError("legacy AutoAttack version=unknown requires --autoattack-amendment")
            entry["posthoc_autoattack_attestation"] = {
                "posthoc_attested": True,
                "evaluation_results_sha256": entry["evidence_digests"]["evaluation_results"],
                "amendment": _amendment(
                    args.autoattack_amendment,
                    result_sha256=entry["evidence_digests"]["evaluation_results"],
                    execution_host=args.execution_host,
                ),
            }
    document = {
        "version": 1,
        "campaign_id": spec.campaign_id,
        "campaign_identity_sha256": store.campaign()["identity_sha256"],
        "source_host": job.host,
        "execution_host": args.execution_host,
        "execution_gpu": args.execution_gpu,
        "execution_gpu_uuid": sequence["execution_gpu_uuid"],
        "scientific_git_sha": spec.git_sha,
        "runtime_git_sha": runtime_git_sha,
        "job": entry,
    }
    document["evidence_sha256"] = canonical_json_sha256(document)
    return document


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--sha", help="bind a template campaign to its immutable scientific Git SHA")
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--state-host", choices=("hamster", "ferret"), required=True)
    parser.add_argument("--evidence", type=Path, action="append", default=[])
    parser.add_argument("--sequence-dir", type=Path)
    parser.add_argument("--job-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execution-host", choices=("hamster", "ferret"))
    parser.add_argument("--execution-gpu", type=int)
    parser.add_argument("--expected-state", choices=tuple(state.value for state in JobState))
    parser.add_argument("--expected-revision", type=int)
    parser.add_argument("--prior-phase-exit", action="append", default=[])
    parser.add_argument("--autoattack-amendment", type=Path)
    parser.add_argument("--auxiliary-sequence-dir", type=Path)
    parser.add_argument("--collect-only", action="store_true", help="print portable evidence without importing it")
    parser.add_argument("--evidence-output", type=Path, help="atomically save collected evidence without clobbering")
    parser.add_argument("--apply", action="store_true", help="perform the otherwise dry-run-only import")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        spec = load_campaign(args.campaign.resolve())
        if args.sha is not None:
            spec = bind_git_sha(spec, args.sha)
        if spec.git_sha is None:
            raise StateError("terminal reassignment import requires a fixed campaign Git SHA or --sha")
        store = CampaignStateStore(args.state_root.resolve())
        if bool(args.evidence) == (args.sequence_dir is not None):
            raise StateError("provide either one or more --evidence files or one --sequence-dir")
        if args.collect_only and (args.sequence_dir is None or args.apply):
            raise StateError("--collect-only requires --sequence-dir and cannot be combined with --apply")
        evidences = (
            [_read_object(path.resolve(), "terminal reassignment evidence") for path in args.evidence]
            if args.evidence
            else [collect(args, spec=spec, store=store)]
        )
        evidence = evidences[0]
        if any(item.get("source_host") != args.state_host for item in evidences):
            raise StateError("terminal evidence job ownership does not match --state-host")
        if args.sequence_dir is not None:
            parse_terminal_evidence(evidence, spec=spec, campaign=store.campaign())
        if args.evidence_output is not None:
            if args.sequence_dir is None:
                raise StateError("--evidence-output is valid only while collecting from --sequence-dir")
            evidence_output = args.evidence_output.resolve()
            if evidence_output.exists() and _read_object(evidence_output, "existing evidence output") != evidence:
                raise StateError(f"evidence output already exists with different content: {evidence_output}")
            if not evidence_output.exists():
                _atomic_json(evidence_output, evidence)
        result = (
            {"status": "collected"}
            if args.collect_only
            else store.import_terminal_reassignments(evidences, spec=spec, dry_run=not args.apply)
        )
    except (CampaignError, StateError) as exc:
        _parser().error(str(exc))
    print(json.dumps({"evidence": evidence if len(evidences) == 1 else evidences, "result": result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
