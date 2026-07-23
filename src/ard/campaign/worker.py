"""One bounded reconciliation pass for a statically assigned host queue."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .gpu import GPUAdmission, GPUSnapshot, admit, inventory
from .launcher import LaunchError, argv_digest, launch_phase, phase_is_live, read_exit_record, read_launch_record
from .schema import CampaignError, CampaignSpec, JobSpec, campaign_identity_sha256, effective_wandb_run_id
from .state import TERMINAL_JOB_STATES, CampaignStateStore, FileLock, JobState, _atomic_json, _read_json


class WorkerError(CampaignError):
    pass


_ACTIVE = frozenset({JobState.TRAINING, JobState.PGD_EVALUATION, JobState.AUTOATTACK})
_PHASE_STATE = {"train": JobState.TRAINING, "pgd": JobState.PGD_EVALUATION, "autoattack": JobState.AUTOATTACK}
PhaseSuccessValidator = Callable[[JobSpec, str, Path], str | None]


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _nonempty_json(path: Path) -> object | None:
    try:
        if path.stat().st_size == 0:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _validate_bundle(bundle: Path) -> tuple[dict[str, object], str | None]:
    required = [bundle / "manifest.json", bundle / "completion.json", bundle / "error-marker.txt"]
    if any(not path.is_file() or path.stat().st_size == 0 for path in required):
        return {}, "terminal bundle is missing or empty"
    manifest = _nonempty_json(bundle / "manifest.json")
    completion = _nonempty_json(bundle / "completion.json")
    if not isinstance(manifest, dict) or not isinstance(completion, dict):
        return {}, "terminal bundle metadata is unreadable"
    if completion.get("status") != "completed":
        return {}, "completion marker is not completed"
    mode = manifest.get("tracking_mode")
    status = manifest.get("status")
    if mode not in {"online", "offline", "offline_sync"}:
        return {}, "manifest tracking mode is missing or invalid"
    if status != "completed":
        return {}, f"{mode} manifest is not completed"
    if mode == "offline_sync" and manifest.get("sync_state") != "synced":
        return {}, "offline-sync manifest is not durably synced"
    marker = (bundle / "error-marker.txt").read_text(encoding="utf-8", errors="replace").lower()
    if "no application error" not in marker:
        return {}, "terminal bundle records an application error"
    return manifest, None


def _validate_train_metrics(bundle: Path) -> str | None:
    metrics = bundle / "metrics.jsonl"
    if not metrics.is_file() or metrics.stat().st_size == 0:
        return "training metrics are missing or empty"
    try:
        rows = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        return f"training metrics are unreadable: {exc}"
    if not rows or not all(isinstance(row, dict) for row in rows):
        return "training metrics are empty or malformed"
    observed = False
    for row in rows:
        assert isinstance(row, dict)
        for key in (
            "train_loss",
            "train_clean_accuracy",
            "train_robust_accuracy",
            "val_clean_accuracy",
            "val_pgd_accuracy",
        ):
            if key in row:
                observed = True
                if not _finite(row[key]):
                    return f"training metric {key} is non-finite"
    return None if observed else "training metrics contain no scientific scalar"


def _validate_evaluation_results(path: Path, *, training_output: Path, phase: str, job: JobSpec) -> str | None:
    from ard.config import load_config
    from ard.config.loader import resolved_config_dict
    from ard.engine import config_digest

    results = _nonempty_json(path)
    if not isinstance(results, list) or len(results) != 2 or not all(isinstance(item, dict) for item in results):
        return "evaluation-results.json must contain exactly two checkpoint results"
    aliases = {str(item.get("checkpoint_alias")) for item in results}
    filenames = {str(item.get("checkpoint_filename")) for item in results}
    if aliases != {"best", "last"} or filenames != {"best.pt", "last.pt"}:
        return "evaluation results must contain exactly best.pt and last.pt"
    training_manifest = _nonempty_json(training_output / "run-bundle" / "manifest.json")
    if not isinstance(training_manifest, dict):
        return "training manifest is missing or invalid"
    try:
        training_config = load_config(training_output / "resolved_config.yaml")
        evaluation_config = load_config(path.parent / "resolved_evaluation_config.yaml")
    except (OSError, ValueError) as exc:
        return f"resolved evaluation lineage is invalid: {exc}"
    selection_attack = training_config.method.selection_attack
    if selection_attack is None:
        return "resolved training selection attack is absent"
    expected = {
        "train_run_id": training_manifest.get("run_id"),
        "config_hash": config_digest(resolved_config_dict(training_config)),
        "threat_hash": selection_attack.identity_sha256(),
    }
    if training_manifest.get("config_hash") != expected["config_hash"]:
        return "training manifest config hash does not match resolved training config"
    for key, value in expected.items():
        if not isinstance(value, str) or not value or any(item.get(key) != value for item in results):
            return f"evaluation results do not match expected {key}"
    for item in results:
        teacher_identity = item.get("teacher_identity")
        if item.get("method") != job.method or not isinstance(teacher_identity, dict):
            return "evaluation method or teacher identity is invalid"
        if teacher_identity.get("registry_id") != job.teacher:
            return "evaluation teacher identity does not match the campaign job"
        if item.get("count") != 10000:
            return "evaluation result count must equal the official CIFAR-10 test size 10000"
        for key in ("clean_accuracy", "pgd_accuracy"):
            value = item.get(key)
            if not _finite(value) or not 0.0 <= float(value) <= 1.0:
                return f"evaluation {key} is not finite in [0, 1]"
        autoattack = item.get("autoattack")
        if phase != "autoattack":
            if autoattack is not None:
                return "PGD-only evaluation unexpectedly contains AutoAttack results"
            continue
        if not evaluation_config.evaluation.autoattack or not isinstance(autoattack, dict):
            return "AutoAttack phase lacks an enabled AutoAttack result"
        expected_autoattack = {
            "seed": evaluation_config.evaluation.seed,
            "norm": "Linf",
            "attack_version": "standard",
            "batch_size": evaluation_config.evaluation.autoattack_batch_size,
        }
        if any(autoattack.get(key) != value for key, value in expected_autoattack.items()):
            return "AutoAttack result identity does not match the resolved evaluation config"
        epsilon = selection_attack.epsilon_value
        if (
            epsilon is None
            or not _finite(autoattack.get("epsilon"))
            or not math.isclose(float(autoattack["epsilon"]), epsilon, rel_tol=0.0, abs_tol=1e-15)
            or not _finite(autoattack.get("autoattack_accuracy"))
            or not 0.0 <= float(autoattack["autoattack_accuracy"]) <= 1.0
            or not isinstance(autoattack.get("version"), str)
            or not autoattack["version"]
        ):
            return "AutoAttack result metrics or threat values are invalid"
        artifact = path.parent / f"autoattack-{item['checkpoint_alias']}.json"
        if _nonempty_json(artifact) != autoattack:
            return "AutoAttack result file is absent or differs from evaluation-results.json"
    return None


def default_phase_success_validator(job: JobSpec, phase: str, output_root: Path) -> str | None:
    """Verify success lineage before allowing a later scientific phase.

    The runner intentionally does not inspect metrics or reinterpret attacks. It
    only checks the terminal artifacts the existing train/evaluate contracts
    already promise to write.
    """
    training_output = output_root / job.output
    phase_output = training_output if phase == "train" else training_output / f"evaluation-{phase}"
    required = [phase_output / "run-bundle" / "manifest.json", phase_output / "run-bundle" / "completion.json"]
    if phase == "train":
        required.extend(
            [
                training_output / "resolved_config.yaml",
                training_output / "best.pt",
                training_output / "last.pt",
            ]
        )
    else:
        required.append(phase_output / "evaluation-results.json")
    missing = [str(path.relative_to(output_root)) for path in required if not path.is_file()]
    if missing:
        return "missing required phase artifacts: " + ", ".join(missing)
    bundle = phase_output / "run-bundle"
    _, failure = _validate_bundle(bundle)
    if failure is not None:
        return failure
    if phase == "train":
        return _validate_train_metrics(bundle)
    return _validate_evaluation_results(
        phase_output / "evaluation-results.json",
        training_output=training_output,
        phase=phase,
        job=job,
    )


class CampaignWorker:
    """Reconcile existing phases first, then launch at most one queued phase.

    A caller may invoke :meth:`run_once` repeatedly (for example from a detached
    host wrapper).  It is deliberately not a scheduler and never launches while
    campaign state is ``unarmed``.
    """

    def __init__(
        self,
        spec: CampaignSpec,
        state: CampaignStateStore,
        *,
        host: str,
        repository: Path,
        output_root: Path,
        inventory_provider: Callable[[], tuple[GPUSnapshot, ...]] = inventory,
        launcher: Callable[..., dict[str, Any]] = launch_phase,
        phase_success_validator: PhaseSuccessValidator = default_phase_success_validator,
        external_processes_enabled: bool | None = None,
        gpu_lock_root: Path | None = None,
    ) -> None:
        if host not in spec.hosts:
            raise WorkerError(f"host is not present in campaign: {host}")
        if spec.git_sha is None:
            raise WorkerError("a worker requires a campaign with a fixed full Git SHA")
        self.spec = spec
        self.state = state
        self.host = host
        self.repository = repository.resolve()
        self.output_root = output_root.resolve()
        self.inventory_provider = inventory_provider
        self.launcher = launcher
        self.phase_success_validator = phase_success_validator
        self.external_processes_enabled = external_processes_enabled
        self.gpu_lock_root = (gpu_lock_root or Path("/tmp/ard-campaign-gpu-locks")).resolve()
        if not self.gpu_lock_root.is_absolute():
            raise WorkerError("gpu_lock_root must be an absolute host-local path")
        self.state.initialize(spec)

    def arm(self) -> None:
        self.state.set_campaign_state("armed")

    def run_once(self) -> dict[str, str]:
        """Perform a short reconciliation pass and return state by job id."""
        self.state.assert_campaign_identity(self.spec)
        phase_finished = self._reconcile_active()
        campaign = self.state.campaign()
        if campaign["state"] != "armed":
            return {job.id: self.state.job(job.id)["state"] for job in self._host_jobs()}
        # A completed detached process gets its own durable reconciliation pass.
        # This prevents a restart from treating an unverified exit as permission
        # to immediately launch a scientific successor phase.
        if not phase_finished:
            self._launch_one_eligible_phase()
        self._maybe_stop_for_scientific_review()
        return {job.id: self.state.job(job.id)["state"] for job in self._host_jobs()}

    def _host_jobs(self) -> tuple[JobSpec, ...]:
        return tuple(job for job in self.spec.jobs if job.host == self.host)

    def _reconcile_active(self) -> bool:
        finished = False
        for job in self._host_jobs():
            record = self.state.job(job.id)
            current = JobState(record["state"])
            if current == JobState.LAUNCHING:
                finished = self._reconcile_launching(job, record) or finished
                continue
            if current not in _ACTIVE:
                continue
            phase = record.get("phase")
            if not isinstance(phase, dict):
                self._block_orphan(job, "active phase record is absent")
                continue
            if phase_is_live(phase):
                self._record_adoption_once(job, current, phase)
                continue
            try:
                exit_record = read_exit_record(Path(str(phase["exit_record"])))
            except (KeyError, LaunchError) as exc:
                self._block_orphan(job, f"exit record invalid: {exc}")
                continue
            if exit_record is None:
                self._block_orphan(job, "process disappeared without an exit record")
                continue
            if exit_record.get("run_id") != job.id or exit_record.get("git_sha") != self.spec.git_sha:
                self._block_orphan(job, "exit record run identity or Git SHA drift")
                continue
            if exit_record.get("phase_argv_digest") != phase.get("phase_argv_digest"):
                self._block_orphan(job, "exit record argv digest drift")
                continue
            self._finish_phase(job, current, int(exit_record["exit_code"]), exit_record)
            finished = True
        return finished

    def _reconcile_launching(self, job: JobSpec, record: dict[str, Any]) -> bool:
        intent = record.get("launch_intent")
        if not isinstance(intent, dict):
            self._block_orphan(job, "launching state has no durable launch intent")
            return True
        try:
            phase_name = str(intent["name"])
            target = _PHASE_STATE[phase_name]
            launch_record_path = Path(str(intent["launch_record"]))
        except (KeyError, TypeError) as exc:
            self._block_orphan(job, f"launching intent is invalid: {exc}")
            return True
        try:
            launched = read_launch_record(launch_record_path)
        except LaunchError as exc:
            self._block_orphan(job, f"launch record is invalid: {exc}")
            return True
        if launched is None:
            return False
        if (
            launched.get("run_id") != job.id
            or launched.get("git_sha") != self.spec.git_sha
            or launched.get("phase_argv_digest") != intent.get("phase_argv_digest")
        ):
            self._block_orphan(job, "launch record identity drift")
            return True
        phase = {"name": phase_name, **launched}
        if phase_is_live(phase):
            self.state.transition_job(job.id, target, phase=phase, launch_intent=None)
            self._record_adoption_once(job, target, phase)
            return False
        try:
            exit_record = read_exit_record(Path(str(intent["exit_record"])))
        except (KeyError, LaunchError) as exc:
            self._block_orphan(job, f"launching exit record invalid: {exc}")
            return True
        if exit_record is None:
            self._block_orphan(job, "launch wrapper disappeared without an exit record")
            return True
        self.state.transition_job(job.id, target, phase=phase, launch_intent=None)
        self._finish_phase(job, target, int(exit_record["exit_code"]), exit_record)
        return True

    def _record_adoption_once(self, job: JobSpec, state: JobState, phase: dict[str, Any]) -> None:
        digest = phase.get("phase_argv_digest")
        record = self.state.job(job.id)
        if record.get("live_phase_digest_evidenced") != digest:
            self.state.append_evidence(job.id, "process_adopted", {"phase": phase.get("name")})
            self.state.transition_job(job.id, state, live_phase_digest_evidenced=digest)

    def _finish_phase(self, job: JobSpec, current: JobState, code: int, exit_record: dict[str, Any]) -> None:
        if code != 0:
            if current == JobState.AUTOATTACK:
                self.state.transition_job(
                    job.id,
                    JobState.PGD_COMPLETED_AUTOATTACK_FAILED,
                    phase_exit=exit_record,
                    autoattack_status="failed",
                )
            else:
                self.state.transition_job(
                    job.id,
                    JobState.FAILED,
                    phase_exit=exit_record,
                    failure="phase returned nonzero",
                )
            self._release_gpu_lease(job, current)
            return
        phase_name = str(self.state.job(job.id).get("phase", {}).get("name", ""))
        validation_failure = self.phase_success_validator(job, phase_name, self.output_root)
        if validation_failure is not None:
            self.state.transition_job(
                job.id,
                JobState.BLOCKED,
                phase_exit=exit_record,
                failure="phase exited zero without required terminal lineage",
                validation_failure=validation_failure,
            )
            self._release_gpu_lease(job, current)
            return
        targets = {
            JobState.TRAINING: JobState.TRAINING_COMPLETED,
            JobState.PGD_EVALUATION: JobState.PGD_COMPLETED,
            JobState.AUTOATTACK: JobState.COMPLETED,
        }
        self.state.transition_job(job.id, targets[current], phase_exit=exit_record)
        self._release_gpu_lease(job, current)

    def _block_orphan(self, job: JobSpec, reason: str) -> None:
        self.state.transition_job(job.id, JobState.BLOCKED, failure="orphaned phase", orphan_reason=reason)

    def _launch_one_eligible_phase(self) -> None:
        for job in sorted(self._host_jobs(), key=self._queue_key):
            record = self.state.job(job.id)
            current = JobState(record["state"])
            if current == JobState.PGD_COMPLETED and job.phases.autoattack is None:
                self.state.transition_job(job.id, JobState.COMPLETED, autoattack_status="not_requested")
                continue
            phase = self._next_phase(job, current)
            if phase is None:
                continue
            if current in {
                JobState.PENDING,
                JobState.WAITING_DEPENDENCY,
                JobState.WAITING_GPU,
                JobState.WAITING_FOR_MEMORY,
            }:
                self.state.transition_job(job.id, JobState.PREFLIGHT)
            if not self._dependencies_complete(job):
                self.state.transition_job(job.id, JobState.WAITING_DEPENDENCY)
                continue
            if self._preflight_and_launch(job, phase):
                return

    def _queue_key(self, job: JobSpec) -> tuple[int, int, str]:
        state = JobState(self.state.job(job.id)["state"])
        phase = self._next_phase(job, state)
        # Once a GPU's training finishes, its mandatory saved-checkpoint PGD
        # must run before that same slot accepts another training job.  Other
        # free GPUs can still begin their initial train phases; AA stays last.
        rank = {"pgd": 0, "train": 1, "autoattack": 2}.get(phase or "autoattack", 3)
        return (rank, job.priority, job.id)

    def _next_phase(self, job: JobSpec, state: JobState) -> str | None:
        if state in {
            JobState.PENDING,
            JobState.PREFLIGHT,
            JobState.WAITING_DEPENDENCY,
            JobState.WAITING_GPU,
            JobState.WAITING_FOR_MEMORY,
        }:
            return "train"
        if state == JobState.TRAINING_COMPLETED:
            return "pgd"
        if state == JobState.PGD_COMPLETED and job.phases.autoattack is not None:
            return "autoattack"
        return None

    def _dependencies_complete(self, job: JobSpec) -> bool:
        for dependency in job.depends_on:
            state = JobState(self.state.job(dependency)["state"])
            if state not in {JobState.COMPLETED, JobState.PGD_COMPLETED_AUTOATTACK_FAILED}:
                return False
        return True

    def _preflight_and_launch(self, job: JobSpec, phase: str) -> bool:
        try:
            snapshots = self.inventory_provider()
        except Exception as exc:  # inventory failures must be evidence, never optimistic admission
            self.state.transition_job(
                job.id,
                JobState.BLOCKED,
                failure="GPU inventory unavailable",
                inventory_error=repr(exc),
            )
            return False
        snapshot = next((item for item in snapshots if item.index == job.gpu), None)
        if snapshot is None:
            self.state.transition_job(job.id, JobState.BLOCKED, failure="assigned GPU absent from inventory")
            return False
        admission = admit(
            snapshot,
            external_process_policy=self.spec.external_process_policy,
            pilot_peak_reserved_mib=job.pilot_peak_reserved_mib,
            campaign_claimed=self._gpu_claimed_by_other_job(job),
            reserved_by_current_run=self._reserved(job),
            external_processes_enabled=self.external_processes_enabled,
        )
        if not admission.allowed:
            target = JobState(admission.state)
            self.state.transition_job(
                job.id,
                target,
                gpu_snapshot=snapshot.json(),
                admission=self._admission_json(admission),
            )
            return False
        lock = self._gpu_lease_lock(snapshot.uuid)
        if not lock.acquire(blocking=False):
            self.state.transition_job(job.id, JobState.WAITING_GPU, failure=None, gpu_lock="held")
            return False
        try:
            if self._gpu_claimed_by_other_job(job) or self._gpu_lease_path(snapshot.uuid).exists():
                self.state.transition_job(job.id, JobState.WAITING_GPU, gpu_lock="host-global lease held")
                return False
            return self._launch(job, phase, snapshot, admission)
        finally:
            lock.release()

    def _launch(self, job: JobSpec, phase: str, snapshot: GPUSnapshot, admission: GPUAdmission) -> bool:
        try:
            argv = self._phase_argv(job, phase)
        except WorkerError as exc:
            self.state.transition_job(job.id, JobState.BLOCKED, failure="unsafe phase argv", launch_error=str(exc))
            return False
        phase_dir = self.output_root / job.output / "campaign-control" / phase
        exit_record = phase_dir / "exit.json"
        launch_record = phase_dir / "launch.json"
        lease_handshake = phase_dir / "lease-handshake.json"
        lease_path = self._gpu_lease_path(snapshot.uuid)
        if exit_record.exists():
            self.state.transition_job(job.id, JobState.BLOCKED, failure="phase exit record already exists")
            return False
        intent: dict[str, object] = {
            "name": phase,
            "phase_argv_digest": argv_digest(argv),
            "exit_record": str(exit_record.resolve()),
            "launch_record": str(launch_record.resolve()),
            "lease_handshake": str(lease_handshake.resolve()),
            "gpu_lease_path": str(lease_path.resolve()),
        }
        self.state.transition_job(
            job.id,
            JobState.LAUNCHING,
            gpu_snapshot=snapshot.json(),
            admission=self._admission_json(admission),
            gpu_uuid=snapshot.uuid,
            launch_intent=intent,
        )
        self._write_gpu_lease(snapshot.uuid, job, phase, intent)
        try:
            launched = self.launcher(
                argv,
                cwd=self.repository,
                stdout_path=phase_dir / "stdout.log",
                stderr_path=phase_dir / "stderr.log",
                exit_record=exit_record,
                launch_record=launch_record,
                gpu_lease_path=lease_path,
                lease_handshake=lease_handshake,
                run_id=job.id,
                git_sha=self.spec.git_sha,
                environment=self._phase_environment(job),
            )
        except Exception as exc:
            # A launcher that got as far as its durable record may have a live
            # wrapper even though its caller died.  Reconciliation will adopt.
            if launch_record.exists():
                self.state.append_evidence(job.id, "launch_return_interrupted", {"error": repr(exc)})
                return True
            self.state.transition_job(job.id, JobState.FAILED, failure="detached launch failed", launch_error=repr(exc))
            self._release_gpu_lease(job, JobState.LAUNCHING)
            return False
        target = _PHASE_STATE[phase]
        self.state.transition_job(
            job.id,
            target,
            phase={"name": phase, **launched},
            gpu_uuid=snapshot.uuid,
            shared_gpu_at_launch=admission.shared_gpu_at_launch,
            launch_intent=None,
        )
        return True

    def _phase_argv(self, job: JobSpec, phase: str) -> tuple[str, ...]:
        raw = {"train": job.phases.train, "pgd": job.phases.pgd_evaluate, "autoattack": job.phases.autoattack}[phase]
        assert raw is not None
        substitutions = {
            "{JOB_OUTPUT_DIR}": str((self.output_root / job.output).resolve()),
            "{CONFIG_PATH}": str((self.repository / job.config).resolve()),
            "{PYTHON}": sys.executable,
        }

        def substitute(token: str) -> str:
            if token in substitutions:
                return substitutions[token]
            output_prefix = "{JOB_OUTPUT_DIR}/"
            if token.startswith(output_prefix):
                return str(Path(substitutions["{JOB_OUTPUT_DIR}"]) / token.removeprefix(output_prefix))
            return token

        argv = tuple(substitute(token) for token in raw)
        if any("{" in token or "}" in token for token in argv):
            raise WorkerError("phase argv contains an unknown brace token")
        return argv

    def _phase_environment(self, job: JobSpec) -> dict[str, str]:
        return {
            "CUDA_VISIBLE_DEVICES": str(job.gpu),
            "PYTHONPATH": str(self.repository / "src"),
            "ARD_JOB_OUTPUT_DIR": str((self.output_root / job.output).resolve()),
            "ARD_RUN_ID": effective_wandb_run_id(self.spec, job),
            "ARD_SEED": str(job.seed),
            "WANDB_ENTITY": job.wandb.entity,
            "WANDB_PROJECT": job.wandb.project,
            "WANDB_GROUP": job.wandb.group,
            # Existing strict experiment configs keep teacher-specific group
            # variables so a wrong teacher/group pairing is visible at resolve
            # time.  A static job has exactly one group, therefore exporting
            # both names is harmless and avoids command-specific shell logic.
            "WANDB_GROUP_CHEN": job.wandb.group,
            "WANDB_GROUP_BARTOLDSON": job.wandb.group,
        }

    def _gpu_claimed_by_other_job(self, requested: JobSpec) -> bool:
        for job in self._host_jobs():
            if job.id == requested.id or job.gpu != requested.gpu:
                continue
            state = JobState(self.state.job(job.id)["state"])
            if state in _ACTIVE or state == JobState.LAUNCHING:
                return True
        return False

    def _gpu_lease_lock(self, gpu_uuid: str) -> FileLock:
        return FileLock(self.gpu_lock_root / f"gpu-{gpu_uuid}.lock")

    def _gpu_lease_path(self, gpu_uuid: str) -> Path:
        return self.gpu_lock_root / f"gpu-{gpu_uuid}.lease.json"

    def _write_gpu_lease(self, gpu_uuid: str, job: JobSpec, phase: str, intent: dict[str, object]) -> None:
        _atomic_json(
            self._gpu_lease_path(gpu_uuid),
            {
                "version": 1,
                "campaign_identity_sha256": campaign_identity_sha256(self.spec),
                "campaign_id": self.spec.campaign_id,
                "job_id": job.id,
                "phase": phase,
                "gpu_uuid": gpu_uuid,
                "state_root": str(self.state.root),
                "launch_intent": intent,
            },
        )

    def _release_gpu_lease(self, job: JobSpec, current: JobState) -> None:
        record = self.state.job(job.id)
        phase = record.get("phase")
        intent = record.get("launch_intent")
        metadata = phase if isinstance(phase, dict) else intent if isinstance(intent, dict) else None
        if not isinstance(metadata, dict):
            return
        gpu_uuid = record.get("gpu_uuid")
        if not isinstance(gpu_uuid, str):
            return
        lock = self._gpu_lease_lock(gpu_uuid)
        lock.acquire()
        try:
            path = self._gpu_lease_path(gpu_uuid)
            if not path.exists():
                return
            try:
                lease = _read_json(path)
            except CampaignError:
                return
            if (
                lease.get("campaign_identity_sha256") == campaign_identity_sha256(self.spec)
                and lease.get("job_id") == job.id
                and lease.get("phase") == metadata.get("name")
            ):
                path.unlink()
        finally:
            lock.release()

    def _reserved(self, job: JobSpec) -> bool:
        for reservation in self.spec.reservations:
            if not (reservation.active and reservation.host == job.host and reservation.gpu == job.gpu):
                continue
            marker = reservation.release_marker
            if marker is None:
                return True
            try:
                payload = _read_json(marker)
            except CampaignError:
                return True
            expected = {
                "status": "completed",
                "run_id": reservation.run_id,
                "training_git_sha": reservation.protected_git_sha,
                "execution_profile": reservation.execution_profile,
                "training_sync": "completed",
                "saved_checkpoint_pgd": "completed",
            }
            if payload != expected:
                return True
        return False

    @staticmethod
    def _admission_json(admission: GPUAdmission) -> dict[str, Any]:
        return {
            "allowed": admission.allowed,
            "state": admission.state,
            "reason": admission.reason,
            "required_free_memory_mib": admission.required_free_memory_mib,
            "shared_gpu_at_launch": admission.shared_gpu_at_launch,
        }

    def _maybe_stop_for_scientific_review(self) -> None:
        core = [job for job in self._host_jobs() if job.core]
        if not core:
            return
        states = [JobState(self.state.job(job.id)["state"]) for job in core]
        # Completion and failure both stop this host controller at the explicit
        # scientific-review boundary.  A failure never unlocks queue extension.
        if all(state in TERMINAL_JOB_STATES for state in states):
            self.state.set_campaign_state("awaiting_scientific_review")
