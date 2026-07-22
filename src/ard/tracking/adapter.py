"""Rank-zero tracking adapters and durable local experiment lineage.

The rest of ARD never imports :mod:`wandb`; this module is the sole boundary.
Local lineage is intentionally written even for disabled development tracking so
that a smoke run remains inspectable without network access.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import torch
import yaml

from ard.config.loader import resolved_config_dict
from ard.config.schema import ExperimentConfig, SeedsConfig
from ard.engine.checkpoint import capture_rng_state, restore_rng_state
from ard.engine.distributed import is_rank_zero, run_rank_zero_phase
from ard.models.teacher import sha256_file


class TrackingError(RuntimeError):
    """A requested tracking contract could not be fulfilled."""


QUALITATIVE_COLUMNS = (
    "sample_id",
    "epoch",
    "clean_image",
    "adversarial_image",
    "perturbation_visualization",
    "true_label",
    "student_clean_prediction",
    "student_adv_prediction",
    "teacher_prediction",
    "teacher_entropy",
    "student_robust_margin_ema",
    "student_unlearnability",
    "joint_risk",
    "kd_weight",
    "clean_correct",
    "robust_correct",
)


class ExperimentTracker(Protocol):
    run_id: str

    def log_metrics(self, values: Mapping[str, Any], *, step: int | None = None) -> None: ...
    def log_table(
        self, name: str, rows: list[Mapping[str, Any]], *, columns: tuple[str, ...] = QUALITATIVE_COLUMNS
    ) -> None: ...
    def log_artifact(self, path: Path, *, name: str, artifact_type: str, aliases: tuple[str, ...] = ()) -> None: ...
    def set_summary(self, values: Mapping[str, Any]) -> None: ...
    def prepare_finish(self, *, status: str = "completed") -> None: ...
    def resume(self, *, checkpoint_run_id: str | None, checkpoint_config_hash: str) -> None: ...
    def finish(self, *, status: str = "completed") -> None: ...


def _json_default(value: Any) -> str:
    return str(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, default=_json_default) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _run_git(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False
        )
    except OSError:
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def collect_environment() -> dict[str, Any]:
    """Collect only reproducibility-relevant, non-secret environment metadata."""
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    }


def _untracked_file_hashes(root: Path) -> dict[str, str]:
    """Hash every Git-untracked regular file without parsing porcelain output."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "-z"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return {}
    if completed.returncode != 0:
        return {}
    files: dict[str, str] = {}
    for encoded_path in (completed.stdout or b"").split(b"\0"):
        if not encoded_path:
            continue
        relative_path = Path(os.fsdecode(encoded_path))
        path = root / relative_path
        if path.is_file():
            files[relative_path.as_posix()] = sha256_file(path)
    return files


def collect_git_state(root: Path) -> dict[str, Any]:
    root = root.resolve()
    sha = _run_git(["-C", str(root), "rev-parse", "HEAD"])
    branch = _run_git(["-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"])
    dirty = _run_git(["-C", str(root), "status", "--porcelain"])
    diff = _run_git(["-C", str(root), "diff", "--binary", "HEAD"])
    return {
        "sha": sha,
        "branch": branch,
        "dirty": bool(dirty),
        "status": dirty or "",
        "diff": diff or "",
        "diff_sha256": hashlib.sha256((diff or "").encode()).hexdigest(),
        "untracked_sha256": _untracked_file_hashes(root),
    }


def _checkout_metadata(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        return {"exists": False, "origin": None, "head": None, "status": None}
    return {
        "exists": True,
        "origin": _run_git(["-C", str(path), "remote", "get-url", "origin"]),
        "head": _run_git(["-C", str(path), "rev-parse", "HEAD"]),
        "status": _run_git(["-C", str(path), "status", "--porcelain"]),
    }


def _external_metadata(root: Path) -> dict[str, Any]:
    lock = root / "external.lock.yaml"
    if not lock.is_file():
        return {"lock_path": str(lock), "sha256": None, "repositories": {}}
    try:
        raw = yaml.safe_load(lock.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise TrackingError("external.lock.yaml is not readable YAML") from exc
    repositories = raw.get("repositories", {}) if isinstance(raw, dict) else {}
    explicit = {
        str(name): {
            "url": value.get("url"),
            "commit": value.get("commit"),
            "checkout": _checkout_metadata(root / ".external" / str(name)),
        }
        for name, value in repositories.items()
        if isinstance(value, dict)
    }
    return {"lock_path": str(lock), "sha256": sha256_file(lock), "repositories": explicit}


def _teacher_metadata(config: ExperimentConfig) -> dict[str, Any] | None:
    teacher = config.teacher
    if teacher is None:
        return None
    actual = None
    if teacher.checkpoint is not None and teacher.checkpoint.is_file():
        actual = sha256_file(teacher.checkpoint)
        if teacher.checkpoint_sha256 is not None and actual != teacher.checkpoint_sha256:
            raise TrackingError("teacher checkpoint declared SHA does not match actual bytes")
    return {
        "source": teacher.source,
        "architecture": teacher.architecture,
        "checkpoint": None if teacher.checkpoint is None else str(teacher.checkpoint),
        "checkpoint_sha256": teacher.checkpoint_sha256,
        "checkpoint_actual_sha256": actual,
        "normalization": teacher.normalization.model_dump(mode="json"),
    }


def stable_run_id(
    config: ExperimentConfig, *, config_hash: str, resume_run_id: str | None = None, git_sha: str | None = None
) -> str:
    explicit = config.tracking.run_id or config.tracker_run_id
    if resume_run_id:
        if explicit and explicit != resume_run_id:
            raise TrackingError("configured tracking run ID does not match the resume checkpoint")
        return resume_run_id
    if explicit:
        return explicit
    # Deliberately deterministic: crashes before the first checkpoint still
    # retain an ID in the local manifest and therefore resume the same run.
    return "ard-" + hashlib.sha256(f"{config_hash}:{git_sha or 'unborn'}".encode()).hexdigest()[:16]


def _rng_preserving_rank_zero_phase(operation: Any, *, phase: str) -> None:
    """Tracking must be observational: restore every RNG stream on every rank."""
    state = capture_rng_state()
    try:
        run_rank_zero_phase(operation, phase=phase)
    finally:
        restore_rng_state(state)


@dataclass
class NullTracker:
    run_id: str

    def log_metrics(self, values: Mapping[str, Any], *, step: int | None = None) -> None:
        del values, step

    def log_table(
        self, name: str, rows: list[Mapping[str, Any]], *, columns: tuple[str, ...] = QUALITATIVE_COLUMNS
    ) -> None:
        del name, rows, columns

    def log_artifact(self, path: Path, *, name: str, artifact_type: str, aliases: tuple[str, ...] = ()) -> None:
        del path, name, artifact_type, aliases

    def set_summary(self, values: Mapping[str, Any]) -> None:
        del values

    def resume(self, *, checkpoint_run_id: str | None, checkpoint_config_hash: str) -> None:
        if checkpoint_run_id != self.run_id:
            raise TrackingError("checkpoint and rank-zero tracker run IDs must match on resume")
        del checkpoint_config_hash

    def finish(self, *, status: str = "completed") -> None:
        del status

    def prepare_finish(self, *, status: str = "completed") -> None:
        del status


class LocalTracker:
    """Durable local bundle plus an optional W&B run owned by rank zero."""

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        output_dir: Path,
        config_hash: str,
        root: Path,
        run_id: str,
        job_type: str = "train",
        wandb_module: Any | None = None,
        training_seed: int | None = None,
        training_seeds: Mapping[str, int] | None = None,
        evaluation_seed: int | None = None,
    ) -> None:
        self.config, self.output_dir, self.config_hash, self.root, self.run_id = (
            config,
            output_dir,
            config_hash,
            root,
            run_id,
        )
        self.bundle_dir = output_dir / "run-bundle"
        self.metrics_path = self.bundle_dir / "metrics.jsonl"
        self.manifest_path = self.bundle_dir / "manifest.json"
        self._wandb_run: Any | None = None
        self._wandb_module: Any | None = None
        self._prepared = False
        self._finalization_id = uuid.uuid4().hex
        resolved_training_seeds = (
            config.seeds if training_seeds is None else SeedsConfig.model_validate(dict(training_seeds))
        )
        if training_seed is not None and training_seed != resolved_training_seeds.model_init:
            raise TrackingError("training_seed compatibility scalar must match training_seeds.model_init")
        git = collect_git_state(root)
        prior: dict[str, Any] | None = None
        if self.manifest_path.is_file():
            try:
                prior = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise TrackingError("existing tracking manifest is unreadable") from exc
            if prior.get("run_id") != run_id or prior.get("config_hash") != config_hash:
                raise TrackingError("resume manifest run ID or config hash does not match checkpoint lineage")
        self._is_resume = prior is not None
        self._prior_manifest = prior
        self.manifest: dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "running",
            "created_at": datetime.now(UTC).isoformat(),
            "tier": config.tier,
            "tracking_mode": config.tracking.mode,
            "job_type": job_type,
            "config_hash": config_hash,
            "protocol_id": config.protocol.id,
            "seed": resolved_training_seeds.model_init,
            "training_seed": resolved_training_seeds.model_init,
            "training_seeds": resolved_training_seeds.model_dump(mode="json"),
            "evaluation_seed": evaluation_seed,
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "git": {key: value for key, value in git.items() if key != "diff"},
            "external": _external_metadata(root),
            "teacher": _teacher_metadata(config),
            "sync_state": "running" if config.tracking.mode == "offline_sync" else None,
            "wandb_initialized": False,
            "summary": {},
            "artifacts": [],
        }
        if prior is not None:
            current_lineage = {key: self.manifest[key] for key in ("git", "external", "teacher")}
            for key, current in current_lineage.items():
                if prior.get(key) != current:
                    raise TrackingError(f"resume manifest lineage drift: {key}")
            for key in (
                "created_at",
                "summary",
                "artifacts",
                "wandb_segments",
                "wandb_initialized",
                "git",
                "external",
                "teacher",
            ):
                if key in prior:
                    self.manifest[key] = prior[key]
            for key in (
                "tier",
                "tracking_mode",
                "job_type",
                "protocol_id",
                "seed",
                "training_seed",
                "training_seeds",
                "world_size",
            ):
                if prior.get(key) != self.manifest.get(key):
                    raise TrackingError(f"resume manifest lineage drift: {key}")
            prior_environment = self.bundle_dir / "environment.json"
            if prior_environment.is_file():
                previous_environment = json.loads(prior_environment.read_text(encoding="utf-8"))
                current_environment = collect_environment()
                if previous_environment != current_environment:
                    raise TrackingError("resume manifest lineage drift: environment")
            events = list(prior.get("resume_events", []))
            events.append({"at": datetime.now(UTC).isoformat(), "git": git["sha"]})
            self.manifest["resume_events"] = events
        self.bundle_dir.mkdir(parents=True, exist_ok=True)
        _write_json(self.bundle_dir / "environment.json", collect_environment())
        _write_json(self.manifest_path, self.manifest)
        (self.bundle_dir / "diff.patch").write_text(git["diff"], encoding="utf-8")
        lock = root / "external.lock.yaml"
        if lock.is_file():
            shutil.copy2(lock, self.bundle_dir / "external.lock.yaml")
        if config.tracking.mode != "disabled":
            try:
                self._start_wandb(wandb_module)
            except Exception:
                self.prepare_finish(status="failed")
                raise

    def attach_resolved_config(self, path: Path) -> None:
        """Copy the exact pre-training resolved config into the durable bundle."""
        if not path.is_file():
            raise TrackingError(f"resolved config is missing: {path}")
        shutil.copy2(path, self.bundle_dir / "resolved_config.yaml")

    def _start_wandb(self, wandb_module: Any | None) -> None:
        module = wandb_module
        if module is None:
            try:
                import wandb

                module = wandb
            except ImportError as exc:
                if self.config.tier in {"repro", "production"}:
                    raise TrackingError("production tracking requires the wandb package") from exc
                # Offline smoke/dev still owns a complete local pending bundle.
                self.manifest["wandb_unavailable"] = True
                _write_json(self.manifest_path, self.manifest)
                return
        kwargs = {
            "project": self.config.tracking.project,
            "entity": self.config.tracking.entity,
            "id": self.run_id,
            "resume": "must" if self._is_resume and self.manifest["wandb_initialized"] else "never",
            "mode": "offline" if self.config.tracking.mode == "offline_sync" else self.config.tracking.mode,
            "name": self.config.tracking.name,
            "group": self.config.tracking.group,
            "job_type": self.manifest["job_type"],
            "dir": str(self.output_dir / "wandb"),
            "config": resolved_config_dict(self.config),
        }
        self._wandb_module = module
        try:
            self._wandb_run = module.init(**{key: value for key, value in kwargs.items() if value is not None})
        except Exception as exc:
            if self.config.tier in {"repro", "production"} or self.config.tracking.mode == "online":
                raise TrackingError("requested W&B tracker could not initialize") from exc
            # Offline/dev/smoke operation must never escape to a live run.  A
            # restricted host can also forbid W&B's local service sockets, in
            # which case the durable pending bundle remains the source of truth.
            self._wandb_run = None
            self._wandb_module = None
            self.manifest["wandb_unavailable"] = True
            _write_json(self.manifest_path, self.manifest)
            return
        self.manifest["wandb_url"] = getattr(self._wandb_run, "url", None)
        self.manifest["wandb_initialized"] = True
        run_directory = getattr(self._wandb_run, "dir", None)
        if run_directory:
            segment = Path(run_directory).resolve()
            if segment.name == "files":
                segment = segment.parent
            entry = {"path": str(segment), "run_id": self.run_id}
            segments = self.manifest.setdefault("wandb_segments", [])
            if entry not in segments:
                segments.append(entry)
        _write_json(self.manifest_path, self.manifest)

    def log_metrics(self, values: Mapping[str, Any], *, step: int | None = None) -> None:
        record = dict(values)
        if step is not None:
            record.setdefault("global_step", step)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=_json_default) + "\n")
        if self._wandb_run is not None:
            self._wandb_run.log(record, step=step)

    def log_table(
        self, name: str, rows: list[Mapping[str, Any]], *, columns: tuple[str, ...] = QUALITATIVE_COLUMNS
    ) -> None:
        if tuple(columns) != QUALITATIVE_COLUMNS:
            raise TrackingError("qualitative tables must use the fixed required column contract")
        missing = [column for row in rows for column in columns if column not in row]
        if missing:
            raise TrackingError("qualitative table is missing required columns: " + ", ".join(sorted(set(missing))))
        panel_dir = self.bundle_dir / "panels"
        panel_dir.mkdir(exist_ok=True)
        serialized: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            converted = dict(row)
            for field in ("clean_image", "adversarial_image", "perturbation_visualization"):
                value = converted[field]
                if isinstance(value, torch.Tensor):
                    from torchvision.utils import save_image

                    asset = panel_dir / f"{name}-{index}-{field}.png"
                    image = (
                        value
                        if field != "perturbation_visualization"
                        else (value.abs() / value.abs().amax().clamp_min(1e-12))
                    )
                    save_image(image.clamp(0, 1), asset)
                    converted[field] = str(asset.relative_to(self.bundle_dir))
            serialized.append(converted)
        (panel_dir / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True, default=_json_default) + "\n" for row in serialized),
            encoding="utf-8",
        )
        if self._wandb_run is not None and self._wandb_module is not None:
            wandb_rows = []
            for row in serialized:
                wandb_row = dict(row)
                for field in ("clean_image", "adversarial_image", "perturbation_visualization"):
                    value = row[field]
                    if value is None:
                        wandb_row[field] = None
                        continue
                    asset = (self.bundle_dir / str(value)).resolve()
                    bundle = self.bundle_dir.resolve()
                    if not asset.is_file() or bundle not in asset.parents:
                        raise TrackingError(f"qualitative image asset is missing or outside run bundle: {value}")
                    wandb_row[field] = self._wandb_module.Image(str(asset))
                wandb_rows.append([wandb_row[column] for column in columns])
            table = self._wandb_module.Table(columns=list(columns), data=wandb_rows)
            self._wandb_run.log({name: table})

    def log_artifact(self, path: Path, *, name: str, artifact_type: str, aliases: tuple[str, ...] = ()) -> None:
        if not path.exists() or not (path.is_file() or path.is_dir()):
            raise FileNotFoundError(f"cannot publish missing local artifact: {path}")
        entry: dict[str, Any] = {
            "name": name,
            "type": artifact_type,
            "aliases": list(aliases),
            "path": str(path.resolve()),
        }
        local: Path | None = None
        created_local = False
        if path.is_file():
            content_digest = sha256_file(path)
            local = self.bundle_dir / "artifacts" / name / content_digest
            if not local.exists():
                local.mkdir(parents=True, exist_ok=False)
                shutil.copy2(path, local / path.name)
                created_local = True
            entry["sha256"] = content_digest
        else:
            digest = hashlib.sha256()
            files = []
            candidates = []
            for item in path.rglob("*"):
                relative_path = item.relative_to(path)
                if (
                    item.is_file()
                    and relative_path.name != "manifest.json"
                    and (not relative_path.parts or relative_path.parts[0] != "artifacts")
                ):
                    candidates.append(item)
            for child in sorted(candidates):
                relative_name = child.relative_to(path).as_posix()
                file_hash = sha256_file(child)
                digest.update(relative_name.encode() + b"\0" + file_hash.encode() + b"\n")
                files.append({"path": relative_name, "sha256": file_hash})
            entry["directory_digest"] = digest.hexdigest()
            entry["files"] = files
            entry["digest_excludes"] = ["manifest.json", "artifacts/"]
            if path.resolve() != self.bundle_dir.resolve():
                local = self.bundle_dir / "artifacts" / name / digest.hexdigest()
                if not local.exists():
                    shutil.copytree(path, local)
                    created_local = True
        if local is not None:
            entry["local_path"] = str(local.relative_to(self.bundle_dir))
        if artifact_type == "run-bundle":
            entry["finalization_id"] = self._finalization_id
        self.manifest["artifacts"].append(entry)
        _write_json(self.manifest_path, self.manifest)
        if self._wandb_run is None or self._wandb_module is None:
            return
        try:
            artifact = self._wandb_module.Artifact(name, type=artifact_type)
            if path.is_file():
                artifact.add_file(str(path), name=path.name)
            else:
                artifact.add_dir(str(path))
            self._wandb_run.log_artifact(artifact, aliases=list(aliases))
        except Exception:
            self.manifest["artifacts"].pop()
            if local is not None and created_local:
                shutil.rmtree(local, ignore_errors=True)
                if local.parent.exists() and not any(local.parent.iterdir()):
                    local.parent.rmdir()
            _write_json(self.manifest_path, self.manifest)
            raise

    def set_summary(self, values: Mapping[str, Any]) -> None:
        self.manifest["summary"].update(dict(values))
        _write_json(self.manifest_path, self.manifest)
        if self._wandb_run is not None:
            self._wandb_run.summary.update(dict(values))

    def resume(self, *, checkpoint_run_id: str | None, checkpoint_config_hash: str) -> None:
        """Cross-check all three identities before a W&B run can be resumed."""
        if checkpoint_run_id != self.run_id or checkpoint_config_hash != self.config_hash:
            raise TrackingError("checkpoint, manifest, and tracker identities must match on resume")

    def validate_terminal_resume(self) -> None:
        """Reject a no-op resume unless its prior terminal lineage is intact."""
        prior = self._prior_manifest
        if prior is None or prior.get("status") not in {"completed", "sync_pending"}:
            raise TrackingError("no-op resume requires a completed prior manifest")
        if not (self.bundle_dir / "completion.json").is_file():
            raise TrackingError("no-op resume requires a completion marker")
        entries = prior.get("artifacts", [])
        if not isinstance(entries, list):
            raise TrackingError("no-op resume prior artifacts are invalid")
        aliases = {alias for entry in entries if isinstance(entry, dict) for alias in entry.get("aliases", [])}
        if (
            not {"best", "last"}.issubset(aliases)
            or not any(isinstance(entry, dict) and entry.get("type") == "sample-stats" for entry in entries)
            or not any(isinstance(entry, dict) and entry.get("type") == "run-bundle" for entry in entries)
        ):
            raise TrackingError("no-op resume prior artifacts are incomplete")
        for entry in entries:
            if not isinstance(entry, dict):
                raise TrackingError("no-op resume prior artifact is invalid")
            local_path = entry.get("local_path")
            if "sha256" in entry:
                path = Path(entry["path"])
                if not path.is_file() or sha256_file(path) != entry["sha256"]:
                    raise TrackingError("no-op resume artifact hash drift")
                if not isinstance(local_path, str):
                    raise TrackingError("no-op resume artifact local copy is missing")
                local_file = self.bundle_dir / local_path / path.name
                if not local_file.is_file() or sha256_file(local_file) != entry["sha256"]:
                    raise TrackingError("no-op resume artifact local hash drift")

    # Short compatibility aliases for local callers created before the public
    # ExperimentTracker interface was frozen.
    def log(self, values: Mapping[str, Any], *, step: int | None = None) -> None:
        self.log_metrics(values, step=step)

    def log_checkpoint(self, path: Path, *, alias: str) -> None:
        self.log_artifact(path, name=f"model-{self.run_id}-{alias}", artifact_type="model", aliases=(alias,))

    def prepare_finish(self, *, status: str = "completed") -> None:
        if self._prepared and status == "completed":
            return
        if self.config.tracking.mode == "offline_sync" and status == "completed":
            self.manifest["sync_state"] = "sync_pending"
            self.manifest["status"] = "sync_pending"
        else:
            self.manifest["status"] = status
        if status == "failed":
            self.manifest["status"] = "failed"
            self.manifest["sync_state"] = (
                "sync_pending"
                if self.config.tracking.mode == "offline_sync" and self.manifest.get("wandb_segments")
                else "failed"
            )
            (self.bundle_dir / "completion.json").unlink(missing_ok=True)
            (self.bundle_dir / "error-marker.txt").write_text("application failure recorded\n", encoding="utf-8")
            self.manifest["artifacts"] = [
                entry
                for entry in self.manifest["artifacts"]
                if not (entry.get("type") == "run-bundle" and entry.get("finalization_id") == self._finalization_id)
            ]
            digest = hashlib.sha256()
            files = []
            for candidate in sorted(self.bundle_dir.rglob("*")):
                relative = candidate.relative_to(self.bundle_dir)
                if not candidate.is_file() or relative.name == "manifest.json" or relative.parts[0] == "artifacts":
                    continue
                name = relative.as_posix()
                file_hash = sha256_file(candidate)
                digest.update(name.encode() + b"\0" + file_hash.encode() + b"\n")
                files.append({"path": name, "sha256": file_hash})
            self.manifest["failure_snapshot"] = {
                "directory_digest": digest.hexdigest(),
                "files": files,
                "digest_excludes": ["manifest.json", "artifacts/"],
            }
        self.manifest["finished_at"] = datetime.now(UTC).isoformat()
        _write_json(self.manifest_path, self.manifest)
        self._prepared = True

    def finish(self, *, status: str = "completed") -> None:
        self.prepare_finish(status=status)
        if self._wandb_run is not None:
            self._wandb_run.finish(exit_code=1 if status == "failed" else 0)


def create_tracker(
    *,
    config: ExperimentConfig,
    output_dir: Path,
    config_hash: str,
    root: Path,
    resume_run_id: str | None = None,
    wandb_module: Any | None = None,
    job_type: str = "train",
    run_id: str | None = None,
    training_seed: int | None = None,
    training_seeds: Mapping[str, int] | None = None,
    evaluation_seed: int | None = None,
) -> ExperimentTracker:
    """Return a no-op on non-zero ranks; only rank zero can initialize W&B."""
    run_id = run_id or stable_run_id(
        config, config_hash=config_hash, resume_run_id=resume_run_id, git_sha=collect_git_state(root)["sha"]
    )
    if not is_rank_zero():
        return NullTracker(run_id)
    return LocalTracker(
        config=config,
        output_dir=output_dir,
        config_hash=config_hash,
        root=root,
        run_id=run_id,
        job_type=job_type,
        wandb_module=wandb_module,
        training_seed=training_seed,
        training_seeds=training_seeds,
        evaluation_seed=evaluation_seed,
    )


def coordinated_create_tracker(
    *,
    config: ExperimentConfig,
    output_dir: Path,
    config_hash: str,
    root: Path,
    resume_run_id: str | None = None,
    job_type: str = "train",
    run_id: str | None = None,
) -> ExperimentTracker:
    """One coordinated, RNG-neutral init phase; rank zero owns the real run."""
    resolved_id = run_id or stable_run_id(
        config, config_hash=config_hash, resume_run_id=resume_run_id, git_sha=collect_git_state(root)["sha"]
    )
    tracker: ExperimentTracker = NullTracker(resolved_id)

    def initialize() -> None:
        nonlocal tracker
        tracker = LocalTracker(
            config=config,
            output_dir=output_dir,
            config_hash=config_hash,
            root=root,
            run_id=resolved_id,
            job_type=job_type,
        )

    _rng_preserving_rank_zero_phase(initialize, phase="tracker init")
    return tracker


def coordinated_tracker_action(
    tracker: ExperimentTracker, *, phase: str, action: Callable[[ExperimentTracker], None]
) -> None:
    """Execute a rank-zero tracking action without perturbing training RNG."""
    _rng_preserving_rank_zero_phase(lambda: action(tracker), phase=phase)


def validate_tracking_guard(config: ExperimentConfig, *, root: Path) -> None:
    """Reject incomplete repro/production lineage before creating any output."""
    if config.tier not in {"repro", "production"}:
        return
    git = collect_git_state(root)
    if not git["sha"]:
        raise TrackingError("repro/production requires a real Git HEAD for lineage")
    if config.tracking.mode not in {"online", "offline_sync"}:
        raise TrackingError("repro/production requires online or offline_sync tracking")
    if config.tier == "production" and (
        not config.tracking.project or not config.tracking.entity or not config.tracking.group
    ):
        raise TrackingError("production requires W&B entity, project, and group")
    lock = root / "external.lock.yaml"
    try:
        raw = yaml.safe_load(lock.read_text(encoding="utf-8"))
        saad = raw["repositories"]["saad"]
        url, commit = saad["url"], saad["commit"]
    except (OSError, TypeError, KeyError, yaml.YAMLError) as exc:
        raise TrackingError("repro/production requires a valid external.lock.yaml saad entry") from exc
    if (
        not isinstance(url, str)
        or not isinstance(commit, str)
        or not commit
        or len(commit) != 40
        or any(c not in "0123456789abcdef" for c in commit)
    ):
        raise TrackingError("external.lock.yaml saad URL/commit is malformed")
    external = root / ".external" / "saad"
    if not external.is_dir():
        raise TrackingError("locked .external/saad checkout is missing")
    origin = _run_git(["-C", str(external), "remote", "get-url", "origin"])
    external_head = _run_git(["-C", str(external), "rev-parse", "HEAD"])
    external_status = _run_git(["-C", str(external), "status", "--porcelain"])

    def normalize(value: str) -> str:
        return value.rstrip("/").removesuffix(".git")

    if origin is None or normalize(origin) != normalize(url) or external_head != commit or external_status:
        raise TrackingError("locked .external/saad origin, commit, or clean state does not match external.lock.yaml")
    if config.tier == "production":
        status = str(git["status"])
        if any(line.startswith("??") for line in status.splitlines()):
            raise TrackingError("production rejects untracked repository files")
        if git["dirty"] and not git["diff"]:
            raise TrackingError("production tracked dirty state requires an exact binary diff")
