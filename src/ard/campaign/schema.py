"""Strict, static campaign definitions.

This module deliberately describes execution *identity* and process commands only.
It does not parse an experiment config or derive scientific parameters from it.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CampaignError(ValueError):
    """A campaign definition or identity is unsafe to use."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_SHA = re.compile(r"^[0-9a-f]{40}$")


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
        raise ValueError("path must be a non-empty safe relative path")
    return path.as_posix()


def _valid_id(value: str) -> str:
    if not _ID.fullmatch(value) or ".." in value:
        raise ValueError("must be a safe run-style identifier")
    return value


class ExecutionProfile(StrictModel):
    id: Literal["ws1_prb128_gb128_localbn_v1"]
    world_size: Literal[1] = 1
    per_rank_batch_size: Literal[128] = 128
    global_batch_size: Literal[128] = 128
    # This exact value is already resolved by ExperimentConfig/tracking for
    # world-size one.  A campaign must not invent a near-equivalent alias.
    batchnorm_mode: Literal["local_per_rank"] = "local_per_rank"


class HostSpec(StrictModel):
    gpus: tuple[int, ...] = Field(min_length=1)

    @field_validator("gpus")
    @classmethod
    def unique_nonnegative_gpus(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(gpu < 0 for gpu in value) or len(set(value)) != len(value):
            raise ValueError("host GPU indexes must be unique non-negative integers")
        return value


class WandbIdentity(StrictModel):
    entity: str = Field(min_length=1)
    project: str = Field(min_length=1)
    group: str = Field(min_length=1)
    run_id: str

    _run_id = field_validator("run_id")(_valid_id)

    @model_validator(mode="after")
    def leaves_room_for_fixed_sha(self) -> WandbIdentity:
        # A runtime effective ID appends '-' plus the immutable SHA prefix.
        if len(self.run_id) > 72:
            raise ValueError("W&B base run_id must leave room for a fixed-SHA suffix")
        return self


class PhaseCommands(StrictModel):
    """Commands are argv arrays; accepting a shell string would lose provenance."""

    train: tuple[str, ...] = Field(min_length=1)
    pgd_evaluate: tuple[str, ...] | None = None
    autoattack: tuple[str, ...] | None = None

    @field_validator("train", "pgd_evaluate", "autoattack")
    @classmethod
    def argv_is_safe(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if value is None:
            return value
        if not all(isinstance(arg, str) and arg and "\x00" not in arg for arg in value):
            raise ValueError("phase commands must be non-empty argv strings without NUL")
        return value


class GPUReservation(StrictModel):
    host: Literal["hamster", "ferret"]
    gpu: int = Field(ge=0)
    run_id: str
    execution_profile: str
    protected_git_sha: str | None = None
    active: bool = True
    release_marker: Path | None = None

    _run_id = field_validator("run_id")(_valid_id)

    @field_validator("protected_git_sha")
    @classmethod
    def protected_full_sha(cls, value: str | None) -> str | None:
        if value is not None and not _SHA.fullmatch(value):
            raise ValueError("reservation protected_git_sha must be a lowercase full 40-hex SHA")
        return value

    @field_validator("release_marker")
    @classmethod
    def absolute_release_marker(cls, value: Path | None) -> Path | None:
        if value is not None and not value.is_absolute():
            raise ValueError("reservation release_marker must be an absolute path")
        return value

    @model_validator(mode="after")
    def release_identity_is_complete(self) -> GPUReservation:
        if self.release_marker is not None and self.protected_git_sha is None:
            raise ValueError("release_marker requires protected_git_sha")
        return self


class JobSpec(StrictModel):
    id: str
    host: Literal["hamster", "ferret"]
    gpu: int = Field(ge=0)
    teacher: str = Field(min_length=1)
    method: str = Field(min_length=1)
    seed: int = Field(ge=0)
    config: str
    output: str
    wandb: WandbIdentity
    phases: PhaseCommands
    depends_on: tuple[str, ...] = ()
    priority: int = 100
    core: bool = True
    pilot_peak_reserved_mib: int | None = Field(default=None, gt=0)

    _id = field_validator("id")(_valid_id)
    _config = field_validator("config")(_safe_relative_path)
    _output = field_validator("output")(_safe_relative_path)

    @field_validator("depends_on")
    @classmethod
    def valid_dependencies(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("dependencies must be unique")
        for dependency in value:
            _valid_id(dependency)
        return value

    @model_validator(mode="after")
    def command_requirements(self) -> JobSpec:
        if self.phases.pgd_evaluate is None:
            raise ValueError("saved-checkpoint PGD evaluation is mandatory for every campaign job")
        return self


class CampaignSpec(StrictModel):
    campaign_id: str
    # Templates may omit this impossible-to-self-reference value.  The runtime
    # binds it exactly once before it may be armed.
    git_sha: str | None = None
    execution_profile: ExecutionProfile
    external_process_policy: Literal["deny", "allow_with_memory_gate"] = "deny"
    hosts: dict[Literal["hamster", "ferret"], HostSpec]
    jobs: tuple[JobSpec, ...] = Field(min_length=1)
    reservations: tuple[GPUReservation, ...] = ()

    _campaign_id = field_validator("campaign_id")(_valid_id)

    @field_validator("git_sha")
    @classmethod
    def full_sha_or_template(cls, value: str | None) -> str | None:
        if value is not None and not _SHA.fullmatch(value):
            raise ValueError("git_sha must be a lowercase full 40-hex SHA")
        return value

    @model_validator(mode="after")
    def static_identity_is_unambiguous(self) -> CampaignSpec:
        ids = [job.id for job in self.jobs]
        if len(set(ids)) != len(ids):
            raise ValueError("job ids must be unique")
        outputs = [job.output for job in self.jobs]
        if len(set(outputs)) != len(outputs):
            raise ValueError("job output paths must be unique")
        wandb_ids = [job.wandb.run_id for job in self.jobs]
        if len(set(wandb_ids)) != len(wandb_ids):
            raise ValueError("W&B run IDs must be unique")
        all_ids = set(ids)
        for job in self.jobs:
            if job.id in job.depends_on:
                raise ValueError(f"job {job.id} cannot depend on itself")
            missing = set(job.depends_on) - all_ids
            if missing:
                raise ValueError(f"job {job.id} has unknown dependencies: {sorted(missing)}")
            if job.gpu not in self.hosts[job.host].gpus:
                raise ValueError(f"job {job.id} is assigned to a GPU not owned by {job.host}")
        for reservation in self.reservations:
            if reservation.gpu not in self.hosts[reservation.host].gpus:
                raise ValueError("reservation GPU is not owned by its host")
        return self


def load_campaign(path: Path) -> CampaignSpec:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CampaignError(f"campaign YAML is not readable: {path}") from exc
    if not isinstance(raw, dict):
        raise CampaignError("campaign YAML must be a mapping")
    try:
        return CampaignSpec.model_validate(raw)
    except ValueError as exc:
        raise CampaignError(str(exc)) from exc


def bind_git_sha(spec: CampaignSpec, sha: str) -> CampaignSpec:
    """Bind a template once; a previously fixed campaign may not drift."""
    if not _SHA.fullmatch(sha):
        raise CampaignError("campaign SHA must be a lowercase full 40-hex SHA")
    if spec.git_sha is not None and spec.git_sha != sha:
        raise CampaignError("campaign SHA drift is forbidden")
    return spec.model_copy(update={"git_sha": sha})


def campaign_identity(spec: CampaignSpec) -> dict[str, object]:
    if spec.git_sha is None:
        raise CampaignError("campaign must bind a full Git SHA before identity is used")
    # This is the immutable runtime manifest, not a short cohort label.  A
    # restart with changed commands, outputs, W&B lineage, reservations, or
    # queue assignment must fail before it can adopt or launch anything.
    return spec.model_dump(mode="json")


def campaign_identity_sha256(spec: CampaignSpec) -> str:
    payload = json.dumps(campaign_identity(spec), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def effective_wandb_run_id(spec: CampaignSpec, job: JobSpec) -> str:
    if spec.git_sha is None:
        raise CampaignError("effective W&B run ID requires a fixed Git SHA")
    value = f"{job.wandb.run_id}-{spec.git_sha[:7]}"
    if not _ID.fullmatch(value) or len(value) > 80:
        raise CampaignError("effective W&B run ID is unsafe or too long")
    return value


def require_aggregation_compatible(records: list[dict[str, object]]) -> None:
    """Reject a ws1/ws2 mix before an aggregation layer can pool it."""
    profiles = {str(record.get("execution_profile")) for record in records}
    if len(profiles) > 1:
        raise CampaignError("runs with different execution profiles must not be aggregated")
