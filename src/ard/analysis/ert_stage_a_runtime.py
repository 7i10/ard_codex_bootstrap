"""Fixed-anchor ERT Stage A continuation runtime.

This module deliberately avoids the historical intervention-screen arm names.
It consumes an immutable epoch-79 parent and one explicit Stage A treatment
specification, then delegates the actual update path to the shared Trainer.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset

from ard.analysis.ert_rslad_rng_sources import (
    RNGSourceSeeds,
    ShuffleAugmentationSeeds,
    reseed_data_stream,
    reseed_other_stream,
    reseed_shuffle_augmentation_stream,
)
from ard.attacks import LinfPGD
from ard.config import load_config
from ard.data import (
    EpochShuffleSampler,
    SampleRef,
    build_train_validation_views,
    collate_indexed,
    data_loader_generator,
    seed_data_loader_worker,
)
from ard.engine import Trainer, get_rank, get_world_size
from ard.engine.checkpoint import load_checkpoint
from ard.models import build_student, build_teacher
from ard.objectives import RSLADObjective
from ard.policies import FixedInterventionMask, RSLADBaselinePolicy
from ard.schedules import build_scheduler
from ard.state import SampleStateStore
from ard.targets import TeacherOnlyTemperatureTargetPolicy
from ard.tracking.adapter import (
    ExperimentTracker,
    collect_git_state,
    create_tracker,
    should_upload_model_artifact,
)


class StageARuntimeError(RuntimeError):
    """Raised when a Stage A parent or treatment contract is invalid."""


def _offline_canary_tracking(tracking: Any) -> Any:
    """Return the registered offline-only tracking view for a public canary.

    The parent is a production configuration, so ``offline_sync`` is the
    only locally durable mode that preserves its schema-level tracking
    contract.  The adapter maps it to W&B's offline mode; the production
    command path never calls this helper.
    """
    return tracking.model_copy(update={"mode": "offline_sync"})


class _CanaryEpochSubset(Dataset[Any]):
    """A bounded real-data view that preserves source-ID/epoch semantics.

    It exists only for the registered public-runtime canary.  Production
    calls never pass the corresponding ``canary_*`` arguments.  Keeping this
    adapter here lets the canary use the same loader, objective, optimizer,
    checkpoint, and router path rather than a private formula reconstruction.
    """

    def __init__(self, dataset: Any, positions: list[int], source_ids: list[int]) -> None:
        if len(positions) != len(source_ids) or not positions:
            raise ValueError("canary positions and source IDs must be non-empty and aligned")
        if len(set(positions)) != len(positions) or len(set(source_ids)) != len(source_ids):
            raise ValueError("canary positions and source IDs must be unique")
        if any(position < 0 or position >= len(dataset) for position in positions):
            raise ValueError("canary position is outside the wrapped dataset")
        self.dataset = dataset
        # ``positions`` are indices in the full SourceIndexedSubset; they
        # must not be replaced by source IDs.  EpochShuffleSampler emits a
        # SampleRef whose index is a position in this bounded view, so
        # ``__getitem__`` maps that position back to the full view while
        # preserving state-update/multiplicity metadata.
        self._positions = tuple(positions)
        # Routing code intentionally discovers this attribute to recover the
        # stable universe.  These IDs remain original train-set source IDs.
        self.indices = tuple(source_ids)

    def __len__(self) -> int:
        return len(self._positions)

    def __getitem__(self, index: int | SampleRef) -> Any:
        reference = index if isinstance(index, SampleRef) else None
        bounded_position = int(index) if reference is None else reference.index
        if bounded_position < 0 or bounded_position >= len(self._positions):
            raise IndexError(f"canary position {bounded_position} is outside the bounded view")
        full_position = self._positions[bounded_position]
        if reference is None:
            return self.dataset[full_position]
        return self.dataset[
            SampleRef(
                index=full_position,
                state_update_mask=reference.state_update_mask,
                multiplicity=reference.multiplicity,
            )
        ]

    def set_epoch(self, epoch: int) -> None:
        setter = getattr(self.dataset, "set_epoch", None)
        if callable(setter):
            setter(epoch)

    def set_augmentation_seed(self, seed: int) -> None:
        setter = getattr(self.dataset, "set_augmentation_seed", None)
        if callable(setter):
            setter(seed)


def _stable_train_labels(dataset: Any) -> dict[int, int]:
    """Recover labels by stable source ID through normal or bounded views.

    ``SourceIndexedSubset`` has two dataset layers above torchvision's
    ``targets``.  The bounded public canary adds one more.  Routing must never
    depend on either incidental wrapper depth or the position-space sampler.
    """

    try:
        source_ids = [int(value) for value in dataset.indices]
    except (AttributeError, TypeError) as exc:
        raise StageARuntimeError("dynamic routing cannot recover the stable train ID universe") from exc
    base = dataset
    seen: set[int] = set()
    while hasattr(base, "dataset"):
        identity = id(base)
        if identity in seen:
            raise StageARuntimeError("dynamic routing dataset wrappers contain a cycle")
        seen.add(identity)
        base = base.dataset
    targets = getattr(base, "targets", None)
    if not isinstance(targets, (list, tuple)):
        raise StageARuntimeError("dynamic routing cannot recover base train labels")
    try:
        labels = {sample_id: int(targets[sample_id]) for sample_id in source_ids}
    except (IndexError, TypeError) as exc:
        raise StageARuntimeError("dynamic routing stable train labels do not join the source universe") from exc
    if len(labels) != len(source_ids):
        raise StageARuntimeError("dynamic routing found duplicate stable train IDs")
    return labels


SAMPLE_KEYED_KL10_ATTACK_IDENTITY_SHA256 = "97a41870008f5946af3b10dd0d7f145324fe5265b12d3c523bf3f8d099623d4d"


def _tracking_run_id(*, run_namespace: str, seed: str, arm: str, source_sha: str) -> str:
    """Build a unique tracker ID without changing scientific identity.

    The orchestrator may retry a job after W&B has already reserved the
    deterministic ID.  A campaign/attempt suffix is therefore added only
    for orchestrated launches; the manifest identity, checkpoint lineage,
    seed, and treatment remain unchanged.  Non-orchestrated runs retain the
    historical deterministic ID for resume compatibility.
    """
    run_id = f"ert-{run_namespace}-{seed}-{arm}-{source_sha[:7]}"
    campaign_id = os.environ.get("ARD_ORCH_CAMPAIGN_ID")
    if campaign_id:
        campaign_hash = hashlib.sha256(campaign_id.encode("utf-8")).hexdigest()[:8]
        run_id = f"{run_id}-orch-{campaign_hash}"
    try:
        attempt = int(os.environ.get("ARD_ORCH_ATTEMPT", "1"))
    except ValueError:
        attempt = 1
    if attempt > 1:
        attempt_id = os.environ.get("ARD_ORCH_ATTEMPT_ID", str(attempt))
        retry_hash = hashlib.sha256(attempt_id.encode("utf-8")).hexdigest()[:8]
        run_id = f"{run_id}-retry-{retry_hash}"
    return run_id


def _prepare_stage_output_dir(output_dir: Path) -> None:
    """Permit an orchestrator-created directory while refusing stale results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.is_dir():
        raise StageARuntimeError(f"Stage A output path is not a directory: {output_dir}")
    allowed = {"orchestration"}
    unexpected = {item.name for item in output_dir.iterdir() if item.name not in allowed}
    if unexpected:
        raise StageARuntimeError(f"Stage A output already contains results: {sorted(unexpected)}")


def _validate_horizons(horizon_epochs: tuple[int, ...], end_epoch: int, *, first_epoch: int = 80) -> None:
    if not horizon_epochs or any(epoch < first_epoch or epoch > end_epoch for epoch in horizon_epochs):
        raise StageARuntimeError(
            f"horizon checkpoints must be at or after first epoch {first_epoch} and no later than the endpoint"
        )
    if len(set(horizon_epochs)) != len(horizon_epochs):
        raise StageARuntimeError("horizon checkpoints must be unique")


def _validate_stage_a_calibration(treatment: StageATreatment, calibration: dict[str, Any]) -> None:
    """Validate only the calibration contract required by this treatment.

    Boundary-distance treatments use their own coefficient artifact and do
    not require the unrelated teacher-softening ``tau`` field.
    """
    if treatment.boundary_intervention is not None:
        required_contract = {
            "pair_margin": "ert_rslad_i100_s2_dynamic_bdd_calibration_v1",
            "detached_boundary_distance": "ert_rslad_i100_s2_dynamic_bdd_calibration_v1",
            "secant_boundary_distance": "ert_rslad_i100_s2_secant_boundary_distance_calibration_v2",
        }[treatment.boundary_intervention]
        if calibration.get("contract") != required_contract:
            raise StageARuntimeError("boundary treatment requires the frozen dynamic-BDD calibration artifact")
        if (
            treatment.boundary_intervention == "secant_boundary_distance"
            and calibration.get("formula_version") != "student_parameter_graph_v2"
        ):
            raise StageARuntimeError("S-BDD calibration does not bind the corrected Student parameter graph")
        if calibration.get("boundary_epsilon") != treatment.boundary_epsilon:
            raise StageARuntimeError("boundary treatment epsilon differs from calibration artifact")
        coefficients = calibration.get("coefficients")
        if (
            not isinstance(coefficients, dict)
            or coefficients.get(treatment.boundary_intervention) != treatment.boundary_coefficient
        ):
            raise StageARuntimeError("boundary treatment coefficient differs from calibration artifact")
    elif treatment.kind == "soft_advkd" and calibration.get("tau") != 2.0:
        raise StageARuntimeError("Stage A calibration tau is not frozen at 2.0")


@dataclass(frozen=True)
class StageATreatment:
    arm: str
    mask_key: str | None
    kind: str
    beta_advce: float | None = None
    advkd_multiplier: float | None = None
    beta_cleance: float | None = None
    clean_wrong_mode: str | None = None
    tau: float | None = None
    selected_attack_epsilon: float | None = None
    selected_attack_step_size: float | None = None
    extra_clean_ce: float | None = None
    bce_adv: float | None = None
    adaptive_advkd_gamma: float | None = None
    margin_coefficient: float | None = None
    margin_target_mode: str | None = None
    margin_gamma: float | None = None
    margin_floor: float | None = None
    margin_cap: float | None = None
    teacher_reliability_gate: bool = False
    iad_inspired: bool = False
    boundary_intervention: str | None = None
    boundary_coefficient: float | None = None
    boundary_epsilon: float = 1e-12
    # This is deliberately a narrow opt-in for the registered I100 online
    # state screen.  Ordinary Stage-A boundary treatments remain fixed-mask
    # only; the dedicated runtime supplies the checkpointable current router.
    online_state_s2: bool = False

    def __post_init__(self) -> None:
        if self.kind not in {"baseline", "advce", "soft_advkd", "advkd_advce", "clean_wrong", "broad"}:
            raise StageARuntimeError(f"unknown Stage A treatment kind: {self.kind}")
        if self.kind == "clean_wrong" and self.clean_wrong_mode not in {
            "clean_ce_only",
            "teacher_clean_gate",
            "clean_kd",
        }:
            raise StageARuntimeError("clean-wrong treatment requires an explicit mode")
        if self.kind in {"advce", "advkd_advce"} and (self.beta_advce is None or self.beta_advce < 0):
            raise StageARuntimeError("AdvCE treatments require a non-negative frozen coefficient")
        if self.kind == "soft_advkd" and self.tau != 2.0:
            raise StageARuntimeError("Stage A softening requires the frozen tau=2.0")
        if self.kind == "clean_wrong" and (self.beta_cleance is None or self.beta_cleance < 0):
            raise StageARuntimeError("clean-wrong treatments require a non-negative frozen coefficient")
        if self.margin_target_mode not in {None, "fixed", "teacher_zero", "teacher_floor", "teacher_abstain"}:
            raise StageARuntimeError("unknown adversarial margin target mode")
        if self.margin_target_mode is not None and self.margin_coefficient is None:
            raise StageARuntimeError("margin target treatments require a frozen margin coefficient")
        if self.margin_coefficient is not None and self.margin_target_mode is None:
            raise StageARuntimeError("margin coefficient requires a target mode")
        if self.margin_target_mode == "fixed" and self.margin_gamma is None:
            raise StageARuntimeError("fixed margin treatment requires gamma")
        if self.margin_target_mode in {"teacher_zero", "teacher_floor", "teacher_abstain"} and self.margin_cap is None:
            raise StageARuntimeError("Teacher margin treatment requires cap")
        if self.margin_target_mode == "teacher_floor" and self.margin_floor is None:
            raise StageARuntimeError("Teacher floor treatment requires floor")
        if self.margin_floor is not None and self.margin_cap is not None and self.margin_floor > self.margin_cap:
            raise StageARuntimeError("margin floor cannot exceed cap")
        if self.boundary_intervention not in {
            None,
            "pair_margin",
            "detached_boundary_distance",
            "secant_boundary_distance",
        }:
            raise StageARuntimeError("unknown boundary intervention")
        if (self.boundary_intervention is None) != (self.boundary_coefficient is None):
            raise StageARuntimeError("boundary intervention and coefficient must be supplied together")
        if self.boundary_intervention is not None and self.mask_key is None and not self.online_state_s2:
            raise StageARuntimeError("boundary intervention requires a fixed selected mask")
        if self.boundary_coefficient is not None and self.boundary_coefficient < 0:
            raise StageARuntimeError("boundary coefficient must be non-negative")
        if self.boundary_epsilon <= 0:
            raise StageARuntimeError("boundary epsilon must be positive")
        if self.kind == "advkd_advce" and (self.advkd_multiplier is None or not 0.0 <= self.advkd_multiplier <= 1.0):
            raise StageARuntimeError("AdvKD/AdvCE treatments require an AdvKD multiplier in [0, 1]")
        if self.kind == "baseline" and any(
            value is not None
            for value in (
                self.beta_advce,
                self.advkd_multiplier,
                self.beta_cleance,
                self.clean_wrong_mode,
                self.tau,
                self.boundary_intervention,
                self.boundary_coefficient,
            )
        ):
            raise StageARuntimeError("baseline treatment cannot carry treatment coefficients")
        if self.selected_attack_epsilon is not None and self.selected_attack_step_size is None:
            raise StageARuntimeError("selected attack epsilon requires a selected step size")
        if self.selected_attack_step_size is not None and self.selected_attack_epsilon is None:
            raise StageARuntimeError("selected attack step size requires a selected epsilon")
        for value in (
            self.selected_attack_epsilon,
            self.selected_attack_step_size,
            self.extra_clean_ce,
            self.bce_adv,
            self.adaptive_advkd_gamma,
            self.margin_coefficient,
            self.margin_gamma,
            self.margin_floor,
            self.margin_cap,
        ):
            if value is not None and value < 0:
                raise StageARuntimeError("broad treatment coefficients must be non-negative")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state_component_sha256(value: Any) -> str:
    """Hash a checkpoint component without child-only lineage metadata."""
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _epoch80_equivalence(payload: dict[str, Any]) -> dict[str, str]:
    required = (
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "rng",
        "sampler_epoch",
        "sampler_state",
        "sample_state",
        "global_step",
    )
    if any(key not in payload for key in required):
        raise StageARuntimeError("epoch-80 checkpoint lacks a parity component")
    return {key: _state_component_sha256(payload[key]) for key in required}


def _validate_shared_prefix_lineage(
    *, prefix_payload: dict[str, Any], experiment_parent_payload: dict[str, Any], experiment_parent_sha256: str
) -> dict[str, Any]:
    """Prove that a capture-only checkpoint belongs to this exact epoch-79 run.

    A matching router universe alone is insufficient: L2/L4 have the same
    CIFAR split and stable IDs, so accepting a foreign seed would silently
    compare different model trajectories.
    """
    prefix_lineage = prefix_payload.get("fork_lineage")
    experiment_config_hash = experiment_parent_payload.get("config_hash")
    if (
        not isinstance(prefix_lineage, dict)
        or not isinstance(experiment_config_hash, str)
        or prefix_lineage.get("parent_checkpoint_sha256") != experiment_parent_sha256
        or prefix_lineage.get("parent_config_hash") != experiment_config_hash
        or prefix_lineage.get("child_config_hash") != prefix_payload.get("config_hash")
    ):
        raise StageARuntimeError("shared-prefix checkpoint does not belong to the requested epoch-79 parent")
    return prefix_lineage


def _checkpoint_component_equivalence(payload: dict[str, Any], *, epoch: int) -> dict[str, str]:
    """Hash the complete resumable state at one shared fork boundary."""
    required = (
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "rng",
        "sampler_epoch",
        "sampler_state",
        "sample_state",
        "global_step",
    )
    if any(key not in payload for key in required):
        raise StageARuntimeError(f"epoch-{epoch} checkpoint lacks a parity component")
    return {key: _state_component_sha256(payload[key]) for key in required}


def _validate_online_state_prefix_lineage(
    *,
    prefix_payload: dict[str, Any],
    experiment_parent_payload: dict[str, Any],
    experiment_parent_sha256: str,
    prefix_epoch: int,
    expected_source_git_sha: str | None = None,
    expected_training_attack_identity_sha256: str | None = None,
) -> dict[str, Any]:
    """Prove an e100 prefix derives from this exact e99 causal parent."""
    lineage = prefix_payload.get("fork_lineage")
    parent_config_hash = experiment_parent_payload.get("config_hash")
    if (
        not isinstance(lineage, dict)
        or not isinstance(parent_config_hash, str)
        or lineage.get("parent_checkpoint_sha256") != experiment_parent_sha256
        or lineage.get("parent_config_hash") != parent_config_hash
        or lineage.get("child_config_hash") != prefix_payload.get("config_hash")
    ):
        raise StageARuntimeError("online shared prefix does not belong to the requested e99 parent")
    identity = lineage.get("online_state_s2")
    state = lineage.get("online_state_s2_state")
    if (
        not isinstance(identity, dict)
        or identity.get("arm") != "prefix"
        or identity.get("prefix_epoch") != prefix_epoch
        or identity.get("original_parent_checkpoint_sha256") != experiment_parent_sha256
        or not isinstance(state, dict)
    ):
        raise StageARuntimeError("online shared prefix lacks immutable e100 router state")
    if expected_source_git_sha is not None and lineage.get("source_git_sha") != expected_source_git_sha:
        raise StageARuntimeError("online shared prefix source Git SHA differs from the frozen child source")
    if (
        expected_training_attack_identity_sha256 is not None
        and lineage.get("training_attack_identity_sha256") != expected_training_attack_identity_sha256
    ):
        raise StageARuntimeError("online shared prefix training attack differs from the frozen child attack")
    return lineage


def _validate_online_state_s2_treatment(*, arm: str, treatment: StageATreatment) -> None:
    """Reject every treatment component outside the registered online screen.

    The public launcher constructs these objects itself, but the runtime is the
    last scientific boundary before an optimizer step.  In particular, an
    online preservation arm may not accidentally inherit a Clean-Wrong, S3,
    margin-floor, or adversarial-KD modifier from a generic Stage-A caller.
    """
    inactive = (
        treatment.beta_advce,
        treatment.advkd_multiplier,
        treatment.beta_cleance,
        treatment.clean_wrong_mode,
        treatment.tau,
        treatment.selected_attack_epsilon,
        treatment.selected_attack_step_size,
        treatment.extra_clean_ce,
        treatment.bce_adv,
        treatment.adaptive_advkd_gamma,
        treatment.margin_coefficient,
        treatment.margin_target_mode,
        treatment.margin_gamma,
        treatment.margin_floor,
        treatment.margin_cap,
    )
    if any(value is not None for value in inactive) or treatment.teacher_reliability_gate or treatment.iad_inspired:
        raise StageARuntimeError("online S2 screen may not stack another Stage-A treatment component")
    if arm in {"prefix", "control"}:
        if treatment.kind != "baseline" or treatment.boundary_intervention is not None:
            raise StageARuntimeError("online prefix/control must be a baseline treatment with router telemetry only")
        return
    expected = {"pmp": "pair_margin", "dbdp": "detached_boundary_distance"}.get(arm)
    expected_coefficient = {"pmp": 0.05380932585058825, "dbdp": 31.649566509850324}.get(arm)
    if (
        expected is None
        or expected_coefficient is None
        or treatment.kind != "broad"
        or treatment.boundary_intervention != expected
        or treatment.boundary_coefficient != expected_coefficient
        or treatment.boundary_epsilon != 1e-12
    ):
        raise StageARuntimeError("online preservation arm does not match its registered boundary treatment")


def _require_attack_identity(actual: dict[str, object], expected: object, *, label: str) -> None:
    if not isinstance(expected, dict) or set(expected) != set(actual):
        raise StageARuntimeError(f"dynamic routing {label} attack contract must contain the complete exact identity")
    if expected != actual:
        raise StageARuntimeError(f"dynamic routing {label} attack contract differs from the parent")


def _epoch80_gate(*, own: dict[str, Any], peer_path: Path, timeout_seconds: float = 600.0) -> None:
    """Block epoch 81 until the paired arm proves common epoch-80 state."""
    deadline = time.monotonic() + timeout_seconds
    peer: dict[str, Any] | None = None
    while peer is None:
        if time.monotonic() >= deadline:
            raise StageARuntimeError(f"timed out waiting for paired epoch-80 state: {peer_path}")
        if peer_path.is_file():
            try:
                peer = _load_json(peer_path)
            except (OSError, json.JSONDecodeError):
                pass
        if peer is None:
            time.sleep(1.0)
    if own.get("components") != peer.get("components"):
        raise StageARuntimeError("fixed/dynamic epoch-80 model/optimizer/scheduler/RNG state differs")
    own_capture = Path(str(own.get("capture_path", "")))
    peer_capture = peer_path.parent / "routing-capture-mask.json"
    if not own_capture.is_file() or not peer_capture.is_file():
        raise StageARuntimeError("paired epoch-80 routing capture artifact is missing")
    if _load_json(own_capture).get("selected_ids_sha256") != _load_json(peer_capture).get("selected_ids_sha256"):
        raise StageARuntimeError("fixed/dynamic epoch-80 capture action differs")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StageARuntimeError(f"expected JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _mask_from_overlay(path: Path, key: str, *, expected_anchor_epoch: int = 79) -> FixedInterventionMask:
    payload = _load_json(path)
    masks = payload.get("masks")
    if payload.get("anchor_epoch") != expected_anchor_epoch or not isinstance(masks, dict):
        raise StageARuntimeError(f"Stage A mask artifact is not an epoch-{expected_anchor_epoch} overlay bundle")
    raw = masks.get(key)
    if not isinstance(raw, dict) or not isinstance(raw.get("selected_ids"), list):
        raise StageARuntimeError(f"Stage A mask key is missing: {key}")
    ids = [item for item in raw["selected_ids"] if isinstance(item, int) and not isinstance(item, bool)]
    if len(ids) != len(raw["selected_ids"]) or len(set(ids)) != len(ids):
        raise StageARuntimeError("Stage A mask contains invalid or duplicate stable IDs")
    digest = hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()
    declared_digest = raw.get("selected_ids_sha256")
    if declared_digest is not None and declared_digest != digest:
        raise StageARuntimeError(f"Stage A mask stable-ID hash mismatch for {key}")
    declared_count = raw.get("selected_count")
    if declared_count is not None and declared_count != len(ids):
        raise StageARuntimeError(f"Stage A mask selected-count mismatch for {key}")
    counts = {int(name): int(value) for name, value in raw.get("selected_class_counts", {}).items()}
    if counts and sum(counts.values()) != len(ids):
        raise StageARuntimeError(f"Stage A mask class-count total mismatch for {key}")
    return FixedInterventionMask(frozenset(ids), digest, counts)


def _validate_mask_lineage(path: Path, *, parent_sha256: str, expected_anchor_epoch: int, key: str) -> None:
    """Validate the campaign mask binding before any training batch is run."""
    payload = _load_json(path)
    if payload.get("contract") == "ert_rslad_i100_s2_rbp_masks_v1":
        if payload.get("parent_checkpoint_sha256") != parent_sha256:
            raise StageARuntimeError("S2 RBP mask parent checkpoint does not match the supplied parent")
        if payload.get("anchor_epoch") != expected_anchor_epoch:
            raise StageARuntimeError("S2 RBP mask anchor epoch does not match the supplied parent")
        raw = payload.get("masks", {}).get(key)
        if not isinstance(raw, dict) or raw.get("namespace") != "train":
            raise StageARuntimeError("S2 RBP training mask namespace is not train")


def _arm_hash(
    parent_hash: str,
    treatment: StageATreatment,
    source_sha: str,
    *,
    continuation_seed: int | None = None,
    rng_source_seeds: RNGSourceSeeds | None = None,
    shuffle_augmentation_seeds: ShuffleAugmentationSeeds | None = None,
    dynamic_s3: dict[str, Any] | None = None,
    online_state_s2: dict[str, Any] | None = None,
) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "parent_config_hash": parent_hash,
                "treatment": treatment.__dict__,
                "continuation_seed": continuation_seed,
                "rng_source_seeds": None if rng_source_seeds is None else rng_source_seeds.as_dict(),
                "shuffle_augmentation_seeds": (
                    None if shuffle_augmentation_seeds is None else shuffle_augmentation_seeds.as_dict()
                ),
                "dynamic_s3": dynamic_s3,
                "online_state_s2": online_state_s2,
                "source_git_sha": source_sha,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def run_stage_a_arm(
    *,
    parent_config_path: Path,
    parent_checkpoint: Path,
    mask_path: Path | None,
    output_dir: Path,
    treatment: StageATreatment,
    calibration: dict[str, Any],
    device: torch.device,
    end_epoch: int,
    horizon_epochs: tuple[int, ...] = (84,),
    run_namespace: str = "stage-a",
    continuation_seed: int | None = None,
    rng_source_seeds: RNGSourceSeeds | None = None,
    shuffle_augmentation_seeds: ShuffleAugmentationSeeds | None = None,
    expected_parent_checkpoint_sha256: str | None = None,
    dynamic_s3_arm: str | None = None,
    dynamic_s3_beta_advce: float | None = None,
    dynamic_s3_attack_contract: dict[str, object] | None = None,
    dynamic_s3_endpoint_contract: dict[str, object] | None = None,
    dynamic_s3_peer_epoch80_state: Path | None = None,
    dynamic_s3_shared_prefix_checkpoint: Path | None = None,
    dynamic_s3_experiment_parent_checkpoint: Path | None = None,
    online_state_s2: dict[str, Any] | None = None,
    resume_epoch: int | None = None,
    mask_anchor_epoch: int | None = None,
    force_sample_keyed_attack: bool = False,
    canary_max_train_batches: int | None = None,
    canary_max_validation_batches: int | None = None,
    canary_source_ids: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    config = load_config(parent_config_path)
    canary_enabled = canary_max_train_batches is not None or canary_max_validation_batches is not None
    if canary_enabled:
        if (
            not isinstance(canary_max_train_batches, int)
            or not isinstance(canary_max_validation_batches, int)
            or canary_max_train_batches <= 0
            or canary_max_validation_batches <= 0
            or canary_max_train_batches > 8
            or canary_max_validation_batches > 2
        ):
            raise StageARuntimeError("public online canary requires bounded 1–8 train and 1–2 validation batches")
        # A bounded public smoke must not create or resume an online
        # production run.  ``offline_sync`` produces an offline W&B segment
        # while remaining valid for the production-tier parent schema.
        config = config.model_copy(update={"tracking": _offline_canary_tracking(config.tracking)})
    if force_sample_keyed_attack:
        # The historical parent config predates the sample-keyed random-start
        # contract.  Keep the checkpoint/config hash as the parent identity,
        # but explicitly bind this continuation's actual attack to the
        # registered sample-keyed KL-PGD10 identity.
        keyed_attack = config.method.attack.model_copy(update={"random_start_keying": "sample_keyed_v1"})
        config = config.model_copy(update={"method": config.method.model_copy(update={"attack": keyed_attack})})
        if keyed_attack.identity_sha256() != SAMPLE_KEYED_KL10_ATTACK_IDENTITY_SHA256:
            raise StageARuntimeError("forced training attack does not match the registered sample-keyed KL10 identity")
    if config.training.deterministic:
        # Stage-A forks are compared at an exact parent boundary.  The
        # ordinary train CLI applies these flags, but this standalone runtime
        # must do so as well or independent CUDA forks can diverge before the
        # first treatment visit.
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if config.method.id != "rslad" or config.method.attack.loss != "kl" or config.method.attack.steps != 10:
        raise StageARuntimeError("Stage A parent is not the observed RSLAD KL-PGD10 run")
    if config.method.attack.kl_target != "teacher_clean":
        raise StageARuntimeError("Stage A parent attack target is not teacher_clean")
    if (
        config.method.selection_attack is None
        or config.method.selection_attack.loss != "ce"
        or config.method.selection_attack.steps != 20
    ):
        raise StageARuntimeError("Stage A parent selection attack is not CE-PGD20")
    payload = torch.load(parent_checkpoint, map_location="cpu", weights_only=False)
    actual_parent_checkpoint_sha256 = _sha256(parent_checkpoint)
    if (
        expected_parent_checkpoint_sha256 is not None
        and actual_parent_checkpoint_sha256 != expected_parent_checkpoint_sha256
    ):
        raise StageARuntimeError("parent checkpoint bytes do not match the registered SHA-256")
    online_arm: str | None = None
    online_prefix_epoch = 100
    online_original_parent_checkpoint: Path | None = None
    online_original_parent_payload: dict[str, Any] | None = None
    online_original_parent_sha256: str | None = None
    online_thresholds_path: Path | None = None
    if online_state_s2 is not None:
        if dynamic_s3_arm is not None:
            raise StageARuntimeError("online S2 and dynamic S3 routing are mutually exclusive")
        if mask_path is not None or treatment.mask_key is not None:
            raise StageARuntimeError("online S2 routing cannot be combined with a fixed treatment mask")
        if not isinstance(online_state_s2, dict):
            raise StageARuntimeError("online S2 routing specification must be a JSON-compatible object")
        online_arm = online_state_s2.get("arm")
        if online_arm not in {"prefix", "control", "pmp", "dbdp"}:
            raise StageARuntimeError("online S2 routing arm must be prefix, control, pmp, or dbdp")
        _validate_online_state_s2_treatment(arm=online_arm, treatment=treatment)
        if online_state_s2.get("prefix_epoch", 100) != 100:
            raise StageARuntimeError("online S2 screen is frozen to a shared epoch-100 prefix")
        raw_original_parent = online_state_s2.get("original_parent_checkpoint")
        if not isinstance(raw_original_parent, str):
            raise StageARuntimeError("online S2 routing requires the exact original e99 parent path")
        online_original_parent_checkpoint = Path(raw_original_parent)
        if not online_original_parent_checkpoint.is_file():
            raise StageARuntimeError("online S2 original e99 parent is unavailable")
        online_original_parent_payload = torch.load(
            online_original_parent_checkpoint, map_location="cpu", weights_only=False
        )
        online_original_parent_sha256 = _sha256(online_original_parent_checkpoint)
        if (
            not isinstance(online_original_parent_payload, dict)
            or online_original_parent_payload.get("epoch") != 99
            or online_original_parent_payload.get("epoch_boundary") != "end"
        ):
            raise StageARuntimeError("online S2 original parent must be the exact e99 end-boundary checkpoint")
        if online_state_s2.get("original_parent_checkpoint_sha256") != online_original_parent_sha256:
            raise StageARuntimeError("online S2 original parent bytes differ from the frozen SHA-256")
        if online_arm == "prefix":
            if actual_parent_checkpoint_sha256 != online_original_parent_sha256:
                raise StageARuntimeError("shared e100 prefix must resume the immutable original e99 parent")
            if resume_epoch not in {None, 99}:
                raise StageARuntimeError("shared e100 prefix must resume exactly from epoch 99")
            if not treatment.online_state_s2:
                raise StageARuntimeError("shared e100 prefix must be an online-observation-only baseline treatment")
        else:
            if (
                not isinstance(payload, dict)
                or payload.get("epoch") != 100
                or payload.get("epoch_boundary") != "end"
                or resume_epoch not in {None, 100}
            ):
                raise StageARuntimeError("online S2 child must resume the exact shared e100 end-boundary checkpoint")
            _validate_online_state_prefix_lineage(
                prefix_payload=payload,
                experiment_parent_payload=online_original_parent_payload,
                experiment_parent_sha256=online_original_parent_sha256,
                prefix_epoch=online_prefix_epoch,
            )
            raw_thresholds = online_state_s2.get("thresholds_path")
            if not isinstance(raw_thresholds, str):
                raise StageARuntimeError("online S2 child requires the frozen e100 threshold artifact")
            online_thresholds_path = Path(raw_thresholds)
            if not online_thresholds_path.is_file():
                raise StageARuntimeError("online S2 frozen threshold artifact is unavailable")
    shared_prefix = dynamic_s3_shared_prefix_checkpoint is not None
    if shared_prefix:
        if dynamic_s3_shared_prefix_checkpoint.resolve() != parent_checkpoint.resolve():
            raise StageARuntimeError("shared-prefix checkpoint must be the actual resume checkpoint")
        if not isinstance(payload, dict) or payload.get("epoch") != 80 or payload.get("epoch_boundary") != "end":
            raise StageARuntimeError("shared-prefix continuation requires an exact epoch-80 end-boundary checkpoint")
        if dynamic_s3_arm not in {"fixed", "dynamic"}:
            raise StageARuntimeError("only fixed/dynamic S3 arms may resume a shared capture prefix")
        if dynamic_s3_experiment_parent_checkpoint is None:
            raise StageARuntimeError("shared-prefix continuation requires the immutable epoch-79 experiment parent")
        experiment_parent_payload = torch.load(
            dynamic_s3_experiment_parent_checkpoint, map_location="cpu", weights_only=False
        )
        if (
            not isinstance(experiment_parent_payload, dict)
            or experiment_parent_payload.get("epoch") != 79
            or experiment_parent_payload.get("epoch_boundary") != "end"
        ):
            raise StageARuntimeError("shared-prefix experiment parent must be an exact epoch-79 boundary")
        start_epoch = 81
        experiment_parent_sha256 = _sha256(dynamic_s3_experiment_parent_checkpoint)
    else:
        if not isinstance(payload, dict) or payload.get("epoch_boundary") != "end":
            raise StageARuntimeError("Stage A requires an exact end-boundary parent")
        payload_epoch = payload.get("epoch")
        if isinstance(payload_epoch, bool) or not isinstance(payload_epoch, int):
            raise StageARuntimeError("Stage A parent has no valid integer epoch")
        if resume_epoch is None:
            resume_epoch = 79
        if resume_epoch != payload_epoch:
            raise StageARuntimeError("resume_epoch does not match the parent checkpoint payload")
        start_epoch = resume_epoch + 1
        experiment_parent_sha256 = _sha256(parent_checkpoint)
    if online_arm is not None:
        assert online_original_parent_sha256 is not None
        experiment_parent_sha256 = online_original_parent_sha256
        # The online router persists one state record per stable ID and its
        # entry/re-entry counters are defined over a complete epoch sequence.
        # This registered screen has no rank-synchronization protocol, so
        # silently accepting a distributed launch would corrupt those
        # per-ID trajectories.
        if get_world_size() != 1:
            raise StageARuntimeError("online S2 state persistence screen requires world_size == 1")
    parent_hash = payload.get("config_hash")
    if not isinstance(parent_hash, str) or len(parent_hash) != 64:
        raise StageARuntimeError("parent checkpoint lacks a valid config hash")
    if end_epoch <= start_epoch:
        raise StageARuntimeError("Stage A endpoint must leave at least one epoch after epoch 79")
    if mask_anchor_epoch is None:
        mask_anchor_epoch = resume_epoch if not shared_prefix else 79
    if mask_anchor_epoch < 0:
        raise StageARuntimeError("mask anchor epoch must be non-negative")
    _validate_horizons(horizon_epochs, end_epoch, first_epoch=start_epoch)
    if not run_namespace or any(char.isspace() for char in run_namespace):
        raise StageARuntimeError("run namespace must be a non-empty token")
    _validate_stage_a_calibration(treatment, calibration)
    source_state = collect_git_state(Path.cwd())
    source_sha = source_state.get("sha")
    if source_state.get("dirty") is not False or not isinstance(source_sha, str):
        raise StageARuntimeError("Stage A runtime requires a clean Git tree")
    if (dynamic_s3_arm is None) != (dynamic_s3_beta_advce is None):
        raise StageARuntimeError("dynamic S3 arm and frozen coefficient must be supplied together")
    if dynamic_s3_arm not in {
        None,
        "baseline",
        "capture",
        "fixed",
        "dynamic",
        "instant",
        "majority3",
        "majority3_exit2",
    }:
        raise StageARuntimeError("unknown dynamic S3 arm")
    if dynamic_s3_beta_advce is not None and dynamic_s3_beta_advce != 0.075:
        raise StageARuntimeError("dynamic S3 recovery requires the frozen AdvCE coefficient 0.075")
    if dynamic_s3_arm is not None:
        if config.method.selection_attack is None:
            raise StageARuntimeError("dynamic S3 requires a saved CE-PGD20 endpoint attack")
        _require_attack_identity(config.method.attack.identity(), dynamic_s3_attack_contract, label="training")
        _require_attack_identity(
            config.method.selection_attack.identity(), dynamic_s3_endpoint_contract, label="endpoint"
        )
        if dynamic_s3_arm in {"fixed", "dynamic"} and not shared_prefix and dynamic_s3_peer_epoch80_state is None:
            raise StageARuntimeError("fixed/dynamic S3 arms require a paired epoch-80 gate path")
    dynamic_s3_identity = (
        None
        if dynamic_s3_arm is None
        else {"arm": dynamic_s3_arm, "beta_advce": dynamic_s3_beta_advce, "timing": "same_step_pre_update"}
    )
    online_state_s2_identity: dict[str, Any] | None = None
    if online_arm is not None:
        assert online_state_s2 is not None and online_original_parent_sha256 is not None
        expected_training = online_state_s2.get("training_attack")
        expected_endpoint = online_state_s2.get("endpoint_attack")
        _require_attack_identity(config.method.attack.identity(), expected_training, label="training")
        if config.method.selection_attack is None:
            raise StageARuntimeError("online S2 routing requires the frozen CE-PGD20 endpoint attack")
        _require_attack_identity(config.method.selection_attack.identity(), expected_endpoint, label="endpoint")
        online_state_s2_identity = {
            "arm": online_arm,
            "timing": "same_step_pre_update",
            "prefix_epoch": online_prefix_epoch,
            "router_margin": "global_true_logit_minus_global_max_nontrue_logit",
            "action_pair": "current_student_strongest_nontrue_reused_for_teacher",
            "original_parent_checkpoint_sha256": online_original_parent_sha256,
            "threshold_artifact_sha256": None if online_thresholds_path is None else _sha256(online_thresholds_path),
        }
    if continuation_seed is not None and (isinstance(continuation_seed, bool) or continuation_seed < 0):
        raise StageARuntimeError("continuation seed must be a non-negative integer")
    if continuation_seed is not None and rng_source_seeds is not None:
        raise StageARuntimeError("continuation_seed cannot be combined with explicit RNG source seeds")
    if continuation_seed is not None and shuffle_augmentation_seeds is not None:
        raise StageARuntimeError("continuation_seed cannot be combined with split shuffle/augmentation seeds")
    if rng_source_seeds is not None and shuffle_augmentation_seeds is not None:
        raise StageARuntimeError("data RNG seeds cannot be combined with split shuffle/augmentation seeds")
    explicit_rng_source_seeds = rng_source_seeds is not None
    if continuation_seed is not None:
        rng_source_seeds = RNGSourceSeeds(
            # Preserve the historical continuation contract: the legacy
            # scalar never changed sampler/augmentation streams.
            data_seed=config.seeds.data_order,
            attack_seed=continuation_seed,
            other_seed=continuation_seed,
        )
    arm_hash = _arm_hash(
        parent_hash,
        treatment,
        source_sha,
        continuation_seed=continuation_seed,
        rng_source_seeds=None if continuation_seed is not None else rng_source_seeds,
        shuffle_augmentation_seeds=shuffle_augmentation_seeds,
        dynamic_s3=dynamic_s3_identity,
        online_state_s2=online_state_s2_identity,
    )
    canary_selected_source_ids: list[int] | None = None
    train_dataset, validation_dataset = build_train_validation_views(
        config.dataset,
        validation_fraction=config.training.validation_fraction,
        split_seed=config.seeds.split,
        augmentation_seed=(
            config.seeds.augmentation
            if not explicit_rng_source_seeds and shuffle_augmentation_seeds is None
            else (
                shuffle_augmentation_seeds.augmentation_seed
                if shuffle_augmentation_seeds is not None
                else rng_source_seeds.data_seed
            )
        ),
    )
    if canary_enabled:
        # Resolve a fixed real-data source-ID subset before constructing the
        # runtime loader.  Prefix and child canaries share this exact ID set;
        # no state outside the bounded canary universe is fabricated.
        try:
            full_train_ids = [int(value) for value in train_dataset.indices]
        except (AttributeError, TypeError) as exc:
            raise StageARuntimeError("public online canary cannot recover stable train IDs") from exc
        source_to_position = {sample_id: position for position, sample_id in enumerate(full_train_ids)}
        if len(source_to_position) != len(full_train_ids):
            raise StageARuntimeError("public online canary found duplicate train source IDs")
        if canary_source_ids is None:
            selector = EpochShuffleSampler(
                len(train_dataset), seed=config.seeds.data_order, rank=0, world_size=1, shuffle=True
            )
            selector.set_epoch(start_epoch)
            positions = [
                int(item.index)
                for _, item in zip(range(canary_max_train_batches * config.training.per_rank_batch_size), selector)
            ]
            selected_source_ids = [full_train_ids[position] for position in positions]
        else:
            selected_source_ids = [int(value) for value in canary_source_ids]
            if len(selected_source_ids) != len(set(selected_source_ids)) or not selected_source_ids:
                raise StageARuntimeError("public online canary source-ID set is empty or duplicated")
            if set(selected_source_ids) - set(source_to_position):
                raise StageARuntimeError("public online canary source IDs are outside the train universe")
            if len(selected_source_ids) > canary_max_train_batches * config.training.per_rank_batch_size:
                raise StageARuntimeError("public online canary source-ID set exceeds its bounded train quota")
        canary_selected_source_ids = list(selected_source_ids)
        positions = [source_to_position[source_id] for source_id in selected_source_ids]
        train_dataset = _CanaryEpochSubset(train_dataset, positions, selected_source_ids)
        validation_positions = list(
            range(min(len(validation_dataset), canary_max_validation_batches * config.training.per_rank_batch_size))
        )
        if not validation_positions:
            raise StageARuntimeError("public online canary validation view is empty")
        validation_dataset = _CanaryEpochSubset(
            validation_dataset,
            validation_positions,
            [int(value) for value in validation_positions],
        )
    dynamic_s3_router = None
    online_state_s2_router = None
    # CIFAR train views retain sparse original source IDs in the subset; both
    # online routers bind every observed row to this exact immutable universe.
    train_labels: dict[int, int] | None = None
    if dynamic_s3_arm is not None or online_arm is not None:
        train_labels = _stable_train_labels(train_dataset)
    if dynamic_s3_arm is not None:
        from ard.analysis.ert_dynamic_s3_recovery import DynamicS3Router

        assert train_labels is not None
        dynamic_s3_router = DynamicS3Router(
            arm=dynamic_s3_arm,
            train_labels=train_labels,
            output_dir=output_dir,
            capture_epoch=80,
        )
    if online_arm is not None:
        from ard.analysis.ert_i100_online_state_s2 import OnlineStateS2Router

        assert train_labels is not None and online_original_parent_sha256 is not None
        online_state_s2_router = OnlineStateS2Router(
            arm=online_arm,
            train_labels=train_labels,
            output_dir=output_dir,
            original_parent_checkpoint_sha256=online_original_parent_sha256,
            prefix_epoch=online_prefix_epoch,
            thresholds_path=online_thresholds_path,
        )
    sampler = EpochShuffleSampler(
        len(train_dataset), seed=config.seeds.data_order, rank=get_rank(), world_size=get_world_size(), shuffle=True
    )
    validation_sampler = EpochShuffleSampler(
        len(validation_dataset),
        seed=config.seeds.data_order,
        rank=get_rank(),
        world_size=get_world_size(),
        shuffle=False,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=config.training.per_rank_batch_size,
        sampler=sampler,
        num_workers=config.training.num_workers,
        collate_fn=collate_indexed,
        generator=data_loader_generator(config.seeds.data_order),
        worker_init_fn=seed_data_loader_worker,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.training.per_rank_batch_size,
        sampler=validation_sampler,
        num_workers=config.training.num_workers,
        collate_fn=collate_indexed,
        generator=data_loader_generator(config.seeds.data_order + 1),
        worker_init_fn=seed_data_loader_worker,
    )
    student = build_student(config.student, tier=config.tier).to(device)
    teacher = build_teacher(config.teacher, tier=config.tier).to(device) if config.teacher is not None else None
    if teacher is None:
        raise StageARuntimeError("Stage A requires a frozen Teacher")
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    optimizer = SGD(
        student.parameters(),
        lr=config.optimizer.learning_rate,
        momentum=config.optimizer.momentum,
        weight_decay=config.optimizer.weight_decay,
        nesterov=config.optimizer.nesterov,
    )
    scheduler = build_scheduler(optimizer, config.scheduler)
    sample_store = SampleStateStore(ema_decay=config.method.student_ema_decay)
    mask = None
    teacher_reliability_mask = None
    if treatment.mask_key is not None:
        if mask_path is None:
            raise StageARuntimeError("selected Stage A treatment requires a mask path")
        _validate_mask_lineage(
            mask_path,
            parent_sha256=actual_parent_checkpoint_sha256,
            expected_anchor_epoch=mask_anchor_epoch,
            key=treatment.mask_key,
        )
        mask = _mask_from_overlay(mask_path, treatment.mask_key, expected_anchor_epoch=mask_anchor_epoch)
        if treatment.teacher_reliability_gate:
            teacher_reliability_mask = _mask_from_overlay(
                mask_path,
                "student_clean_wrong_teacher_clean_correct",
                expected_anchor_epoch=mask_anchor_epoch,
            )
    target_policy = (
        TeacherOnlyTemperatureTargetPolicy(target_temperature=treatment.tau, baseline_temperature=1.0)
        if treatment.kind == "soft_advkd"
        else None
    )
    objective = RSLADObjective(
        temperature=config.method.temperature,
        temperature_squared=config.method.temperature_squared,
    )
    _prepare_stage_output_dir(output_dir)
    shared_epoch80: dict[str, Any] | None = None
    online_prefix_state: dict[str, Any] | None = None
    if shared_prefix:
        if dynamic_s3_router is None:
            raise StageARuntimeError("shared-prefix continuation requires a dynamic S3 router")
        prefix_lineage = _validate_shared_prefix_lineage(
            prefix_payload=payload,
            experiment_parent_payload=experiment_parent_payload,
            experiment_parent_sha256=experiment_parent_sha256,
        )
        if not isinstance(prefix_lineage.get("dynamic_s3_state"), dict):
            raise StageARuntimeError("shared-prefix checkpoint lacks persisted dynamic S3 capture state")
        prefix_identity = prefix_lineage.get("dynamic_s3")
        if (
            not isinstance(prefix_identity, dict)
            or prefix_identity.get("arm") != "capture"
            or prefix_identity.get("beta_advce") != 0.075
        ):
            raise StageARuntimeError("shared-prefix checkpoint does not attest the frozen treated capture arm")
        prefix_dynamic = prefix_lineage["dynamic_s3_state"]
        if prefix_dynamic.get("arm") != "capture":
            raise StageARuntimeError("shared-prefix checkpoint was not produced by the capture-only arm")
        dynamic_s3_router.adopt_capture_state(prefix_dynamic)
        raw_state = prefix_dynamic.get("state_paths", {}).get("80")
        if not isinstance(raw_state, dict) or not isinstance(raw_state.get("path"), str):
            raise StageARuntimeError("shared-prefix checkpoint lacks its epoch-80 state artifact binding")
        prefix_state_path = Path(raw_state["path"])
        if not prefix_state_path.is_file() or raw_state.get("sha256") != _sha256(prefix_state_path):
            raise StageARuntimeError("shared-prefix epoch-80 state artifact hash does not match checkpoint lineage")
        child_state_path = output_dir / "dynamic-state" / "epoch-80.parquet"
        child_state_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(prefix_state_path, child_state_path)
        if _sha256(child_state_path) != raw_state["sha256"]:
            raise StageARuntimeError("copied shared-prefix epoch-80 state artifact hash mismatch")
        dynamic_s3_router.register_existing_epoch_state(epoch=80, path=child_state_path)
        shared_epoch80 = {
            "path": str(parent_checkpoint.resolve()),
            "checkpoint_sha256": _sha256(parent_checkpoint),
            "components": _epoch80_equivalence(payload),
            "capture_path": str((output_dir / "routing-capture-mask.json").resolve()),
            "shared_prefix": True,
        }
        _write_json_atomic(output_dir / "epoch80-routing-state.json", shared_epoch80)
    if online_arm is not None and online_arm != "prefix":
        if (
            online_state_s2_router is None
            or online_original_parent_payload is None
            or online_original_parent_sha256 is None
        ):
            raise StageARuntimeError("online S2 child router/prefix lineage was not constructed")
        prefix_lineage = _validate_online_state_prefix_lineage(
            prefix_payload=payload,
            experiment_parent_payload=online_original_parent_payload,
            experiment_parent_sha256=online_original_parent_sha256,
            prefix_epoch=online_prefix_epoch,
            expected_source_git_sha=source_sha,
            expected_training_attack_identity_sha256=config.method.attack.identity_sha256(),
        )
        prefix_router_state = prefix_lineage.get("online_state_s2_state")
        if not isinstance(prefix_router_state, dict):
            raise StageARuntimeError("shared e100 checkpoint lacks online router state")
        raw_materialized_state_path = online_state_s2.get("prefix_state_materialized_path")
        materialized_state_path = None
        if raw_materialized_state_path is not None:
            if not isinstance(raw_materialized_state_path, str):
                raise StageARuntimeError("online shared-prefix state materialization path must be a string")
            materialized_state_path = Path(raw_materialized_state_path)
        online_state_s2_router.adopt_prefix_state(
            prefix_router_state,
            materialized_state_path=materialized_state_path,
            prefix_checkpoint_sha256=actual_parent_checkpoint_sha256,
            source_git_sha=source_sha,
            training_attack_identity_sha256=config.method.attack.identity_sha256(),
        )
        online_prefix_state = {
            "path": str(parent_checkpoint.resolve()),
            "checkpoint_sha256": _sha256(parent_checkpoint),
            "components": _checkpoint_component_equivalence(payload, epoch=online_prefix_epoch),
            "threshold_artifact_sha256": None if online_thresholds_path is None else _sha256(online_thresholds_path),
            "shared_prefix": True,
        }
        _write_json_atomic(output_dir / "epoch100-online-state.json", online_prefix_state)
    # Include the immutable source revision so a prior canary or interrupted
    # launch can never collide with a new production arm using the same label.
    run_id = _tracking_run_id(
        run_namespace=run_namespace,
        seed=str(config.seeds.model_init),
        arm=treatment.arm,
        source_sha=source_sha,
    )
    tracked_config = config.model_copy(
        update={
            "output_dir": output_dir,
            "tracking": config.tracking.model_copy(
                update={
                    "run_id": run_id,
                    "name": f"ert-{run_namespace}-{config.seeds.model_init}-{treatment.arm}",
                    "group": f"ert-{run_namespace}-{config.teacher.registry_id or 'teacherless'}",
                }
            ),
            "tracker_run_id": run_id,
        }
    )
    if online_arm is not None and tracked_config.tracking.artifact_retention != "metrics_only":
        raise StageARuntimeError("online S2 screen requires the registered W&B metrics-only artifact policy")
    trainer = Trainer(
        model=student,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        attack=LinfPGD(config.method.attack),
        selection_attack=LinfPGD(config.method.selection_attack),
        objective=objective,
        policy=RSLADBaselinePolicy(),
        device=device,
        output_dir=output_dir,
        config_hash=arm_hash,
        seed=config.seeds.train_attack,
        evaluation_attack_seed=config.seeds.evaluation_attack,
        tracker_run_id=run_id,
        teacher=teacher,
        sample_store=sample_store,
        target_policy=target_policy,
        intervention_mask=mask,
        adversarial_kd_multiplier=treatment.advkd_multiplier,
        clean_ce_coefficient=treatment.beta_cleance,
        clean_wrong_mode=treatment.clean_wrong_mode,
        clean_wrong_attack_skip=treatment.kind == "clean_wrong",
        selected_attack_epsilon=treatment.selected_attack_epsilon,
        selected_attack_step_size=treatment.selected_attack_step_size,
        extra_clean_ce_coefficient=treatment.extra_clean_ce,
        adversarial_bce_coefficient=treatment.bce_adv,
        adaptive_advkd_gamma=treatment.adaptive_advkd_gamma,
        margin_coefficient=treatment.margin_coefficient,
        margin_target_mode=treatment.margin_target_mode,
        margin_gamma=treatment.margin_gamma,
        margin_floor=treatment.margin_floor,
        margin_cap=treatment.margin_cap,
        teacher_clean_reliability_mask=teacher_reliability_mask,
        iad_inspired=treatment.iad_inspired,
        boundary_intervention=treatment.boundary_intervention,
        boundary_coefficient=treatment.boundary_coefficient,
        boundary_epsilon=treatment.boundary_epsilon,
        dynamic_s3_router=dynamic_s3_router,
        online_state_s2_router=online_state_s2_router,
        # The baseline diagnostic arm observes the exact same state but
        # deliberately ignores the decision, so it never receives AdvCE.
        adversarial_ce_coefficient=(
            dynamic_s3_beta_advce
            if dynamic_s3_arm in {"capture", "fixed", "dynamic", "instant", "majority3", "majority3_exit2"}
            else treatment.beta_advce
        ),
        observation_profile="teacher_response",
    )
    # Restore the complete epoch-79 optimizer/scheduler/RNG/sampler/sample
    # state under the parent identity, then switch only the child output hash.
    state = load_checkpoint(
        parent_checkpoint,
        model=student,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        sampler=sampler,
        expected_config_hash=parent_hash,
        device=device,
    )
    trainer.global_step = state.global_step
    trainer.best_metric = float("-inf")
    trainer.selection_metadata = dict(state.selection_metadata)
    trainer.selection_metadata["scope"] = "stage_a_post_parent"
    trainer.selection_metadata["selected_epoch"] = None
    trainer.tracker_run_id = run_id
    trainer.sample_state = state.sample_state
    sample_store.load_state_dict(state.sample_state)
    if rng_source_seeds is not None:
        # The epoch-79 optimizer/sampler/sample state remains the exact parent
        # state. Only the explicitly selected post-resume streams are rebound.
        reseed_other_stream(rng_source_seeds.other_seed)
        if explicit_rng_source_seeds:
            if loader.generator is None:
                raise StageARuntimeError("training DataLoader must expose its data-side generator")
            reseed_data_stream(
                dataset=train_dataset,
                sampler=sampler,
                loader_generator=loader.generator,
                seed=rng_source_seeds.data_seed,
            )
        trainer.seed = rng_source_seeds.attack_seed
    if shuffle_augmentation_seeds is not None:
        reseed_other_stream(shuffle_augmentation_seeds.other_seed)
        reseed_shuffle_augmentation_stream(
            dataset=train_dataset,
            sampler=sampler,
            shuffle_seed=shuffle_augmentation_seeds.shuffle_seed,
            augmentation_seed=shuffle_augmentation_seeds.augmentation_seed,
        )
        trainer.seed = shuffle_augmentation_seeds.attack_seed
    trainer.fork_lineage = {
        "kind": "ert_stage_a_treatment_v1",
        "arm": treatment.arm,
        "parent_checkpoint_sha256": _sha256(parent_checkpoint),
        "parent_config_hash": parent_hash,
        "parent_epoch": int(payload["epoch"]),
        "experiment_parent_checkpoint_sha256": experiment_parent_sha256,
        "experiment_parent_epoch": (99 if online_arm is not None else 79 if shared_prefix else int(payload["epoch"])),
        "shared_prefix_checkpoint_sha256": _sha256(parent_checkpoint) if shared_prefix else None,
        "shared_prefix": shared_prefix,
        "online_state_s2": online_state_s2_identity,
        "online_shared_prefix_checkpoint_sha256": (
            _sha256(parent_checkpoint) if online_arm is not None and online_arm != "prefix" else None
        ),
        "child_config_hash": arm_hash,
        "calibration_sha256": calibration.get("artifact_sha256"),
        "training_attack_identity_sha256": config.method.attack.identity_sha256(),
        "training_attack_random_start_keying": config.method.attack.random_start_keying,
        "source_git_sha": source_sha,
        "rng_source_seeds": None if rng_source_seeds is None else rng_source_seeds.as_dict(),
        "shuffle_augmentation_seeds": (
            None if shuffle_augmentation_seeds is None else shuffle_augmentation_seeds.as_dict()
        ),
        "dynamic_s3": dynamic_s3_identity,
    }
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                "parent_config": str(parent_config_path.resolve()),
                "parent_checkpoint": str(parent_checkpoint.resolve()),
                "parent_config_hash": parent_hash,
                "experiment_parent_checkpoint": (
                    None
                    if dynamic_s3_experiment_parent_checkpoint is None
                    else str(dynamic_s3_experiment_parent_checkpoint.resolve())
                ),
                "experiment_parent_checkpoint_sha256": experiment_parent_sha256,
                "shared_prefix": shared_prefix,
                "treatment": treatment.__dict__,
                "calibration": calibration,
                "training_attack_identity_sha256": config.method.attack.identity_sha256(),
                "child_config_hash": arm_hash,
                "run_namespace": run_namespace,
                "continuation_seed": continuation_seed,
                "rng_source_seeds": None if rng_source_seeds is None else rng_source_seeds.as_dict(),
                "shuffle_augmentation_seeds": (
                    None if shuffle_augmentation_seeds is None else shuffle_augmentation_seeds.as_dict()
                ),
                "wandb_artifact_retention": config.tracking.artifact_retention,
                "horizon_epochs": list(horizon_epochs),
                "dynamic_s3": (
                    None
                    if dynamic_s3_arm is None
                    else {"arm": dynamic_s3_arm, "beta_advce": dynamic_s3_beta_advce, "timing": "same_step_pre_update"}
                ),
                "online_state_s2": online_state_s2_identity,
                "online_shared_prefix": online_prefix_state,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    tracker: ExperimentTracker | None = None
    try:
        tracker = create_tracker(
            config=tracked_config,
            output_dir=output_dir,
            config_hash=arm_hash,
            root=Path.cwd(),
            job_type="train",
            run_id=run_id,
            training_seed=config.seeds.model_init,
            evaluation_seed=config.seeds.evaluation_attack,
        )
        tracker.attach_resolved_config(output_dir / "resolved_config.yaml")
    except Exception:
        if tracker is not None:
            tracker.finish(status="failed")
        raise
    metrics_path = output_dir / "epoch-metrics.jsonl"
    horizon_dir = output_dir / "checkpoints"
    horizon_paths: dict[str, Path] = {}
    dynamic_epoch80: dict[str, Any] | None = shared_epoch80
    online_epoch100: dict[str, Any] | None = online_prefix_state

    def record(metrics: dict[str, float], improved: bool) -> None:
        nonlocal dynamic_epoch80, online_epoch100
        row = {"epoch": trainer.current_epoch, "global_step": trainer.global_step, **metrics, "improved": improved}
        if dynamic_s3_router is not None:
            state = dynamic_s3_router.epoch_statistics.get(trainer.current_epoch, {})
            row.update({f"routing_{key}": value for key, value in state.items()})
        if online_state_s2_router is not None:
            state = online_state_s2_router.epoch_statistics.get(trainer.current_epoch, {})
            row.update({f"online_state_{key}": value for key, value in state.items()})
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        tracker.log_metrics(row, step=trainer.global_step)
        if trainer.current_epoch in horizon_epochs:
            horizon_dir.mkdir(exist_ok=True)
            horizon_path = horizon_dir / f"epoch-{trainer.current_epoch}.pt"
            shutil.copy2(output_dir / "last.pt", horizon_path)
            horizon_paths[str(trainer.current_epoch)] = horizon_path
        if dynamic_s3_router is not None and trainer.current_epoch == 80:
            horizon_dir.mkdir(exist_ok=True)
            epoch80_path = horizon_dir / "epoch-80.pt"
            shutil.copy2(output_dir / "last.pt", epoch80_path)
            payload = torch.load(epoch80_path, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict):
                raise StageARuntimeError("epoch-80 dynamic routing checkpoint is malformed")
            dynamic_epoch80 = {
                "path": str(epoch80_path.resolve()),
                "checkpoint_sha256": _sha256(epoch80_path),
                "components": _epoch80_equivalence(payload),
                "capture_path": str((output_dir / "routing-capture-mask.json").resolve()),
            }
            _write_json_atomic(output_dir / "epoch80-routing-state.json", dynamic_epoch80)
            if dynamic_s3_peer_epoch80_state is not None:
                _epoch80_gate(own=dynamic_epoch80, peer_path=dynamic_s3_peer_epoch80_state)
        if (
            online_state_s2_router is not None
            and online_arm == "prefix"
            and trainer.current_epoch == online_prefix_epoch
        ):
            horizon_dir.mkdir(exist_ok=True)
            epoch100_path = horizon_dir / "epoch-100.pt"
            if not epoch100_path.is_file():
                raise StageARuntimeError("shared e100 prefix horizon checkpoint was not materialized")
            epoch100_payload = torch.load(epoch100_path, map_location="cpu", weights_only=False)
            if not isinstance(epoch100_payload, dict):
                raise StageARuntimeError("shared e100 prefix checkpoint is malformed")
            online_epoch100 = {
                "path": str(epoch100_path.resolve()),
                "checkpoint_sha256": _sha256(epoch100_path),
                "components": _checkpoint_component_equivalence(epoch100_payload, epoch=online_prefix_epoch),
                "shared_prefix": True,
            }
            _write_json_atomic(output_dir / "epoch100-online-state.json", online_epoch100)

    trainer.fork_lineage = {
        **trainer.fork_lineage,
        "child_tracker_run_id": run_id,
    }
    tracker.attach_fork_lineage(trainer.fork_lineage)

    try:
        trainer.fit(
            loader,
            validation_loader=validation_loader,
            epochs=end_epoch,
            start_epoch=start_epoch,
            on_epoch_end=record,
        )
        dynamic_s3_artifacts = None if dynamic_s3_router is None else dynamic_s3_router.finalize()
        online_state_s2_artifacts = None if online_state_s2_router is None else online_state_s2_router.finalize()
        tracker.set_summary(
            {
                "best_metric": trainer.best_metric,
                "best_epoch": trainer.selection_metadata.get("selected_epoch"),
                "last_pgd_accuracy": trainer.selection_metadata.get("last_pgd_accuracy"),
                "last_clean_accuracy": trainer.selection_metadata.get("last_clean_accuracy"),
                "stage": run_namespace,
                "arm": treatment.arm,
                "horizon_epochs": list(horizon_epochs),
                "online_state_s2_active_count": (
                    None
                    if online_state_s2_router is None
                    else online_state_s2_router.epoch_statistics.get(trainer.current_epoch, {}).get(
                        "active_treatment_count"
                    )
                ),
            }
        )
        if should_upload_model_artifact(tracked_config.tracking.artifact_retention, is_final=True):
            tracker.log_artifact(
                output_dir / "last.pt", name=f"model-{run_id}-last", artifact_type="model", aliases=("last",)
            )
            tracker.log_artifact(
                output_dir / "best.pt", name=f"model-{run_id}-best", artifact_type="model", aliases=("best",)
            )
        if dynamic_s3_artifacts is not None and "artifacts" in dynamic_s3_artifacts:
            artifacts = dynamic_s3_artifacts["artifacts"]
            tracker.log_artifact(
                Path(artifacts["capture"]["path"]),
                name=f"dynamic-s3-capture-{run_id}",
                artifact_type="routing-input",
                aliases=("capture",),
            )
            tracker.log_artifact(
                Path(artifacts["state"]["path"]),
                name=f"dynamic-s3-state-{run_id}",
                artifact_type="sample-state",
                aliases=("state",),
            )
        if dynamic_epoch80 is not None:
            tracker.log_artifact(
                output_dir / "epoch80-routing-state.json",
                name=f"dynamic-s3-epoch80-state-{run_id}",
                artifact_type="routing-input",
                aliases=("epoch-80-routing-state",),
            )
        # Online-state rows and frozen threshold artifacts deliberately remain
        # local hash-bound lineage.  The campaign's W&B policy is metrics-only
        # and must not upload 45k-ID routing tables or checkpoints.
        horizon_manifest = {
            str(epoch): {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "epoch": int(epoch),
            }
            for epoch, path in sorted(horizon_paths.items(), key=lambda item: int(item[0]))
        }
        if set(horizon_manifest) != {str(epoch) for epoch in horizon_epochs}:
            missing = sorted(set(horizon_epochs) - {int(epoch) for epoch in horizon_manifest})
            raise StageARuntimeError(f"Trainer did not produce requested horizon checkpoints: {missing}")
        (output_dir / "horizon-checkpoints.json").write_text(
            json.dumps(horizon_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if tracked_config.tracking.artifact_retention == "full":
            for epoch, path in sorted(horizon_paths.items(), key=lambda item: int(item[0])):
                tracker.log_artifact(
                    path,
                    name=f"model-{run_id}-epoch-{epoch}",
                    artifact_type="model",
                    aliases=(f"epoch-{epoch}",),
                )
        tracker.prepare_finish()
        tracker.finish()
    except Exception:
        tracker.finish(status="failed")
        raise
    return {
        "arm": treatment.arm,
        "seed": config.seeds.model_init,
        "output_dir": str(output_dir.resolve()),
        "config_hash": arm_hash,
        "parent_checkpoint_sha256": _sha256(parent_checkpoint),
        "continuation_seed": continuation_seed,
        "rng_source_seeds": None if rng_source_seeds is None else rng_source_seeds.as_dict(),
        "last_checkpoint_sha256": _sha256(output_dir / "last.pt"),
        "best_checkpoint_sha256": _sha256(output_dir / "best.pt"),
        "horizon_checkpoints": {
            epoch: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for epoch, path in sorted(horizon_paths.items(), key=lambda item: int(item[0]))
        },
        "dynamic_s3": dynamic_s3_artifacts,
        "dynamic_s3_epoch80": dynamic_epoch80,
        "online_state_s2": online_state_s2_artifacts,
        "online_state_s2_epoch100": online_epoch100,
        "canary": (
            None
            if not canary_enabled
            else {
                "enabled": True,
                "max_train_batches": canary_max_train_batches,
                "max_validation_batches": canary_max_validation_batches,
                "source_ids": canary_selected_source_ids,
            }
        ),
    }
