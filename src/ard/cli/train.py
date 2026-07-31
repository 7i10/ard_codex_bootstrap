"""Command-line entry point for the composed M1 training path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import torch
import yaml
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from ard.analysis import write_sample_parquet
from ard.analysis.frozen_oracle import FrozenRiskLookup, load_frozen_risk_lookup
from ard.attacks import LinfPGD
from ard.config import ExperimentConfig, load_config, save_resolved_config
from ard.config.loader import resolved_config_dict
from ard.config.schema import validate_global_batch_size
from ard.data import (
    EpochShuffleSampler,
    IndexedBatch,
    build_train_validation_views,
    collate_indexed,
)
from ard.engine import Trainer, config_digest, get_rank, get_world_size
from ard.engine.checkpoint import validate_resume_checkpoint
from ard.engine.distributed import (
    barrier,
    initialize_from_env,
    is_rank_zero,
    run_rank_zero_phase,
    run_rank_zero_value,
    teardown,
    wrap_ddp,
)
from ard.models import build_student, build_teacher
from ard.objectives import DistillationObjective, PGDATObjective, RSLADObjective, TRADESObjective
from ard.policies import (
    EntropyOnlyPolicy,
    HardFallbackPolicy,
    JointDownweightPolicy,
    JointRiskPolicy,
    RSLADBaselinePolicy,
    StudentRiskPolicy,
    WeightPolicy,
)
from ard.protocols import ensure_local_trainable
from ard.schedules import build_scheduler
from ard.state import SampleStateStore
from ard.targets import TeacherTargetPolicy, UniformSofteningTeacherTargetPolicy
from ard.tracking import (
    ExperimentTracker,
    LocalTracker,
    TrackingError,
    coordinated_create_tracker,
    coordinated_tracker_action,
    validate_tracking_guard,
)
from ard.tracking.diagnostics import TrainingDiagnostics

_RESEARCH_DESIGN_RELATIVE_PATH = Path("configs/analysis/logging_only_history_confirmatory_v1.yaml")
_RESEARCH_GATE_ATTESTATION_RELATIVE_PATH = Path(
    "configs/analysis/logging_only_history_confirmatory_v1_gate_attestation.yaml"
)
_RESEARCH_DESIGN_SHA256 = "d653d9ef08cfa94976a0e3279166b47543d16f3eaadb69810769470b77838c12"
_RESEARCH_GATE_ATTESTATION_SHA256 = "6207cce0fe70b2ef41fa3607e10b4eae47df5da6fd351fe2b68c2077b4c01cc5"
_RESEARCH_ALLOCATION = {
    "decision": "start_first_replication_only",
    "teacher": "bartoldson2024_adversarial_wrn94_16",
    "method": "rslad_logging_only",
    "training_protocol": "controlled_cifar10_r18_v1",
    "seed": 1,
    "epochs": 200,
    "world_size": 1,
    "per_rank_batch_size": 128,
    "global_batch_size": 128,
    "tracking": {
        "entity": "shunsuke-n-waseda-university",
        "project": "single-teacher-ard",
        "group": "bartoldson-cifar10-r18-ws1",
        "run_id": "bart-rslad-logging-only-s1-confirm-v1",
    },
    "output_dir": "outputs/scientific/bart-rslad-logging-only-s1-confirm-v1",
}
_FROZEN_MASK_ARTIFACTS = {
    "outcome_informed": {
        "run_id": "bart-oracle-soft-s0-05cd0c6",
        "roles": {
            "train_manifest_sha256": "c101cb59a233e3d4272aac929c62788a0c31c4efb05fde92c3838e0810fb4eac",
            "pgd_results_sha256": "b0f74de111f283cc1213bea60eb0cf03f1f2a96923dfd07d5f02763fd5f2788d",
            "autoattack_results_sha256": "e76fef5503fa8d3b7d531988e87582b7ab80210fbe9e124418c7c06274f5e87f",
        },
    },
    "random_1": {
        "run_id": "bart-rand1-soft-s0-05cd0c6",
        "roles": {
            "train_manifest_sha256": "0374ff5d22cb02604758a367bdd22819a51699ac6f72987dba69ffbc689e6f39",
            "pgd_results_sha256": "489ea0e2e1665cfe7232fb777d0e85818e3a14d4d23308266a16fead5bdc0652",
            "autoattack_results_sha256": "6b5e922d7c253143d35fb58ed457a89168ff32ddb78949d8f79db3626bcb2d69",
        },
    },
    "random_2": {
        "run_id": "bart-rand2-soft-s0-05cd0c6",
        "roles": {
            "train_manifest_sha256": "afd8bb6334ae6353ff5c89f89e8ae278124b2f147b1524cb0bac673c76b1077c",
            "pgd_results_sha256": "43516cd5acef8c5b59f28159abdb58167f44d5cfe785b8f49c59dbf933d9a0b6",
            "autoattack_results_sha256": "ac12e918485917902c2664d81c2672a322a2bcebe1ea7081fd970322bcfc4ce0",
        },
    },
    "random_3": {
        "run_id": "bart-rand3-soft-s0-05cd0c6",
        "roles": {
            "train_manifest_sha256": "1bd0bd0a773a3f4e3e3951f22485197070289336ebaed055176b8c4f5fd7ad27",
            "pgd_best_results_sha256": "a37b5302447100139e78b9b543042ac836c9e77a5831f8bb1e9b905c71310c16",
            "pgd_last_results_sha256": "363084e38745f816030755cdecf1b8d72d78705a04325a6b9bde51581a89240b",
            "autoattack_best_results_sha256": "89c4004c23accb766ec7e66a9711468762e8932b9c9600a9fc500cf847106056",
            "autoattack_last_results_sha256": "b6ba77e32f54651239e7026aae169aa15b128023c71103072e3be820657daf52",
        },
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an ARD student model.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a strict YAML experiment configuration.",
    )
    parser.add_argument("overrides", nargs="*", help="Dot-path YAML overrides such as training.epochs=2")
    parser.add_argument("--output", type=Path, help="Override output_dir")
    parser.add_argument("--resume", type=Path, help="Resume an epoch-boundary checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and save config without constructing training")
    return parser


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def _guard_output(output_dir: Path, *, resume: Path | None, config_hash: str) -> None:
    """Reject collisions before the resolved config or checkpoint can be written."""
    if resume is None:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"refusing to overwrite existing output directory without --resume: {output_dir}")
        return
    resume = resume.resolve()
    if resume.parent != output_dir:
        raise ValueError("resume checkpoint must live in the selected output directory")
    validate_resume_checkpoint(resume, expected_config_hash=config_hash)


def _validate_research_design(config: ExperimentConfig, *, world_size: int = 1) -> None:
    """Fail before tracker initialization unless immutable design and evidence match."""
    design = config.research_design
    if design is None:
        return
    root = Path(__file__).resolve().parents[3]
    design_path = design.manifest.resolve()
    expected_design_path = (root / _RESEARCH_DESIGN_RELATIVE_PATH).resolve()
    if design_path != expected_design_path:
        raise ValueError("research design must use the immutable preregistration manifest")
    if not design_path.is_file():
        raise FileNotFoundError(f"research design manifest is missing: {design_path}")
    digest = hashlib.sha256(design_path.read_bytes()).hexdigest()
    if digest != _RESEARCH_DESIGN_SHA256 or design.sha256 != _RESEARCH_DESIGN_SHA256:
        raise ValueError("research design preregistration SHA-256 does not match the immutable bytes")
    try:
        payload = yaml.safe_load(design_path.read_text(encoding="utf-8"))
        scope = payload["scope"]
        seed_roles = payload["seed_roles"]
        launch = payload["launch_sequence"]["first_and_only_automatic_run"]
    except (OSError, TypeError, KeyError, yaml.YAMLError) as exc:
        raise ValueError("research design manifest lacks its frozen launch contract") from exc
    if not all(isinstance(value, Mapping) for value in (payload, scope, seed_roles, launch)):
        raise ValueError("research design launch contract must use mappings")
    if (
        payload.get("design_id") != design.id
        or payload.get("status") != "frozen_before_seed_1"
        or seed_roles.get(1) != "sequential_replication_gate"
    ):
        raise ValueError("research design is not frozen for the seed-1 replication gate")
    if config.teacher is None:
        raise ValueError("research design launch requires a registered teacher")
    expected_seed = launch.get("seed")
    seed_fields = ("model_init", "data_order", "augmentation", "train_attack", "qualitative_panel")
    actual = {
        "teacher": config.teacher.registry_id,
        "method": config.method.id,
        "training_protocol": config.protocol.id,
        "seed": config.seeds.model_init,
        "epochs": config.training.epochs,
        "world_size": world_size,
        "per_rank_batch_size": config.training.per_rank_batch_size,
        "global_batch_size": config.training.global_batch_size,
    }
    expected = {
        "teacher": launch.get("teacher"),
        "method": scope.get("method"),
        "training_protocol": scope.get("training_protocol"),
        "seed": expected_seed,
        "epochs": launch.get("epochs"),
        "world_size": launch.get("world_size"),
        "per_rank_batch_size": launch.get("per_rank_batch_size"),
        "global_batch_size": launch.get("global_batch_size"),
    }
    if actual != expected or any(getattr(config.seeds, field) != expected_seed for field in seed_fields):
        raise ValueError("resolved run does not match the frozen first replication allocation")
    tracking = {
        "entity": config.tracking.entity,
        "project": config.tracking.project,
        "group": config.tracking.group,
        "run_id": config.tracking.run_id,
    }
    if (
        tracking != _RESEARCH_ALLOCATION["tracking"]
        or config.output_dir.as_posix() != _RESEARCH_ALLOCATION["output_dir"]
    ):
        raise ValueError("resolved run does not match the canonical logging-only tracking/output identity")
    attestation_path = (root / _RESEARCH_GATE_ATTESTATION_RELATIVE_PATH).resolve()
    if not attestation_path.is_file():
        raise FileNotFoundError(f"research gate attestation is missing: {attestation_path}")
    if hashlib.sha256(attestation_path.read_bytes()).hexdigest() != _RESEARCH_GATE_ATTESTATION_SHA256:
        raise ValueError("research gate attestation SHA-256 does not match the immutable evidence bytes")
    try:
        attestation = yaml.safe_load(attestation_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, yaml.YAMLError) as exc:
        raise ValueError("research gate attestation is unreadable") from exc
    if not isinstance(attestation, Mapping):
        raise ValueError("research gate attestation must be a mapping")
    _validate_research_gate_evidence(
        attestation=attestation,
    )


def _validate_research_gate_evidence(
    *,
    attestation: Mapping[object, object],
) -> None:
    """Fail closed unless terminal seed-0 evidence permits the frozen allocation."""

    common_trajectory = attestation.get("common_trajectory")
    frozen_mask = attestation.get("frozen_mask")
    parity = attestation.get("optimization_parity")
    allocation = attestation.get("allocation")
    if not all(isinstance(value, Mapping) for value in (common_trajectory, frozen_mask, parity, allocation)):
        raise ValueError("research gate attestation lacks structured evidence")
    common_trajectory = cast(Mapping[object, object], common_trajectory)
    frozen_mask = cast(Mapping[object, object], frozen_mask)
    parity = cast(Mapping[object, object], parity)
    allocation = cast(Mapping[object, object], allocation)
    if (
        attestation.get("schema_version") != 1
        or attestation.get("attestation_id") != "logging_only_history_confirmatory_v1_gate_attestation"
        or attestation.get("status") != "eligible_for_first_replication"
        or attestation.get("preregistered_design_sha256") != _RESEARCH_DESIGN_SHA256
    ):
        raise ValueError("research design gate is not eligible for the first replication")
    if (
        common_trajectory.get("checkpoint_scientific_git_sha") != "2d54b8230b8d14d13c1ea7472ccba53491b4d38d"
        or common_trajectory.get("analysis_git_sha") != "d3c59b19788f915d82047b5f2722e9070b664517"
        or common_trajectory.get("analysis_source_sha256")
        != "2a5956e42bb65395d195c1db73b90660b2e0af65414612f2da013332eff2bec2"
    ):
        raise ValueError("research design gate lacks exact common-trajectory Git identities")

    expected_common = {
        "chen": {
            "run_id": "prod-chen-rslad-s0-2d54b82",
            "report_sha256": "cb5305182bb942b9f9d44036c67700c9d8fff54116ad3f7111be0a73f65016fa",
            "lineage_sha256": "f485f72341b276351098e97514e4becc46b278a005694be2672811c5aaf5a808",
        },
        "bartoldson": {
            "run_id": "prod-bart-rslad-s0-2d54b82",
            "report_sha256": "d44ee166f8866b77067ebd07757d394a060242c9cf1cdc5d4513f127897981f8",
            "lineage_sha256": "9b6ea091dc9ed4ff81bb579bf05d6650ac8e6d4ab6104981c446f29069e4a64e",
        },
    }
    for teacher in ("chen", "bartoldson"):
        evidence = common_trajectory.get(teacher)
        if not isinstance(evidence, Mapping):
            raise ValueError(f"research design gate lacks {teacher} common-trajectory evidence")
        interval = evidence.get("delta_auroc_ci_95")
        delta_auroc = evidence.get("delta_auroc_vs_best_current")
        delta_log_loss = evidence.get("delta_log_loss")
        if (
            evidence.get("run_id") != expected_common[teacher]["run_id"]
            or evidence.get("train_count") != 45_000
            or evidence.get("history_gate") != "go"
            or evidence.get("report_sha256") != expected_common[teacher]["report_sha256"]
            or evidence.get("lineage_sha256") != expected_common[teacher]["lineage_sha256"]
            or not isinstance(interval, list)
            or len(interval) != 2
            or not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in interval)
            or not all(math.isfinite(float(value)) for value in interval)
            or float(interval[0]) <= 0.0
            or isinstance(delta_auroc, bool)
            or not isinstance(delta_auroc, (int, float))
            or not math.isfinite(float(delta_auroc))
            or float(delta_auroc) < 0.02
            or isinstance(delta_log_loss, bool)
            or not isinstance(delta_log_loss, (int, float))
            or not math.isfinite(float(delta_log_loss))
            or float(delta_log_loss) >= 0.0
        ):
            raise ValueError(f"research design gate rejects incomplete or failed {teacher} History evidence")

    arms = frozen_mask.get("arms")
    if (
        frozen_mask.get("all_arms_terminal") is not True
        or frozen_mask.get("scientific_git_sha") != "05cd0c66367e399dde266bd898c3ddc4097ca95c"
        or not isinstance(arms, Mapping)
        or set(arms) != set(_FROZEN_MASK_ARTIFACTS)
    ):
        raise ValueError("research design gate lacks terminal four-arm frozen-mask evidence")
    best_metrics: dict[str, tuple[float, float]] = {}
    for arm_name, arm in arms.items():
        if not isinstance(arm_name, str) or not isinstance(arm, Mapping):
            raise ValueError("research design frozen-mask arm evidence must use mappings")
        best = arm.get("best")
        if not isinstance(best, Mapping):
            raise ValueError("research design frozen-mask arm lacks best metrics")
        clean, aa = best.get("clean"), best.get("aa")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in (clean, aa)
        ):
            raise ValueError("research design frozen-mask arm metrics must be finite")
        assert isinstance(clean, (int, float)) and not isinstance(clean, bool)
        assert isinstance(aa, (int, float)) and not isinstance(aa, bool)
        best_metrics[arm_name] = (float(clean), float(aa))
        expected_arm = _FROZEN_MASK_ARTIFACTS[arm_name]
        digest_roles = {key: value for key, value in arm.items() if isinstance(key, str) and key.endswith("sha256")}
        if arm.get("run_id") != expected_arm["run_id"] or digest_roles != expected_arm["roles"]:
            raise ValueError("research design frozen-mask artifact identities or roles do not match")

    mask_clean, mask_aa = best_metrics["outcome_informed"]
    random_values = [best_metrics[f"random_{index}"] for index in (1, 2, 3)]
    random_clean = sum(value[0] for value in random_values) / 3.0
    random_aa = sum(value[1] for value in random_values) / 3.0
    go = (
        mask_aa >= random_aa + 0.005
        and all(mask_aa > value[1] for value in random_values)
        and random_clean - mask_clean <= 0.005
    )
    observed_decision = "go" if go else ("no_go" if mask_aa <= random_aa else "inconclusive")
    if frozen_mask.get("decision") != observed_decision:
        raise ValueError("research design frozen-mask decision does not match terminal metrics")

    if allocation != _RESEARCH_ALLOCATION:
        raise ValueError("research design gate allocation drifted from the frozen launch sequence")
    expected_parity = {
        "status": "passed",
        "scientific_git_sha": "d3c59b19788f915d82047b5f2722e9070b664517",
        "command": (
            "PYTHONPATH=src /home/shunsukenaito/.conda/envs/adv/bin/python -m pytest -q "
            "tests/integration/test_checkpoint_resume.py::"
            "test_rslad_logging_only_cuda_parity_with_random_start_pgd"
        ),
        "nodeid": (
            "tests/integration/test_checkpoint_resume.py::test_rslad_logging_only_cuda_parity_with_random_start_pgd"
        ),
        "result": {"passed": 1, "failed": 0, "skipped": 0},
        "source_digests": {
            "test_checkpoint_resume_py_sha256": "cf4e1e3ac0b7222b8c437c3572f388ad2dcee9629bcc79bb957236252e86b24d",
            "trainer_py_sha256": "8b3044a6faf8ffecfae999f7ed064fd5d1b6799be30bfafe9903e5883a0792e0",
            "pgd_py_sha256": "7775955573b1c95bea81b6ee043a9cfa9f9a0d4961f61a0cd23e81fa3e5925c2",
            "rslad_py_sha256": "646d95f8da74c4fc920cacf4ea353ec7104059305d4f2c154973643c9920d3da",
            "sample_store_py_sha256": "dc529e9f736bdd6f722e2cff0d6481cb13302a4cd0087e83ca9d31d0921ffe02",
        },
    }
    if parity != expected_parity:
        raise ValueError("research design gate lacks exact RSLAD/logging-only optimization parity")


def _research_claim_identity(config: ExperimentConfig, *, config_hash: str) -> dict[str, object]:
    """Identity written once for the single allowed fresh confirmatory allocation."""
    return {
        "research_design_sha256": _RESEARCH_DESIGN_SHA256,
        "gate_attestation_sha256": _RESEARCH_GATE_ATTESTATION_SHA256,
        "config_hash": config_hash,
        "tracking_run_id": config.tracking.run_id,
        "output_dir": config.output_dir.as_posix(),
    }


def _claim_research_allocation(
    config: ExperimentConfig,
    *,
    output_dir: Path,
    resume: Path | None,
    config_hash: str,
) -> None:
    """Atomically reserve a fresh allocation; resume must prove the same claim."""
    if output_dir.resolve() != config.output_dir.resolve():
        raise ValueError("research allocation claim must use the canonical output directory")
    expected = _research_claim_identity(config, config_hash=config_hash)
    claim_path = output_dir / ".research-allocation-claim.json"
    if resume is None:
        try:
            output_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise FileExistsError("research allocation was already claimed; use its exact --resume checkpoint") from exc
        descriptor = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(expected, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
        except BaseException:
            # Keep the empty allocation directory as a consumed fresh claim if
            # the process dies after mkdir; this path must never start a second
            # fresh run with the same W&B identity.
            raise
        return
    if not claim_path.is_file():
        raise ValueError("research resume requires the original allocation claim")
    try:
        observed = json.loads(claim_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("research allocation claim is unreadable") from exc
    if observed != expected:
        raise ValueError("research resume allocation claim does not match run/config identity")


def _resume_tracker_id(path: Path | None) -> str | None:
    """Read only the stable run identity before tracker initialization."""
    if path is None:
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("resume checkpoint must be a mapping")
    run_id = payload.get("tracker_run_id")
    if run_id is not None and not isinstance(run_id, str):
        raise ValueError("resume checkpoint tracker_run_id must be a string or null")
    return run_id


def _terminal_resume_requested(*, output_dir: Path, resume: Path | None, epochs: int) -> bool:
    """Read-only terminal-resume preflight before any run-bundle write."""
    if resume is None:
        return False
    manifest_path = output_dir / "run-bundle" / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrackingError("existing tracking manifest is unreadable") from exc
    if manifest.get("status") not in {"completed", "sync_pending"}:
        return False
    payload = torch.load(resume, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("epoch"), int):
        raise TrackingError("terminal resume checkpoint is invalid")
    if payload["epoch"] + 1 != epochs:
        raise TrackingError("terminal resume requires a checkpoint at the completed epoch boundary exactly")
    return True


def _selection_metric(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"checkpoint selection metadata lacks {name}")
    return float(value)


def _build_method(
    config: ExperimentConfig,
) -> tuple[DistillationObjective, WeightPolicy | None, SampleStateStore | None, TeacherTargetPolicy | None]:
    """Compose the M2 outer objective and optional policy without branching a loop."""
    method = config.method
    if method.id == "pgd_at":
        return PGDATObjective(), None, None, None
    if method.id == "trades":
        return (
            TRADESObjective(
                beta=method.trades_beta,
                temperature=method.temperature,
                temperature_squared=method.temperature_squared,
            ),
            None,
            None,
            None,
        )
    if method.id == "rslad":
        return (
            RSLADObjective(temperature=method.temperature, temperature_squared=method.temperature_squared),
            RSLADBaselinePolicy(),
            None,
            None,
        )
    if method.id == "rslad_logging_only":
        return (
            RSLADObjective(temperature=method.temperature, temperature_squared=method.temperature_squared),
            RSLADBaselinePolicy(),
            SampleStateStore(ema_decay=method.student_ema_decay),
            None,
        )
    if method.id == "rslad_entropy":
        return (
            RSLADObjective(temperature=method.temperature, temperature_squared=method.temperature_squared),
            EntropyOnlyPolicy(),
            None,
            None,
        )
    if method.id in {"rslad_student", "rslad_joint", "rslad_frozen_oracle_softening"}:
        assert method.target_policy is not None
        policy = StudentRiskPolicy() if method.id == "rslad_student" else JointRiskPolicy()
        sample_store = (
            None
            if method.id == "rslad_frozen_oracle_softening"
            else SampleStateStore(ema_decay=method.student_ema_decay)
        )
        return (
            RSLADObjective(temperature=method.temperature, temperature_squared=method.temperature_squared),
            policy,
            sample_store,
            UniformSofteningTeacherTargetPolicy(rho_max=method.target_policy.rho_max),
        )
    if method.id == "rslad_joint_downweight":
        return (
            RSLADObjective(temperature=method.temperature, temperature_squared=method.temperature_squared),
            JointDownweightPolicy(),
            SampleStateStore(ema_decay=method.student_ema_decay),
            None,
        )
    if method.id == "rslad_hard_fallback":
        return (
            RSLADObjective(temperature=method.temperature, temperature_squared=method.temperature_squared),
            HardFallbackPolicy(),
            SampleStateStore(ema_decay=method.student_ema_decay),
            None,
        )
    raise RuntimeError(f"unsupported validated method: {method.id}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config, args.overrides)
    if args.output is not None:
        config = config.model_copy(update={"output_dir": args.output})
    output_dir = config.output_dir.resolve()
    if config.dataset.split != "train":
        raise ValueError("training only accepts the official train split; official test data is evaluation-only")
    ensure_local_trainable(config.protocol.id)
    device, initialized_distributed = initialize_from_env(config.training.device)
    tracker: ExperimentTracker | None = None
    tracking_completed = False
    terminal_resume = False
    try:
        validate_global_batch_size(
            per_rank_batch_size=config.training.per_rank_batch_size,
            global_batch_size=config.training.global_batch_size,
            world_size=get_world_size(),
        )
        config_hash = config_digest(resolved_config_dict(config))

        def _validate_output_guard() -> None:
            validate_tracking_guard(config, root=Path.cwd())
            _validate_research_design(config, world_size=get_world_size())
            _guard_output(output_dir, resume=args.resume, config_hash=config_hash)

        run_rank_zero_phase(_validate_output_guard, phase="output guard")
        terminal_resume = run_rank_zero_value(
            lambda: _terminal_resume_requested(
                output_dir=output_dir, resume=args.resume, epochs=config.training.epochs
            ),
            phase="terminal resume preflight",
        )
        if args.dry_run:
            if not terminal_resume and config.research_design is None:
                run_rank_zero_phase(
                    lambda: save_resolved_config(config, output_dir / "resolved_config.yaml"),
                    phase="resolved-config write",
                )
                barrier()
            if is_rank_zero():
                print(json.dumps(resolved_config_dict(config), sort_keys=True))
            return 0
        if config.research_design is not None:
            run_rank_zero_phase(
                lambda: _claim_research_allocation(
                    config,
                    output_dir=output_dir,
                    resume=args.resume,
                    config_hash=config_hash,
                ),
                phase="research allocation claim",
            )
        if not terminal_resume:
            run_rank_zero_phase(
                lambda: save_resolved_config(config, output_dir / "resolved_config.yaml"),
                phase="resolved-config write",
            )
        barrier()

        # The run ID is deterministic on fresh starts and taken from the
        # checkpoint on resume.  All ranks independently derive the same ID;
        # create_tracker then makes only rank zero stateful.
        resumed_run_id = _resume_tracker_id(args.resume)
        tracker = coordinated_create_tracker(
            config=config,
            output_dir=output_dir,
            config_hash=config_hash,
            root=Path.cwd(),
            resume_run_id=resumed_run_id,
        )
        active_tracker = tracker

        def _attach_resume(active_tracker: ExperimentTracker) -> None:
            if isinstance(active_tracker, LocalTracker):
                active_tracker.attach_resolved_config(output_dir / "resolved_config.yaml")
            if args.resume is not None:
                active_tracker.resume(checkpoint_run_id=resumed_run_id, checkpoint_config_hash=config_hash)

        coordinated_tracker_action(tracker, phase="tracker attach/resume", action=_attach_resume)
        if config.research_design is not None and args.resume is None:
            design_path = config.research_design.manifest.resolve()
            gate_attestation_path = (
                Path(__file__).resolve().parents[3] / _RESEARCH_GATE_ATTESTATION_RELATIVE_PATH
            ).resolve()

            def _record_research_design(active_tracker: ExperimentTracker) -> None:
                active_tracker.log_artifact(
                    design_path,
                    name=f"research-design-{active_tracker.run_id}",
                    artifact_type="analysis-input",
                    aliases=("input",),
                )
                active_tracker.log_artifact(
                    gate_attestation_path,
                    name=f"research-gate-attestation-{active_tracker.run_id}",
                    artifact_type="analysis-input",
                    aliases=("input",),
                )

            coordinated_tracker_action(
                active_tracker,
                phase="research design artifact",
                action=_record_research_design,
            )

        _seed_everything(config.seeds.model_init + get_rank())
        if config.training.deterministic:
            torch.use_deterministic_algorithms(True)
        train_dataset, validation_dataset = build_train_validation_views(
            config.dataset,
            validation_fraction=config.training.validation_fraction,
            split_seed=config.seeds.split,
            augmentation_seed=config.seeds.augmentation,
        )
        frozen_risk_lookup: FrozenRiskLookup | None = None
        if config.method.id == "rslad_frozen_oracle_softening":
            assert config.method.frozen_oracle_manifest is not None
            assert config.method.frozen_oracle_manifest_sha256 is not None
            assert config.teacher is not None
            assert config.teacher.checkpoint_sha256 is not None
            raw_targets = getattr(train_dataset.dataset.dataset, "targets", None)
            if not isinstance(raw_targets, (list, tuple)):
                raise ValueError("frozen oracle training requires immutable source training labels")
            train_labels = {int(sample_id): int(raw_targets[sample_id]) for sample_id in train_dataset.indices}
            frozen_risk_lookup = load_frozen_risk_lookup(
                config.method.frozen_oracle_manifest,
                expected_sha256=config.method.frozen_oracle_manifest_sha256,
                expected_dataset_name=config.dataset.name,
                expected_num_classes=config.dataset.num_classes,
                expected_train_labels=train_labels,
                expected_attack_identity=config.method.attack.identity(),
                expected_teacher_checkpoint_sha256=config.teacher.checkpoint_sha256,
            )
            if args.resume is None:
                frozen_input_path = config.method.frozen_oracle_manifest

                def _record_frozen_oracle_input(active_tracker: ExperimentTracker) -> None:
                    active_tracker.log_artifact(
                        frozen_input_path,
                        name=f"frozen-oracle-input-{active_tracker.run_id}",
                        artifact_type="analysis-input",
                        aliases=("input",),
                    )

                coordinated_tracker_action(
                    active_tracker,
                    phase="frozen oracle input artifact",
                    action=_record_frozen_oracle_input,
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
        loader = cast(
            DataLoader[IndexedBatch],
            DataLoader(
                train_dataset,
                batch_size=config.training.per_rank_batch_size,
                sampler=sampler,
                num_workers=config.training.num_workers,
                collate_fn=collate_indexed,
            ),
        )
        validation_loader = cast(
            DataLoader[IndexedBatch],
            DataLoader(
                validation_dataset,
                batch_size=config.training.per_rank_batch_size,
                sampler=validation_sampler,
                num_workers=config.training.num_workers,
                collate_fn=collate_indexed,
            ),
        )
        student: nn.Module = build_student(config.student, tier=config.tier).to(device)
        if initialized_distributed:
            student = wrap_ddp(student, device)
        teacher = None if config.teacher is None else build_teacher(config.teacher, tier=config.tier)
        optimizer = SGD(
            student.parameters(),
            lr=config.optimizer.learning_rate,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
            nesterov=config.optimizer.nesterov,
        )
        scheduler = build_scheduler(optimizer, config.scheduler)
        selection_attack_config = config.method.selection_attack
        assert selection_attack_config is not None  # resolved by MethodConfig validation
        objective, policy, sample_store, target_policy = _build_method(config)
        diagnostics = (
            None
            if config.tracking.diagnostics_mode == "off"
            else TrainingDiagnostics.for_ids(
                list(train_dataset.indices),
                seed=config.seeds.qualitative_panel,
                size=config.tracking.panel_size,
                mode=config.tracking.diagnostics_mode,
            )
        )
        trainer = Trainer(
            model=student,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            attack=LinfPGD(config.method.attack),
            selection_attack=LinfPGD(selection_attack_config),
            objective=objective,
            policy=policy,
            device=device,
            output_dir=output_dir,
            config_hash=config_hash,
            seed=config.seeds.train_attack,
            evaluation_attack_seed=config.seeds.evaluation_attack,
            tracker_run_id=active_tracker.run_id,
            teacher=teacher,
            sample_store=sample_store,
            target_policy=target_policy,
            policy_warmup_epochs=(
                config.method.student_policy_warmup_epochs
                if config.method.id
                in {
                    "rslad_student",
                    "rslad_joint",
                    "rslad_joint_downweight",
                    "rslad_hard_fallback",
                }
                else 0
            ),
            oracle_mask=config.method.oracle_mask,
            frozen_risk_lookup=frozen_risk_lookup,
            diagnostics=diagnostics,
            observe_teacher_signals=config.method.id == "rslad_logging_only",
        )
        start_epoch = 0
        if args.resume is not None:
            start_epoch = trainer.resume(args.resume, sampler=sampler).next_epoch
        if start_epoch >= config.training.epochs:

            def validate_noop_resume(active_tracker: ExperimentTracker) -> None:
                if isinstance(active_tracker, LocalTracker):
                    active_tracker.validate_terminal_resume()

            coordinated_tracker_action(
                active_tracker,
                phase="tracker no-op resume validation",
                action=validate_noop_resume,
            )
            tracking_completed = True
            if is_rank_zero():
                print(json.dumps({"output_dir": str(output_dir), "history": []}, sort_keys=True))
            return 0

        def _record_epoch(metrics: Mapping[str, float], improved: bool) -> None:
            # Every rank enters one phase after checkpoint writes.  The best
            # conditional lives inside the rank-zero closure, never around a
            # collective, preserving DDP progress and RNG parity.
            def record(active_tracker: ExperimentTracker) -> None:
                values = dict(metrics)
                values["epoch"] = trainer.current_epoch
                active_tracker.log_metrics(values, step=trainer.global_step)
                publish_models = (
                    trainer.current_epoch + 1 == config.training.epochs
                    or (trainer.current_epoch + 1) % config.tracking.artifact_interval_epochs == 0
                )
                if publish_models:
                    active_tracker.log_artifact(
                        output_dir / "last.pt",
                        name=f"model-{active_tracker.run_id}-last",
                        artifact_type="model",
                        aliases=("last",),
                    )
                    active_tracker.log_artifact(
                        output_dir / "best.pt",
                        name=f"model-{active_tracker.run_id}-best",
                        artifact_type="model",
                        aliases=("best",),
                    )
                sparse = (
                    trainer.current_epoch == 0
                    or improved
                    or trainer.current_epoch + 1 == config.training.epochs
                    or (trainer.current_epoch + 1) % config.tracking.panel_interval_epochs == 0
                )
                if diagnostics is not None and diagnostics.mode == "panel" and sparse and diagnostics.panel_rows:
                    rows: list[Mapping[str, object]] = [row for row in diagnostics.panel_rows]
                    active_tracker.log_table(f"panel-epoch-{trainer.current_epoch}", rows)

            coordinated_tracker_action(active_tracker, phase="tracker epoch", action=record)

        history = trainer.fit(
            loader,
            validation_loader=validation_loader,
            epochs=config.training.epochs,
            start_epoch=start_epoch,
            on_epoch_end=_record_epoch,
        )
        stats_path: Path | None = None
        if diagnostics is not None:
            scalar_rows = [
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"clean_image", "adversarial_image", "perturbation_visualization"}
                }
                for _, row in sorted(diagnostics.all_rows.items())
            ]
            if sample_store is not None:
                for row in scalar_rows:
                    sample_id = int(row["sample_id"])
                    record = sample_store.records.get(sample_id)
                    if record is not None:
                        row.update(
                            {
                                "student_last_margin": record.last_margin,
                                "student_robust_margin_ema": record.margin_ema,
                                "student_seen": record.seen,
                                "student_robust_correct_frequency": record.robust_correct_frequency,
                                "student_forgetting_count": record.forgetting_count,
                                "student_first_robustly_learned_epoch": record.first_robustly_learned_epoch,
                                "student_current_correct_streak": record.current_correct_streak,
                                "student_longest_correct_streak": record.longest_correct_streak,
                                "student_margin_mean": record.margin_mean,
                                "student_margin_variance": record.margin_variance,
                                "student_margin_slope": record.margin_slope,
                                "student_margin_time_sum": record.margin_time_sum,
                                "student_margin_time_squared_sum": record.margin_time_squared_sum,
                                "student_margin_time_margin_sum": record.margin_time_margin_sum,
                                "student_history_statistics_complete": record.history_statistics_complete,
                                "teacher_clean_entropy": record.teacher_clean_entropy,
                                "teacher_clean_true_probability": record.teacher_clean_true_probability,
                                "teacher_clean_max_wrong_probability": record.teacher_clean_max_wrong_probability,
                                "teacher_clean_prediction": record.teacher_clean_prediction,
                                "teacher_clean_correct": record.teacher_clean_correct,
                                "teacher_adversarial_entropy": record.teacher_adversarial_entropy,
                                "teacher_adversarial_true_probability": (record.teacher_adversarial_true_probability),
                                "teacher_adversarial_max_wrong_probability": (
                                    record.teacher_adversarial_max_wrong_probability
                                ),
                                "teacher_adversarial_prediction": record.teacher_adversarial_prediction,
                                "teacher_adversarial_correct": record.teacher_adversarial_correct,
                                "teacher_clean_to_adversarial_margin_response": (
                                    record.teacher_clean_to_adversarial_margin_response
                                ),
                                "teacher_clean_to_adversarial_js_response": (
                                    record.teacher_clean_to_adversarial_js_response
                                ),
                            }
                        )
            stats_path = output_dir / "sample-stats-train.parquet"

            def _write_sample_statistics() -> None:
                assert stats_path is not None
                write_sample_parquet(scalar_rows, stats_path)

            run_rank_zero_phase(
                _write_sample_statistics,
                phase="sample statistics write",
            )

        def _finalize(active_tracker: ExperimentTracker) -> None:
            best_epoch = trainer.selection_metadata["selected_epoch"]
            selected_clean = trainer.selection_metadata.get("selected_clean_accuracy")
            selected_pgd = trainer.selection_metadata.get("selected_pgd_accuracy")
            last_clean = trainer.selection_metadata.get("last_clean_accuracy")
            last_pgd = trainer.selection_metadata.get("last_pgd_accuracy")
            selected_clean = _selection_metric(selected_clean, name="selected clean accuracy")
            selected_pgd = _selection_metric(selected_pgd, name="selected PGD accuracy")
            last_clean = _selection_metric(last_clean, name="last clean accuracy")
            last_pgd = _selection_metric(last_pgd, name="last PGD accuracy")
            if stats_path is not None:
                active_tracker.log_artifact(
                    stats_path, name=f"sample-stats-{active_tracker.run_id}", artifact_type="sample-stats"
                )
            active_tracker.set_summary(
                {
                    "best_metric": trainer.best_metric,
                    "best_epoch": best_epoch,
                    "best_clean_accuracy": selected_clean,
                    "best_pgd_accuracy": selected_pgd,
                    "last_clean_accuracy": last_clean,
                    "last_pgd_accuracy": last_pgd,
                    "robust_overfit_gap": selected_pgd - last_pgd,
                }
            )
            bundle = output_dir / "run-bundle"
            (bundle / "completion.json").write_text(
                json.dumps({"status": "completed", "output_dir": str(output_dir)}) + "\n", encoding="utf-8"
            )
            (bundle / "error-marker.txt").write_text("no application error recorded\n", encoding="utf-8")
            active_tracker.prepare_finish()
            active_tracker.log_artifact(bundle, name=f"run-bundle-{active_tracker.run_id}", artifact_type="run-bundle")
            active_tracker.finish()

        coordinated_tracker_action(active_tracker, phase="tracker finish", action=_finalize)
        tracking_completed = True
        if is_rank_zero():
            print(json.dumps({"output_dir": str(output_dir), "history": history}, sort_keys=True))
        return 0
    finally:
        if tracker is not None and not tracking_completed and not terminal_resume:
            # On an exception retain an explicit failed local manifest for
            # offline recovery.
            try:
                coordinated_tracker_action(
                    tracker, phase="tracker failure manifest", action=lambda active: active.finish(status="failed")
                )
            except Exception:
                pass
        if initialized_distributed:
            teardown()


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess
    raise SystemExit(main())
