"""Command-line entry point for the composed M1 training path."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import torch
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
    FixedInterventionMask,
    HardFallbackPolicy,
    JointDownweightPolicy,
    JointRiskPolicy,
    RSLADBaselinePolicy,
    StudentRiskPolicy,
    WeightPolicy,
    load_fixed_intervention_mask,
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
from ard.tracking.adapter import collect_git_state
from ard.tracking.diagnostics import TrainingDiagnostics


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


def _screen_identity_sha256(manifest: Mapping[str, object]) -> str:
    arms = manifest.get("arms")
    if not isinstance(arms, list):
        raise ValueError("completed intervention screen manifest arms must be a list")
    identity = {
        "schema_version": 1,
        "kind": "common_state_intervention_v1",
        "parent_checkpoint_sha256": manifest.get("parent_checkpoint_sha256"),
        "parent_run_id": manifest.get("parent_run_id"),
        "fork_git_sha": manifest.get("fork_git_sha"),
        "arms": [
            {
                "arm": item.get("arm"),
                "config_hash": item.get("config_hash"),
                "run_id": item.get("run_id"),
                "output": item.get("output"),
            }
            for item in sorted(
                (item for item in arms if isinstance(item, Mapping)), key=lambda item: str(item.get("arm"))
            )
        ],
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _validate_completed_screen(
    *, path: Path, config: ExperimentConfig, config_hash: str, lineage: Mapping[str, object]
) -> None:
    screen_path = path.parent / "screen-complete.json"
    try:
        manifest = json.loads(screen_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("intervention arm requires a readable completed screen manifest") from exc
    if not isinstance(manifest, Mapping) or manifest.get("status") != "complete":
        raise ValueError("intervention arm requires a completed screen manifest")
    if manifest.get("screen_id") != lineage.get("screen_id") or _screen_identity_sha256(manifest) != manifest.get(
        "screen_id"
    ):
        raise ValueError("completed screen manifest identity does not match the fork checkpoint")
    arms = manifest.get("arms")
    if not isinstance(arms, list) or len(arms) != 5:
        raise ValueError("completed screen manifest must contain all five registered arms")
    entries = [entry for entry in arms if isinstance(entry, Mapping)]
    names = [entry.get("arm") for entry in entries]
    run_ids = [entry.get("run_id") for entry in entries]
    if len(entries) != 5 or set(names) != {"C", "HS", "RS", "HD", "RD"} or len(set(run_ids)) != 5:
        raise ValueError("completed screen manifest has duplicate or missing sibling identities")
    expected_arm = config.intervention.arm if config.intervention is not None else None
    own = next((entry for entry in entries if entry.get("arm") == expected_arm), None)
    if (
        not isinstance(own, Mapping)
        or own.get("config_hash") != config_hash
        or own.get("run_id") != lineage.get("child_tracker_run_id")
    ):
        raise ValueError("completed screen manifest does not bind this arm config and tracking identity")
    launch_state = collect_git_state(Path.cwd())
    launch_git = launch_state.get("sha")
    if (
        launch_state.get("dirty") is not False
        or launch_git != lineage.get("fork_git_sha")
        or manifest.get("fork_git_sha") != launch_git
    ):
        raise ValueError("intervention launch Git SHA must exactly match the clean fork Git SHA")


def _validate_intervention_resume(path: Path | None, config: ExperimentConfig, *, config_hash: str) -> None:
    """Keep the registered fork separate from both ordinary resume and parent best."""
    if path is None:
        if config.intervention is not None:
            raise ValueError("intervention arms must start from a registered common-state fork checkpoint")
        return
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("resume checkpoint must be a mapping")
    lineage = payload.get("fork_lineage")
    if config.intervention is None:
        if lineage is not None:
            raise ValueError("a common-state intervention fork cannot resume as an ordinary run")
        return
    if not isinstance(lineage, Mapping):
        raise ValueError("intervention arms require a registered common-state fork lineage")
    parent = config.intervention.parent
    expected = {
        "kind": "common_state_intervention_v1",
        "arm": config.intervention.arm,
        "parent_checkpoint_sha256": parent.checkpoint_sha256,
        "parent_raw_config_sha256": parent.raw_config_sha256,
        "parent_git_sha": parent.git_sha,
        "parent_epoch": 99,
        "parent_world_size": 1,
        "parent_teacher_checkpoint_sha256": parent.teacher_checkpoint_sha256,
        "parent_sample_state_records": 45000,
        "post_fork_best_scope": True,
    }
    if any(lineage.get(key) != value for key, value in expected.items()):
        raise ValueError("intervention fork lineage does not exactly match the configured parent")
    metadata = payload.get("selection_metadata")
    if not isinstance(metadata, Mapping) or metadata.get("scope") != "post_fork_best":
        raise ValueError("intervention fork must retain the post-fork best selection scope")
    epoch = payload.get("epoch")
    if epoch == 99 and payload.get("best_metric") != float("-inf"):
        raise ValueError("initial intervention fork must reset best selection to the post-fork scope")
    if not isinstance(epoch, int) or epoch < 99:
        raise ValueError("intervention fork checkpoint epoch is invalid")
    _validate_completed_screen(path=path, config=config, config_hash=config_hash, lineage=lineage)


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
            (
                UniformSofteningTeacherTargetPolicy(rho_max=config.intervention.uniform_target_softening_rho)
                if config.intervention is not None and config.intervention.kind == "uniform_target_softening"
                else None
            ),
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
            _guard_output(output_dir, resume=args.resume, config_hash=config_hash)
            _validate_intervention_resume(args.resume, config, config_hash=config_hash)

        run_rank_zero_phase(_validate_output_guard, phase="output guard")
        terminal_resume = run_rank_zero_value(
            lambda: _terminal_resume_requested(
                output_dir=output_dir, resume=args.resume, epochs=config.training.epochs
            ),
            phase="terminal resume preflight",
        )
        if args.dry_run:
            if not terminal_resume:
                run_rank_zero_phase(
                    lambda: save_resolved_config(config, output_dir / "resolved_config.yaml"),
                    phase="resolved-config write",
                )
                barrier()
            if is_rank_zero():
                print(json.dumps(resolved_config_dict(config), sort_keys=True))
            return 0
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
                if isinstance(active_tracker, LocalTracker) and config.intervention is not None:
                    payload = torch.load(args.resume, map_location="cpu", weights_only=False)
                    assert isinstance(payload, Mapping) and isinstance(payload.get("fork_lineage"), Mapping)
                    active_tracker.attach_fork_lineage(payload["fork_lineage"])
                active_tracker.resume(checkpoint_run_id=resumed_run_id, checkpoint_config_hash=config_hash)

        coordinated_tracker_action(tracker, phase="tracker attach/resume", action=_attach_resume)
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
        intervention_mask: FixedInterventionMask | None = None
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
        if config.intervention is not None and config.intervention.mask is not None:
            raw_targets = getattr(train_dataset.dataset.dataset, "targets", None)
            if not isinstance(raw_targets, (list, tuple)):
                raise ValueError("intervention training requires immutable source training labels")
            train_labels = {int(sample_id): int(raw_targets[sample_id]) for sample_id in train_dataset.indices}
            mask = config.intervention.mask
            intervention_mask = load_fixed_intervention_mask(
                mask.path,
                expected_sha256=mask.sha256,
                expected_selected_ids_sha256=mask.selected_ids_sha256,
                expected_selected_count=mask.selected_count,
                expected_class_counts=mask.selected_class_counts,
                expected_provenance=mask.provenance.model_dump(mode="json"),
                train_labels=train_labels,
                num_classes=config.dataset.num_classes,
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
        if config.observation.records_student_history and sample_store is None:
            sample_store = SampleStateStore(ema_decay=config.method.student_ema_decay)
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
            intervention_mask=intervention_mask,
            adversarial_kd_multiplier=(
                config.intervention.adversarial_kd_multiplier
                if config.intervention is not None and config.intervention.kind == "adversarial_kd_downweight"
                else None
            ),
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
            observation_profile=config.observation.profile,
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
