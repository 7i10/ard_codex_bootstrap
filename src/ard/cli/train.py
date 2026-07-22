"""Command-line entry point for the composed M1 training path."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from ard.analysis import write_sample_parquet
from ard.attacks import LinfPGD
from ard.config import ExperimentConfig, load_config, save_resolved_config
from ard.config.loader import resolved_config_dict
from ard.config.schema import validate_global_batch_size
from ard.data import (
    EpochShuffleSampler,
    IndexedBatch,
    build_dataset,
    collate_indexed,
    stratified_train_validation_split,
)
from ard.engine import Trainer, config_digest, get_rank, get_world_size
from ard.engine.checkpoint import validate_resume_checkpoint
from ard.engine.distributed import barrier, initialize_from_env, is_rank_zero, run_rank_zero_phase, teardown
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
from ard.state import SampleStateStore
from ard.targets import TeacherTargetPolicy, UniformSofteningTeacherTargetPolicy
from ard.tracking import (
    ExperimentTracker,
    LocalTracker,
    coordinated_create_tracker,
    coordinated_tracker_action,
    validate_tracking_guard,
)
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
    if method.id == "rslad_entropy":
        return (
            RSLADObjective(temperature=method.temperature, temperature_squared=method.temperature_squared),
            EntropyOnlyPolicy(),
            None,
            None,
        )
    if method.id in {"rslad_student", "rslad_joint"}:
        assert method.target_policy is not None
        policy = StudentRiskPolicy() if method.id == "rslad_student" else JointRiskPolicy()
        return (
            RSLADObjective(temperature=method.temperature, temperature_squared=method.temperature_squared),
            policy,
            SampleStateStore(ema_decay=method.student_ema_decay),
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
    device, initialized_distributed = initialize_from_env(config.training.device)
    tracker: ExperimentTracker | None = None
    tracking_completed = False
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

        run_rank_zero_phase(_validate_output_guard, phase="output guard")
        run_rank_zero_phase(
            lambda: save_resolved_config(config, output_dir / "resolved_config.yaml"),
            phase="resolved-config write",
        )
        barrier()
        if args.dry_run:
            if is_rank_zero():
                print(json.dumps(resolved_config_dict(config), sort_keys=True))
            return 0

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

        _seed_everything(config.seeds.model_init + get_rank())
        if config.training.deterministic:
            torch.use_deterministic_algorithms(True)
        dataset = build_dataset(config.dataset)
        train_dataset, validation_dataset = stratified_train_validation_split(
            dataset, validation_fraction=config.training.validation_fraction, seed=config.seeds.split
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
            student = DistributedDataParallel(student, device_ids=[device.index] if device.type == "cuda" else None)
        teacher = None if config.teacher is None else build_teacher(config.teacher, tier=config.tier)
        optimizer = SGD(
            student.parameters(),
            lr=config.optimizer.learning_rate,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
            nesterov=config.optimizer.nesterov,
        )
        scheduler = StepLR(optimizer, step_size=1, gamma=1.0)
        selection_attack_config = config.method.selection_attack
        assert selection_attack_config is not None  # resolved by MethodConfig validation
        objective, policy, sample_store, target_policy = _build_method(config)
        diagnostics = TrainingDiagnostics.for_ids(
            list(train_dataset.indices), seed=config.seeds.qualitative_panel, size=config.tracking.panel_size
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
            policy_warmup_epochs=(config.method.student_policy_warmup_epochs if sample_store is not None else 0),
            oracle_mask=config.method.oracle_mask,
            diagnostics=diagnostics,
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
            coordinated_tracker_action(
                active_tracker,
                phase="tracker no-op resume finish",
                action=lambda active: active.finish(),
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
                active_tracker.log_artifact(
                    output_dir / "last.pt",
                    name=f"model-{active_tracker.run_id}-last",
                    artifact_type="model",
                    aliases=("last",),
                )
                if improved:
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
                if sparse and diagnostics.panel_rows:
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
        scalar_rows = [
            {
                key: value
                for key, value in row.items()
                if key not in {"clean_image", "adversarial_image", "perturbation_visualization"}
            }
            for _, row in sorted(diagnostics.all_rows.items())
        ]
        stats_path = output_dir / "sample-stats-train.parquet"

        def _write_sample_statistics() -> None:
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
        if tracker is not None and not tracking_completed:
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
