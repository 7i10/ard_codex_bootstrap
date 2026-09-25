"""Separate saved-checkpoint evaluation command.

This process constructs only the student and a configured CE PGD attacker.  It
does not construct a teacher, objective, policy, optimizer, or sample state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import DataLoader, Subset

from ard.attacks import LinfPGD
from ard.config import ExperimentConfig, save_resolved_config
from ard.config.loader import load_evaluation_config, load_resolved_config_for_evaluation, resolved_config_dict
from ard.config.schema import TrainingConfig, training_execution_identity
from ard.data import EpochShuffleSampler, IndexedBatch, build_dataset, collate_indexed
from ard.engine import config_digest
from ard.evaluation import (
    evaluate_saved_checkpoint,
    load_saved_student_checkpoint,
    run_autoattack,
    validate_checkpoint_lineage,
)
from ard.models import build_student
from ard.tracking import (
    LocalTracker,
    create_tracker,
    reject_stale_remote_run_id,
    should_upload_run_bundle,
    validate_tracking_guard,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate saved ARD student checkpoints with clean and PGD accuracy.")
    parser.add_argument("--config", type=Path, required=True, help="Strict evaluation experiment config.")
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--checkpoint-dir", type=Path, help="Training output directory containing best.pt / last.pt.")
    location.add_argument("--checkpoint", type=Path, help="One explicit saved student checkpoint.")
    parser.add_argument(
        "--output", type=Path, help="Evaluation artifact directory (defaults below checkpoint directory)."
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        help="Resolved training config paired with the checkpoint (defaults to sibling resolved_config.yaml).",
    )
    parser.add_argument(
        "--allow-autoattack",
        action="store_true",
        help="Acknowledge an explicitly configured full AutoAttack run (never used by tests).",
    )
    parser.add_argument(
        "--weights",
        choices=("model", "ema"),
        default="model",
        help=(
            "Which checkpointed weight set to evaluate: 'model' (default) is the raw trained student, "
            "matching every non-EMA method and the official ADR code's plain 'ADR' table rows. 'ema' "
            "evaluates the EMA/weight-averaged shadow model an adr/adr_trades run checkpoints (matching "
            "the official ADR code's '--ema'-flagged 'ADR + WA' rows) or a plain pgd_at/trades run with "
            "training.weight_ema_decay set (plan 0102); only a checkpoint from one of those two carries it."
        ),
    )
    parser.add_argument("overrides", nargs="*", help="Dot-path YAML overrides")
    return parser


def _checkpoint_paths(
    *, checkpoint: Path | None, checkpoint_dir: Path | None, selection: str, weights: str = "model"
) -> tuple[Path, ...]:
    if checkpoint is not None:
        resolved = checkpoint.resolve()
        # "best.pt" is selected on the raw student's validation accuracy,
        # never the EMA's -- evaluating it with weights="ema" would silently
        # report EMA weights at a student-selected epoch, which is not the
        # official ADR code's "ADR + WA" quantity (EMA weights at an
        # EMA-reselected best epoch, see best-ema.pt). Only an explicit
        # single-file --checkpoint can hit this, since --checkpoint-dir is
        # remapped to best-ema.pt below.
        if weights == "ema" and resolved.name == "best.pt":
            raise ValueError(
                "--weights=ema cannot evaluate best.pt: its epoch was selected on the student's own validation "
                "accuracy, not the EMA's. Point --checkpoint at best-ema.pt instead (the EMA's own "
                "independently-selected best epoch), or evaluate last.pt."
            )
        return (resolved,)
    assert checkpoint_dir is not None
    directory = checkpoint_dir.resolve()
    best_name = "best-ema.pt" if weights == "ema" else "best.pt"
    names = {"best": (best_name,), "last": ("last.pt",), "both": (best_name, "last.pt")}[selection]
    paths = tuple(directory / name for name in names)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("requested saved checkpoints are missing: " + ", ".join(missing))
    return paths


def _evaluation_run_id_weights_suffix(weights: str) -> str:
    """Hash-input suffix for the requested weight set.

    Empty for "model" so every evaluation run ID minted before --weights
    existed remains re-derivable byte-for-byte (~20 archived IDs depend on
    this, see docs/archive/ard-distillation-2026/EXPERIMENT_DASHBOARD.md and
    scripts/aggregate_controlled_trades_fix_official_test.py's pinned
    "pgd_only_run_id"). Only a non-default weight set changes the identity.
    """
    return "" if weights == "model" else f":{weights}"


def _attack_identity(attack: Any) -> dict[str, object]:
    return attack.identity()


def _dataset_identity(dataset: Any, *, observed: dict[str, object] | None = None) -> dict[str, object]:
    fingerprints = {
        "cifar10": "c58f30108f718f92721af3b95e74349a",
        "cifar100": "eb9058c3a382ffc7106e4002c42a8d85",
    }
    fingerprint = fingerprints.get(dataset.name)
    if dataset.name == "synthetic_cifar":
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "name": dataset.name,
                    "num_samples": dataset.num_samples,
                    "num_classes": dataset.num_classes,
                    "image_size": dataset.image_size,
                    "seed": dataset.seed,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
    verification: dict[str, object] | None = None
    if dataset.name == "tiny_imagenet":
        fingerprint = dataset.content_sha256
        if observed is not None:
            observed_fingerprint = observed.get("observed_sha256")
            verification = observed
            if not isinstance(observed_fingerprint, str):
                raise ValueError("Tiny-ImageNet adapter did not expose a content digest")
            fingerprint = observed_fingerprint
        else:
            verification = {
                "algorithm": "tiny-imagenet-visible-v1",
                "expected_sha256": dataset.content_sha256,
                "verification": "expected-unverified",
            }
    if dataset.name == "imagenet":
        # Same shape as tiny_imagenet immediately above, but the manifest is
        # path/label/size-keyed, not a full byte hash -- see
        # ard.data.datasets.ImageNetDataset's own docstring (plan 0099's
        # "Design question") for why, and plan 0100's scientific review for
        # why this branch was missing until now (every evaluation of an
        # ImageNet-trained checkpoint failed fast here).
        fingerprint = dataset.content_sha256
        if observed is not None:
            observed_fingerprint = observed.get("observed_sha256")
            verification = observed
            if not isinstance(observed_fingerprint, str):
                raise ValueError("ImageNet adapter did not expose a content digest")
            fingerprint = observed_fingerprint
        else:
            verification = {
                "algorithm": "imagenet-manifest-v1",
                "expected_sha256": dataset.content_sha256,
                "verification": "expected-unverified",
            }
    if fingerprint is None:
        raise ValueError("evaluation dataset requires an explicit portable content fingerprint")
    identity: dict[str, object] = {
        "name": dataset.name,
        "split": dataset.split,
        "classes": dataset.num_classes,
        "image_size": dataset.image_size,
        "version": {
            "synthetic_cifar": "ard-synthetic-v1",
            "cifar10": "torchvision-cifar10",
            "cifar100": "torchvision-cifar100",
            "tiny_imagenet": "tiny-layout-v1",
            "imagenet": "imagenet-1k-layout-v1",
        }[dataset.name],
        "content_fingerprint": fingerprint,
    }
    if verification is not None:
        identity["content_verification"] = verification
    return identity


def evaluation_loader(dataset: Any, *, seed: int, batch_size: int, num_workers: int) -> DataLoader[IndexedBatch]:
    """The clean/PGD evaluation loader: every sample once, in dataset order.

    Shared with ``scripts/evaluate_external_checkpoint.py`` so a foreign model's
    PGD random starts (drawn per batch from one seeded generator) come from the
    same batches, in the same order, as this project's own checkpoints get.
    """
    sampler = EpochShuffleSampler(len(dataset), seed=seed, shuffle=False)
    return cast(
        DataLoader[IndexedBatch],
        DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_indexed,
        ),
    )


def _autoattack_loader(
    dataset: Any, *, sample_count: int | None, seed: int, batch_size: int, num_workers: int
) -> DataLoader[IndexedBatch]:
    """The (possibly subsetted) loader AutoAttack materializes in full.

    ``sample_count=None`` (every existing CIFAR config) attacks every image
    the dataset yields, unchanged from before this function existed. A
    fixed-seed uniform random subset, never a prefix -- see
    ``EvaluationConfig.autoattack_sample_count``'s docstring for why a
    prefix would silently bias an ImageNet-scale pass toward the first few
    classes only (``ImageNetDataset``'s own sample ordering is grouped by
    class).
    """
    autoattack_dataset: Any = dataset
    if sample_count is not None and sample_count < len(dataset):
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)[:sample_count].tolist()
        autoattack_dataset = Subset(dataset, indices)
    return cast(
        DataLoader[IndexedBatch],
        DataLoader(
            autoattack_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_indexed,
        ),
    )


def _validate_evaluation_tracking_identity(config: ExperimentConfig, training_config: ExperimentConfig) -> None:
    if config.protocol.id != training_config.protocol.id:
        raise ValueError("evaluation protocol ID must match resolved training config")
    if config.method != training_config.method:
        raise ValueError("evaluation method identity must exactly match resolved training config")
    evaluation_teacher = None if config.teacher is None else config.teacher.model_dump(mode="json")
    training_teacher = None if training_config.teacher is None else training_config.teacher.model_dump(mode="json")
    # A checkpoint's absolute location is execution metadata, not scientific
    # teacher identity. Cross-host evaluation may relocate the same verified
    # bytes; every other teacher field, including checkpoint_sha256,
    # normalization and threat model, must remain exact.
    for teacher in (evaluation_teacher, training_teacher):
        if teacher is not None:
            teacher.pop("checkpoint", None)
    if evaluation_teacher != training_teacher:
        raise ValueError("evaluation teacher identity must exactly match resolved training config")
    if config.seeds != training_config.seeds:
        raise ValueError("evaluation training seeds must exactly match resolved training config")
    if training_config.tier not in {"repro", "pilot", "production"}:
        return
    if config.tier != training_config.tier:
        raise ValueError("evaluation tier may not downgrade the resolved training tier")
    for field in ("project", "entity", "group"):
        if getattr(config.tracking, field) != getattr(training_config.tracking, field):
            raise ValueError(f"evaluation tracking {field} must match resolved training config")


def _evaluation_tracker_config(
    config: ExperimentConfig, training_config: ExperimentConfig, *, output_dir: Path
) -> ExperimentConfig:
    """Compose evaluation metadata without accepting training identity from the evaluation file."""
    return training_config.model_copy(update={"evaluation": config.evaluation, "output_dir": output_dir})


def _evaluation_preflight_config(config: ExperimentConfig, training_config: ExperimentConfig) -> ExperimentConfig:
    """Use portable local paths without changing canonical training lineage."""
    return training_config.model_copy(update={"teacher": config.teacher})


def _throughput_protocol_identity(training: TrainingConfig) -> dict[str, bool]:
    """Identity entries for the throughput options, present only when enabled.

    ``cudnn_benchmark`` and ``compile`` change kernel selection and fusion, so
    numerics differ from an eager, deterministic run and the two must never be
    pooled silently. They are emitted only when true: every run that predates
    them (or leaves them at their false defaults) keeps a byte-identical
    training_protocol_identity, so re-evaluating an old checkpoint still
    aggregates with its already-recorded rows, while an enabled run's identity
    differs and the aggregator's mixed-identity guard refuses to pool it.
    """
    identity: dict[str, bool] = {}
    if training.cudnn_benchmark:
        identity["cudnn_benchmark"] = True
    if training.compile:
        identity["compile"] = True
    return identity


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.checkpoint_dir is not None:
        checkpoint_parent = args.checkpoint_dir.resolve()
    else:
        assert args.checkpoint is not None
        checkpoint_parent = args.checkpoint.resolve().parent
    training_config_path = args.train_config or (checkpoint_parent / "resolved_config.yaml")
    if not training_config_path.is_file():
        raise FileNotFoundError(f"sibling resolved training config is missing: {training_config_path}")
    resolved_training = load_resolved_config_for_evaluation(training_config_path)
    training_config = resolved_training.config
    config = load_evaluation_config(args.config, training_config=training_config, overrides=args.overrides)
    if config.evaluation.dataset is None:
        raise ValueError("evaluation requires evaluation.dataset with an official val or test split")
    if config.evaluation.autoattack and not args.allow_autoattack:
        raise ValueError("AutoAttack is opt-in: rerun this separate evaluation process with --allow-autoattack")
    if (
        args.weights == "ema"
        and training_config.method.adr is None
        and training_config.training.weight_ema_decay is None
    ):
        raise ValueError(
            "--weights=ema requires either an adr/adr_trades training run or training.weight_ema_decay set "
            f"(plan 0102's plain weight-EMA); this checkpoint's method is {training_config.method.id!r} with "
            "no weight_ema_decay configured, so it carries no EMA state"
        )
    checkpoints = _checkpoint_paths(
        checkpoint=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        selection=config.evaluation.checkpoints,
        weights=args.weights,
    )
    _validate_evaluation_tracking_identity(config, training_config)
    # The training config remains the canonical lineage/config-hash source.
    # Only use the evaluation host's already-verified checkpoint location for
    # the local RobustBench preflight.
    preflight_config = _evaluation_preflight_config(config, training_config)
    # check_remote_run_collision=False: preflight_config still carries the
    # *training* run's own tracking.run_id/output_dir here (reused only for
    # the RobustBench/external-lock/git-dirty checks below), not this
    # evaluation's own identity. That training run's W&B history is
    # expected to exist -- checking it here would reject every ordinary
    # evaluation. The evaluation's own run ID is checked separately, once
    # its output_dir exists, right before create_tracker.
    validate_tracking_guard(
        preflight_config, root=Path.cwd(), output_dir=preflight_config.output_dir, check_remote_run_collision=False
    )
    evaluation_dataset = config.evaluation.dataset
    training_dataset = training_config.dataset
    if (evaluation_dataset.name, evaluation_dataset.num_classes, evaluation_dataset.image_size) != (
        training_dataset.name,
        training_dataset.num_classes,
        training_dataset.image_size,
    ):
        raise ValueError("evaluation dataset family/classes/image size must match the resolved training config")
    if config.student != training_config.student:
        raise ValueError("evaluation student identity and normalization must match the resolved training config")
    training_selection_attack = training_config.method.selection_attack
    assert training_selection_attack is not None
    evaluation_attack = config.evaluation.attack or training_selection_attack
    if _attack_identity(evaluation_attack) != _attack_identity(training_selection_attack):
        raise ValueError("evaluation attack must exactly match the resolved training selection attack")
    expected_config_hash = resolved_training.raw_config_hash
    checkpoint_payloads = [
        validate_checkpoint_lineage(checkpoint, expected_config_hash=expected_config_hash) for checkpoint in checkpoints
    ]
    if args.weights == "ema":
        # The filename-based check in _checkpoint_paths only catches a
        # literal "best.pt"; a student-selected checkpoint hand-copied or
        # renamed to anything else would slip past it. last.pt carries no
        # selection at all (every epoch's "last" is unambiguous), so only
        # non-last.pt files need their own selection_metadata to actually
        # say "ema".
        for checkpoint, payload in zip(checkpoints, checkpoint_payloads, strict=True):
            if checkpoint.name == "last.pt":
                continue
            selection_metadata = payload.get("selection_metadata")
            source = selection_metadata.get("selection_source") if isinstance(selection_metadata, Mapping) else None
            if source != "ema":
                raise ValueError(
                    f"--weights=ema requires a checkpoint selected on the EMA's own accuracy, but "
                    f"{checkpoint.name} was not (selection_source={source!r}) -- point at best-ema.pt instead"
                )
    train_run_id = checkpoint_payloads[0].get("tracker_run_id")
    if not isinstance(train_run_id, str):
        raise ValueError("saved checkpoint lacks a stable tracking run ID")
    checkpoint_world_size = checkpoint_payloads[0]["world_size"]
    if isinstance(checkpoint_world_size, bool) or not isinstance(checkpoint_world_size, int):
        raise ValueError("saved checkpoint has an invalid world size")
    if any(
        payload.get("tracker_run_id") != train_run_id
        or payload.get("config_hash") != expected_config_hash
        or payload.get("world_size") != checkpoint_world_size
        for payload in checkpoint_payloads
    ):
        raise ValueError("requested checkpoints do not share the same training run/config/world-size identity")
    execution_identity = training_execution_identity(
        training=training_config.training,
        world_size=checkpoint_world_size,
    )
    default_evaluation_dirname = "evaluation" if args.weights == "model" else f"evaluation-{args.weights}"
    output_dir = (args.output or ((args.checkpoint_dir or args.checkpoint.parent) / default_evaluation_dirname)).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing evaluation output: {output_dir}")
    tracker_config = _evaluation_tracker_config(config, training_config, output_dir=output_dir)
    evaluation_hash = config_digest(resolved_config_dict(tracker_config))
    output_dir.mkdir(parents=True, exist_ok=False)
    save_resolved_config(tracker_config, output_dir / "resolved_evaluation_config.yaml")
    (output_dir / "evaluation-lineage.json").write_text(
        json.dumps(
            {
                "training_config": str(training_config_path.resolve()),
                "training_config_hash": expected_config_hash,
                "training_runtime_config_hash": config_digest(resolved_config_dict(training_config)),
                "training_config_migration": resolved_training.migration,
                "evaluation_config_hash": evaluation_hash,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint_set = ",".join(path.name for path in checkpoints)
    weights_suffix = _evaluation_run_id_weights_suffix(args.weights)
    evaluation_run_id = (
        "eval-"
        + hashlib.sha256(f"{train_run_id}:{evaluation_hash}:{checkpoint_set}{weights_suffix}".encode()).hexdigest()[
            :20
        ]
    )
    # This evaluation's own real identity, now that output_dir exists and is
    # confirmed empty (line above): the counterpart the training-config-based
    # check above deliberately skipped. evaluation_run_id is deterministic in
    # (train_run_id, evaluation_hash, checkpoint_set, weights) -- a re-run
    # from a fresh --output dir after an earlier attempt reached create_tracker
    # would otherwise collide exactly like decision 0004.
    reject_stale_remote_run_id(
        mode=tracker_config.tracking.mode,
        run_id=evaluation_run_id,
        entity=tracker_config.tracking.entity,
        project=tracker_config.tracking.project,
        output_dir=output_dir,
    )
    evaluation_tracker = create_tracker(
        config=tracker_config,
        preflight_config=preflight_config,
        output_dir=output_dir,
        config_hash=evaluation_hash,
        root=Path.cwd(),
        job_type="evaluation",
        run_id=evaluation_run_id,
        training_seed=training_config.seeds.model_init,
        training_seeds=training_config.seeds.model_dump(mode="json"),
        evaluation_seed=config.evaluation.seed,
        training_execution=execution_identity,
    )
    try:
        if isinstance(evaluation_tracker, LocalTracker):
            evaluation_tracker.attach_resolved_config(output_dir / "resolved_evaluation_config.yaml")
        device = torch.device("cuda" if training_config.training.device == "cuda" else "cpu")
        if training_config.training.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = build_dataset(config.evaluation.dataset)
        evaluation_dataset_identity = _dataset_identity(config.evaluation.dataset, observed=dataset.content_identity)
        loader = evaluation_loader(
            dataset,
            seed=config.evaluation.seed,
            batch_size=training_config.training.per_rank_batch_size,
            num_workers=training_config.training.num_workers,
        )
        attack_config = evaluation_attack
        threat_model = attack_config.identity()
        threat_hash = attack_config.identity_sha256()
        attack = LinfPGD(attack_config)
        evaluation_protocol_identity = {
            "seed": config.evaluation.seed,
            "loader_batch_size": training_config.training.per_rank_batch_size,
            "attack": threat_model,
            "autoattack": {
                "enabled": config.evaluation.autoattack,
                "batch_size": config.evaluation.autoattack_batch_size,
                "sample_count": config.evaluation.autoattack_sample_count,
            },
        }
    except Exception:
        try:
            evaluation_tracker.finish(status="failed")
        except Exception:
            pass
        raise
    results: list[dict[str, Any]] = []
    try:
        for checkpoint, checkpoint_payload in zip(checkpoints, checkpoint_payloads, strict=True):
            payload_world_size = checkpoint_payload["world_size"]
            assert isinstance(payload_world_size, int) and not isinstance(payload_world_size, bool)
            student = build_student(training_config.student, tier=training_config.tier)
            result = evaluate_saved_checkpoint(
                checkpoint=checkpoint,
                model=student,
                loader=loader,
                attack=attack,
                device=device,
                seed=config.evaluation.seed,
                output_dir=output_dir,
                panel_size=config.evaluation.panel_size,
                write_sample_stats=config.evaluation.write_sample_stats,
                weights_key=args.weights,
            )
            alias = checkpoint.stem
            autoattack_result = None
            if config.evaluation.autoattack:
                # Explicitly separate from PGD and reached only from this saved-
                # checkpoint CLI process; tests inject the adapter and never call it.
                # Must restore the same weight set the PGD/clean pass above just
                # evaluated -- otherwise --weights=ema would report AA numbers
                # for the raw student instead.
                load_saved_student_checkpoint(checkpoint, student, weights_key=args.weights)
                student.to(device).eval()
                autoattack_loader = _autoattack_loader(
                    dataset,
                    sample_count=config.evaluation.autoattack_sample_count,
                    seed=config.evaluation.seed,
                    batch_size=training_config.training.per_rank_batch_size,
                    num_workers=training_config.training.num_workers,
                )
                batches = [batch.to(device) for batch in autoattack_loader]
                images = torch.cat([batch.images for batch in batches])
                labels = torch.cat([batch.labels for batch in batches])
                epsilon = attack_config.epsilon_value
                assert epsilon is not None  # resolved by AttackConfig validation
                autoattack_result = run_autoattack(
                    model=student,
                    images=images,
                    labels=labels,
                    norm=attack_config.norm,
                    epsilon=epsilon,
                    seed=config.evaluation.seed,
                    output_path=output_dir / f"autoattack-{alias}.json",
                    batch_size=config.evaluation.autoattack_batch_size,
                )
            results.append(
                {
                    "checkpoint": result.checkpoint,
                    "checkpoint_alias": alias,
                    "checkpoint_filename": checkpoint.name,
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "weights": args.weights,
                    # Which weights this checkpoint's OWN selection (if any)
                    # was made on -- "ema" only for best-ema.pt, "model" for
                    # every other file (best.pt/last.pt/epoch-*.pt are always
                    # selected, or simply not selected at all, on the
                    # student). Read from the checkpoint's own
                    # selection_metadata rather than assumed, so this stays
                    # correct if a checkpoint is renamed or hand-copied.
                    "selection_weights": cast(Mapping[str, object], checkpoint_payload.get("selection_metadata", {})).get(
                        "selection_source", "model"
                    ),
                    "threat_model": threat_model,
                    "threat_hash": threat_hash,
                    "train_run_id": train_run_id,
                    "dataset": config.evaluation.dataset.name,
                    "dataset_identity": evaluation_dataset_identity,
                    "training_dataset_identity": _dataset_identity(training_config.dataset),
                    "dataset_provenance": {
                        "root": None if config.evaluation.dataset.root is None else str(config.evaluation.dataset.root)
                    },
                    "student": training_config.student.architecture,
                    "student_identity": training_config.student.model_dump(mode="json"),
                    "method": resolved_training.migration["source_method_id"],
                    "runtime_method": training_config.method.id,
                    "method_identity": training_config.method.model_dump(mode="json"),
                    "training_config_migration": resolved_training.migration,
                    "training_protocol_identity": {
                        "id": training_config.protocol.id,
                        "epochs": training_config.training.epochs,
                        "optimizer": training_config.optimizer.model_dump(mode="json"),
                        "deterministic": training_config.training.deterministic,
                        "validation_fraction": training_config.training.validation_fraction,
                        "scheduler": training_config.scheduler.model_dump(mode="json"),
                        "execution": execution_identity,
                        # Plan 0102 scientific review, P2 finding 4: these
                        # two training-level fields (plan 0101's epsilon
                        # warmup, plan 0102's plain weight EMA) previously
                        # appeared in neither this identity nor
                        # method_identity, so summarize_checkpoint_groups's
                        # "cannot aggregate mixed experiment identities"
                        # guard could not see the one thing distinguishing
                        # two otherwise-identical arms and would pool them.
                        # No aggregator has ever run over a config carrying
                        # either field (confirmed: no aggregator exists yet
                        # for the imagenet-stage01 contract), so adding them
                        # here does not change the identity of any already-
                        # recorded result.
                        "epsilon_warmup_epochs": training_config.training.epsilon_warmup_epochs,
                        "weight_ema_decay": training_config.training.weight_ema_decay,
                        **_throughput_protocol_identity(training_config.training),
                    },
                    "evaluation_protocol_identity": evaluation_protocol_identity,
                    "teacher": None if training_config.teacher is None else training_config.teacher.architecture,
                    "teacher_identity": (
                        None if training_config.teacher is None else training_config.teacher.model_dump(mode="json")
                    ),
                    "seed": config.evaluation.seed,
                    "training_seed": training_config.seeds.model_init,
                    "training_seeds": training_config.seeds.model_dump(mode="json"),
                    "evaluation_seed": config.evaluation.seed,
                    "config_hash": expected_config_hash,
                    "clean_accuracy": result.clean_accuracy,
                    "autoattack": autoattack_result,
                    "pgd_accuracy": result.pgd_accuracy,
                    "count": result.count,
                    "sample_stats": None if result.sample_stats is None else str(result.sample_stats),
                    "panel": str(result.panel),
                }
            )
            evaluation_tracker.log_metrics(
                {
                    "checkpoint": result.checkpoint,
                    "eval_clean_accuracy": result.clean_accuracy,
                    "eval_pgd_accuracy": result.pgd_accuracy,
                }
            )
            if device.type == "cuda":
                # A8 (docs/BACKLOG.md): release this checkpoint's cached
                # allocator blocks before the next one starts. Complementary
                # to decision packet 0010's Option A (batching the AutoAttack
                # recomputation forward), not a substitute for it -- this
                # reduces cross-checkpoint fragmentation within one process,
                # it does not shrink any single allocation.
                torch.cuda.empty_cache()
        (output_dir / "evaluation-results.json").write_text(
            json.dumps(results, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        for path in (
            output_dir / "resolved_evaluation_config.yaml",
            output_dir / "evaluation-lineage.json",
            output_dir / "evaluation-results.json",
        ):
            evaluation_tracker.log_artifact(
                path, name=f"evaluation-{path.stem}-{evaluation_tracker.run_id}", artifact_type="evaluation"
            )
        for item in results:
            for field in ("panel", "sample_stats"):
                value = item[field]
                if value is not None:
                    path = Path(value)
                    evaluation_tracker.log_artifact(
                        path, name=f"evaluation-{path.stem}-{evaluation_tracker.run_id}", artifact_type="evaluation"
                    )
        bundle = output_dir / "run-bundle"
        (bundle / "completion.json").write_text(
            json.dumps({"status": "completed", "results": len(results)}) + "\n", encoding="utf-8"
        )
        (bundle / "error-marker.txt").write_text("no application error recorded\n", encoding="utf-8")
        evaluation_tracker.set_summary({"evaluation_checkpoints": [result["checkpoint"] for result in results]})
        evaluation_tracker.prepare_finish()
        if should_upload_run_bundle(tracker_config.tracking.artifact_retention):
            evaluation_tracker.log_artifact(
                bundle, name=f"run-bundle-{evaluation_tracker.run_id}", artifact_type="run-bundle"
            )
        evaluation_tracker.finish()
    except Exception:
        try:
            evaluation_tracker.finish(status="failed")
        except Exception:
            pass
        raise
    print(json.dumps({"output_dir": str(output_dir), "results": results}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - subprocess CLI
    raise SystemExit(main())
