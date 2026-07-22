"""Separate saved-checkpoint evaluation command.

This process constructs only the student and a configured CE PGD attacker.  It
does not construct a teacher, objective, policy, optimizer, or sample state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import DataLoader

from ard.attacks import LinfPGD
from ard.config import ExperimentConfig, load_config, save_resolved_config
from ard.config.loader import resolved_config_dict
from ard.config.schema import training_execution_identity
from ard.data import EpochShuffleSampler, IndexedBatch, build_dataset, collate_indexed
from ard.engine import config_digest
from ard.evaluation import (
    evaluate_saved_checkpoint,
    load_saved_student_checkpoint,
    run_autoattack,
    validate_checkpoint_lineage,
)
from ard.models import build_student
from ard.tracking import LocalTracker, create_tracker, validate_tracking_guard


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
    parser.add_argument("overrides", nargs="*", help="Dot-path YAML overrides")
    return parser


def _checkpoint_paths(*, checkpoint: Path | None, checkpoint_dir: Path | None, selection: str) -> tuple[Path, ...]:
    if checkpoint is not None:
        return (checkpoint.resolve(),)
    assert checkpoint_dir is not None
    directory = checkpoint_dir.resolve()
    names = {"best": ("best.pt",), "last": ("last.pt",), "both": ("best.pt", "last.pt")}[selection]
    paths = tuple(directory / name for name in names)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("requested saved checkpoints are missing: " + ", ".join(missing))
    return paths


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
        }[dataset.name],
        "content_fingerprint": fingerprint,
    }
    if verification is not None:
        identity["content_verification"] = verification
    return identity


def _validate_evaluation_tracking_identity(config: ExperimentConfig, training_config: ExperimentConfig) -> None:
    if config.protocol.id != training_config.protocol.id:
        raise ValueError("evaluation protocol ID must match resolved training config")
    if config.method != training_config.method:
        raise ValueError("evaluation method identity must exactly match resolved training config")
    if config.teacher != training_config.teacher:
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config, args.overrides)
    if config.evaluation.dataset is None:
        raise ValueError("evaluation requires evaluation.dataset with an official val or test split")
    if config.evaluation.autoattack and not args.allow_autoattack:
        raise ValueError("AutoAttack is opt-in: rerun this separate evaluation process with --allow-autoattack")
    checkpoints = _checkpoint_paths(
        checkpoint=args.checkpoint, checkpoint_dir=args.checkpoint_dir, selection=config.evaluation.checkpoints
    )
    training_config_path = args.train_config or (checkpoints[0].parent / "resolved_config.yaml")
    if not training_config_path.is_file():
        raise FileNotFoundError(f"sibling resolved training config is missing: {training_config_path}")
    training_config = load_config(training_config_path)
    _validate_evaluation_tracking_identity(config, training_config)
    validate_tracking_guard(training_config, root=Path.cwd())
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
    expected_config_hash = config_digest(resolved_config_dict(training_config))
    checkpoint_payloads = [
        validate_checkpoint_lineage(checkpoint, expected_config_hash=expected_config_hash) for checkpoint in checkpoints
    ]
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
    output_dir = (args.output or ((args.checkpoint_dir or args.checkpoint.parent) / "evaluation")).resolve()
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
                "evaluation_config_hash": evaluation_hash,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint_set = ",".join(path.name for path in checkpoints)
    evaluation_run_id = (
        "eval-" + hashlib.sha256(f"{train_run_id}:{evaluation_hash}:{checkpoint_set}".encode()).hexdigest()[:20]
    )
    evaluation_tracker = create_tracker(
        config=tracker_config,
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
        sampler = EpochShuffleSampler(len(dataset), seed=config.evaluation.seed, shuffle=False)
        loader = cast(
            DataLoader[IndexedBatch],
            DataLoader(
                dataset,
                batch_size=training_config.training.per_rank_batch_size,
                sampler=sampler,
                num_workers=training_config.training.num_workers,
                collate_fn=collate_indexed,
            ),
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
            )
            alias = checkpoint.stem
            autoattack_result = None
            if config.evaluation.autoattack:
                # Explicitly separate from PGD and reached only from this saved-
                # checkpoint CLI process; tests inject the adapter and never call it.
                load_saved_student_checkpoint(checkpoint, student)
                student.to(device).eval()
                batches = [batch.to(device) for batch in loader]
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
                    "method": training_config.method.id,
                    "method_identity": training_config.method.model_dump(mode="json"),
                    "training_protocol_identity": {
                        "id": training_config.protocol.id,
                        "epochs": training_config.training.epochs,
                        "optimizer": training_config.optimizer.model_dump(mode="json"),
                        "deterministic": training_config.training.deterministic,
                        "validation_fraction": training_config.training.validation_fraction,
                        "scheduler": training_config.scheduler.model_dump(mode="json"),
                        "execution": execution_identity,
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
