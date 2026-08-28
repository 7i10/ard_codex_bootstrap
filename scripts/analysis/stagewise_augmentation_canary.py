#!/usr/bin/env python3
"""One-batch runtime/resume canary for a stage-wise augmentation fork."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
from pathlib import Path

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader

from ard.attacks import LinfPGD
from ard.cli.train import _build_method, _seed_everything
from ard.config import load_config
from ard.config.loader import resolved_config_dict
from ard.data import (
    EpochCropReTransform,
    EpochCropshiftTransform,
    EpochIdbhWeakTransform,
    EpochShuffleSampler,
    build_train_validation_views,
    collate_indexed,
)
from ard.engine import Trainer, config_digest
from ard.engine.checkpoint import REQUIRED_KEYS
from ard.models import build_student, build_teacher
from ard.schedules import build_scheduler
from ard.state import SampleStateStore


class OneBatchLoader:
    """Expose one real DataLoader batch while retaining Trainer loader fields."""

    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader
        self.dataset = loader.dataset
        self.sampler = loader.sampler

    def __iter__(self):
        yield next(iter(self.loader))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--switch", type=int, choices=(50, 75, 100, 125, 150), required=True)
    parser.add_argument("--late-policy", choices=("crop_re", "idbh_weak"), required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if config.dataset.stagewise_switch_epoch != args.switch or config.dataset.stagewise_late_policy != args.late_policy:
        raise ValueError("canary policy does not match config")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or REQUIRED_KEYS.difference(payload):
        raise ValueError("canary checkpoint is incomplete")
    config_hash = config_digest(resolved_config_dict(config))
    if payload.get("config_hash") != config_hash:
        raise ValueError("canary checkpoint/config hash mismatch")
    lineage = payload.get("fork_lineage")
    if (
        not isinstance(lineage, dict)
        or lineage.get("switch_epoch") != args.switch
        or lineage.get("late_policy") != args.late_policy
    ):
        raise ValueError("canary fork lineage mismatch")
    if payload.get("epoch") != args.switch - 1:
        raise ValueError("canary checkpoint is not the exact switch-boundary parent")

    _seed_everything(config.seeds.model_init)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, _ = build_train_validation_views(
        config.dataset,
        validation_fraction=config.training.validation_fraction,
        split_seed=config.seeds.split,
        augmentation_seed=config.seeds.augmentation,
    )
    transform = train_dataset.dataset.transform
    raw = train_dataset.dataset.dataset
    if not hasattr(transform, "set_epoch"):
        raise TypeError("stage-wise dataset did not expose an epoch-aware transform")
    prefix = EpochCropshiftTransform(augmentation_seed=config.seeds.augmentation, high=11)
    late = (
        EpochCropReTransform(augmentation_seed=config.seeds.augmentation, high=11)
        if args.late_policy == "crop_re"
        else EpochIdbhWeakTransform(augmentation_seed=config.seeds.augmentation, high=11)
    )
    sample_id = int(train_dataset.indices[0])
    image = raw[sample_id][0]
    for epoch, reference in ((args.switch - 1, prefix), (args.switch, late)):
        transform.set_epoch(epoch)
        reference.set_epoch(epoch)
        actual = transform(image, source_id=sample_id)
        expected = reference(image, source_id=sample_id)
        if not torch.equal(actual, expected):
            raise AssertionError(f"augmentation policy mismatch at epoch {epoch}")
    transform.set_epoch(args.switch)

    sampler = EpochShuffleSampler(len(train_dataset), seed=config.seeds.data_order, rank=0, world_size=1, shuffle=True)
    sampler.set_epoch(args.switch)
    loader = DataLoader(
        train_dataset,
        batch_size=8,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_indexed,
    )
    one_batch = OneBatchLoader(loader)
    student = build_student(config.student, tier=config.tier).to(device)
    teacher = build_teacher(config.teacher, tier=config.tier) if config.teacher is not None else None
    optimizer = SGD(
        student.parameters(),
        lr=config.optimizer.learning_rate,
        momentum=config.optimizer.momentum,
        weight_decay=config.optimizer.weight_decay,
        nesterov=config.optimizer.nesterov,
    )
    scheduler = build_scheduler(optimizer, config.scheduler)
    objective, policy, sample_store, target_policy = _build_method(config)
    if config.observation.records_student_history and sample_store is None:
        sample_store = SampleStateStore(ema_decay=config.method.student_ema_decay)
    selection_attack_config = config.method.selection_attack
    if selection_attack_config is None:
        raise ValueError("canary config has no selection attack")
    with tempfile.TemporaryDirectory(prefix="stagewise-canary-") as directory:
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
            output_dir=Path(directory),
            config_hash=config_hash,
            seed=config.seeds.train_attack,
            evaluation_attack_seed=config.seeds.evaluation_attack,
            tracker_run_id=payload.get("tracker_run_id"),
            teacher=teacher,
            sample_store=sample_store,
            target_policy=target_policy,
            observation_profile=config.observation.profile,
            checkpoint_epochs=(),
        )
        state = trainer.resume(args.checkpoint, sampler=sampler)
        if state.next_epoch != args.switch:
            raise AssertionError(f"resume next epoch {state.next_epoch} != {args.switch}")
        trainer.current_epoch = args.switch
        metrics = trainer.train_epoch(one_batch)
        if not all(torch.isfinite(torch.tensor(value)) for value in metrics.values()):
            raise AssertionError("canary metrics are non-finite")
        gradients = [parameter.grad for parameter in trainer.model.parameters() if parameter.grad is not None]
        if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
            raise AssertionError("canary gradients are missing or non-finite")
    print(
        {
            "checkpoint_sha256": sha256(args.checkpoint),
            "switch": args.switch,
            "late_policy": args.late_policy,
            "next_epoch": state.next_epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "batch_metrics": metrics,
            "status": "pass",
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
