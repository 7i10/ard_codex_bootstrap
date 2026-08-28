#!/usr/bin/env python3
"""Materialize exact single-switch parents from sparse CROPSHIFT checkpoints.

The source controls save sparse checkpoints at file labels 49, 99 and 149.
This tool continues CROPSHIFT only until the checkpoint immediately before a
requested switch (50, 75, 100, 125 or 150).  The full optimizer, scheduler,
RNG, sampler and sample-state payload is restored; no W&B run is created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import torch
import yaml
from torch.optim import SGD
from torch.utils.data import DataLoader

from ard.attacks import LinfPGD
from ard.cli.train import _build_method, _seed_everything
from ard.config import load_config
from ard.data import (
    EpochShuffleSampler,
    build_train_validation_views,
    collate_indexed,
    data_loader_generator,
    seed_data_loader_worker,
)
from ard.engine import Trainer
from ard.models import build_student, build_teacher
from ard.schedules import build_scheduler
from ard.state import SampleStateStore

RUNS = {
    1: Path(
        "/home/islab/workspace-local/shunsuke.naito/ard-runs/ard_codex_bootstrap/"
        "ert-rslad-static-trajstab-v1/cropshift-s1-r2"
    ),
    2: Path(
        "/home/islab/workspace-local/shunsuke.naito/ard-runs/ard_codex_bootstrap/"
        "ert-rslad-static-trajstab-v1/cropshift-s2-r1"
    ),
}

# A checkpoint file labelled ``N`` contains the state at the end of payload
# epoch N-1.  Use the nearest earlier sparse control checkpoint and continue
# with CROPSHIFT until payload epoch ``boundary - 1``.
SOURCE_LABELS = {
    50: 49,
    75: 49,
    100: 99,
    125: 99,
    150: 149,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def historical_config_hash(path: Path) -> str:
    """Hash the saved resolved mapping exactly as the historical run did."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("resolved source config must be a mapping")
    canonical = deepcopy(raw)
    tracking = canonical.get("tracking")
    if isinstance(tracking, dict):
        # This is the post-hoc operational-retention exception already used by
        # the checkpoint loader; it is not part of trajectory identity.
        tracking.pop("artifact_retention", None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(1, 2), required=True)
    parser.add_argument("--boundary", type=int, choices=tuple(SOURCE_LABELS), required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_root = RUNS[args.seed]
    config_path = run_root / "run-bundle" / "resolved_config.yaml"
    source_label = SOURCE_LABELS[args.boundary]
    sparse = run_root / f"epoch-{source_label:03d}.pt"
    target_epoch = args.boundary - 1
    target_name = f"epoch-{args.boundary:03d}.pt"
    output_dir = args.output_root / f"seed{args.seed}" / f"s{args.boundary}"
    if not config_path.is_file() or not sparse.is_file():
        raise FileNotFoundError(f"missing source config/checkpoint: {config_path} / {sparse}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite materialized parent directory: {output_dir}")

    payload = torch.load(sparse, map_location="cpu", weights_only=False)
    expected_source_payload = source_label - 1
    if not isinstance(payload, dict) or payload.get("epoch") != expected_source_payload:
        raise ValueError(f"sparse checkpoint must contain payload epoch {expected_source_payload}")
    source_config = load_config(config_path)
    expected_config_hash = historical_config_hash(config_path)
    if payload.get("config_hash") != expected_config_hash:
        raise ValueError("source config hash does not match sparse checkpoint")
    if payload.get("world_size") != 1:
        raise ValueError("parent materialization requires the historical single-rank checkpoint")
    expected_lr = 0.1 if args.boundary <= 100 else 0.01
    actual_lr = float(payload["optimizer"]["param_groups"][0]["lr"])
    if abs(actual_lr - expected_lr) > 1e-12:
        raise ValueError(f"unexpected sparse parent LR: {actual_lr}")
    if payload["scheduler"]["last_epoch"] != source_label:
        raise ValueError("sparse parent scheduler boundary is inconsistent")

    config = source_config
    _seed_everything(config.seeds.model_init)
    torch.use_deterministic_algorithms(True)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_dataset, validation_dataset = build_train_validation_views(
        config.dataset,
        validation_fraction=config.training.validation_fraction,
        split_seed=config.seeds.split,
        augmentation_seed=config.seeds.augmentation,
    )
    sampler = EpochShuffleSampler(
        len(train_dataset), seed=config.seeds.data_order, rank=0, world_size=1, shuffle=True
    )
    validation_sampler = EpochShuffleSampler(
        len(validation_dataset), seed=config.seeds.data_order, rank=0, world_size=1, shuffle=False
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
        raise ValueError("historical CROPSHIFT config has no selection attack")

    output_dir.mkdir(parents=True)
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
        config_hash=expected_config_hash,
        seed=config.seeds.train_attack,
        evaluation_attack_seed=config.seeds.evaluation_attack,
        tracker_run_id=payload.get("tracker_run_id"),
        teacher=teacher,
        sample_store=sample_store,
        target_policy=target_policy,
        observation_profile=config.observation.profile,
        checkpoint_epochs=(args.boundary,),
    )
    state = trainer.resume(sparse, sampler=sampler)
    if state.next_epoch != source_label:
        raise ValueError(f"resume next epoch {state.next_epoch} != source epoch {source_label}")
    history = trainer.fit(
        loader,
        validation_loader=validation_loader,
        epochs=target_epoch + 1,
        start_epoch=state.next_epoch,
    )
    target = output_dir / target_name
    if not target.is_file():
        raise RuntimeError(f"materialization did not create {target}")
    target_payload = torch.load(target, map_location="cpu", weights_only=False)
    if not isinstance(target_payload, dict) or target_payload.get("epoch") != target_epoch:
        raise RuntimeError("materialized checkpoint has the wrong payload epoch")
    result = {
        "seed": args.seed,
        "boundary": args.boundary,
        "source_checkpoint": str(sparse.resolve()),
        "source_checkpoint_sha256": sha256(sparse),
        "target_checkpoint": str(target.resolve()),
        "target_checkpoint_sha256": sha256(target),
        "target_payload_epoch": target_payload["epoch"],
        "target_global_step": target_payload["global_step"],
        "target_scheduler_last_epoch": target_payload["scheduler"]["last_epoch"],
        "target_learning_rate": target_payload["optimizer"]["param_groups"][0]["lr"],
        "target_config_hash": target_payload["config_hash"],
        "tracker_run_id": target_payload.get("tracker_run_id"),
        "history": history,
    }
    (output_dir / "materialization.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
