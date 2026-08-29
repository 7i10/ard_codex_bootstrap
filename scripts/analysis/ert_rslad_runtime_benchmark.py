"""Bounded runtime benchmark for the core canonical CIFAR RSLAD workload.

This script is deliberately separate from the production trainer.  It measures
the core hot path on a real CIFAR/Teacher configuration without writing
checkpoints, W&B state, or scientific run artifacts.  It intentionally omits
the production diagnostics panel and per-sample history bookkeeping so their
overhead is not mistaken for model/attack time.  One invocation measures one
candidate so compile cold-start and cache effects remain explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader

from ard.attacks import AttackRequest, LinfPGD
from ard.config import load_config
from ard.data import (
    EpochShuffleSampler,
    build_train_validation_views,
    collate_indexed,
    data_loader_generator,
    seed_data_loader_worker,
)
from ard.models import build_student, build_teacher
from ard.objectives import RSLADObjective


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(
    config: Any,
    *,
    workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
) -> DataLoader[Any]:
    train, _ = build_train_validation_views(
        config.dataset,
        validation_fraction=config.training.validation_fraction,
        split_seed=config.seeds.split,
        augmentation_seed=config.seeds.augmentation,
    )
    train.set_epoch(100)
    sampler = EpochShuffleSampler(len(train), seed=config.seeds.data_order, shuffle=True)
    kwargs: dict[str, Any] = {
        "batch_size": config.training.per_rank_batch_size,
        "sampler": sampler,
        "num_workers": workers,
        "collate_fn": collate_indexed,
        "generator": data_loader_generator(config.seeds.data_order),
        "worker_init_fn": seed_data_loader_worker,
        "pin_memory": pin_memory,
    }
    if workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(train, **kwargs)


def _compile_model(model: torch.nn.Module, mode: str) -> tuple[torch.nn.Module, float]:
    if mode == "eager":
        return model, 0.0
    started = time.perf_counter()
    compiled = torch.compile(model, mode=mode, fullgraph=False)
    return compiled, time.perf_counter() - started


def _load_student_checkpoint(student: torch.nn.Module, checkpoint: Path | None) -> str | None:
    if checkpoint is None:
        return None
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model"), Mapping):
        raise ValueError("checkpoint must contain a model state mapping")
    student.load_state_dict(payload["model"])
    return hashlib.sha256(checkpoint.read_bytes()).hexdigest()


def _timed_batch(
    *,
    iterator: Any,
    model: torch.nn.Module,
    teacher: torch.nn.Module,
    attack: LinfPGD,
    objective: RSLADObjective,
    optimizer: SGD,
    device: torch.device,
    attack_generator: torch.Generator,
    non_blocking: bool,
) -> dict[str, float]:
    marks: dict[str, float] = {}
    started = time.perf_counter()
    batch = next(iterator)
    marks["data"] = time.perf_counter() - started

    _sync(device)
    started = time.perf_counter()
    images = batch.images.to(device, non_blocking=non_blocking)
    labels = batch.labels.to(device, non_blocking=non_blocking)
    _sync(device)
    marks["h2d"] = time.perf_counter() - started

    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        _sync(device)
        started = time.perf_counter()
        teacher_clean = teacher(images.float()).detach().float()
        _sync(device)
    marks["teacher_clean"] = time.perf_counter() - started

    _sync(device)
    started = time.perf_counter()
    attack_result = attack.generate(
        AttackRequest(
            inputs=images,
            labels=labels,
            student=model,
            teacher=teacher,
            target_logits=teacher_clean,
            generator=attack_generator,
        )
    )
    _sync(device)
    marks["pgd10"] = time.perf_counter() - started

    _sync(device)
    started = time.perf_counter()
    with torch.no_grad():
        _ = teacher(attack_result.adversarial.float()).detach().float()
    _sync(device)
    marks["teacher_adv"] = time.perf_counter() - started

    _sync(device)
    started = time.perf_counter()
    adversarial_logits = model(attack_result.adversarial)
    clean_logits = model(images)
    terms = objective(
        student_logits=adversarial_logits,
        labels=labels,
        teacher_logits=teacher_clean,
        clean_student_logits=clean_logits,
    )
    loss = terms.total.mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("benchmark loss is non-finite")
    loss.backward()
    optimizer.step()
    _sync(device)
    marks["outer"] = time.perf_counter() - started
    return marks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--candidate",
        choices=("eager", "default", "reduce-overhead", "max-autotune-no-cudagraphs", "max-autotune"),
        default="eager",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--warmup-batches", type=int, default=8)
    parser.add_argument("--measure-batches", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--non-blocking", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup_batches < 0 or args.measure_batches < 1 or args.workers < 0:
        raise ValueError("invalid benchmark length or worker count")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device(args.device)
    config = load_config(args.config)
    _seed(args.seed)
    if config.training.deterministic:
        torch.use_deterministic_algorithms(True)
    loader = _loader(
        config,
        workers=args.workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    student = build_student(config.student, tier="production").to(device)
    checkpoint_sha256 = _load_student_checkpoint(student, args.checkpoint)
    teacher = build_teacher(config.teacher, tier="production").to(device)
    teacher.eval()
    model, compile_seconds = _compile_model(student, args.candidate)
    model.train()
    attack = LinfPGD(config.method.attack)
    objective = RSLADObjective(
        temperature=config.method.temperature,
        temperature_squared=config.method.temperature_squared,
    )
    optimizer = SGD(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        momentum=config.optimizer.momentum,
        weight_decay=config.optimizer.weight_decay,
        nesterov=config.optimizer.nesterov,
    )
    attack_generator = torch.Generator(device=device).manual_seed(config.seeds.train_attack)
    iterator = iter(loader)
    for _ in range(args.warmup_batches):
        _timed_batch(
            iterator=iterator,
            model=model,
            teacher=teacher,
            attack=attack,
            objective=objective,
            optimizer=optimizer,
            device=device,
            attack_generator=attack_generator,
            non_blocking=args.non_blocking,
        )
    measurements: list[dict[str, float]] = []
    for _ in range(args.measure_batches):
        measurements.append(
            _timed_batch(
                iterator=iterator,
                model=model,
                teacher=teacher,
                attack=attack,
                objective=objective,
                optimizer=optimizer,
                device=device,
                attack_generator=attack_generator,
                non_blocking=args.non_blocking,
            )
        )
    keys = ("data", "h2d", "teacher_clean", "pgd10", "teacher_adv", "outer")
    medians = {key: float(torch.tensor([row[key] for row in measurements]).median()) for key in keys}
    totals = [sum(row[key] for key in keys) for row in measurements]
    seconds = float(torch.tensor(totals).median())
    examples = config.training.per_rank_batch_size
    result = {
        "schema_version": 1,
        "kind": "ert_rslad_runtime_benchmark_v2",
        "config": str(args.config),
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "candidate": args.candidate,
        "device": str(device),
        "seed": args.seed,
        "warmup_batches": args.warmup_batches,
        "measure_batches": args.measure_batches,
        "workers": args.workers,
        "pin_memory": bool(args.pin_memory),
        "non_blocking": bool(args.non_blocking),
        "persistent_workers": bool(args.persistent_workers),
        "prefetch_factor": args.prefetch_factor if args.workers > 0 else None,
        "compile_seconds": compile_seconds,
        "segment_median_seconds": medians,
        "median_batch_seconds": seconds,
        "images_per_second": examples / seconds,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "ARD_NUM_WORKERS")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
