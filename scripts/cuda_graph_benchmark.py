"""Peak memory and throughput of ``training.cuda_graph`` vs the eager step (plan 0105).

Synthetic data, random weights, the real ``Trainer.fit`` (panel diagnostics,
``step_diagnostics=False``, a selection pass every epoch), one process per arm.
Not a parity test (that is tests/integration/test_cuda_graph_training_step.py):
it reports where the memory and the time go.

Peak memory is the process-wide peak over the whole run (training, the last
partial batch, validation), not the per-epoch training peak the trainer
records: ``torch.cuda.reset_peak_memory_stats`` is disabled for the run.
``--keep-graph-pool`` reproduces the pre-2026-10-08 behaviour (the graph's
private pool stays allocated through the last partial batch and validation).

Examples (Hamster GPU1 only, next to a production job: keep every process small)::

    PYTHONPATH=src python scripts/cuda_graph_benchmark.py --architecture efficientnet_b0_imagenet \\
        --image-size 224 --batch-size 16 --mode graph --batches 6 --epochs 2

The EfficientNet-B0 224 px / batch 128 measurement needs a FREE 24 GB GPU (about 20 GB with the
old behaviour); see docs/plans/0105-cuda-graph-training-step.md for the command.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from ard.attacks import LinfPGD
from ard.cli.train import _seed_everything
from ard.config.schema import AttackConfig, AwpConfig, MixedBatchConfig, ModelConfig, NormalizationConfig
from ard.data import EpochShuffleSampler, IndexedDataset, SyntheticCIFAR, collate_indexed
from ard.distillation.crop_keys import CropKeyedBatch
from ard.distillation.soft_label_bank import SoftLabelBank, SoftLabelBankTeacher, _EpochArrays, compress_probabilities
from ard.distillation.trainer_hooks import DistillationTargetHooks
from ard.engine.trainer import Trainer
from ard.models import build_student
from ard.models.teacher import TeacherAdapter, TeacherMetadata
from ard.objectives import PGDATObjective, RSLADObjective
from ard.policies import RSLADBaselinePolicy
from ard.tracking.diagnostics import TrainingDiagnostics

METHODS = ("pgd_at", "rslad_bank", "rslad_advt_bank", "rslad_online", "mixed_batch", "awp")
NUM_CLASSES = 1000
TOP_K = 10


class _Batches:
    """Pre-collated, pinned batches with a real sampler (fit reads its epoch state)."""

    def __init__(self, batches: list[Any], size: int) -> None:
        self.batches = batches
        self.sampler = EpochShuffleSampler(size, seed=5, shuffle=True)
        self.dataset = None

    def __iter__(self) -> Any:
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def _batches(count: int, batch: int, image_size: int, *, partial: int, seed: int, keyed: bool) -> _Batches:
    sizes = [batch] * count + ([partial] if partial else [])
    dataset = SyntheticCIFAR(size=sum(sizes), num_classes=NUM_CLASSES, image_size=image_size, seed=seed)
    indexed = IndexedDataset(dataset)
    out, start = [], 0
    for size in sizes:
        base = collate_indexed([indexed[index] for index in range(start, start + size)])
        if keyed:
            keys = torch.zeros(size, 6, dtype=torch.int64)
            keys[:, 1] = base.sample_ids
            base = CropKeyedBatch(
                base.images, base.labels, base.sample_ids, base.state_update_mask, base.multiplicity, keys
            )
        out.append(base.pin_memory())
        start += size
    return _Batches(out, sum(sizes))


def _bank(size: int, epochs: int) -> SoftLabelBank:
    ids = np.arange(size, dtype=np.int64)
    bank = SoftLabelBank(Path("in-memory"), {"top_k": TOP_K, "num_classes": NUM_CLASSES, "epoch_records": {}}, ids)
    generator = torch.Generator().manual_seed(77)
    keys = np.zeros((size, 6), dtype=np.int32)
    keys[:, 1] = ids
    for epoch in range(epochs):
        index, prob16, residual16 = compress_probabilities(
            (torch.randn(size, NUM_CLASSES, generator=generator) * 2.0).softmax(dim=1), TOP_K
        )
        bank._epochs[epoch] = _EpochArrays(
            index=index.numpy().astype(np.uint16), prob=prob16.numpy(), residual=residual16.numpy(), keys=keys
        )
    return bank


def _teacher(architecture: str) -> TeacherAdapter:
    from ard.models.imagenet_teacher_registry import build_imagenet_teacher_architecture

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4321)
        model = build_imagenet_teacher_architecture(architecture)
    metadata = TeacherMetadata(
        architecture=architecture,
        num_classes=NUM_CLASSES,
        normalization=NormalizationConfig(profile="imagenet_standard"),
        checkpoint_sha256="0" * 64,
    )
    return TeacherAdapter(model, metadata)


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cuda")
    if args.nondeterministic:
        torch.use_deterministic_algorithms(False)
    else:
        torch.use_deterministic_algorithms(True)
    if args.keep_graph_pool:
        Trainer._release_cuda_graph = lambda self: None  # type: ignore[method-assign]
    # Process-wide peaks: the trainer resets the peak at every epoch start.
    torch.cuda.reset_peak_memory_stats = lambda *a, **k: None  # type: ignore[assignment]
    _seed_everything(1234)
    student = build_student(
        ModelConfig(
            architecture=args.architecture,  # type: ignore[arg-type]
            num_classes=NUM_CLASSES,
            pretrained=False,
            normalization=NormalizationConfig(profile="imagenet_standard"),
        ),
        tier="dev",
    ).to(device)
    optimizer = SGD(student.parameters(), lr=0.025, momentum=0.9, weight_decay=1e-4, nesterov=True)
    method = args.method
    distillation = method.startswith("rslad")
    budget = {"epsilon": "4/255", "step_size": args.step_size, "steps": args.steps, "random_start": True}
    attack = AttackConfig(**budget, **({"loss": "kl", "kl_target": "teacher_clean"} if distillation else {}))
    train_size = args.batches * args.batch_size + args.partial
    kwargs: dict[str, Any] = {}
    if distillation:
        teacher: Any
        if method == "rslad_online":
            teacher = _teacher(args.teacher)
        else:
            teacher = SoftLabelBankTeacher(
                _bank(train_size, args.epochs),
                online_teacher=_teacher(args.teacher) if method == "rslad_advt_bank" else None,
            )
        kwargs = {
            "objective": RSLADObjective(temperature=1.0, temperature_squared=True),
            "policy": RSLADBaselinePolicy(),
            "teacher": teacher,
            "distillation_hooks": DistillationTargetHooks(
                adversarial_teacher_target=method == "rslad_advt_bank", temperature=1.0
            ),
        }
    else:
        kwargs = {
            "objective": PGDATObjective(),
            "mixed_batch": MixedBatchConfig(adversarial_fraction=0.5, adversarial_weight=0.3)
            if method == "mixed_batch"
            else None,
            "awp": AwpConfig(gamma=0.01) if method == "awp" else None,
        }
    with TemporaryDirectory() as output:
        trainer = Trainer(
            model=student,
            optimizer=optimizer,
            scheduler=MultiStepLR(optimizer, milestones=[1], gamma=0.1),
            scaler=None,
            attack=LinfPGD(attack),
            selection_attack=LinfPGD(
                AttackConfig(epsilon="4/255", step_size=args.step_size, steps=args.selection_steps, random_start=True)
            ),
            device=device,
            output_dir=Path(output),
            config_hash="c" * 64,
            seed=11,
            tracker_run_id="cuda-graph-benchmark",
            diagnostics=TrainingDiagnostics.for_ids(list(range(train_size)), seed=0, size=24, mode="panel"),
            step_diagnostics=False,
            cuda_graph=args.mode == "graph",
            student_architecture=args.architecture,
            **kwargs,
        )
        loader = _batches(
            args.batches, args.batch_size, args.image_size, partial=args.partial, seed=3, keyed=method.endswith("bank")
        )
        validation = _batches(
            args.validation_batches, args.batch_size, args.image_size, partial=0, seed=99, keyed=False
        )
        torch.cuda.synchronize()
        started = time.perf_counter()
        history = trainer.fit(loader, validation_loader=validation, epochs=args.epochs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    return {
        "architecture": args.architecture,
        "method": method,
        "mode": args.mode,
        "keep_graph_pool": args.keep_graph_pool,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "deterministic": not args.nondeterministic,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        # The last epoch's rate (its first full batch eager, the second captured, the rest replayed).
        "train_images_per_second": [row["train_images_per_second"] for row in history],
        "wall_seconds": elapsed,
        "audit": [row.get("train_cuda_graph_replays") for row in history],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--architecture", default="mobilenetv4_conv_small_imagenet")
    parser.add_argument("--method", choices=METHODS, default="pgd_at")
    parser.add_argument("--teacher", default="resnet50_imagenet", help="in-step teacher (rslad_online / advT)")
    parser.add_argument("--mode", choices=("eager", "graph"), required=True)
    parser.add_argument("--keep-graph-pool", action="store_true", help="pre-2026-10-08 memory behaviour")
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches", type=int, default=40, help="full training batches per epoch")
    parser.add_argument("--partial", type=int, default=0, help="size of a last partial batch (0: none)")
    parser.add_argument("--validation-batches", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--step-size", default="8/765")
    parser.add_argument("--selection-steps", type=int, default=10)
    parser.add_argument("--nondeterministic", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run(args)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
