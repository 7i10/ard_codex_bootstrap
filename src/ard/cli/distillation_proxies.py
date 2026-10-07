"""Cheap (student, teacher) distillation proxies; one JSON per pair (see ard.distillation.proxies).

The student is a finished Phase 1 PGD-AT run: ``--student-checkpoint`` (its
``best.pt``/``last.pt``, SHA-256 checked when given) next to the run's
``resolved_config.yaml``, which fixes the dataset, partition, augmentation
seed and the selection attack used for ``x + dS``.  ``ARD_IMAGENET_ROOT`` style
paths in that resolved config are already absolute; pass
``--dataset-root`` to read the same (digest-checked) training set elsewhere.

    python -m ard.cli.distillation_proxies \\
        --student-checkpoint $RUNS/plan0104-mobilenetv4-random-init-lr0025-v1/outputs/train/last.pt \\
        --teacher singh2023_convnext_b_convstem=$CK/singh2023_revisiting_at/convnext_b_cvst/convnext_b_cvst_robust.pt \\
        --output-dir $OUT/proxies --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from ard.cli.build_soft_label_bank import teacher_config_for
from ard.config.schema import ExperimentConfig
from ard.data.datasets import build_train_validation_views
from ard.distillation.proxies import (
    PROXY_SUBSET_SEED,
    FixedEpochSubset,
    compute_pair_proxies,
    ids_digest,
    proxy_subset_ids,
)
from ard.models import build_student
from ard.models.imagenet_teacher_registry import spec_for
from ard.models.teacher import build_teacher


def load_student(checkpoint: Path, expected_sha256: str | None) -> tuple[torch.nn.Module, ExperimentConfig, str]:
    data = checkpoint.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise SystemExit(f"student checkpoint SHA-256 mismatch: expected {expected_sha256}, got {digest}")
    config = ExperimentConfig.model_validate(
        yaml.safe_load((checkpoint.parent / "resolved_config.yaml").read_text(encoding="utf-8"))
    )
    # Our own epoch-boundary checkpoint (NumPy RNG state needs the full unpickler); bytes hashed above.
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    student = build_student(config.student, tier=config.tier)
    student.load_state_dict(payload["model"], strict=True)
    return student, config, digest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--student-checkpoint-sha256")
    parser.add_argument("--teacher", action="append", required=True, help="REGISTRY_ID=CHECKPOINT (repeatable)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, help="override the resolved config's dataset.root")
    parser.add_argument("--subset-size", type=int, default=10000)
    parser.add_argument("--crop-epoch", type=int, default=0, help="training-crop epoch to view the subset at")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--attack-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device)
    student, config, student_sha = load_student(args.student_checkpoint, args.student_checkpoint_sha256)
    dataset = (
        config.dataset if args.dataset_root is None else config.dataset.model_copy(update={"root": args.dataset_root})
    )
    train_view, _ = build_train_validation_views(
        dataset,
        validation_fraction=config.training.validation_fraction,
        split_seed=config.seeds.split,
        augmentation_seed=config.seeds.augmentation,
        train_image_size=config.training.train_image_size,
        jpeg_draft_decode=config.training.jpeg_draft_decode,
    )
    ids = proxy_subset_ids(train_view, size=args.subset_size)
    attack = config.method.selection_attack
    assert attack is not None
    student = student.to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = config.tracking.run_id or args.student_checkpoint.parent.parent.parent.name
    for entry in args.teacher:
        registry_id, _, path = entry.partition("=")
        output = args.output_dir / f"proxies-{run_id}-{args.student_checkpoint.stem}-{registry_id}.json"
        if output.exists():
            print(f"skip existing {output}")
            continue
        teacher = build_teacher(teacher_config_for(registry_id, Path(path)), tier="production").to(device)
        loader: DataLoader[Any] = DataLoader(
            FixedEpochSubset(train_view, ids, epoch=args.crop_epoch),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
        metrics = compute_pair_proxies(
            student, teacher, loader, attack_config=attack, device=device, attack_seed=args.attack_seed
        )
        record = {
            "contract": "ard-distillation-proxies-v1",
            "official_evaluation": False,
            "student": {
                "run_id": run_id,
                "architecture": config.student.architecture,
                "checkpoint": args.student_checkpoint.name,
                "checkpoint_sha256": student_sha,
                "config_protocol": config.protocol.id,
            },
            "teacher": {"registry_id": registry_id, "checkpoint_sha256": spec_for(registry_id).checkpoint_sha256},
            "subset": {
                "partition": "training (validation_fraction split by seeds.split)",
                "seed": PROXY_SUBSET_SEED,
                "size": len(ids),
                "ids_sha256": ids_digest(ids),
                "crop_epoch": args.crop_epoch,
                "augmentation_seed": config.seeds.augmentation,
                "dataset_content_sha256": config.dataset.content_sha256,
            },
            "attack_identity": attack.identity(),
            "attack_identity_sha256": attack.identity_sha256(),
            "attack_seed": args.attack_seed,
            "metrics": metrics,
        }
        output.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(output), **metrics}, sort_keys=True), flush=True)
        del teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
