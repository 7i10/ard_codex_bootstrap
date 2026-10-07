"""Teacher sanity and throughput probe for registered ImageNet teachers.

Accuracy mode (default): clean and PGD-10 (CE, Linf 4/255, step 8/765, random
start) on a fixed 5,000-image class-stratified subset of the official val split
named by ``--config``'s ``evaluation.dataset`` (digest checked), under the ARD
eval transform and the teacher's native resize interpolation; plus forward
throughput.  ``--throughput-only`` runs just the forward img/s probe (random
pixels, no data access).  Not an official evaluation.

    python -m ard.cli.teacher_sanity --config <phase2 config> \\
        --teacher salman2020_resnet50_linf_eps4=$CK/madrylab/resnet50_linf_eps4.0.ckpt \\
        --output $OUT/teacher-sanity-salman.json --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ard.cli.build_soft_label_bank import teacher_config_for
from ard.config.loader import load_config
from ard.data.datasets import ImageNetEvalTransform, build_raw_dataset
from ard.distillation.proxies import ids_digest
from ard.distillation.teacher_sanity import (
    SANITY_ATTACK,
    SANITY_SUBSET_SEED,
    NativeEvalTransform,
    _SubsetView,
    accuracy_under_pgd,
    forward_throughput,
    sanity_subset_ids,
)
from ard.models.imagenet_teacher_registry import eval_interpolation, spec_for
from ard.models.teacher import build_teacher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="config whose evaluation.dataset is the official val split")
    parser.add_argument("--teacher", action="append", required=True, help="REGISTRY_ID=CHECKPOINT (repeatable)")
    parser.add_argument("--output", type=Path, required=True, help="JSON result (never overwritten)")
    parser.add_argument("--subset-size", type=int, default=5000)
    parser.add_argument("--preprocessing", choices=("ard", "native", "both"), default="both")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--throughput-batch-size", type=int, default=128)
    parser.add_argument("--throughput-seconds", type=float, default=60.0)
    parser.add_argument("--throughput-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("overrides", nargs="*", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    device = torch.device(args.device)
    raw = ids = None
    dataset_identity: dict[str, Any] | None = None
    if not args.throughput_only:
        if args.config is None:
            raise SystemExit("--config is required unless --throughput-only")
        config = load_config(args.config, args.overrides)
        val = config.evaluation.dataset
        if val is None or val.name != "imagenet" or val.split != "val" or val.content_sha256 is None:
            raise SystemExit("evaluation.dataset must be the digest-pinned official ImageNet val split")
        raw = build_raw_dataset(val)
        ids = sanity_subset_ids(list(raw.targets), size=args.subset_size)  # type: ignore[attr-defined]
        dataset_identity = {
            "name": val.name,
            "split": val.split,
            "content_sha256": val.content_sha256,
            "subset_seed": SANITY_SUBSET_SEED,
            "subset_size": len(ids),
            "subset_ids_sha256": ids_digest(ids),
        }
    results: dict[str, Any] = {
        "contract": "ard-teacher-sanity-v1",
        "official_evaluation": False,
        "dataset": dataset_identity,
        "attack_identity": SANITY_ATTACK.identity(),
        "attack_identity_sha256": SANITY_ATTACK.identity_sha256(),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "teachers": {},
    }
    for entry in args.teacher:
        registry_id, _, path = entry.partition("=")
        spec = spec_for(registry_id)
        teacher = build_teacher(teacher_config_for(registry_id, Path(path)), tier="production").to(device)
        record: dict[str, Any] = {
            "checkpoint_sha256": spec.checkpoint_sha256,
            "normalization": spec.normalization().model_dump(mode="json"),
            "native_eval_interpolation": spec.native_eval_interpolation,
            "published_clean_accuracy": spec.published_clean_accuracy,
            "published_robust_accuracy": spec.published_robust_accuracy,
            "published_robust_attack": spec.published_robust_attack,
        }
        if raw is not None and ids is not None:
            transforms = {
                "ard": ImageNetEvalTransform(image_size=224),
                "native": NativeEvalTransform(interpolation=eval_interpolation(spec)),
            }
            chosen = ("ard", "native") if args.preprocessing == "both" else (args.preprocessing,)
            for name in chosen:
                loader = DataLoader(
                    _SubsetView(raw, ids, transforms[name]),
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    shuffle=False,
                )
                record[f"accuracy_{name}"] = accuracy_under_pgd(teacher, loader, device=device)
                print(registry_id, name, record[f"accuracy_{name}"], flush=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        record["forward_throughput"] = forward_throughput(
            teacher, device=device, batch_size=args.throughput_batch_size, seconds=args.throughput_seconds
        )
        print(registry_id, "throughput", record["forward_throughput"], flush=True)
        results["teachers"][registry_id] = record
        del teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
