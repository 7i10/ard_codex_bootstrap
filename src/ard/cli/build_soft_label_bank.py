"""Precompute FKD-style teacher soft-label banks on the exact training crops.

The training view is built exactly as ``ard.cli.train`` builds it (same
dataset, split seed, validation fraction, augmentation seed, image size and
``jpeg_draft_decode``), wrapped so every image carries its crop key.  Each
epoch is decoded once and fed to every requested teacher, so N teachers cost
one decode pass.  One bank directory per teacher:
``<output-root>/<registry_id>-aug<seed>-k<K>/``.

Epochs are resumable (an existing epoch directory is verified and skipped) and
can be split across processes with ``--epochs``; ``--finalize`` then writes the
manifest once all epochs exist and prints its SHA-256 for
``distillation.bank.manifest_sha256``.

Example (one GPU, both Phase-2 teachers of MobileNetV4-S, seed 0, K=10)::

    ARD_SEED=0 ... python -m ard.cli.build_soft_label_bank \\
        --config configs/scientific/imagenet_mobilenetv4_rslad_phase2_salman_r50_online.yaml \\
        --teacher salman2020_resnet50_linf_eps4=$CK/madrylab/resnet50_linf_eps4.0.ckpt \\
        --teacher singh2023_convnext_b_convstem=$CK/singh2023_revisiting_at/convnext_b_cvst/convnext_b_cvst_robust.pt \\
        --top-k 10 --epochs 0-49 --output-root $BANKS --device cuda --batch-size 256 --num-workers 16
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ard.config.loader import load_config
from ard.config.schema import ExperimentConfig, TeacherConfig
from ard.data.datasets import build_train_validation_views
from ard.distillation.crop_keys import CropKeyedSubset, collate_crop_keyed
from ard.distillation.soft_label_bank import (
    SENTINEL_COUNT,
    BankEpochWriter,
    SoftLabelBankError,
    bank_identity,
    compute_pixel_sentinel,
    finalize_bank,
    sha256_file,
    write_source_ids,
)
from ard.models.imagenet_teacher_registry import spec_for
from ard.models.teacher import build_teacher


def parse_epochs(text: str) -> list[int]:
    epochs: list[int] = []
    for part in text.split(","):
        if "-" in part:
            start, end = (int(value) for value in part.split("-", 1))
            epochs.extend(range(start, end + 1))
        else:
            epochs.append(int(part))
    if not epochs or min(epochs) < 0:
        raise ValueError("epochs must be non-negative, e.g. 0-49 or 0,1,2")
    return sorted(set(epochs))


def teacher_config_for(registry_id: str, checkpoint: Path) -> TeacherConfig:
    spec = spec_for(registry_id)
    return TeacherConfig(
        source="imagenet_registry",
        registry_id=registry_id,  # type: ignore[arg-type]
        architecture=spec.architecture,  # type: ignore[arg-type]
        num_classes=1000,
        normalization=spec.normalization(),
        preprocessing_owner="teacher_adapter",
        checkpoint=checkpoint,
        checkpoint_sha256=spec.checkpoint_sha256,
        threat_norm="linf",
        threat_epsilon="4/255",
    )


def bank_directory(output_root: Path, config: ExperimentConfig, registry_id: str, top_k: int) -> Path:
    return output_root / f"{registry_id}-aug{config.seeds.augmentation}-k{top_k}"


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _build_environment(device: torch.device, batch_size: int) -> dict[str, Any]:
    import PIL
    import torchvision

    return {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "pillow": PIL.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "batch_size": batch_size,
    }


def _prepare_directory(directory: Path, identity: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    identity_path = directory / "identity.json"
    payload = json.dumps(identity, sort_keys=True, indent=2) + "\n"
    if identity_path.exists():
        if identity_path.read_text(encoding="utf-8") != payload:
            raise SoftLabelBankError(f"{directory} was started for a different identity; use a new output root")
    else:
        identity_path.write_text(payload, encoding="utf-8")


def _epoch_complete(directory: Path, epoch: int) -> bool:
    meta_path = directory / f"epoch-{epoch:03d}" / "meta.json"
    if not meta_path.is_file():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for name, digest in meta["files"].items():
        if sha256_file(meta_path.parent / name) != digest:
            raise SoftLabelBankError(f"existing bank epoch {epoch} in {directory} is corrupt ({name})")
    return True


def build_banks(
    config: ExperimentConfig,
    teachers: dict[str, TeacherConfig],
    *,
    output_root: Path,
    top_k: int,
    epochs: list[int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    finalize: bool,
    teacher_modules: dict[str, torch.nn.Module] | None = None,
    log: Any = print,
) -> dict[str, str | None]:
    train_dataset, _ = build_train_validation_views(
        config.dataset,
        validation_fraction=config.training.validation_fraction,
        split_seed=config.seeds.split,
        augmentation_seed=config.seeds.augmentation,
        train_image_size=config.training.train_image_size,
        jpeg_draft_decode=config.training.jpeg_draft_decode,
    )
    keyed = CropKeyedSubset(train_dataset)
    source_ids = list(train_dataset.indices)
    if source_ids != sorted(source_ids):
        raise SoftLabelBankError("training partition indices are not sorted; bank row order would be ambiguous")
    directories: dict[str, Path] = {}
    modules: dict[str, torch.nn.Module] = {}
    for registry_id, teacher_config in teachers.items():
        run_config = config.model_copy(update={"teacher": teacher_config})
        directory = bank_directory(output_root, config, registry_id, top_k)
        _prepare_directory(directory, bank_identity(run_config))
        write_source_ids(directory, source_ids)
        directories[registry_id] = directory
        module = (teacher_modules or {}).get(registry_id)
        if module is None:
            module = build_teacher(teacher_config, tier=config.tier)
        modules[registry_id] = module.to(device).eval()
    ids_array = np.asarray(source_ids, dtype=np.int64)
    num_classes = config.dataset.num_classes
    for epoch in epochs:
        pending = [rid for rid in teachers if not _epoch_complete(directories[rid], epoch)]
        if not pending:
            log(f"epoch {epoch}: complete for every teacher, skipped")
            continue
        keyed.set_epoch(epoch)
        loader = DataLoader(
            keyed,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_crop_keyed,
            pin_memory=device.type == "cuda",
        )
        writers = {
            rid: BankEpochWriter(
                directories[rid], epoch=epoch, source_ids=ids_array, top_k=top_k, num_classes=num_classes
            )
            for rid in pending
        }
        started = time.perf_counter()
        seen = 0
        sentinel = compute_pixel_sentinel(keyed, source_ids[:SENTINEL_COUNT], epoch=epoch)
        for writer in writers.values():
            writer.pixel_sentinel = sentinel
        for batch in loader:
            images = batch.images.to(device, non_blocking=True).float()
            assert batch.crop_keys is not None
            for rid in pending:
                with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                    probabilities = F.softmax(modules[rid](images).float(), dim=1)
                writers[rid].add(batch.sample_ids, batch.labels, batch.crop_keys, probabilities)
            seen += batch.images.shape[0]
        for rid, writer in writers.items():
            meta = writer.close()
            log(
                f"epoch {epoch} {rid}: {seen} crops, {seen / max(time.perf_counter() - started, 1e-9):.1f} img/s, "
                f"teacher top-1 on crops {meta['teacher_top1_accuracy_on_crops']:.4f}, "
                f"true-class mass {meta['teacher_true_class_mass_on_crops']:.4f}, "
                f"top-{top_k} coverage {meta['teacher_top_k_mass_coverage']:.4f}"
            )
    digests: dict[str, str | None] = {}
    for registry_id, directory in directories.items():
        if not finalize:
            digests[registry_id] = None
            continue
        identity = json.loads((directory / "identity.json").read_text(encoding="utf-8"))
        all_epochs = sorted(int(path.name.split("-")[1]) for path in directory.glob("epoch-[0-9][0-9][0-9]"))
        digests[registry_id] = finalize_bank(
            directory,
            identity=identity,
            epochs=all_epochs,
            top_k=top_k,
            source_git_sha=_git_sha(),
            build_environment=_build_environment(device, batch_size),
        )
        log(f"{registry_id}: manifest {directory / 'manifest.json'} sha256 {digests[registry_id]}")
    return digests


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="Phase-2 training config (fixes the crops)")
    parser.add_argument(
        "--teacher",
        action="append",
        default=[],
        help="REGISTRY_ID=CHECKPOINT (repeatable); default: the config's own teacher",
    )
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--epochs", type=parse_epochs, required=True, help="e.g. 0-49")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--finalize", action="store_true", help="write manifests after the requested epochs")
    parser.add_argument("overrides", nargs="*", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config, args.overrides)
    teachers: dict[str, TeacherConfig] = {}
    for entry in args.teacher:
        registry_id, _, path = entry.partition("=")
        if not path:
            raise SystemExit(f"--teacher expects REGISTRY_ID=CHECKPOINT, got {entry!r}")
        teachers[registry_id] = teacher_config_for(registry_id, Path(path))
    if not teachers:
        if config.teacher is None or config.teacher.registry_id is None:
            raise SystemExit("the config has no registered teacher; pass --teacher")
        teachers[config.teacher.registry_id] = config.teacher
    digests = build_banks(
        config,
        teachers,
        output_root=args.output_root,
        top_k=args.top_k,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=torch.device(args.device),
        finalize=args.finalize,
    )
    print(json.dumps(digests, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
