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

The default is contract ``ard-distillation-proxies-v1`` (unchanged output,
``proxies-<run>-<ckpt>-<teacher>.json``).  ``--contract v2`` selects
``ard-distillation-proxies-v2``: ``--attack ce`` and/or ``--attack rslad``
(repeatable; both in one pass share the teacher's own PGD-10), one JSON per
(pair, attack) named ``proxies-v2-<run>-<ckpt>-<teacher>-<attack>.json``, and
``--student-teacher NAME=<run dir or last.pt>`` adds one of our own Phase 1
checkpoints (built from its ``resolved_config.yaml``, frozen, eval mode) as a
proxy-only teacher:

    python -m ard.cli.distillation_proxies --contract v2 --attack ce --attack rslad \\
        --student-checkpoint $RUNS/plan0104-mobilenetv4-random-init-lr0025-v1/outputs/train/last.pt \\
        --teacher salman2020_resnet50_linf_eps4=$CK/madrylab/resnet50_linf_eps4.0.ckpt \\
        --student-teacher efficientnet_b0_phase1=$RUNS/plan0103-phase1-efficientnet-b0-random-v1 \\
        --output-dir $OUT/proxies-v2 --device cuda
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
from torch import nn
from torch.utils.data import DataLoader

from ard.cli.build_soft_label_bank import teacher_config_for
from ard.config.schema import AttackConfig, ExperimentConfig
from ard.data.datasets import build_train_validation_views
from ard.distillation.proxies import (
    METRIC_DEFINITIONS_V2,
    PROXIES_V1_CONTRACT,
    PROXIES_V2_CONTRACT,
    PROXY_SUBSET_SEED,
    SEED_OFFSETS,
    STUDENT_ATTACKS,
    FixedEpochSubset,
    FrozenEvalModel,
    compute_pair_proxies,
    compute_pair_proxies_v2,
    ids_digest,
    proxy_subset_ids,
    rslad_attack_from_config,
    validate_rslad_attack,
)
from ard.models import build_student
from ard.models.imagenet_teacher_registry import spec_for
from ard.models.teacher import build_teacher

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RSLAD_CONFIG = (
    REPOSITORY_ROOT / "configs" / "scientific" / "imagenet_mobilenetv4_rslad_phase2_salman_r50_online.yaml"
)


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


def resolve_run_checkpoint(path: Path) -> Path:
    """``<run dir>`` -> ``<run dir>/outputs/train/last.pt``; a directory holding ``last.pt`` or a file as given."""
    if path.is_file():
        return path
    for candidate in (path / "outputs" / "train" / "last.pt", path / "last.pt"):
        if candidate.is_file():
            return candidate
    raise SystemExit(f"no last.pt under {path} (looked in outputs/train/ and the directory itself)")


def load_same_recipe_teacher(path: Path, student_config: ExperimentConfig) -> tuple[nn.Module, dict[str, Any]]:
    """One of our own Phase 1 checkpoints as a proxy-only teacher (frozen, eval)."""
    checkpoint = resolve_run_checkpoint(path)
    model, config, digest = load_student(checkpoint, None)
    if config.student.num_classes != student_config.student.num_classes:
        raise SystemExit(f"{checkpoint}: class count differs from the student's")
    if config.dataset.content_sha256 != student_config.dataset.content_sha256:
        raise SystemExit(f"{checkpoint}: trained on a different dataset digest than the student")
    run_id = config.tracking.run_id or checkpoint.parent.parent.parent.name
    return FrozenEvalModel(model), {
        "source": "phase1_run_checkpoint",
        "run_id": run_id,
        "architecture": config.student.architecture,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "config_protocol": config.protocol.id,
        "selection_attack_identity_sha256": (
            None if config.method.selection_attack is None else config.method.selection_attack.identity_sha256()
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--student-checkpoint-sha256")
    parser.add_argument("--teacher", action="append", default=[], help="REGISTRY_ID=CHECKPOINT (repeatable)")
    parser.add_argument(
        "--student-teacher",
        action="append",
        default=[],
        help="v2 only: NAME=<Phase 1 run dir or last.pt> (repeatable); our own checkpoint as a proxy-only teacher",
    )
    parser.add_argument("--contract", choices=("v1", "v2"), default="v1")
    parser.add_argument(
        "--attack",
        action="append",
        choices=STUDENT_ATTACKS,
        help="v2 only: student attack for x+dS (repeatable; default ce)",
    )
    parser.add_argument(
        "--rslad-attack-config",
        type=Path,
        default=DEFAULT_RSLAD_CONFIG,
        help="v2 --attack rslad: Phase 2 RSLAD config whose method.attack is the attack identity",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, help="override the resolved config's dataset.root")
    parser.add_argument("--subset-size", type=int, default=10000)
    parser.add_argument("--crop-epoch", type=int, default=0, help="training-crop epoch to view the subset at")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--attack-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser


def _validate_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.teacher and not args.student_teacher:
        parser.error("at least one --teacher or --student-teacher is required")
    if args.contract == "v1" and (args.student_teacher or args.attack):
        parser.error("--student-teacher and --attack require --contract v2")
    entries = [*args.teacher, *args.student_teacher]
    if any(not name or not path for name, _, path in (entry.partition("=") for entry in entries)):
        parser.error("teachers are NAME=PATH")
    names = [entry.partition("=")[0] for entry in entries]
    if len(set(names)) != len(names):
        parser.error(f"teacher names must be unique: {names}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_arguments(parser, args)
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
    student_record = {
        "run_id": run_id,
        "architecture": config.student.architecture,
        "checkpoint": args.student_checkpoint.name,
        "checkpoint_sha256": student_sha,
        "config_protocol": config.protocol.id,
    }
    subset_record = {
        "partition": "training (validation_fraction split by seeds.split)",
        "seed": PROXY_SUBSET_SEED,
        "size": len(ids),
        "ids_sha256": ids_digest(ids),
        "crop_epoch": args.crop_epoch,
        "augmentation_seed": config.seeds.augmentation,
        "dataset_content_sha256": config.dataset.content_sha256,
    }

    def loader() -> DataLoader[Any]:
        return DataLoader(
            FixedEpochSubset(train_view, ids, epoch=args.crop_epoch),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )

    if args.contract == "v2":
        return _main_v2(args, student, config, student_sha, attack, run_id, student_record, subset_record, loader)
    for entry in args.teacher:
        registry_id, _, path = entry.partition("=")
        output = args.output_dir / f"proxies-{run_id}-{args.student_checkpoint.stem}-{registry_id}.json"
        if output.exists():
            print(f"skip existing {output}")
            continue
        teacher = build_teacher(teacher_config_for(registry_id, Path(path)), tier="production").to(device)
        metrics = compute_pair_proxies(
            student, teacher, loader(), attack_config=attack, device=device, attack_seed=args.attack_seed
        )
        record = {
            "contract": PROXIES_V1_CONTRACT,
            "official_evaluation": False,
            "student": student_record,
            "teacher": {"registry_id": registry_id, "checkpoint_sha256": spec_for(registry_id).checkpoint_sha256},
            "subset": subset_record,
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


def _main_v2(
    args: argparse.Namespace,
    student: nn.Module,
    config: ExperimentConfig,
    student_sha: str,
    selection_attack: AttackConfig,
    run_id: str,
    student_record: dict[str, Any],
    subset_record: dict[str, Any],
    loader: Any,
) -> int:
    device = torch.device(args.device)
    requested = list(dict.fromkeys(args.attack or ["ce"]))
    attack_configs: dict[str, AttackConfig] = {}
    provenance: dict[str, Any] = {}
    for name in requested:
        if name == "ce":
            attack_configs[name] = selection_attack
            provenance[name] = {"source": "student resolved_config.yaml method.selection_attack"}
        else:
            rslad_attack, origin = rslad_attack_from_config(args.rslad_attack_config)
            validate_rslad_attack(rslad_attack, selection_attack)
            attack_configs[name] = rslad_attack
            provenance[name] = origin | {
                "target": "teacher clean logits, full online softmax (online_teacher target source; no bank truncation)"
            }
    teachers: list[tuple[str, str, str]] = [("registry", *entry.partition("=")[::2]) for entry in args.teacher]
    teachers += [("phase1", *entry.partition("=")[::2]) for entry in args.student_teacher]
    for kind, name, path in teachers:
        outputs = {
            attack: args.output_dir / f"proxies-v2-{run_id}-{args.student_checkpoint.stem}-{name}-{attack}.json"
            for attack in requested
        }
        missing = {attack: config_ for attack, config_ in attack_configs.items() if not outputs[attack].exists()}
        for attack in sorted(set(requested) - set(missing)):
            print(f"skip existing {outputs[attack]}")
        if not missing:
            continue
        if kind == "registry":
            teacher: nn.Module = build_teacher(teacher_config_for(name, Path(path)), tier="production")
            teacher_record: dict[str, Any] = {
                "source": "imagenet_registry",
                "name": name,
                "registry_id": name,
                "checkpoint_sha256": spec_for(name).checkpoint_sha256,
            }
        else:
            teacher, teacher_record = load_same_recipe_teacher(Path(path), config)
            teacher_record = {"name": name} | teacher_record
        teacher = teacher.to(device)
        results = compute_pair_proxies_v2(
            student,
            teacher,
            loader(),
            selection_attack=selection_attack,
            student_attacks=missing,
            device=device,
            attack_seed=args.attack_seed,
        )
        for attack, metrics in results.items():
            student_attack = attack_configs[attack]
            record = {
                "contract": PROXIES_V2_CONTRACT,
                "official_evaluation": False,
                "student": student_record,
                "teacher": teacher_record,
                "self_pair": teacher_record["checkpoint_sha256"] == student_sha,
                "subset": subset_record,
                "student_attack": {
                    "name": attack,
                    "identity": student_attack.identity(),
                    "identity_sha256": student_attack.identity_sha256(),
                    "provenance": provenance[attack],
                },
                "teacher_own_attack": {
                    "identity": selection_attack.identity(),
                    "identity_sha256": selection_attack.identity_sha256(),
                    "provenance": "student resolved_config.yaml method.selection_attack, run on the teacher",
                },
                "attack_identity": student_attack.identity(),
                "attack_identity_sha256": student_attack.identity_sha256(),
                "attack_seed": args.attack_seed,
                "seed_offsets": dict(SEED_OFFSETS),
                "metric_definitions": METRIC_DEFINITIONS_V2,
                "metrics": metrics,
            }
            outputs[attack].write_text(json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"output": str(outputs[attack]), **metrics}, sort_keys=True), flush=True)
        del teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
