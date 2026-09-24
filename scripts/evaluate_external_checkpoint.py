"""Evaluate a third-party checkpoint under this project's threat identity.

Why this exists rather than `ard.cli.evaluate`.  That command validates that a
checkpoint and a resolved config come from the same run, by hash.  A checkpoint
downloaded from someone else's paper has no such lineage, and manufacturing one
so it passes would defeat the guard rather than satisfy it.  This script is
explicit instead: it says the weights are foreign, records where they came from,
and reuses only the parts that make the *measurement* ours -- the official test
split, the threat identity, and the pinned AutoAttack.

What it therefore tests is the evaluation stack, not the training stack.  If a
published number reproduces here, this project's official-test AutoAttack
pipeline agrees with the field's.  If it does not, that is the more important
result and no comparison built on this pipeline can be trusted until it is
explained.

ImageNet mode (plan 0103, external anchor).  ``--dataset imagenet`` takes every
measurement setting from one of this project's own scientific configs
(``--protocol-config``): the evaluation split and its manifest hash, the PGD
threat identity, the loader batch size and workers, the evaluation seed and the
student architecture and normalization.  Clean and PGD accuracy on the full
split then go through ``ard.evaluation.evaluate_loaded_model``, the same loop
``ard.cli.evaluate`` uses for this project's checkpoints, and AutoAttack uses the
same seeded subset helper.  So a foreign ImageNet model is measured exactly as
our own run with that config would be.  The config's environment variables
(``ARD_IMAGENET_ROOT``, ``ARD_NUM_WORKERS``, ...) must be set, as for training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import torch

from ard.attacks import LinfPGD
from ard.config import load_config
from ard.config.schema import DatasetConfig
from ard.data.datasets import _to_tensor, build_dataset
from ard.evaluation import evaluate_loaded_model, run_autoattack
from ard.models import build_student
from ard.models.registry import build_architecture

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_foreign_state_dict(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    # DataParallel prefixes are a packaging detail, not a difference in weights.
    return {key.removeprefix("module."): value for key, value in state.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", choices=("cifar10", "imagenet"), default="cifar10")
    # CIFAR-only flags default to None so ImageNet mode can refuse them instead of
    # silently ignoring them (there, all of these come from --protocol-config).
    parser.add_argument("--architecture", default=None)
    parser.add_argument("--cifar10-root", type=Path, default=None)
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=None,
        help="imagenet only: this project's scientific config whose evaluation settings define the measurement",
    )
    parser.add_argument(
        "--autoattack-sample-count",
        type=int,
        default=None,
        help="imagenet only: seeded uniform AutoAttack subset size (decision 0014 direction setting: 500)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", required=True, help="where these weights came from, recorded verbatim")
    parser.add_argument("--published-robust-accuracy", type=float, default=None)
    parser.add_argument("--published-clean-accuracy", type=float, default=None, help="imagenet only")
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--allow-autoattack", action="store_true", required=True, help="AutoAttack is never run implicitly"
    )
    args = parser.parse_args()
    if args.dataset == "imagenet":
        cifar_only = ("architecture", "cifar10_root", "epsilon", "seed", "batch_size")
        given = [f"--{name.replace('_', '-')}" for name in cifar_only if getattr(args, name) is not None]
        if given:
            parser.error(f"{', '.join(given)} are CIFAR-only; in ImageNet mode they come from --protocol-config")
        return _main_imagenet(args)
    if args.cifar10_root is None:
        parser.error("--cifar10-root is required for --dataset cifar10")
    if args.protocol_config is not None or args.autoattack_sample_count is not None:
        parser.error("--protocol-config/--autoattack-sample-count are ImageNet-only")
    args.architecture = args.architecture or "saad_resnet18_cifar_v1"
    args.epsilon = 8.0 / 255.0 if args.epsilon is None else args.epsilon
    args.seed = 0 if args.seed is None else args.seed
    args.batch_size = 128 if args.batch_size is None else args.batch_size

    model = build_architecture(args.architecture, num_classes=10)
    state = _load_foreign_state_dict(args.checkpoint)
    # strict: a silently ignored key would mean a different network.
    model.load_state_dict(state, strict=True)
    model.to(args.device).eval()

    # The official test split, through this project's own loader, so that a
    # reproduced number vouches for that loader and not merely for torchvision.
    dataset = build_dataset(
        DatasetConfig(name="cifar10", root=args.cifar10_root, split="test", num_classes=10),
        transform=_to_tensor,
    )
    items = [dataset[i] for i in range(len(dataset))]  # (image, label, source_id)
    images = torch.stack([item[0] for item in items]).to(args.device)
    labels = torch.tensor([int(item[1]) for item in items]).to(args.device)
    if images.shape[0] != 10_000:
        raise SystemExit(f"expected the official 10,000-example test split, got {images.shape[0]}")
    if float(images.min()) < 0.0 or float(images.max()) > 1.0:
        raise SystemExit("inputs must be raw pixels in [0,1]; this model applies no normalisation of its own")

    with torch.no_grad():
        clean = model(images).argmax(1).eq(labels).float().mean().item()

    args.output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = run_autoattack(
        model=model,
        images=images,
        labels=labels,
        norm="linf",
        epsilon=args.epsilon,
        output_path=args.output / "autoattack.log",
        seed=args.seed,
        batch_size=args.batch_size,
    )
    record = {
        "contract": "ard_external_checkpoint_evaluation_v1",
        "lineage": "FOREIGN: these weights were not produced by this repository and have no run lineage here",
        "source": args.source,
        "checkpoint_sha256": _sha256(args.checkpoint),
        "architecture": args.architecture,
        "dataset": "cifar10 official test, 10000 examples",
        "normalization": "none; raw pixels in [0,1]",
        "threat": {"norm": "linf", "epsilon": args.epsilon, "attack": "AutoAttack standard"},
        "seed": args.seed,
        "clean_accuracy": clean,
        "autoattack": result,
    }
    if args.published_robust_accuracy is not None:
        # The key is "autoattack_accuracy" (src/ard/evaluation/autoattack.py).  Reading a
        # name that is not there used to default to NaN, so difference_pp was NaN on every
        # run and a failed calibration was indistinguishable from a passing one.  The whole
        # point of this field is to fail loudly, so an absent key is now an error.
        if "autoattack_accuracy" not in result:
            raise SystemExit("AutoAttack result carries no autoattack_accuracy; refusing to report a difference")
        measured = float(result["autoattack_accuracy"])
        record["published_robust_accuracy"] = args.published_robust_accuracy
        record["difference_pp"] = (measured - args.published_robust_accuracy) * 100.0
    (args.output / "result.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


def _load_state_dict_safely(path: Path) -> tuple[dict[str, torch.Tensor], str]:
    """Tensors only (``weights_only=True``): a downloaded file cannot run code.

    Returns the state dict and which entry it came from, recorded so a checkpoint
    that also carries e.g. EMA weights cannot be read from the wrong entry unnoticed.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise SystemExit(f"expected a state dict mapping in {path}, got {type(payload).__name__}")
    if all(isinstance(value, torch.Tensor) for value in payload.values()):
        state, key = payload, "<root>"
    elif isinstance(payload.get("state_dict"), dict):
        state, key = payload["state_dict"], "state_dict"
    else:
        raise SystemExit(f"cannot tell which entry of {path} holds the weights: {sorted(payload)}")
    return {name.removeprefix("module."): value for name, value in state.items()}, key


def _git_identity() -> dict[str, Any]:
    def git(*args: str) -> str:
        completed = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True)
        return completed.stdout.strip()

    status = git("status", "--porcelain")
    return {"source_git_sha": git("rev-parse", "HEAD"), "dirty": bool(status), "status": status}


def _environment_identity(device: torch.device) -> dict[str, Any]:
    import timm

    return {
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "timm": timm.__version__,
        # Ada/Ampere run convolutions in TF32 under PyTorch defaults; Turing has
        # no TF32, so the same flags mean full fp32 there. Recorded as a covariate.
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }


def _binomial_se(accuracy: float, count: int) -> float:
    return math.sqrt(max(accuracy * (1.0 - accuracy), 0.0) / count)


def _main_imagenet(args: argparse.Namespace) -> int:
    if args.protocol_config is None:
        raise SystemExit("--protocol-config is required for --dataset imagenet")
    if args.autoattack_sample_count is None or args.autoattack_sample_count < 1:
        raise SystemExit("--autoattack-sample-count must be set explicitly for ImageNet (no full 50k pass is implied)")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory {args.output}")
    # Deferred import: ard.cli.evaluate is a CLI module; its two loader helpers
    # are reused so a foreign model is fed exactly the batches our own runs get.
    from ard.cli.evaluate import _autoattack_loader, evaluation_loader

    config = load_config(args.protocol_config)
    if config.evaluation.dataset.name != "imagenet":
        raise SystemExit(f"--protocol-config evaluates {config.evaluation.dataset.name!r}, not imagenet")
    # Same resolution rule and guard as ard.cli.evaluate: the evaluation attack is
    # the training selection attack unless given, and must equal it exactly.
    selection_attack = config.method.selection_attack
    if selection_attack is None:
        raise SystemExit("--protocol-config has no method.selection_attack")
    attack_config = config.evaluation.attack or selection_attack
    if attack_config.identity() != selection_attack.identity():
        raise SystemExit("evaluation attack must exactly match the config's selection attack")

    # build_dataset re-derives the manifest hash and raises if it differs from
    # the config's pinned content_sha256; a config without a pin is refused.
    dataset = build_dataset(config.evaluation.dataset)
    identity = dict(dataset.content_identity or {})
    if identity.get("verification") != "computed-and-matched":
        raise SystemExit("the evaluation split must be pinned by content_sha256 and verified against it")

    # Same architecture and normalization as our run; only the initialization
    # differs, because the weights come from the checkpoint, not timm.
    student_config = config.student.model_copy(update={"pretrained": False})
    model = build_student(student_config, tier=config.tier)
    state, state_key = _load_state_dict_safely(args.checkpoint)
    # strict: a silently ignored key would mean a different network.
    model.model.load_state_dict(state, strict=True)
    device = torch.device(args.device)

    seed = config.evaluation.seed
    batch_size = config.training.per_rank_batch_size
    num_workers = config.training.num_workers
    args.output.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "contract": "ard_external_checkpoint_evaluation_imagenet_v1",
        "lineage": "FOREIGN: these weights were not produced by this repository and have no run lineage here",
        "evaluation_kind": "direction setting (decision 0014 style AutoAttack subset); NOT an official test",
        "source": args.source,
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_state_key": state_key,
        "code": _git_identity(),
        "environment": _environment_identity(device),
        "protocol_config": str(args.protocol_config),
        "protocol_config_sha256": _sha256(args.protocol_config),
        "student_identity": student_config.model_dump(mode="json"),
        "dataset": "imagenet val",
        "dataset_identity": identity,
        "threat_model": attack_config.identity(),
        "threat_hash": attack_config.identity_sha256(),
        "seed": seed,
        "loader_batch_size": batch_size,
    }

    def write(record: dict[str, Any]) -> None:
        (args.output / "result.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    measured = evaluate_loaded_model(
        model=model,
        result_name=args.checkpoint.name,
        artifact_stem="external",
        loader=evaluation_loader(dataset, seed=seed, batch_size=batch_size, num_workers=num_workers),
        attack=LinfPGD(attack_config),
        device=device,
        seed=seed,
        output_dir=args.output,
        panel_size=config.evaluation.panel_size,
        # Per-sample rows allow a paired comparison against our own run.
        write_sample_stats=True,
    )
    record.update(
        {
            "clean_accuracy": measured.clean_accuracy,
            "pgd_accuracy": measured.pgd_accuracy,
            "count": measured.count,
            "clean_accuracy_se": _binomial_se(measured.clean_accuracy, measured.count),
            "pgd_accuracy_se": _binomial_se(measured.pgd_accuracy, measured.count),
            "sample_stats": None if measured.sample_stats is None else measured.sample_stats.name,
        }
    )
    if args.published_clean_accuracy is not None:
        # The only guard against a same-shaped but different network variant
        # (padding, BN eps): it must reproduce its own published clean number.
        record["published_clean_accuracy"] = args.published_clean_accuracy
        record["clean_difference_pp"] = (measured.clean_accuracy - args.published_clean_accuracy) * 100.0
    # Written before AutoAttack so a failure there cannot lose the full-split pass.
    write({**record, "autoattack": "pending"})

    model.to(device).eval()
    autoattack_loader = _autoattack_loader(
        dataset,
        sample_count=args.autoattack_sample_count,
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    batches = [batch.to(device) for batch in autoattack_loader]
    images = torch.cat([batch.images for batch in batches])
    labels = torch.cat([batch.labels for batch in batches])
    epsilon = attack_config.epsilon_value
    assert epsilon is not None  # resolved by AttackConfig validation
    autoattack_result: dict[str, Any] = run_autoattack(
        model=model,
        images=images,
        labels=labels,
        norm=attack_config.norm,
        epsilon=epsilon,
        seed=seed,
        output_path=args.output / "autoattack-external.json",
        batch_size=config.evaluation.autoattack_batch_size,
    )
    record["autoattack"] = {
        "sample_count": int(labels.numel()),
        "sample_ids": [int(value) for batch in batches for value in batch.sample_ids.tolist()],
        "batch_size": config.evaluation.autoattack_batch_size,
        "result": autoattack_result,
    }
    if args.published_robust_accuracy is not None:
        if "autoattack_accuracy" not in autoattack_result:
            raise SystemExit("AutoAttack result carries no autoattack_accuracy; refusing to report a difference")
        robust = float(autoattack_result["autoattack_accuracy"])
        record["published_robust_accuracy"] = args.published_robust_accuracy
        # A 500-image estimate against a published figure of unknown size and
        # preprocessing: the difference is only meaningful next to its SE.
        record["robust_difference_pp"] = (robust - args.published_robust_accuracy) * 100.0
        record["autoattack_accuracy_se"] = _binomial_se(robust, int(labels.numel()))
    write(record)
    print(json.dumps({key: value for key, value in record.items() if key != "autoattack"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
