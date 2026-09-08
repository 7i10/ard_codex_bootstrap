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
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from ard.config.schema import DatasetConfig
from ard.data.datasets import _to_tensor, build_dataset
from ard.evaluation import run_autoattack
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
    parser.add_argument("--architecture", default="saad_resnet18_cifar_v1")
    parser.add_argument("--cifar10-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", required=True, help="where these weights came from, recorded verbatim")
    parser.add_argument("--published-robust-accuracy", type=float, default=None)
    parser.add_argument("--epsilon", type=float, default=8.0 / 255.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--allow-autoattack", action="store_true", required=True, help="AutoAttack is never run implicitly"
    )
    args = parser.parse_args()

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


if __name__ == "__main__":
    raise SystemExit(main())
