"""Fold a structurally reparameterized training run into its deployed plain network (plan 0107).

``python -m ard.cli.fold_reparam --run-dir <training output>`` reads ``last.pt``, ``best.pt`` and
``best-ema.pt`` (those present; ``last.pt`` is required), folds every weight set they carry (``model`` and,
with weight EMA, ``ema``) into the plain architecture, and writes them to ``<run-dir>/folded/`` with a
byte-identical copy of ``resolved_config.yaml``. Every other checkpoint key (config hash, tracking run ID,
world size, selection record, epoch, optimizer state, ...) is kept unchanged, so ``ard.cli.evaluate
--checkpoint-dir <run-dir>/folded`` validates the same training lineage and evaluates the plain network.
Each folded checkpoint gains a ``reparameterization`` record (contract, architectures, source file and
SHA-256, per-weight-set equivalence check). A folded checkpoint cannot be resumed: its ``model`` keys are the
plain architecture's, so the strict load of a resume fails.

Equivalence check (fail closed): on a fixed seeded batch of random images at the training image size, the
unfolded model and the folded plain model are both evaluated in eval mode in float64 (the folded weights as
stored, i.e. after the float32 cast). Their logits must agree to ``FLOAT64_RELATIVE_TOLERANCE`` of the largest
logit magnitude. Float64 removes float32 accumulation noise, so the check sees only folding error and the
float32 rounding of the stored weights. The float32 difference is recorded too, for information. Both are
measured on CPU. They say nothing about GPU agreement: on a GPU, cuDNN (TF32 convolutions, and the algorithm
cudnn_benchmark picks) makes unfolded and folded outputs differ by about 1e-3 relative per conv, the same noise
class as the algorithm choice itself. Any agreement reported for a GPU evaluation must be measured on the GPU.

Provenance: the manifest and every ``reparameterization`` record carry ``source_git_sha`` (HEAD of the source
tree that ran the fold), ``source_git_dirty``, and the torch and timm versions. A dirty source tree is refused
unless ``--allow-dirty-source`` is passed (then ``source_git_dirty`` is true).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from ard.config.loader import load_resolved_config_for_evaluation
from ard.evaluation import validate_checkpoint_lineage
from ard.models import build_student
from ard.models.reparam import FOLD_CONTRACT, REPARAM_ARCHITECTURES, deployed_student_config, fold_student

CHECKPOINT_NAMES = ("last.pt", "best.pt", "best-ema.pt")
WEIGHT_SETS = ("model", "ema")
FLOAT64_RELATIVE_TOLERANCE = 1e-5
CHECK_IMAGES = 2
CHECK_SEED = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fold a reparameterized training run into its plain deployed network.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Training output directory (resolved_config.yaml).")
    parser.add_argument(
        "--output-dir", type=Path, help="Destination (default <run-dir>/folded); must be empty or absent."
    )
    parser.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help="Fold from a source tree with uncommitted changes (recorded as source_git_dirty=true).",
    )
    return parser


def source_identity(*, allow_dirty: bool) -> dict[str, Any]:
    """HEAD SHA and cleanliness of the source tree running the fold, plus the torch and timm versions."""
    import timm

    root = Path(__file__).resolve().parents[3]
    try:
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"], check=True, capture_output=True, text=True
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("folding requires a readable source-tree Git identity") from exc
    if len(sha) != 40 or any(character not in "0123456789abcdef" for character in sha):
        raise ValueError(f"source-tree Git SHA must be an exact lowercase commit SHA, got {sha!r}")
    if dirty and not allow_dirty:
        raise ValueError(
            "the source tree has uncommitted changes; fold from a pinned clean worktree or pass --allow-dirty-source"
        )
    return {
        "source_git_sha": sha,
        "source_git_dirty": dirty,
        "torch_version": torch.__version__,
        "timm_version": timm.__version__,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_difference(reference: torch.Tensor, other: torch.Tensor) -> tuple[float, float, float]:
    reference, other = reference.double(), other.double()
    max_abs_diff = float((reference - other).abs().max())
    max_abs_output = float(reference.abs().max())
    return max_abs_diff, max_abs_output, max_abs_diff / max(max_abs_output, 1e-12)


@torch.no_grad()
def equivalence_check(training: nn.Module, plain: nn.Module, *, image_size: int) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(CHECK_SEED)
    images = torch.rand(CHECK_IMAGES, 3, image_size, image_size, generator=generator)
    results: dict[str, Any] = {"images": CHECK_IMAGES, "image_size": image_size, "seed": CHECK_SEED}
    for dtype, label in ((torch.float64, "float64"), (torch.float32, "float32")):
        unfolded = copy.deepcopy(training).to(dtype).eval()
        folded = copy.deepcopy(plain).to(dtype).eval()
        diff, magnitude, relative = _relative_difference(unfolded(images.to(dtype)), folded(images.to(dtype)))
        results[label] = {"max_abs_diff": diff, "max_abs_logit": magnitude, "relative": relative}
    results["float64_relative_tolerance"] = FLOAT64_RELATIVE_TOLERANCE
    if not results["float64"]["relative"] <= FLOAT64_RELATIVE_TOLERANCE:
        raise ValueError(f"folded and unfolded networks differ beyond tolerance: {results}")
    return results


def _atomic_save(payload: Mapping[str, Any], path: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def fold_run(run_dir: Path, output_dir: Path | None = None, *, allow_dirty_source: bool = False) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    output_dir = (output_dir or run_dir / "folded").resolve()
    config_path = run_dir / "resolved_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"resolved training config is missing: {config_path}")
    resolved = load_resolved_config_for_evaluation(config_path)
    config = resolved.config
    architecture = config.student.architecture
    spec = REPARAM_ARCHITECTURES.get(architecture)
    if spec is None:
        raise ValueError(f"{architecture!r} is not a reparameterized architecture; nothing to fold")
    if not (run_dir / "last.pt").is_file():
        raise FileNotFoundError(f"last.pt is required: {run_dir / 'last.pt'}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing fold output: {output_dir}")
    deployed = deployed_student_config(config.student)
    provenance = source_identity(allow_dirty=allow_dirty_source)
    sources = [run_dir / name for name in CHECKPOINT_NAMES if (run_dir / name).is_file()]
    payloads = [validate_checkpoint_lineage(path, expected_config_hash=resolved.raw_config_hash) for path in sources]
    for path, payload in zip(sources, payloads, strict=True):
        if "reparameterization" in payload:
            raise ValueError(f"{path.name} is already folded")
        if "auxiliary_batchnorm" in payload:
            raise ValueError(f"{path.name} carries split-BN auxiliary state, which folding does not define")

    manifest: dict[str, Any] = {
        "contract": FOLD_CONTRACT,
        "training_architecture": architecture,
        "deployed_architecture": deployed.architecture,
        "extra_branches": spec.extra_branches,
        "training_config_hash": resolved.raw_config_hash,
        "resolved_config_sha256": _sha256(config_path),
        **provenance,
        "checkpoints": [],
    }
    # Fold and check everything in memory first, so a failed check leaves no partial output directory.
    folded_payloads: list[tuple[Path, dict[str, Any]]] = []
    for source, payload in zip(sources, payloads, strict=True):
        folded_payload = dict(payload)
        checks: dict[str, Any] = {}
        weight_sets = [key for key in WEIGHT_SETS if isinstance(payload.get(key), dict)]
        for key in weight_sets:
            with torch.random.fork_rng(devices=[]):
                training = build_student(config.student, tier=config.tier)
            weights = payload[key]
            assert isinstance(weights, dict)
            training.load_state_dict(weights, strict=True)
            training.eval()
            plain = fold_student(training, config.student).eval()
            checks[key] = equivalence_check(training, plain, image_size=config.dataset.image_size)
            folded_payload[key] = plain.state_dict()
        record = {
            "contract": FOLD_CONTRACT,
            "training_architecture": architecture,
            "deployed_architecture": deployed.architecture,
            "extra_branches": spec.extra_branches,
            "source_checkpoint_filename": source.name,
            "source_checkpoint_sha256": _sha256(source),
            "folded_weight_sets": weight_sets,
            "equivalence_check": checks,
            **provenance,
        }
        folded_payload["reparameterization"] = record
        folded_payloads.append((source, folded_payload))
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, output_dir / "resolved_config.yaml")
    for source, folded_payload in folded_payloads:
        destination = output_dir / source.name
        _atomic_save(folded_payload, destination)
        manifest["checkpoints"].append(
            {
                **cast(dict[str, Any], folded_payload["reparameterization"]),
                "folded_checkpoint_sha256": _sha256(destination),
            }
        )
    (output_dir / "fold-manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = fold_run(args.run_dir, args.output_dir, allow_dirty_source=args.allow_dirty_source)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - subprocess CLI
    raise SystemExit(main())
