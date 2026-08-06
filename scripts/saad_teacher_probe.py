#!/usr/bin/env python3
"""Emit and compare deterministic cross-runtime RobustBench teacher logits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_exclusive(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")


def fixed_inputs():
    import numpy as np
    import torch

    count = 4 * 3 * 32 * 32
    array = (np.arange(count, dtype=np.float32) / np.float32(count - 1)).reshape(4, 3, 32, 32)
    return torch.from_numpy(array), hashlib.sha256(array.tobytes(order="C")).hexdigest()


def emit(args: argparse.Namespace) -> int:
    checkpoint = args.checkpoint.resolve()
    robustbench_root = args.robustbench_root.resolve()
    if not checkpoint.is_file():
        raise RuntimeError("teacher checkpoint is absent")
    observed_checkpoint = sha256_file(checkpoint)
    if observed_checkpoint != args.checkpoint_sha256:
        raise RuntimeError("teacher checkpoint SHA-256 mismatch")
    sys.path.insert(0, str(robustbench_root))
    import autoattack.state as autoattack_state
    import robustbench
    import torch
    from robustbench.utils import load_model

    if not Path(robustbench.__file__).resolve().is_relative_to(robustbench_root):
        raise RuntimeError("RobustBench was not imported from the pinned checkout")
    autoattack_path = Path(autoattack_state.__file__).resolve()
    if robustbench_root in autoattack_path.parents:
        raise RuntimeError("official autoattack.state was not selected")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the teacher probe")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inputs, input_sha256 = fixed_inputs()
    with tempfile.TemporaryDirectory(prefix="ard-saad-teacher-probe-", dir=args.output.parent) as temporary:
        model_dir = Path(temporary)
        staged = model_dir / "cifar10" / "Linf" / f"{args.teacher_name}.pt"
        staged.parent.mkdir(parents=True)
        staged.symlink_to(checkpoint)
        model = (
            load_model(
                model_name=args.teacher_name,
                model_dir=model_dir,
                dataset="cifar10",
                threat_model="Linf",
            )
            .cuda()
            .eval()
        )
        with torch.no_grad():
            logits = model(inputs.cuda()).float().cpu()
    if logits.shape != (4, 10) or not torch.isfinite(logits).all():
        raise RuntimeError("teacher probe logits must be finite with shape [4, 10]")
    write_exclusive(
        args.output,
        {
            "schema_version": 1,
            "teacher_name": args.teacher_name,
            "checkpoint_sha256": observed_checkpoint,
            "fixed_input": {
                "formula": "float32(arange(4*3*32*32)/(4*3*32*32-1)).reshape(4,3,32,32)",
                "sha256": input_sha256,
            },
            "runtime": {
                "python": sys.version,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "sources": {
                "probe_sha256": sha256_file(Path(__file__)),
                "robustbench_path": str(Path(robustbench.__file__).resolve()),
                "robustbench_sha256": sha256_file(Path(robustbench.__file__).resolve()),
                "autoattack_state_path": str(autoattack_path),
                "autoattack_state_sha256": sha256_file(autoattack_path),
            },
            "logits": logits.tolist(),
            "argmax": logits.argmax(dim=1).tolist(),
        },
    )
    return 0


def compare(args: argparse.Namespace) -> int:
    import numpy as np

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    for key in ("teacher_name", "checkpoint_sha256", "fixed_input"):
        if reference.get(key) != candidate.get(key):
            raise RuntimeError(f"teacher probe identity mismatch: {key}")
    if reference["runtime"]["torch"] != args.reference_torch:
        raise RuntimeError("reference torch version mismatch")
    if candidate["runtime"]["torch"] != args.candidate_torch:
        raise RuntimeError("candidate torch version mismatch")
    reference_logits = np.asarray(reference["logits"], dtype=np.float64)
    candidate_logits = np.asarray(candidate["logits"], dtype=np.float64)
    if reference_logits.shape != (4, 10) or candidate_logits.shape != (4, 10):
        raise RuntimeError("teacher probe logits must have shape [4, 10]")
    if not np.isfinite(reference_logits).all() or not np.isfinite(candidate_logits).all():
        raise RuntimeError("teacher probe logits must be finite")
    delta = np.abs(reference_logits - candidate_logits)
    max_abs = float(delta.max())
    mean_abs = float(delta.mean())
    argmax_equal = reference["argmax"] == candidate["argmax"]
    passed = bool(np.allclose(reference_logits, candidate_logits, rtol=args.rtol, atol=args.atol)) and argmax_equal
    value = {
        "schema_version": 1,
        "passed": passed,
        "teacher_name": reference["teacher_name"],
        "checkpoint_sha256": reference["checkpoint_sha256"],
        "fixed_input": reference["fixed_input"],
        "contract": {
            "reference_torch": args.reference_torch,
            "candidate_torch": args.candidate_torch,
            "fixed_input_count": 4,
            "atol": args.atol,
            "rtol": args.rtol,
            "require_argmax_equal": True,
        },
        "observed": {"max_abs": max_abs, "mean_abs": mean_abs, "argmax_equal": argmax_equal},
        "reference": {"identity": {"path": str(args.reference.resolve()), "sha256": sha256_file(args.reference)}},
        "candidate": {"identity": {"path": str(args.candidate.resolve()), "sha256": sha256_file(args.candidate)}},
        "probe_source_sha256": reference["sources"]["probe_sha256"],
    }
    write_exclusive(args.output, value)
    if not passed or not math.isfinite(max_abs) or not math.isfinite(mean_abs):
        raise RuntimeError(f"teacher probe comparison failed: max_abs={max_abs}, argmax_equal={argmax_equal}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    emit_parser = subparsers.add_parser("emit")
    emit_parser.add_argument("--teacher-name", required=True)
    emit_parser.add_argument("--checkpoint", type=Path, required=True)
    emit_parser.add_argument("--checkpoint-sha256", required=True)
    emit_parser.add_argument("--robustbench-root", type=Path, required=True)
    emit_parser.add_argument("--output", type=Path, required=True)
    emit_parser.set_defaults(handler=emit)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--reference", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--reference-torch", required=True)
    compare_parser.add_argument("--candidate-torch", required=True)
    compare_parser.add_argument("--atol", type=float, required=True)
    compare_parser.add_argument("--rtol", type=float, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    compare_parser.set_defaults(handler=compare)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
