#!/usr/bin/env python3
"""Bounded old-vs-sample-keyed random-start benchmark and contract report."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ard.attacks.pgd import sample_keyed_random_start


def _measure(fn, *, repeats: int, device: torch.device) -> float:
    values = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        values.append(time.perf_counter() - start)
    return sum(values) / len(values)


def _one_case(*, batch: int, resolution: int, device: torch.device, repeats: int) -> dict[str, object]:
    clean = torch.full((batch, 3, resolution, resolution), 0.5, device=device)
    source_ids = torch.arange(batch, dtype=torch.long)
    generator = torch.Generator(device=device).manual_seed(123)

    def old() -> None:
        output = torch.empty_like(clean)
        output.uniform_(-1.0, 1.0, generator=generator)

    def keyed() -> None:
        sample_keyed_random_start(clean, source_ids, attack_seed=7, epoch=100)

    old_seconds = _measure(old, repeats=repeats, device=device)
    keyed_seconds = _measure(keyed, repeats=repeats, device=device)
    return {
        "batch": batch,
        "resolution": resolution,
        "device": str(device),
        "repeats": repeats,
        "old_seconds": old_seconds,
        "sample_keyed_seconds": keyed_seconds,
        "overhead_fraction": (keyed_seconds / old_seconds) - 1.0 if old_seconds else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")
    device = torch.device(args.device)
    rows = [
        _one_case(batch=128, resolution=32, device=device, repeats=args.repeats),
        _one_case(batch=1, resolution=224, device=device, repeats=args.repeats),
    ]
    payload = {
        "schema_version": 1,
        "kind": "ert_rslad_sample_keyed_attack_rng_benchmark",
        "algorithm": "sample_keyed_v1",
        "key_fields": ["attack_seed", "epoch", "source_id", "stream_tag", "restart_index"],
        "rows": rows,
        "torch_version": torch.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
