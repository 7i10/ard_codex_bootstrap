#!/usr/bin/env python3
"""Cheap deterministic canary for the source-keyed static augmentation views."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image

from ard.data import EpochCropshiftTransform, EpochSourceTransform


def _digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def run() -> dict[str, object]:
    image = Image.frombytes("RGB", (32, 32), bytes((index % 256 for index in range(32 * 32 * 3))) )
    canonical = EpochSourceTransform(augmentation_seed=11)
    cropshift = EpochCropshiftTransform(augmentation_seed=11, high=11)
    canonical.set_epoch(7)
    cropshift.set_epoch(7)
    c1, c2 = canonical(image, source_id=123), canonical(image, source_id=123)
    s1, s2 = cropshift(image, source_id=123), cropshift(image, source_id=123)
    if not torch.equal(c1, c2) or not torch.equal(s1, s2):
        raise AssertionError("source/epoch keyed transform is not deterministic")
    for name, tensor in (("canonical", c1), ("cropshift", s1)):
        if tensor.shape != (3, 32, 32) or tensor.dtype != torch.float32:
            raise AssertionError(f"{name} shape/dtype contract failed")
        if float(tensor.min()) < 0 or float(tensor.max()) > 1:
            raise AssertionError(f"{name} pixel range contract failed")
    arbitrary = EpochCropshiftTransform(augmentation_seed=11, high=11)(Image.new("RGB", (48, 40)), source_id=0)
    if arbitrary.shape != (3, 40, 48):
        raise AssertionError("arbitrary-resolution CropShift contract failed")
    return {
        "status": "pass",
        "source_id": 123,
        "epoch": 7,
        "canonical_sha256": _digest(c1),
        "cropshift_sha256": _digest(s1),
        "arbitrary_shape": list(arbitrary.shape),
        "augmentation_only_difference": not torch.equal(c1, s1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
