#!/usr/bin/env python3
"""Bounded deterministic canary for the nested static augmentation policies.

This is deliberately a data/RNG audit only.  It uses no model, dataset split,
or outcome and therefore can run before the long production trajectories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch
import torchvision
from PIL import Image

from ard.data import EpochCropReTransform, EpochCropshiftTransform, EpochIdbhWeakTransform
from ard.data.datasets import _apply_cropshift, _layer_generator, _to_tensor


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(tensor: torch.Tensor) -> str:
    return _sha256_bytes(tensor.detach().cpu().contiguous().numpy().tobytes())


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _canary_image() -> Image.Image:
    values = bytes((index * 37 + 11) % 256 for index in range(32 * 32 * 3))
    return Image.frombytes("RGB", (32, 32), values)


def run(*, sample_count: int = 4096) -> dict[str, object]:
    image = _canary_image()
    policies = {
        "cropshift": EpochCropshiftTransform(augmentation_seed=17, high=11),
        "crop_re": EpochCropReTransform(augmentation_seed=17, high=11),
        "idbh_weak": EpochIdbhWeakTransform(augmentation_seed=17, high=11),
    }
    epoch = 7
    ids = list(range(sample_count))

    # Every policy is deterministic and does not consume process-global RNG.
    hashes: dict[str, str] = {}
    for name, policy in policies.items():
        policy.set_epoch(epoch)
        before = torch.get_rng_state()
        first = policy(image, source_id=123)
        middle = policy(image, source_id=123)
        after = torch.get_rng_state()
        if not torch.equal(first, middle):
            raise AssertionError(f"{name} is not source/epoch deterministic")
        if not torch.equal(before, after):
            raise AssertionError(f"{name} consumed global torch RNG")
        if first.shape != (3, 32, 32) or first.dtype != torch.float32:
            raise AssertionError(f"{name} shape/dtype contract failed")
        if float(first.min()) < 0.0 or float(first.max()) > 1.0:
            raise AssertionError(f"{name} pixel-domain contract failed")
        hashes[name] = _digest(first)

    # Worker/sampler order cannot change a source-keyed view.
    for name, policy in policies.items():
        policy.set_epoch(epoch)
        forward = {source_id: _digest(policy(image, source_id=source_id)) for source_id in ids[:64]}
        reverse = {source_id: _digest(policy(image, source_id=source_id)) for source_id in reversed(ids[:64])}
        if forward != reverse:
            raise AssertionError(f"{name} depends on iteration order")

    # The CropShift prefix remains byte-identical whenever the erase layer's
    # Bernoulli draw declines to apply.  This also guards layer isolation.
    policies["crop_re"].set_epoch(epoch)
    prefix_equal_ids: list[int] = []
    erase_expected = 0
    erase_observed_difference = 0
    for source_id in ids[:256]:
        prefix_generator = torch.Generator().manual_seed(17 + 1_000_003 * epoch + 10_007 * source_id)
        prefix = _to_tensor(_apply_cropshift(image, generator=prefix_generator, high=11))
        layer_generator = _layer_generator(augmentation_seed=17, epoch=epoch, source_id=source_id, layer="erase")
        expected = float(torch.rand((), generator=layer_generator).item()) < 0.5
        observed = policies["crop_re"](image, source_id=source_id)
        if not expected:
            prefix_equal_ids.append(source_id)
            if not torch.equal(prefix, observed):
                raise AssertionError("CROP_RE changed the spatial prefix without erasing")
        else:
            erase_expected += 1
            if not torch.equal(prefix, observed):
                erase_observed_difference += 1
    if erase_expected and erase_observed_difference == 0:
        raise AssertionError("RandomErasing was never observable on the canary image")

    # Distribution checks use only pre-treatment RNG draws, never outcomes.
    erase_count = 0
    color_counts = [0] * 8
    for source_id in ids:
        erase_generator = _layer_generator(augmentation_seed=17, epoch=epoch, source_id=source_id, layer="erase")
        if float(torch.rand((), generator=erase_generator).item()) < 0.5:
            erase_count += 1
        color_generator = _layer_generator(augmentation_seed=17, epoch=epoch, source_id=source_id, layer="colorshape")
        operation = min(int(float(torch.rand((), generator=color_generator).item()) * 8), 7)
        color_counts[operation] += 1
    erase_rate = erase_count / sample_count
    color_rates = [count / sample_count for count in color_counts]
    if abs(erase_rate - 0.5) > 0.04:
        raise AssertionError(f"erase application rate is out of tolerance: {erase_rate:.4f}")
    if max(abs(rate - 0.125) for rate in color_rates) > 0.03:
        raise AssertionError(f"ColorShape operation frequencies are out of tolerance: {color_rates}")

    source_sha = Path(".external/DA-Alone-Improves-AT/src/data/idbh.py").read_bytes()
    return {
        "schema_version": 1,
        "status": "pass",
        "source_git_sha": _git_sha(),
        "torchvision_version": torchvision.__version__,
        "random_erasing_defaults": {
            "p": 0.5,
            "scale": [0.02, 0.33],
            "ratio": [0.3, 3.3],
            "value": 0,
            "inplace": False,
        },
        "upstream_idbh": {
            "commit": "38b740aeffe5933c16869a126c6972ef443a8352",
            "path": ".external/DA-Alone-Improves-AT/src/data/idbh.py",
            "sha256": _sha256_bytes(source_sha),
            "colorshape": "color",
        },
        "transform_order": {
            "crop_re": ["RandomHorizontalFlip/CropShift", "ToTensor", "RandomErasing"],
            "idbh_weak": ["RandomHorizontalFlip/CropShift", "ColorShape(color)", "ToTensor", "RandomErasing"],
        },
        "rng_contract": {
            "spatial": "augmentation_seed + 1000003*epoch + 10007*source_id",
            "named_layers": ["colorshape", "erase"],
            "global_torch_rng_unchanged": True,
            "order_independent": True,
        },
        "canary": {
            "augmentation_seed": 17,
            "epoch": epoch,
            "source_id": 123,
            "output_sha256": hashes,
            "spatial_prefix_equal_without_erase_ids": prefix_equal_ids,
            "sample_count": sample_count,
            "erase_application_count": erase_count,
            "erase_application_rate": erase_rate,
            "colorshape_operation_counts": color_counts,
            "colorshape_operation_rates": color_rates,
            "erase_observable_difference_count_first_256": erase_observed_difference,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sample-count", type=int, default=4096)
    args = parser.parse_args()
    payload = json.dumps(run(sample_count=args.sample_count), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
