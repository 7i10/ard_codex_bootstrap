"""Teacher sanity: clean / PGD-10 accuracy on a fixed official-val subset, and forward throughput.

Not an official evaluation: a 5,000-image class-stratified subset (5 per class)
of the official ImageNet val split, PGD-10 CE (Linf 4/255, step 8/765, random
start, eval mode) from this project's ``LinfPGD``.  Two preprocessings:
``ard`` (this project's ``ImageNetEvalTransform``: bilinear shorter side 256,
centre crop 224 -- what an ARD evaluation sees) and ``native`` (the teacher's
upstream resize interpolation, e.g. bicubic for Singh et al.).  The teacher's
own normalization is applied inside its adapter in both.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as transform_functional

from ard.attacks import AttackRequest, LinfPGD
from ard.config.schema import AttackConfig
from ard.data.datasets import ImageNetEvalTransform, _to_tensor, train_probe_ids

SANITY_SUBSET_SEED = 20261008
SANITY_ATTACK = AttackConfig(
    loss="ce", epsilon="4/255", step_size="8/765", steps=10, random_start=True, student_mode="eval", teacher_mode="eval"
)


class NativeEvalTransform:
    """Resize shorter side to round(224*256/224)=256 with the teacher's interpolation, centre crop 224."""

    def __init__(self, *, interpolation: Any, image_size: int = 224) -> None:
        self.interpolation = interpolation
        self.image_size = image_size

    def __call__(self, image: Any) -> torch.Tensor:
        resized = transform_functional.resize(
            image, round(self.image_size * 256 / 224), interpolation=self.interpolation
        )
        return _to_tensor(transform_functional.center_crop(resized, self.image_size))


class _SubsetView(Dataset[tuple[torch.Tensor, int, int]]):
    def __init__(self, raw: Any, ids: Sequence[int], transform: Any) -> None:
        self.raw, self.ids, self.transform = raw, list(ids), transform

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, int, int]:
        source_id = self.ids[position]
        image, label = self.raw[source_id]
        return self.transform(image), int(label), int(source_id)


def sanity_subset_ids(targets: Sequence[int], *, size: int = 5000, seed: int = SANITY_SUBSET_SEED) -> list[int]:
    return train_probe_ids(list(range(len(targets))), targets, size=size, seed=seed)


def accuracy_under_pgd(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    *,
    device: torch.device,
    attack_config: AttackConfig = SANITY_ATTACK,
    attack_seed: int = 0,
) -> dict[str, float]:
    attack = LinfPGD(attack_config)
    generator = torch.Generator(device=device).manual_seed(attack_seed)
    clean = robust = count = 0
    model.eval()
    for images, labels, _ in loader:
        images, labels = images.to(device).float(), labels.to(device)
        with torch.no_grad():
            clean += int((model(images).argmax(1) == labels).sum().item())
        adversarial = attack.generate(
            AttackRequest(inputs=images, labels=labels, student=model, generator=generator, stream_tag="teacher_sanity")
        ).adversarial
        with torch.no_grad():
            robust += int((model(adversarial).argmax(1) == labels).sum().item())
        count += labels.shape[0]
    return {"count": float(count), "clean_accuracy": clean / count, "pgd10_accuracy": robust / count}


def forward_throughput(
    model: torch.nn.Module,
    *,
    device: torch.device,
    batch_size: int = 128,
    image_size: int = 224,
    warmup_batches: int = 10,
    seconds: float = 60.0,
) -> dict[str, float]:
    """Eval-mode, no-grad FP32 forward images/s on random pixels (decode-free upper bound)."""
    model.eval()
    inputs = torch.rand(batch_size, 3, image_size, image_size, device=device)
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        for _ in range(warmup_batches):
            model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        batches = 0
        while True:
            model(inputs)
            batches += 1
            if device.type == "cuda" and batches % 5 == 0:
                torch.cuda.synchronize(device)
            if batches % 5 == 0 and time.perf_counter() - started >= seconds:
                break
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
    result = {
        "batch_size": float(batch_size),
        "batches": float(batches),
        "seconds": elapsed,
        "images_per_second": batches * batch_size / elapsed,
    }
    if device.type == "cuda":
        result["cuda_peak_allocated_bytes"] = float(torch.cuda.max_memory_allocated(device))
    return result


__all__ = [
    "ImageNetEvalTransform",
    "NativeEvalTransform",
    "SANITY_ATTACK",
    "accuracy_under_pgd",
    "forward_throughput",
    "sanity_subset_ids",
]
