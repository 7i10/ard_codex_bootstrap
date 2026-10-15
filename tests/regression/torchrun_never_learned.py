"""Two-rank gloo check of the never-learned guard (debugging 0030, review P2-5c).

Asserted, with an odd-sized validation set (the two-rank sampler pads one image,
which is masked out):
* ``validate_epoch``'s ``distinct_predictions`` / ``logit_std_over_images`` /
  ``num_examples`` under DDP equal a single-process computation over the whole set;
* a constant-output student makes BOTH ranks raise ``NeverLearnedError`` after
  epoch 1, and rank 0 writes ``never-learned.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from ard.attacks import LinfPGD
from ard.config.schema import AttackConfig
from ard.data import (
    EpochShuffleSampler,
    IndexedDataset,
    SyntheticCIFAR,
    collate_indexed,
    stratified_train_validation_split,
)
from ard.engine.distributed import initialize_from_env, teardown, unwrap_model, wrap_ddp
from ard.engine.never_learned import NeverLearnedError
from ard.engine.trainer import Trainer
from ard.objectives import PGDATObjective


class _ConstantOutput(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3 * 4 * 4, 3)
        nn.init.zeros_(self.linear.weight)
        self.linear.weight.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.linear(images.flatten(1))


def _healthy() -> nn.Module:
    torch.manual_seed(31)
    return nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(128, 3))


def _trainer(model: nn.Module, device: torch.device, output: Path) -> Trainer:
    selection = AttackConfig(
        epsilon="1/255", step_size="1/255", steps=1, random_start=True, student_mode="eval", teacher_mode="eval"
    )
    return Trainer(
        model=model,
        optimizer=SGD([p for p in model.parameters() if p.requires_grad], lr=0.05),
        scheduler=None,
        scaler=None,
        attack=LinfPGD(AttackConfig(epsilon="1/255", step_size="1/255", steps=1, random_start=True)),
        selection_attack=LinfPGD(selection),
        objective=PGDATObjective(),
        device=device,
        output_dir=output,
        config_hash="e" * 64,
        seed=7,
        tracker_run_id="ddp-never-learned",
    )


def main() -> None:
    output_root = Path(sys.argv[1])
    device, initialized = initialize_from_env("cpu")
    assert initialized
    try:
        rank, world_size = dist.get_rank(), dist.get_world_size()
        assert world_size == 2
        dataset = IndexedDataset(SyntheticCIFAR(size=60, num_classes=3, image_size=4, seed=13))
        train, validation = stratified_train_validation_split(dataset, validation_fraction=0.25, seed=13)
        assert len(validation) == 15  # odd: the two-rank sampler pads one example

        def loader(data, *, shuffle: bool) -> DataLoader:
            sampler = EpochShuffleSampler(len(data), seed=13, rank=rank, world_size=world_size, shuffle=shuffle)
            return DataLoader(data, batch_size=4, sampler=sampler, collate_fn=collate_indexed)

        # 1. DDP statistics equal the single-process statistics over all 15 images.
        healthy = wrap_ddp(_healthy().to(device), device)
        metrics = _trainer(healthy, device, output_root / "healthy").validate_epoch(loader(validation, shuffle=False))
        reference_model = unwrap_model(healthy).eval()
        whole = next(iter(DataLoader(validation, batch_size=len(validation), collate_fn=collate_indexed)))
        with torch.no_grad():
            logits = reference_model(whole.images).double()
        assert metrics["num_examples"] == 15.0, metrics
        assert metrics["distinct_predictions"] == float(logits.argmax(1).unique().numel()), metrics
        expected_std = float(logits.std(0, unbiased=False).mean())
        # 1e-6 relative: the reference forward batches 15 images at once, DDP
        # batches 4, and fp32 convolution results differ in the last bits.
        assert abs(metrics["logit_std_over_images"] - expected_std) <= 1e-6 * expected_std, (
            metrics,
            expected_std,
        )

        # 2. A constant-output student: every rank raises after epoch 1.
        torch.manual_seed(5)
        constant = wrap_ddp(_ConstantOutput().to(device), device)
        output = output_root / "constant"
        trainer = _trainer(constant, device, output)
        raised = False
        try:
            trainer.fit(loader(train, shuffle=True), validation_loader=loader(validation, shuffle=False), epochs=4)
        except NeverLearnedError:
            raised = True
        assert trainer.current_epoch == 1, trainer.current_epoch
        gathered: list[object] = [None] * world_size
        dist.all_gather_object(gathered, (raised, sorted(trainer.never_learned_observations.items())))
        assert all(item[0] for item in gathered), gathered  # type: ignore[index]
        assert gathered[0] == gathered[1], gathered
        dist.barrier()
        if rank == 0:
            marker = json.loads((output / "never-learned.json").read_text(encoding="utf-8"))
            assert set(marker["observations"]) == {"0", "1"} and marker["validation_images"] == 15.0, marker
            print("NEVER_LEARNED_DDP_OK", flush=True)
    finally:
        teardown()


if __name__ == "__main__":
    main()
