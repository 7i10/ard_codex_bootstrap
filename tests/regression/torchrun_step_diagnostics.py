"""Two-rank gloo check that ``step_diagnostics=False`` changes no DDP training state.

The skipped post-step forward is a no-grad DDP forward, so with the default
it is the one that broadcasts rank 0's BatchNorm buffers after each step.
With the option off that broadcast moves to the next DDP forward (the next
step's eval-mode training attack, or validation). The student has
BatchNorm, the training attack runs the student in eval mode (so it reads
the running statistics), and each rank sees different data (so the ranks'
local running statistics really diverge between broadcasts).

Asserted:
* every rank's final model state and rank 0's ``last.pt``/``best.pt``
  model/optimizer/RNG (all ranks)/selection state are identical for
  ``step_diagnostics`` True and False;
* the per-rank buffer invariant: with True the BN running statistics are
  equal across ranks when ``train_epoch`` returns; with False they are not
  (the weaker invariant) but become equal -- to rank 0's unchanged values --
  after the next DDP forward.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from ard.attacks import LinfPGD
from ard.config.schema import AttackConfig
from ard.data import EpochShuffleSampler, IndexedDataset, SyntheticCIFAR, collate_indexed
from ard.engine.distributed import initialize_from_env, teardown, wrap_ddp
from ard.engine.trainer import Trainer
from ard.objectives import PGDATObjective

EPOCHS = 2


def _student() -> nn.Module:
    torch.manual_seed(31)
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 3),
    )


def _loader(dataset: IndexedDataset, *, rank: int, world_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=4,
        sampler=EpochShuffleSampler(len(dataset), seed=13, rank=rank, world_size=world_size, shuffle=shuffle),
        collate_fn=collate_indexed,
    )


def _trainer(output: Path, device: torch.device, *, step_diagnostics: bool) -> Trainer:
    model = wrap_ddp(_student().to(device), device)
    assert model.broadcast_buffers
    optimizer = SGD(model.parameters(), lr=0.05, momentum=0.9)
    return Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        attack=LinfPGD(
            AttackConfig(
                epsilon="4/255", step_size="2/255", steps=2, random_start=True, student_mode="eval", teacher_mode="eval"
            )
        ),
        selection_attack=LinfPGD(
            AttackConfig(
                epsilon="2/255", step_size="1/255", steps=2, random_start=True, student_mode="eval", teacher_mode="eval"
            )
        ),
        objective=PGDATObjective(),
        device=device,
        output_dir=output,
        config_hash="d" * 64,
        seed=19,
        tracker_run_id="ddp-step-diagnostics",
        step_diagnostics=step_diagnostics,
    )


def _digest(value: Any) -> str:
    digest = hashlib.sha256()

    def feed(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(f"T|{tensor.dtype}|{tuple(tensor.shape)}|".encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(f"D{len(item)}|".encode())
            for key in sorted(item, key=repr):
                digest.update(f"{key!r}:".encode())
                feed(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(f"L{len(item)}|".encode())
            for element in item:
                feed(element)
        elif isinstance(item, float):
            digest.update(f"F{item.hex()}|".encode())
        else:
            digest.update(f"{type(item).__name__}:{item!r}|".encode())

    feed(value)
    return digest.hexdigest()


def _gathered_running_mean(model: nn.Module) -> list[torch.Tensor]:
    batchnorm = model.module[1]
    assert isinstance(batchnorm, nn.BatchNorm2d)
    gathered = [torch.empty_like(batchnorm.running_mean) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, batchnorm.running_mean.detach().clone())
    return gathered


def main() -> None:
    output_root = Path(sys.argv[1])
    device, initialized = initialize_from_env("cpu")
    assert initialized
    try:
        rank, world_size = dist.get_rank(), dist.get_world_size()
        assert world_size == 2
        # Disjoint shards of one dataset: every rank normalizes different data.
        dataset = IndexedDataset(SyntheticCIFAR(size=32, num_classes=3, image_size=4, seed=13))
        validation = IndexedDataset(SyntheticCIFAR(size=8, num_classes=3, image_size=4, seed=14))

        final_state: dict[bool, str] = {}
        for step_diagnostics in (True, False):
            torch.manual_seed(1000)
            trainer = _trainer(output_root / f"fit-{step_diagnostics}", device, step_diagnostics=step_diagnostics)
            history = trainer.fit(
                _loader(dataset, rank=rank, world_size=world_size, shuffle=True),
                validation_loader=_loader(validation, rank=rank, world_size=world_size, shuffle=False),
                epochs=EPOCHS,
            )
            assert len(history) == EPOCHS
            assert ("train_clean_accuracy" in history[-1]) is step_diagnostics
            final_state[step_diagnostics] = _digest(trainer.model.module.state_dict())
        # Every rank (not only rank 0) ends with the same model and buffers.
        assert final_state[True] == final_state[False], f"rank {rank} final model state differs"

        if rank == 0:
            for name in ("last.pt", "best.pt"):
                payloads = {
                    flag: torch.load(output_root / f"fit-{flag}" / name, map_location="cpu", weights_only=False)
                    for flag in (True, False)
                }
                for key in ("model", "optimizer", "rng", "sampler_state", "global_step", "selection_metadata"):
                    assert _digest(payloads[True][key]) == _digest(payloads[False][key]), f"{name}:{key}"
                assert len(payloads[True]["rng"]) == world_size

        # Per-rank BatchNorm buffer invariant at train_epoch return.
        for step_diagnostics in (True, False):
            torch.manual_seed(2000)
            trainer = _trainer(output_root / f"epoch-{step_diagnostics}", device, step_diagnostics=step_diagnostics)
            trainer.train_epoch(_loader(dataset, rank=rank, world_size=world_size, shuffle=True))
            gathered = _gathered_running_mean(trainer.model)
            if step_diagnostics:
                # The post-step no-grad forward already broadcast rank 0's buffers.
                assert torch.equal(gathered[0], gathered[1])
            else:
                # Weaker invariant: the last train-mode forward updated each
                # rank's running statistics from its own shard and nothing
                # has broadcast them yet ...
                assert not torch.equal(gathered[0], gathered[1])
                rank_zero_before = gathered[0].clone()
                # ... until the next DDP forward, which broadcasts rank 0's
                # unchanged values before anything reads them.
                with torch.no_grad():
                    trainer.model.eval()
                    trainer.model(torch.rand(2, 3, 4, 4, device=device))
                    trainer.model.train()
                gathered = _gathered_running_mean(trainer.model)
                assert torch.equal(gathered[0], gathered[1])
                assert torch.equal(gathered[0], rank_zero_before)
        dist.barrier()
        if rank == 0:
            print("STEP_DIAGNOSTICS_DDP_OK", flush=True)
    finally:
        teardown()


if __name__ == "__main__":
    main()
