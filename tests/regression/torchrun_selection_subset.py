"""Two-rank gloo check of ``training.selection_subset_size`` (plan 0103 Phase 2).

Asserted:
* every rank's final model state, and rank 0's ``last.pt`` model / optimizer / RNG, are
  identical with and without the option (training unchanged under DDP);
* the final full-split and complement passes count every held-out image exactly once
  (sampler padding is masked out) and every rank returns the same reduced row;
* with best.pt held at epoch 0, every rank reloads it for the final pass and restores
  its live weights.
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
from ard.data import (
    EpochShuffleSampler,
    IndexedDataset,
    SourceIndexedSubset,
    SyntheticCIFAR,
    build_selection_subset_view,
    collate_indexed,
    stratified_train_validation_split,
)
from ard.engine.distributed import initialize_from_env, teardown, wrap_ddp
from ard.engine.trainer import Trainer
from ard.objectives import PGDATObjective
from ard.policies import selected_ids_sha256

EPOCHS = 3
SUBSET_SIZE = 3


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


def _loader(dataset: SourceIndexedSubset, *, rank: int, world_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=4,
        sampler=EpochShuffleSampler(len(dataset), seed=13, rank=rank, world_size=world_size, shuffle=shuffle),
        collate_fn=collate_indexed,
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
        subset = build_selection_subset_view(validation, size=SUBSET_SIZE, seed=13)
        metadata = {
            "size": len(subset),
            "ids_sha256": selected_ids_sha256(tuple(subset.indices)),
            "full_split_size": len(validation),
            "seed": 13,
            "sampling": "class_stratified",
            "source": "held_out_validation_split",
        }

        def loaders() -> tuple[DataLoader, DataLoader, DataLoader]:
            return (
                _loader(train, rank=rank, world_size=world_size, shuffle=True),
                _loader(validation, rank=rank, world_size=world_size, shuffle=False),
                _loader(subset, rank=rank, world_size=world_size, shuffle=False),
            )

        final_state: dict[bool, str] = {}
        rows: dict[bool, list[dict[str, Any]]] = {}
        for enabled in (False, True):
            torch.manual_seed(1000)
            model = wrap_ddp(_student().to(device), device)
            optimizer = SGD(model.parameters(), lr=0.05, momentum=0.9)
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                scheduler=None,
                scaler=None,
                attack=LinfPGD(AttackConfig(epsilon="4/255", step_size="2/255", steps=2, random_start=True)),
                selection_attack=LinfPGD(
                    AttackConfig(
                        epsilon="2/255",
                        step_size="1/255",
                        steps=2,
                        random_start=True,
                        student_mode="eval",
                        teacher_mode="eval",
                    )
                ),
                objective=PGDATObjective(),
                device=device,
                output_dir=output_root / f"fit-{enabled}",
                config_hash="d" * 64,
                seed=19,
                tracker_run_id="ddp-selection-subset",
                selection_subset=metadata if enabled else None,
                selection_subset_ids=list(subset.indices) if enabled else None,
            )
            train_loader, full_loader, subset_loader = loaders()
            if enabled:
                # Falling subset numbers hold best.pt at epoch 0, so the final pass reloads it.
                falling = iter([0.9, 0.8, 0.7])

                def fake(
                    loader: DataLoader, *, model: nn.Module | None = None, _subset: DataLoader = subset_loader
                ) -> dict[str, float]:
                    assert loader is _subset
                    # Iterate like a real pass: a loader without its own generator draws
                    # its base seed from the global RNG, as the plain run's passes do.
                    for _ in loader:
                        pass
                    value = next(falling)
                    return {"clean_accuracy": value, "pgd_accuracy": value}

                trainer.validate_epoch = fake  # type: ignore[method-assign]
                history = trainer.fit(
                    train_loader,
                    validation_loader=subset_loader,
                    full_validation_loader=full_loader,
                    epochs=EPOCHS,
                )
            else:
                history = trainer.fit(train_loader, validation_loader=full_loader, epochs=EPOCHS)
            rows[enabled] = history
            final_state[enabled] = _digest(trainer.model.module.state_dict())
        assert final_state[True] == final_state[False], f"rank {rank} final model state differs"

        final = rows[True][-1]
        assert final["val_full_num_examples"] == 15, final
        assert final["val_complement_num_examples"] == 15 - SUBSET_SIZE, final
        assert final["best_val_full_epoch"] == 0
        gathered: list[Any] = [None] * world_size
        dist.all_gather_object(
            gathered, {key: value for key, value in final.items() if "full" in key or "complement" in key}
        )
        assert gathered[0] == gathered[1], gathered

        if rank == 0:
            plain = torch.load(output_root / "fit-False" / "last.pt", map_location="cpu", weights_only=False)
            subset_last = torch.load(output_root / "fit-True" / "last.pt", map_location="cpu", weights_only=False)
            for key in ("model", "optimizer", "rng", "sampler_state", "global_step"):
                assert _digest(plain[key]) == _digest(subset_last[key]), key
            record = subset_last["selection_metadata"]["full_split_final"]
            assert record["best_epoch"] == 0 and record["num_examples"] == 15
            assert record["best_pgd_accuracy"] == final["best_val_full_pgd_accuracy"]
        dist.barrier()
        if rank == 0:
            print("SELECTION_SUBSET_DDP_OK", flush=True)
    finally:
        teardown()


if __name__ == "__main__":
    main()
