"""Inject a rank-zero checkpoint serialization failure in the real train CLI."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

from ard.cli.train import main

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
original_save = torch.save


def failing_save(*args, **kwargs):
    target = args[1] if len(args) > 1 else kwargs.get("f")
    # all_gather_object pickles tensor storage through torch.save(BytesIO).
    # Inject only at the actual atomic checkpoint file boundary.
    if local_rank == 0 and isinstance(target, (str, Path)):
        raise OSError("injected checkpoint write failure")
    return original_save(*args, **kwargs)


torch.save = failing_save
try:
    raise SystemExit(main())
except RuntimeError as exc:
    print(f"ARD_CHECKPOINT_FAILURE_RANK={local_rank}: {exc}", file=sys.stderr, flush=True)
    raise
