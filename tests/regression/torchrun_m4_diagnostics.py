"""Two-rank padded diagnostics dedup oracle."""

from __future__ import annotations

import torch.distributed as dist

from ard.engine.distributed import initialize_from_env, teardown
from ard.tracking.diagnostics import TrainingDiagnostics

_, initialized = initialize_from_env("cpu")
assert initialized and dist.get_world_size() == 2
rank = dist.get_rank()
try:
    diagnostics = TrainingDiagnostics(panel_ids=(0, 1, 2))
    diagnostics.record(sample_id=rank, valid=True, epoch=0, scalar=rank)
    diagnostics.record(sample_id=2, valid=rank == 0, epoch=0, scalar=100 + rank)
    diagnostics.flush()
    assert set(diagnostics.all_rows) == {0, 1, 2}
    assert diagnostics.all_rows[2]["scalar"] == 100
    assert all("rank" not in row and "order" not in row for row in diagnostics.all_rows.values())
    replicas: list[dict | None] = [None, None]
    dist.all_gather_object(replicas, diagnostics.all_rows)
    assert replicas[0] == replicas[1] == diagnostics.all_rows
finally:
    teardown()
