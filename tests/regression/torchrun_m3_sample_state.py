"""Two-rank oracle for deterministic sparse sample-state merging."""

from __future__ import annotations

import torch.distributed as dist

from ard.engine.distributed import initialize_from_env, teardown
from ard.state import SampleStateStore

device, initialized = initialize_from_env("cpu")
assert initialized and dist.get_world_size() == 2
rank = dist.get_rank()
try:
    # Same valid original ID on both ranks is deliberately adversarial input:
    # rank/order canonicalization must not update its EMA twice.  The padding
    # analogue is omitted before this queue is materialized by the trainer.
    local = [
        {
            "sample_id": 7,
            "margin": 0.25 if rank == 0 else -0.75,
            "robust_correct": rank == 0,
            "update": 3,
            "rank": rank,
            "order": 0,
        },
        {
            "sample_id": rank,
            "margin": -0.1 * (rank + 1),
            "robust_correct": False,
            "update": 3,
            "rank": rank,
            "order": 1,
        },
    ]
    gathered: list[list[dict[str, object]] | None] = [None, None]
    dist.all_gather_object(gathered, local)
    store = SampleStateStore(ema_decay=0.9)
    store.merge_pending(gathered)
    assert store.pending == []
    assert store.records[7].seen == 1
    assert store.records[7].margin_ema == 0.25
    assert store.records[7].robust_correct_count == 1
    expected = store.state_dict()
    replicas: list[dict[str, object] | None] = [None, None]
    dist.all_gather_object(replicas, expected)
    assert replicas[0] == replicas[1] == expected
finally:
    teardown()
