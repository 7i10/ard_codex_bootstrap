"""Two-rank oracle for global-valid entropy weights and DDP gradients."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from ard.engine.distributed import initialize_from_env, reduce_min, reduce_sums, teardown
from ard.policies import EntropyOnlyPolicy, PolicyContext

device, initialized = initialize_from_env("cpu")
assert initialized and dist.get_world_size() == 2
rank = dist.get_rank()
try:
    model = DistributedDataParallel(nn.Linear(1, 1, bias=False).to(device))
    with torch.no_grad():
        model.module.weight.fill_(1.0)

    if rank == 0:
        entropy = torch.tensor([0.1], device=device)
        valid = torch.tensor([True], device=device)
        base = torch.tensor([2.0], device=device)
    else:
        entropy = torch.tensor([0.4, 0.01], device=device)
        valid = torch.tensor([True, False], device=device)
        base = torch.tensor([3.0, 99.0], device=device)

    weights = (
        EntropyOnlyPolicy()
        .weights(
            {"teacher_entropy": entropy},
            context=PolicyContext(valid_mask=valid, global_min=reduce_min),
            num_classes=3,
        )
        .kd_weight
    )
    expected = torch.tensor([0.0], device=device) if rank == 0 else torch.tensor([1.5, 0.0], device=device)
    torch.testing.assert_close(weights, expected, rtol=0, atol=1e-6)

    output = model(torch.ones(1, 1, device=device)).reshape(())
    global_count = reduce_sums(valid.sum(dtype=torch.float64)).clamp_min(1.0)
    loss = (base * weights * valid).sum() * output * (dist.get_world_size() / global_count)
    loss.backward()
    # Global oracle: mean([2*0, 3*1.5]) = 2.25.
    torch.testing.assert_close(model.module.weight.grad, torch.tensor([[2.25]], device=device), rtol=0, atol=1e-6)
finally:
    teardown()
