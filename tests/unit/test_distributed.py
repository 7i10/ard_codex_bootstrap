from __future__ import annotations

import pytest
import torch

from ard.engine import distributed

pytestmark = pytest.mark.t1


def test_torchrun_cpu_environment_initializes_gloo_with_local_rank(monkeypatch) -> None:
    observed: dict[str, str] = {}
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(distributed.dist, "init_process_group", lambda **kwargs: observed.update(kwargs))

    device, initialized = distributed.initialize_from_env("cpu")

    assert initialized and device == torch.device("cpu")
    assert observed == {"backend": "gloo", "init_method": "env://"}


def test_reduce_max_uses_cross_rank_max_without_mutating_input(monkeypatch: pytest.MonkeyPatch) -> None:
    value = torch.tensor([2.0, 11.0])
    observed: list[object] = []

    def fake_all_reduce(reduced: torch.Tensor, *, op: object) -> None:
        observed.append(op)
        reduced.copy_(torch.tensor([3.0, 17.0]))

    monkeypatch.setattr(distributed, "distributed_ready", lambda: True)
    monkeypatch.setattr(distributed.dist, "all_reduce", fake_all_reduce)

    reduced = distributed.reduce_max(value)

    assert torch.equal(value, torch.tensor([2.0, 11.0]))
    assert torch.equal(reduced, torch.tensor([3.0, 17.0]))
    assert observed == [distributed.dist.ReduceOp.MAX]
