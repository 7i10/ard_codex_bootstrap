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
