"""CUDA RNG restoration across a changed visible-device set.

A checkpoint owns the CUDA devices that were visible when it was written.  A
world_size-1 production run pinned with ``CUDA_VISIBLE_DEVICES`` saves one
device state and may be resumed on a host exposing both GPUs; the reverse
direction cannot restore the saved stream and must fail closed rather than
resume with a partially restored RNG.
"""

from __future__ import annotations

import random

import pytest
import torch

from ard.engine.checkpoint import restore_rng_state

pytestmark = pytest.mark.t1


def _cpu_state(cuda_states: list[torch.Tensor]) -> dict[str, object]:
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda_states,
        "numpy": None,
    }


def _stub_cuda(monkeypatch: pytest.MonkeyPatch, *, visible: int) -> list[tuple[torch.Tensor, int]]:
    """Present exactly ``visible`` CUDA devices and record per-device restores."""
    restored: list[tuple[torch.Tensor, int]] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: visible)
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state, device=None: restored.append((state, device)))
    monkeypatch.setattr(
        torch.cuda, "set_rng_state_all", lambda states: [restored.append((state, i)) for i, state in enumerate(states)]
    )
    return restored


def test_single_device_checkpoint_restores_only_saved_devices_on_a_two_device_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved = [torch.zeros(16, dtype=torch.uint8)]
    restored = _stub_cuda(monkeypatch, visible=2)
    restore_rng_state(_cpu_state(saved))
    assert [device for _, device in restored] == [0]
    assert torch.equal(restored[0][0], saved[0])


def test_more_saved_cuda_states_than_visible_devices_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    saved = [torch.zeros(16, dtype=torch.uint8), torch.ones(16, dtype=torch.uint8)]
    restored = _stub_cuda(monkeypatch, visible=1)
    with pytest.raises(RuntimeError, match=r"2 CUDA RNG states but only 1 CUDA device"):
        restore_rng_state(_cpu_state(saved))
    assert restored == []


def test_matching_device_count_restores_every_saved_device(monkeypatch: pytest.MonkeyPatch) -> None:
    saved = [torch.zeros(16, dtype=torch.uint8), torch.ones(16, dtype=torch.uint8)]
    restored = _stub_cuda(monkeypatch, visible=2)
    restore_rng_state(_cpu_state(saved))
    assert [device for _, device in restored] == [0, 1]
    assert torch.equal(restored[1][0], saved[1])


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="requires at least two CUDA devices"
)
def test_real_single_device_state_restores_device_zero_and_leaves_peers_untouched() -> None:
    original = torch.cuda.get_rng_state_all()
    try:
        torch.cuda.manual_seed_all(1234)
        saved = [torch.cuda.get_rng_state(0)]
        torch.cuda.manual_seed_all(4321)
        peer_before = torch.cuda.get_rng_state(1)
        restore_rng_state(_cpu_state(saved))
        assert torch.equal(torch.cuda.get_rng_state(0), saved[0])
        assert torch.equal(torch.cuda.get_rng_state(1), peer_before)
    finally:
        torch.cuda.set_rng_state_all(original)
