"""Plan 0099 item 4: TrainingConfig.amp / GradScaler wiring.

``amp: false`` (the default, and what every existing config that predates
this plan implicitly has, since none of them mention this field) must
reproduce today's exact trainer construction: ``scaler=None``, unconditionally
and regardless of device. Same "default reproduces the exact prior behavior"
discipline plan 0098 used for ``AdrConfig.lambda_source`` -- see
tests/unit/test_adr_config_lambda_source.py. Real mixed-precision GPU
numerics are out of scope for this unit suite: construction/wiring only,
mirroring this project's convention for its other CUDA-only paths.
"""

from __future__ import annotations

import torch

from ard.cli.train import _build_grad_scaler
from ard.config.schema import TrainingConfig


def test_default_training_config_has_amp_disabled() -> None:
    config = TrainingConfig(per_rank_batch_size=2, global_batch_size=2)
    assert config.amp is False


def test_a_config_that_never_mentions_amp_still_validates_and_defaults_false() -> None:
    config = TrainingConfig(epochs=1, per_rank_batch_size=2, global_batch_size=2, device="cpu")
    assert config.amp is False


def test_amp_false_reproduces_todays_exact_scaler_none_regardless_of_device() -> None:
    assert _build_grad_scaler(amp=False, device=torch.device("cpu")) is None
    assert _build_grad_scaler(amp=False, device=torch.device("cuda")) is None


def test_amp_true_constructs_a_real_but_disabled_gradscaler_on_cpu() -> None:
    """Construction/wiring only: a disabled GradScaler on CPU is a safe no-op, not an error."""
    scaler = _build_grad_scaler(amp=True, device=torch.device("cpu"))
    assert scaler is not None
    assert isinstance(scaler, torch.amp.GradScaler)
    assert scaler.is_enabled() is False
    loss = torch.tensor(2.0, requires_grad=True)
    assert torch.equal(scaler.scale(loss), loss)


def test_amp_true_on_a_cuda_device_constructs_an_enabled_gradscaler() -> None:
    scaler = _build_grad_scaler(amp=True, device=torch.device("cuda"))
    assert scaler is not None
    assert isinstance(scaler, torch.amp.GradScaler)
    assert scaler.is_enabled() is True
