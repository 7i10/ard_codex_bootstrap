"""The [0, 1] pixel guard survives torch.compile's production (inductor) path.

``PixelNormalization`` enforces the range with ``torch._assert_async``. These
tests prove the assertion is neither dead-code-eliminated nor dropped by
inductor in the three graph shapes training produces, and that on CUDA it
kills the process with a device-side assert in eager and compiled mode.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch import nn

from ard.config.schema import NormalizationConfig
from ard.models.registry import FixtureCNN, PixelModel

pytestmark = pytest.mark.t3

MESSAGE = r"model adapter expects pixels in \[0, 1\]"


def _model() -> nn.Module:
    torch.manual_seed(0)
    normalization = NormalizationConfig(
        profile="custom", mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25), provenance="unit-test"
    )
    return PixelModel(FixtureCNN(3), normalization)


def _batches() -> tuple[torch.Tensor, torch.Tensor]:
    good = torch.rand(2, 3, 8, 8)
    bad = good.clone()
    bad[1, 0, 3, 3] = 1.5
    return good, bad


def _eval_no_grad(model: nn.Module, pixels: torch.Tensor) -> None:
    model.eval()
    with torch.no_grad():
        model(pixels)


def _train_backward(model: nn.Module, pixels: torch.Tensor) -> None:
    model.train()
    model.zero_grad(set_to_none=True)
    model(pixels).sum().backward()


def _input_gradient(model: nn.Module, pixels: torch.Tensor) -> None:
    # The attack shape: gradient w.r.t. the input only.
    model.eval()
    delta = pixels.clone().requires_grad_(True)
    (gradient,) = torch.autograd.grad(model(delta).sum(), delta)
    assert gradient.shape == pixels.shape


VARIANTS: dict[str, Callable[[nn.Module, torch.Tensor], None]] = {
    "eval_no_grad": _eval_no_grad,
    "train_backward": _train_backward,
    "input_gradient": _input_gradient,
}


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_inductor_compiled_pixel_guard_fires_in_every_training_graph_shape(variant: str) -> None:
    torch._dynamo.reset()
    run = VARIANTS[variant]
    compiled = torch.compile(_model(), backend="inductor")
    good, bad = _batches()
    run(compiled, good)
    with pytest.raises(RuntimeError, match=MESSAGE):
        run(compiled, bad)
    # Still armed on the next forward after a failure (cached graph reused).
    run(compiled, good)
    with pytest.raises(RuntimeError, match=MESSAGE):
        run(compiled, bad)


_CUDA_SCRIPT = """
import sys
import torch
from ard.config.schema import NormalizationConfig
from ard.models.registry import FixtureCNN, PixelModel

model = PixelModel(
    FixtureCNN(3),
    NormalizationConfig(profile="custom", mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25), provenance="unit-test"),
).cuda()
if sys.argv[1] == "inductor":
    model = torch.compile(model, backend="inductor")
good = torch.rand(2, 3, 8, 8, device="cuda")
model(good)
torch.cuda.synchronize()
print("in-range ok", flush=True)
bad = good.clone()
bad[1, 0, 3, 3] = 1.5
model(bad)
torch.cuda.synchronize()
print("guard did not fire", flush=True)
"""


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("mode", ["eager", "inductor"])
def test_cuda_pixel_guard_is_a_fatal_device_side_assert(mode: str) -> None:
    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    result = subprocess.run(
        [sys.executable, "-c", _CUDA_SCRIPT, mode],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        timeout=300,
    )
    assert "in-range ok" in result.stdout, result.stderr
    assert "guard did not fire" not in result.stdout
    assert result.returncode != 0
    assert "device-side assert" in result.stderr
    # Observed with torch 2.11 on an RTX 4090 (2026-09-26): the message text
    # survives in both modes. Eager prints it from
    # ``TensorCompare.cu: _assert_async_cuda_kernel ... Assertion `model
    # adapter expects pixels in [0, 1]` failed``; inductor prints it from the
    # generated Triton kernel (``/tmp/torchinductor_*/...py: unknown: ...
    # Assertion `model adapter expects pixels in [0, 1]` failed``). Both then
    # raise ``torch.AcceleratorError`` (a RuntimeError): "CUDA error:
    # device-side assert triggered".
    assert "model adapter expects pixels in [0, 1]" in result.stderr
