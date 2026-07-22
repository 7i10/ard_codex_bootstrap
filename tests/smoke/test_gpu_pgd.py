from __future__ import annotations

import pytest
import torch
from torch import nn

from ard.attacks import AttackRequest, LinfPGD
from ard.config.schema import AttackConfig

pytestmark = [pytest.mark.t3, pytest.mark.gpu]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_single_gpu_pgd_smoke() -> None:
    device = torch.device("cuda")
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 3)).to(device)
    inputs = torch.rand(2, 3, 4, 4, device=device)
    labels = torch.tensor([0, 1], device=device)
    result = LinfPGD(AttackConfig(steps=1)).generate(AttackRequest(inputs=inputs, labels=labels, student=model))
    assert result.adversarial.is_cuda
