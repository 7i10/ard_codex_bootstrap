from __future__ import annotations

import torch

from ard.analysis.ert_stage_a_endpoint import _probability_margin


def test_probability_margin_uses_true_class_vs_strongest_wrong_class() -> None:
    logits = torch.tensor([[2.0, 1.0, -1.0], [-1.0, 3.0, 0.0]])
    labels = torch.tensor([0, 2])
    margin = _probability_margin(logits, labels)
    assert margin.shape == (2,)
    assert margin[0] > 0
    assert margin[1] < 0
