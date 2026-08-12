from __future__ import annotations

import torch

from ard.analysis.ert_stage_a_endpoint import _probability_margin, split_identity


def test_probability_margin_uses_true_class_vs_strongest_wrong_class() -> None:
    logits = torch.tensor([[2.0, 1.0, -1.0], [-1.0, 3.0, 0.0]])
    labels = torch.tensor([0, 2])
    margin = _probability_margin(logits, labels)
    assert margin.shape == (2,)
    assert margin[0] > 0
    assert margin[1] < 0


class _Raw:
    targets = (0, 1, 0, 1)


class _Indexed:
    dataset = _Raw()


class _View:
    dataset = _Indexed()
    indices = [1, 2]


def test_split_identity_is_stable_and_records_source_ids() -> None:
    result = split_identity(_View(), split="validation")
    assert result["name"] == "validation"
    assert result["count"] == 2
    assert result["class_counts"] == {"0": 1, "1": 1}
    assert len(result["sample_id_label_sha256"]) == 64
