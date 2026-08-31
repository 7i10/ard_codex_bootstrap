from __future__ import annotations

import pytest
import torch

from ard.attacks import LinfPGD
from ard.attacks.base import AttackRequest
from ard.attacks.pgd import sample_keyed_random_start
from ard.config.schema import AttackConfig


def _mapped(values: torch.Tensor, source_ids: torch.Tensor) -> dict[int, torch.Tensor]:
    return {int(source_id): value for source_id, value in zip(source_ids.tolist(), values, strict=True)}


def test_sample_keyed_random_start_is_order_and_partition_invariant() -> None:
    clean = torch.full((8, 3, 4, 5), 0.5)
    ids = torch.tensor([10, 2, 31, 7, 19, 4, 25, 1], dtype=torch.long)
    kwargs = {"attack_seed": 17, "epoch": 4, "stream_tag": "train_pgd", "restart_index": 0}
    expected = _mapped(sample_keyed_random_start(clean, ids, **kwargs), ids)
    reversed_ids = ids.flip(0)
    reversed_values = sample_keyed_random_start(clean.flip(0), reversed_ids, **kwargs)
    assert all(torch.equal(expected[int(source_id)], value) for source_id, value in zip(reversed_ids, reversed_values))
    split = torch.cat(
        (
            sample_keyed_random_start(clean[:3], ids[:3], **kwargs),
            sample_keyed_random_start(clean[3:], ids[3:], **kwargs),
        )
    )
    assert all(torch.equal(expected[int(source_id)], value) for source_id, value in zip(ids, split))


def test_sample_keyed_random_start_is_rank_invariant_and_key_sensitive() -> None:
    clean = torch.full((6, 3, 2, 2), 0.5)
    ids = torch.arange(6, dtype=torch.long)
    kwargs = {"attack_seed": 5, "epoch": 3, "stream_tag": "train_pgd", "restart_index": 0}
    whole = _mapped(sample_keyed_random_start(clean, ids, **kwargs), ids)
    rank0_ids, rank1_ids = ids[::2], ids[1::2]
    rank0 = sample_keyed_random_start(clean[::2], rank0_ids, **kwargs)
    rank1 = sample_keyed_random_start(clean[1::2], rank1_ids, **kwargs)
    assert all(torch.equal(whole[int(source_id)], value) for source_id, value in zip(rank0_ids, rank0))
    assert all(torch.equal(whole[int(source_id)], value) for source_id, value in zip(rank1_ids, rank1))
    assert not torch.equal(whole[0], sample_keyed_random_start(clean[:1], ids[:1], **{**kwargs, "epoch": 4})[0])
    assert not torch.equal(whole[0], sample_keyed_random_start(clean[:1], ids[:1], **{**kwargs, "attack_seed": 6})[0])
    assert not torch.equal(whole[0], sample_keyed_random_start(clean[:1], torch.tensor([99]), **kwargs)[0])


@pytest.mark.parametrize("shape", [(2, 3, 32, 32), (1, 3, 224, 224)])
def test_sample_keyed_random_start_supports_arbitrary_resolution(shape: tuple[int, ...]) -> None:
    clean = torch.full(shape, 0.5)
    ids = torch.arange(shape[0], dtype=torch.long)
    result = sample_keyed_random_start(clean, ids, attack_seed=1, epoch=0)
    assert result.shape == shape
    assert result.dtype == clean.dtype
    assert torch.isfinite(result).all()
    assert float(result.min()) >= -1.0 and float(result.max()) <= 1.0


def test_sample_keyed_attack_requires_key_fields_and_preserves_selection_identity() -> None:
    config = AttackConfig(
        epsilon="8/255",
        step_size="2/255",
        steps=1,
        random_start=True,
        loss="kl",
        kl_target="teacher_clean",
        random_start_keying="sample_keyed_v1",
    )
    attack = LinfPGD(config)
    student = torch.nn.Flatten()
    request = AttackRequest(inputs=torch.zeros(1, 1, 1, 1), labels=torch.zeros(1, dtype=torch.long), student=student)
    with pytest.raises(ValueError, match="sample-keyed"):
        attack.generate(request)
    selection = config.model_copy(update={"loss": "ce", "kl_target": None, "random_start_keying": "batch"})
    assert "random_start_keying" not in selection.identity()
    assert config.identity()["random_start_keying"] == "sample_keyed_v1"
