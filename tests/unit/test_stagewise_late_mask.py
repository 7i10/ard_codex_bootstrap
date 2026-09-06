"""A masked late augmentation policy must leave everyone else untouched.

Plan 0096 compares who should receive the richer augmentation after the
learning-rate decay.  That comparison is only paired if an image outside the
mask gets exactly the view it would have got under a run with no mask at all --
bit for bit, not merely in distribution.  Both branches read the same
source-keyed stream, so this holds by construction; the test exists because the
construction is easy to break and the breakage would be invisible in any metric.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from ard.data.datasets import EpochStagewiseAugmentationTransform


def _transform(mask: frozenset[int] | None) -> EpochStagewiseAugmentationTransform:
    return EpochStagewiseAugmentationTransform(
        augmentation_seed=1234, switch_epoch=100, late_policy="idbh_weak", late_mask=mask
    )


def _image(seed: int) -> Image.Image:
    """The transforms take PIL images, as the CIFAR-10 loader hands them over."""
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8))


@pytest.mark.parametrize("epoch", [99, 100, 150])
def test_unmasked_samples_get_the_view_they_would_have_got_with_no_mask(epoch: int) -> None:
    treated = frozenset({3, 7, 11})
    masked, unmasked = _transform(treated), _transform(None)
    for transform in (masked, unmasked):
        transform.set_epoch(epoch)
    for source_id in (0, 1, 2, 4, 5):  # none of these is in the mask
        assert source_id not in treated
        with_mask = masked(_image(source_id), source_id=source_id)
        without = unmasked(_image(source_id), source_id=source_id)
        if epoch < 100:
            assert torch.equal(with_mask, without), "before the switch nothing may differ at all"
        else:
            # After the switch the unmasked run applies the late policy to
            # everyone, so these must differ; the point of the mask is that the
            # masked run keeps the prefix policy for them.
            prefix_only = _transform(frozenset({999}))
            prefix_only.set_epoch(epoch)
            assert torch.equal(with_mask, prefix_only(_image(source_id), source_id=source_id))


def test_masked_samples_get_the_late_policy() -> None:
    treated = frozenset({3, 7, 11})
    masked, unmasked = _transform(treated), _transform(None)
    for transform in (masked, unmasked):
        transform.set_epoch(120)
    for source_id in sorted(treated):
        assert torch.equal(masked(_image(source_id), source_id=source_id), unmasked(_image(source_id), source_id=source_id))


def test_before_the_switch_the_mask_changes_nothing() -> None:
    masked, unmasked = _transform(frozenset({3})), _transform(None)
    for transform in (masked, unmasked):
        transform.set_epoch(50)
    for source_id in (3, 4):
        assert torch.equal(masked(_image(source_id), source_id=source_id), unmasked(_image(source_id), source_id=source_id))


def test_an_empty_mask_is_refused() -> None:
    """An empty mask would silently mean "treat nobody", which is a different arm."""
    with pytest.raises(ValueError, match="non-empty"):
        _transform(frozenset())
