"""A masked late augmentation policy must leave everyone else untouched.

Plan 0096 compares who should receive the richer augmentation after the
learning-rate decay.  The property that makes the comparison worth running is
that an image outside the mask gets exactly the CropShift view, so any two arms
agree bit for bit on every image neither of them treats.

The claim is *not* that it matches a run with no mask: there every image is
switched, so an untreated image would get the late policy.  An earlier version of
this file said that, and the docstring it was copied from said it too.

The tests below compare against the prefix transform itself rather than against
another masked transform, because the second comparison holds for any
implementation whose prefix branch is a function of (seed, epoch, source id) --
including broken ones -- and it also exercises an interleaved sequence, so an
implementation that shared state between samples would fail.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from ard.data.datasets import EpochCropshiftTransform, EpochStagewiseAugmentationTransform


def _transform(mask: frozenset[int] | None) -> EpochStagewiseAugmentationTransform:
    return EpochStagewiseAugmentationTransform(
        augmentation_seed=1234, switch_epoch=100, late_policy="idbh_weak", late_mask=mask
    )


def _image(seed: int) -> Image.Image:
    """The transforms take PIL images, as the CIFAR-10 loader hands them over."""
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8))


def _prefix_transform() -> EpochCropshiftTransform:
    return EpochCropshiftTransform(augmentation_seed=1234, high=11)


@pytest.mark.parametrize("epoch", [99, 100, 150])
def test_untreated_images_get_exactly_the_prefix_view(epoch: int) -> None:
    treated = frozenset({3, 7, 11})
    masked, prefix = _transform(treated), _prefix_transform()
    masked.set_epoch(epoch)
    prefix.set_epoch(epoch)
    for source_id in (0, 1, 2, 4, 5):
        assert source_id not in treated
        assert torch.equal(
            masked(_image(source_id), source_id=source_id), prefix(_image(source_id), source_id=source_id)
        )


def test_two_arms_agree_on_every_image_neither_treats() -> None:
    """The interleaving is the point: shared state between samples would break it."""
    left, right = _transform(frozenset({3, 7, 11})), _transform(frozenset({1, 5, 9}))
    for transform in (left, right):
        transform.set_epoch(120)
    sequence = [0, 3, 1, 7, 2, 11, 4, 5, 9, 6]
    untreated = [sid for sid in sequence if sid not in {3, 7, 11} | {1, 5, 9}]
    left_views = {sid: left(_image(sid), source_id=sid) for sid in sequence}
    right_views = {sid: right(_image(sid), source_id=sid) for sid in sequence}
    assert untreated, "the sequence must contain images neither arm treats"
    for sid in untreated:
        assert torch.equal(left_views[sid], right_views[sid]), f"arms disagree on untreated image {sid}"


def test_masked_samples_get_the_late_policy() -> None:
    treated = frozenset({3, 7, 11})
    masked, unmasked = _transform(treated), _transform(None)
    for transform in (masked, unmasked):
        transform.set_epoch(120)
    for source_id in sorted(treated):
        assert torch.equal(
            masked(_image(source_id), source_id=source_id), unmasked(_image(source_id), source_id=source_id)
        )


def test_before_the_switch_the_mask_changes_nothing() -> None:
    masked, unmasked = _transform(frozenset({3})), _transform(None)
    for transform in (masked, unmasked):
        transform.set_epoch(50)
    for source_id in (3, 4):
        assert torch.equal(
            masked(_image(source_id), source_id=source_id), unmasked(_image(source_id), source_id=source_id)
        )


def test_an_empty_mask_is_refused() -> None:
    """An empty mask would silently mean "treat nobody", which is a different arm."""
    with pytest.raises(ValueError, match="non-empty"):
        _transform(frozenset())
