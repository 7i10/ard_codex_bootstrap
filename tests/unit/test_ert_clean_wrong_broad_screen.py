from __future__ import annotations

import json
from pathlib import Path

import pytest

from ard.analysis.ert_clean_wrong_broad_screen import (
    ARMS,
    CleanWrongScreenError,
    arm_by_name,
    fixed_clean_wrong_mask,
    validate_arm,
)


def test_screen_has_exact_frozen_sixteen_arms() -> None:
    assert tuple(arm.name for arm in ARMS) == tuple(f"C{i}" for i in range(16))
    assert arm_by_name("C3").advkd_multiplier == 0.5
    assert arm_by_name("C13").adaptive_pressure
    assert arm_by_name("C15").iad_inspired


def test_screen_rejects_step_larger_than_selected_epsilon() -> None:
    with pytest.raises(CleanWrongScreenError, match="step exceeds"):
        validate_arm(arm_by_name("C0").__class__("bad", "2/255", "4/255"))


def test_fixed_clean_wrong_mask_is_epoch79_and_hash_bound(tmp_path: Path) -> None:
    path = tmp_path / "mask.json"
    path.write_text(
        json.dumps(
            {
                "contract": "ert_state_overlay_v1",
                "anchor_epoch": 79,
                "masks": {
                    "student_clean_wrong": {
                        "selected_ids": [1, 3],
                        "selected_class_counts": {"0": 1, "1": 1},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    record = fixed_clean_wrong_mask(path, run="L2")
    assert record["selected_ids"] == [1, 3]
    assert record["selected_count"] == 2
