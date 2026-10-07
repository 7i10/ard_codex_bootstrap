"""Two-rank gloo run of ``training.selection_subset_size`` (see torchrun_selection_subset.py)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.t3, pytest.mark.regression]


def test_two_rank_selection_subset_trains_identically_and_counts_the_held_out_split_once(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(root / "tests" / "regression" / "torchrun_selection_subset.py"),
            str(tmp_path),
        ],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
    assert "SELECTION_SUBSET_DDP_OK" in completed.stdout
