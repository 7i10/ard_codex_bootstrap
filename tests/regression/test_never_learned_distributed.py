"""Two-rank gloo run of the never-learned guard (see torchrun_never_learned.py)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.t3, pytest.mark.regression]


def test_two_rank_never_learned_statistics_match_and_every_rank_raises(tmp_path: Path) -> None:
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
            str(root / "tests" / "regression" / "torchrun_never_learned.py"),
            str(tmp_path),
        ],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr[-6000:]
    assert "NEVER_LEARNED_DDP_OK" in completed.stdout
