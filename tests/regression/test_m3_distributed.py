from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.t2, pytest.mark.regression]


def test_two_rank_pending_sample_state_merge_is_deterministic_and_duplicate_safe() -> None:
    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(root / "tests" / "regression" / "torchrun_m3_sample_state.py"),
        ],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("case", ("success", "failure"))
def test_two_rank_terminal_resume_preflight_is_rank_zero_owned_and_common(tmp_path: Path, case: str) -> None:
    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(root / "tests" / "regression" / "torchrun_m3_terminal_resume.py"),
            case,
            "--output",
            str(tmp_path / case),
        ],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
