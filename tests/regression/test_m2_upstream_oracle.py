"""Optional clean-room differential against the pinned clone, never an import dependency."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from ard.objectives.kl import target_to_student_kl
from ard.signals import shannon_entropy
from scripts.run_saad_upstream import verified_saad_clone

pytestmark = [pytest.mark.t2, pytest.mark.upstream]


@pytest.mark.skipif(
    os.environ.get("ARD_RUN_SAAD_ORACLE") != "1", reason="set ARD_RUN_SAAD_ORACLE=1 for local-clone differential"
)
def test_pinned_saad_subprocess_oracle_matches_shannon_kl() -> None:
    root = Path(__file__).resolve().parents[2]
    try:
        clone = verified_saad_clone(root)
    except FileNotFoundError:
        pytest.skip("pinned local SAAD clone is unavailable")
    student = [[0.2, -0.1, 0.7], [1.0, 0.3, -0.5]]
    teacher = [[0.9, 0.0, -0.3], [-0.2, 0.4, 0.8]]
    code = (
        "import json, torch; from utils import samplewise_kl_div, samplewise_renyi_entropy; "
        f"s=torch.tensor({student}); t=torch.tensor({teacher}); "
        "print(json.dumps({'kl': samplewise_kl_div(s,t).tolist(), 'entropy': samplewise_renyi_entropy(t,1).tolist()}))"
    )
    oracle = subprocess.run([sys.executable, "-c", code], cwd=clone, text=True, capture_output=True)
    if oracle.returncode:
        pytest.skip("pinned SAAD oracle dependencies are unavailable: " + oracle.stderr.splitlines()[-1])
    observed = json.loads(oracle.stdout)
    local_student, local_teacher = torch.tensor(student), torch.tensor(teacher)
    assert torch.allclose(
        torch.tensor(observed["kl"]),
        target_to_student_kl(
            student_logits=local_student,
            target_logits=local_teacher,
            temperature=1.0,
            temperature_squared=True,
        ),
        atol=1e-7,
        rtol=0,
    )
    assert torch.allclose(torch.tensor(observed["entropy"]), shannon_entropy(local_teacher), atol=1e-7, rtol=0)
