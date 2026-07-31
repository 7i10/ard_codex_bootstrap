from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _finalizer_module() -> ModuleType:
    path = Path("scripts/campaign/finalize_protected_run.py").resolve()
    spec = importlib.util.spec_from_file_location("ard_campaign_finalizer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
@pytest.mark.t1
def test_finalizer_exposes_active_python_console_scripts_on_path() -> None:
    finalizer = _finalizer_module()
    original = {"PATH": os.pathsep.join(("/usr/local/bin", "/usr/bin")), "KEEP": "value"}

    updated = finalizer._with_active_python_on_path(original)

    assert updated["PATH"].split(os.pathsep)[0] == str(Path(sys.executable).resolve().parent)
    assert updated["PATH"].split(os.pathsep)[1:] == ["/usr/local/bin", "/usr/bin"]
    assert updated["KEEP"] == "value"
    assert original["PATH"] == os.pathsep.join(("/usr/local/bin", "/usr/bin"))


@pytest.mark.unit
@pytest.mark.t1
def test_finalizer_evaluates_with_training_resolved_config() -> None:
    finalizer = _finalizer_module()
    train_output = Path("/runs/protected/train")
    evaluation_output = train_output / "evaluation-pgd"

    command = finalizer._evaluation_command(train_output, evaluation_output)

    assert command[command.index("--config") + 1] == str(train_output / "resolved_config.yaml")
    assert command[command.index("--checkpoint-dir") + 1] == str(train_output)
    assert command[command.index("--output") + 1] == str(evaluation_output)
