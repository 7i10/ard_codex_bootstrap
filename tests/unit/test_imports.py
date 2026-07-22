"""Bootstrap checks for package namespaces and CLI argument handling."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

NAMESPACE_MODULES = (
    "ard.cli",
    "ard.config",
    "ard.data",
    "ard.models",
    "ard.attacks",
    "ard.objectives",
    "ard.signals",
    "ard.policies",
    "ard.state",
    "ard.engine",
    "ard.evaluation",
    "ard.tracking",
    "ard.analysis",
)


def test_namespace_imports() -> None:
    for module_name in NAMESPACE_MODULES:
        __import__(module_name)


def test_cli_help() -> None:
    env = {"PYTHONPATH": str(Path(__file__).parents[2] / "src")}
    for module_name in ("ard.cli.train", "ard.cli.evaluate"):
        result = subprocess.run(
            [sys.executable, "-m", module_name, "--help"],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()
        assert "--config" in result.stdout
