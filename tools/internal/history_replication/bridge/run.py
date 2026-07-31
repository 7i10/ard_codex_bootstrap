#!/usr/bin/env python3
"""CLI wrapper for the bounded cross-version observation bridge."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def _runtime_root(arguments: list[str]) -> tuple[Path, list[str]]:
    """Extract a runtime worktree before importing its ``ard`` package."""
    try:
        index = arguments.index("--runtime-root")
    except ValueError:
        return Path(__file__).resolve().parents[4], arguments
    if index + 1 == len(arguments):
        raise SystemExit("--runtime-root requires a worktree path")
    root = Path(arguments[index + 1]).resolve()
    remaining = arguments[:index] + arguments[index + 2 :]
    if not (root / "src" / "ard" / "engine" / "trainer.py").is_file():
        raise SystemExit(f"--runtime-root is not an ARD runtime worktree: {root}")
    return root, remaining


def _load_main(runtime_root: Path):
    """Load bridge code here, but resolve ``ard`` imports from target runtime."""
    if any(name == "ard" or name.startswith("ard.") for name in sys.modules):
        raise SystemExit("refusing bridge launch because an ard module was already imported")
    implementation = Path(__file__).with_name("cross_version_bridge.py").resolve()
    if not implementation.is_file():
        raise SystemExit(f"bridge implementation is missing: {implementation}")
    os.environ["ARD_BRIDGE_RUNTIME_ROOT"] = str(runtime_root)
    os.environ["ARD_BRIDGE_WRAPPER_PATH"] = str(Path(__file__).resolve())
    sys.path.insert(0, str(runtime_root / "src"))
    specification = importlib.util.spec_from_file_location("_ard_observation_bridge", implementation)
    if specification is None or specification.loader is None:
        raise SystemExit("cannot load bridge implementation")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.main


if __name__ == "__main__":
    root, remaining = _runtime_root(sys.argv[1:])
    raise SystemExit(_load_main(root)(remaining))
