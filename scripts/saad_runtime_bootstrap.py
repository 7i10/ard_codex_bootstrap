#!/usr/bin/env python3
"""Python-3.11 import-order bridge for the pinned SAAD oracle.

RobustBench must see the official AutoAttack package.  SAAD itself carries a
local ``autoattack`` package, which is intentionally restored immediately
before executing upstream ``saad.py``.  This script records both origins; it
does not patch either checkout.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import runpy
import sys


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_identity(module):
    path = os.path.realpath(getattr(module, "__file__", ""))
    if not path or not os.path.isfile(path):
        raise RuntimeError(f"module has no source file: {module!r}")
    return {"path": path, "sha256": sha256(path)}


def under(path, root):
    try:
        return os.path.commonpath([os.path.realpath(path), os.path.realpath(root)]) == os.path.realpath(root)
    except ValueError:
        return False


def write_exclusive(path, value):
    with open(path, "x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")


def read_runtime_lock(path):
    """Read only ``name==version`` and pinned direct-reference lock lines."""
    packages = {}
    direct = {}
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "==" in line:
                name, version = line.split("==", 1)
                if name == "python":
                    continue
                packages[name] = version
            elif " @ " in line:
                name, source = line.split(" @ ", 1)
                direct[name] = source
            else:
                raise RuntimeError(f"unsupported runtime lock line: {line}")
    return packages, direct


def verify_runtime_lock(path):
    packages, direct = read_runtime_lock(path)
    observed = {}
    for name, expected in sorted(packages.items()):
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            raise RuntimeError(f"pinned runtime package is absent: {name}")
        if actual != expected:
            raise RuntimeError(f"runtime package drift for {name}: expected {expected}, got {actual}")
        observed[name] = actual
    observed_direct = {}
    for name, expected_source in sorted(direct.items()):
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            raise RuntimeError(f"pinned direct-reference package is absent: {name}")
        raw_direct_url = distribution.read_text("direct_url.json")
        if not raw_direct_url:
            raise RuntimeError(f"direct-reference provenance is absent for {name}")
        try:
            direct_url = json.loads(raw_direct_url)
            commit = direct_url["vcs_info"]["commit_id"]
        except (KeyError, TypeError, ValueError):
            raise RuntimeError(f"direct-reference provenance is malformed for {name}")
        expected_commit = expected_source.rsplit("@", 1)[-1]
        if commit != expected_commit:
            raise RuntimeError(f"direct-reference drift for {name}: expected {expected_commit}, got {commit}")
        observed_direct[name] = {"source": expected_source, "commit": commit}
    return {"packages": observed, "direct_references": observed_direct}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saad-stage", required=True)
    parser.add_argument("--saad-root", required=True)
    parser.add_argument("--robustbench-root", required=True)
    parser.add_argument("--runtime-lock", required=True)
    parser.add_argument("--expected-python", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("upstream_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not args.upstream_args or args.upstream_args[0] != "--":
        parser.error("upstream arguments must follow --")
    stage = os.path.realpath(args.saad_stage)
    saad_root = os.path.realpath(args.saad_root)
    robustbench_root = os.path.realpath(args.robustbench_root)
    if platform.python_version() != args.expected_python:
        raise RuntimeError(f"runtime Python drift: expected {args.expected_python}, got {platform.python_version()}")
    runtime_lock = verify_runtime_lock(args.runtime_lock)
    if not os.path.isfile(os.path.join(stage, "saad.py")):
        raise RuntimeError("SAAD stage has no saad.py")
    # Keep SAAD out of the initial import path.  RobustBench's optional
    # AutoAttack state import therefore proves the runtime package origin.
    sys.path = [robustbench_root] + [entry for entry in sys.path if not under(entry or os.getcwd(), stage)]
    import autoattack.state as official_state
    import robustbench  # noqa: F401

    official = module_identity(official_state)
    if under(official["path"], stage):
        raise RuntimeError("official autoattack.state was shadowed by SAAD")
    robustbench_identity = module_identity(sys.modules["robustbench"])
    if not under(robustbench_identity["path"], robustbench_root):
        raise RuntimeError("RobustBench was not imported from the pinned checkout")
    # RobustBench remains loaded.  Its AutoAttack modules are deliberately
    # removed so SAAD's documented local final evaluator wins for saad.py.
    for name in list(sys.modules):
        if name == "autoattack" or name.startswith("autoattack."):
            del sys.modules[name]
    sys.path.insert(0, stage)
    import autoattack as saad_autoattack

    saad_identity = module_identity(saad_autoattack)
    if not under(saad_identity["path"], saad_root) or not hasattr(saad_autoattack, "AutoAttack"):
        raise RuntimeError("SAAD local autoattack.AutoAttack was not selected")
    write_exclusive(
        args.provenance,
        {
            "python": {"full": sys.version, "version": platform.python_version()},
            "runtime_lock": runtime_lock,
            "robustbench": robustbench_identity,
            "official_autoattack_state": official,
            "saad_autoattack": saad_identity,
            "saad_stage": stage,
            "saad_root": saad_root,
            "robustbench_root": robustbench_root,
        },
    )
    os.chdir(stage)
    sys.argv = [os.path.join(stage, "saad.py")] + args.upstream_args[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
