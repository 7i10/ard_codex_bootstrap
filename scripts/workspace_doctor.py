#!/usr/bin/env python3
"""Report compact operational health for the tracked ARD workspace contract."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ard.workspace import WorkspaceContract, load_workspace_contract  # noqa: E402


def command(argv: list[str], *, timeout: float = 10.0) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(argv, text=True, capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(argv, returncode=127, stdout="", stderr=str(exc))


def parse_csv(text: str) -> list[list[str]]:
    return [row.strip().split(", ") for row in text.splitlines() if row.strip()]


def git_summary(repo_root: Path) -> dict[str, Any]:
    head = command(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    status = command(["git", "-C", str(repo_root), "status", "--porcelain"])
    return {
        "head": head.stdout.strip() if head.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "status_error": (status.stderr.strip() or None) if status.returncode != 0 else None,
    }


def torch_summary(python: Path) -> dict[str, Any]:
    result = command(
        [
            str(python),
            "-c",
            "import json, torch; print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda, "
            "'cuda_available': torch.cuda.is_available()}))",
        ]
    )
    if result.returncode != 0:
        return {"status": "unavailable", "error": result.stderr.strip() or result.stdout.strip()}
    try:
        return {"status": "ok", **json.loads(result.stdout)}
    except json.JSONDecodeError:
        return {"status": "unavailable", "error": "invalid torch probe JSON"}


def gpu_summary() -> dict[str, Any]:
    devices = command(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"])
    compute = command(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,gpu_uuid,used_memory",
            "--format=csv,noheader",
        ]
    )
    if devices.returncode != 0:
        return {"status": "unavailable", "error": devices.stderr.strip() or devices.stdout.strip()}
    return {
        "status": "ok",
        "devices": [
            dict(zip(("index", "uuid", "name"), row, strict=True)) for row in parse_csv(devices.stdout) if len(row) == 3
        ],
        "compute_processes": [
            dict(zip(("pid", "process_name", "gpu_uuid", "used_memory"), row, strict=True))
            for row in parse_csv(compute.stdout)
            if len(row) == 4
        ],
    }


def active_processes() -> dict[str, list[dict[str, str]]]:
    result = command(["ps", "-eo", "pid=,args="])
    patterns = {
        "orchestrators": "orchestrate.py",
        "scientific": "ard.cli.train|ard.cli.evaluate|ert_stage_a_endpoint",
    }
    records: dict[str, list[dict[str, str]]] = {name: [] for name in patterns}
    if result.returncode != 0:
        return records
    for line in result.stdout.splitlines():
        pid, _, argv = line.strip().partition(" ")
        if not pid or "workspace_doctor.py" in argv:
            continue
        for name, pattern in patterns.items():
            if any(term in argv for term in pattern.split("|")):
                records[name].append({"pid": pid, "argv": argv})
    return records


def stale_worktrees(repo_root: Path) -> list[str]:
    result = command(["git", "-C", str(repo_root), "worktree", "list", "--porcelain"])
    return [line.removeprefix("worktree ") for line in result.stdout.splitlines() if line.startswith("worktree ")]


def legacy_roots(contract: WorkspaceContract) -> list[dict[str, Any]]:
    roots = contract.values.get("historical_roots", {})
    if not isinstance(roots, dict):
        return []
    return [
        {"name": name, "path": value, "exists": Path(value).exists()}
        for name, value in sorted(roots.items())
        if isinstance(value, str)
    ]


def active_old_path_hits(repo_root: Path) -> list[str]:
    # This is deliberately a *future runtime* scan, not a report that every
    # historical analysis script has been rewritten.  Frozen analysis helpers
    # and provenance documents legitimately preserve historical absolute
    # paths; current shared launch/runtime owners must not.
    targets = [
        repo_root / "src" / "ard" / "workspace.py",
        repo_root / "scripts" / "workspace_paths.py",
        repo_root / ".agents" / "skills" / "multi-gpu-experiment-orchestrator",
        repo_root / ".agents" / "skills" / "production-launch-gate",
        repo_root / ".agents" / "skills" / "run-on-ferret",
    ]
    existing = [str(path) for path in targets if path.exists()]
    if not existing:
        return []
    result = command(
        [
            "rg",
            "-l",
            "/home/islab/workspace-local/shunsuke.naito/|/home/shunsukenaito/workspace-local/ard-runs",
            *existing,
        ]
    )
    return sorted(line for line in result.stdout.splitlines() if line)[:50]


def doctor(contract: WorkspaceContract) -> dict[str, Any]:
    paths = {
        field: {"path": contract.values[field], "exists": contract.path(field).exists()}
        for field in ("repo_root", "dataset_root", "ard_dataset_root", "imagenet_root", "runtime_root")
    }
    runtime = {field: {"path": str(path), "exists": path.exists()} for field, path in contract.runtime_paths.items()}
    processes = active_processes()
    warnings: list[str] = []
    if not all(row["exists"] for row in runtime.values()):
        warnings.append("canonical_runtime_layout_missing")
    if any(row["exists"] for row in legacy_roots(contract)):
        warnings.append("historical_roots_present_read_only")
    if active_old_path_hits(contract.path("repo_root")):
        warnings.append("future_active_old_path_literals_present")
    if processes["orchestrators"] or processes["scientific"]:
        warnings.append("active_processes_present")
    return {
        "schema_version": 1,
        "status": "warn" if warnings else "ok",
        "workspace_registry": str(contract.registry_path),
        "workspace_registry_sha256": contract.registry_sha256,
        "paths": paths,
        "runtime_layout": runtime,
        "git": git_summary(contract.path("repo_root")),
        "python": {"path": contract.values["python"], "exists": contract.path("python").exists()},
        "torch": torch_summary(contract.path("python")),
        "gpu": gpu_summary(),
        "active_processes": processes,
        "worktrees": stale_worktrees(contract.path("repo_root")),
        "historical_roots": legacy_roots(contract),
        "active_old_path_hits": active_old_path_hits(contract.path("repo_root")),
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--json", action="store_true", help="emit the machine-readable doctor record")
    parser.add_argument("--verbose", action="store_true", help="show all arrays rather than a compact summary")
    parser.add_argument("--ensure-runtime", action="store_true", help="explicitly create the registered runtime layout")
    args = parser.parse_args()
    try:
        contract = load_workspace_contract(args.registry)
        if args.ensure_runtime:
            contract.ensure_runtime_layout()
        report = doctor(contract)
    except ValueError as exc:
        report = {"schema_version": 1, "status": "fail", "error": str(exc)}
    if args.json:
        print(json.dumps(report, indent=2 if args.verbose else None, sort_keys=True))
    else:
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "git": report.get("git", {}).get("head"),
                    "runtime_root": report.get("paths", {}).get("runtime_root", {}).get("path"),
                    "warnings": report.get("warnings", []),
                },
                sort_keys=True,
            )
        )
    return 0 if report.get("status") != "fail" else 2


if __name__ == "__main__":
    raise SystemExit(main())
