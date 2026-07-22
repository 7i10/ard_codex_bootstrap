#!/usr/bin/env python3
"""Run only the tests implicated by a Git diff, reusing valid deterministic passes."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ard.testing.cache import CacheRecord, PassCache, fingerprint  # noqa: E402
from ard.testing.gpu_lock import GPULock  # noqa: E402
from ard.testing.impact import select  # noqa: E402

CACHE_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "PYTHONHASHSEED",
    "WANDB_MODE",
    "PYTHONPATH",
    "ARD_TEST_SEED",
    "ARD_TEST_FIXTURE_VERSION",
    "ARD_RUN_SAAD_ORACLE",
    "ARD_TRADES_SOURCE_EVIDENCE",
)


class VerificationError(RuntimeError):
    """The gate cannot determine a complete, trustworthy test selection."""


def _git(root: Path, args: list[str]) -> str:
    completed = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown Git error"
        raise VerificationError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout


def changed_paths(root: Path, base: str | None = None) -> tuple[str, ...]:
    """Include untracked paths and support an unborn repository without inventing HEAD."""
    if _git(root, ["rev-parse", "--is-inside-work-tree"]).strip() != "true":
        raise VerificationError(f"not a Git work tree: {root}")
    has_head = (
        subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"], cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ).returncode
        == 0
    )
    paths: set[str] = set()
    if has_head:
        if base:
            _git(root, ["rev-parse", "--verify", f"{base}^{{commit}}"])
        diff_base = base or "HEAD"
        paths.update(filter(None, _git(root, ["diff", "--name-only", f"{diff_base}..HEAD"]).splitlines()))
        paths.update(filter(None, _git(root, ["diff", "--name-only"]).splitlines()))
        paths.update(filter(None, _git(root, ["diff", "--cached", "--name-only"]).splitlines()))
    else:
        if base:
            raise VerificationError("--base cannot be used while HEAD is unborn")
        paths.update(filter(None, _git(root, ["ls-files", "--others", "--exclude-standard"]).splitlines()))
        paths.update(filter(None, _git(root, ["diff", "--cached", "--name-only"]).splitlines()))
    # Untracked files matter even with a real HEAD; git diff intentionally omits them.
    paths.update(filter(None, _git(root, ["ls-files", "--others", "--exclude-standard"]).splitlines()))
    return tuple(sorted(paths))


def available_tests(root: Path) -> tuple[str, ...]:
    return tuple(sorted(str(path.relative_to(root)) for path in (root / "tests").glob("**/test_*.py")))


def command_for(test: str) -> tuple[str, ...]:
    return (sys.executable, "-m", "pytest", "-q", test)


def build_test_environment(root: Path) -> dict[str, str]:
    """Run source-layout tests without requiring an editable install."""
    environment = os.environ.copy()
    source = str(root / "src")
    environment["PYTHONPATH"] = source + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    # Cache identity must not rely on a developer's ambient hash seed or
    # implicit fixture revision.  Explicit defaults also make subprocess test
    # fixtures reproducible when invoked by Make.
    environment.setdefault("PYTHONHASHSEED", "0")
    environment.setdefault("ARD_TEST_SEED", "0")
    environment.setdefault("ARD_TEST_FIXTURE_VERSION", "1")
    return environment


def gate_relevant_paths(root: Path) -> tuple[str, ...]:
    """Return all inputs whose changes invalidate broad tier/failed results."""
    paths: set[str] = set()
    for directory in ("src", "configs", "scripts", "tests"):
        base = root / directory
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}:
                paths.add(path.relative_to(root).as_posix())
    for relative in (".gitignore", "Makefile", "external.lock.yaml", "pyproject.toml"):
        if (root / relative).is_file():
            paths.add(relative)
    return tuple(sorted(paths))


def command_cacheable(
    *, root: Path, command: tuple[str, ...], requested_tier: str | None, environment: dict[str, str]
) -> bool:
    """Ask pytest whether this command can collect any T4/T5-marked item."""
    if requested_tier in {"T4", "T5"}:
        return False
    try:
        pytest_index = command.index("pytest")
    except ValueError:
        return False
    arguments = list(command[pytest_index + 1 :])
    marker_expression: str | None = None
    try:
        marker_index = arguments.index("-m")
    except ValueError:
        marker_index = -1
    if marker_index >= 0:
        if marker_index + 1 >= len(arguments):
            return False
        marker_expression = arguments[marker_index + 1]
        arguments[marker_index + 1] = f"({marker_expression}) and (t4 or t5)"
    else:
        arguments.extend(("-m", "t4 or t5"))
    if "--collect-only" not in arguments:
        arguments.append("--collect-only")
    collection = (*command[: pytest_index + 1], *arguments)
    completed = subprocess.run(
        collection, cwd=root, env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    if completed.returncode == 0:
        return False
    # Pytest exit code 5 means no selected T4/T5 tests. Errors fail closed.
    return completed.returncode == 5


def command_selects_marker(*, root: Path, command: tuple[str, ...], marker: str, environment: dict[str, str]) -> bool:
    """Fail closed when pytest collection cannot prove a marker is absent."""
    try:
        pytest_index = command.index("pytest")
    except ValueError:
        return False
    arguments = list(command[pytest_index + 1 :])
    try:
        marker_index = arguments.index("-m")
    except ValueError:
        marker_index = -1
    if marker_index >= 0:
        if marker_index + 1 >= len(arguments):
            return True
        arguments[marker_index + 1] = f"({arguments[marker_index + 1]}) and {marker}"
    else:
        arguments.extend(("-m", marker))
    if "--collect-only" not in arguments:
        arguments.append("--collect-only")
    collection = (*command[: pytest_index + 1], *arguments)
    completed = subprocess.run(
        collection, cwd=root, env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    if completed.returncode == 5:
        return False
    return True


def exclude_scientific_markers(command: tuple[str, ...]) -> tuple[str, ...]:
    """Restrict a pytest command to non-scientific tests without changing paths."""
    try:
        pytest_index = command.index("pytest")
    except ValueError:
        return command
    arguments = list(command[pytest_index + 1 :])
    expression = "not t4 and not t5"
    try:
        marker_index = arguments.index("-m")
    except ValueError:
        arguments.extend(("-m", expression))
    else:
        if marker_index + 1 >= len(arguments):
            raise VerificationError("pytest -m requires a marker expression")
        arguments[marker_index + 1] = f"({arguments[marker_index + 1]}) and ({expression})"
    return (*command[: pytest_index + 1], *arguments)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--changed", action="store_true", help="select tests for the current Git diff")
    source.add_argument("--tier", choices=("T0", "T1", "T2", "T3", "T4", "T5"))
    source.add_argument("--failed", action="store_true", help="re-run the latest failed commands")
    source.add_argument("--smoke", action="store_true", help="run the bounded synthetic smoke suite")
    parser.add_argument("--base", help="Git base revision for --changed")
    parser.add_argument("--force", action="store_true", help="ignore a matching pass-cache entry")
    parser.add_argument(
        "--non-scientific", action="store_true", help="exclude T4/T5 from every selected pytest command"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.non_scientific and args.tier in {"T4", "T5"}:
        parser.error("--non-scientific cannot be combined with --tier T4 or T5")
    root = args.root.resolve()
    cache = PassCache(root / ".cache" / "test-gate" / "results.jsonl")
    tests = available_tests(root)
    if args.failed:
        commands = list(cache.failed_commands())
        relevant = list(gate_relevant_paths(root))
    elif args.smoke:
        commands = [(sys.executable, "-m", "pytest", "-q", "tests/smoke", "-m", "smoke")]
        relevant = list(gate_relevant_paths(root))
    elif args.tier:
        marker = args.tier.lower()
        commands = [(sys.executable, "-m", "pytest", "-q", "-m", marker)]
        relevant = list(gate_relevant_paths(root))
    else:
        try:
            paths = changed_paths(root, args.base)
        except VerificationError as exc:
            parser.error(str(exc))
        selection = select(paths, tests)
        print("changed=" + ",".join(paths))
        print("tiers=" + ",".join(selection.tiers))
        commands = [command_for(test) for test in selection.tests]
        relevant = list(paths) + list(selection.tests)
    if not commands:
        print("no impacted tests")
        return 0
    for command in commands:
        if args.non_scientific:
            command = exclude_scientific_markers(command)
        run_environment = build_test_environment(root)
        cacheable = command_cacheable(root=root, command=command, requested_tier=args.tier, environment=run_environment)
        key = fingerprint(
            root=root,
            command=command,
            relevant_paths=relevant,
            environment={key: run_environment[key] for key in CACHE_ENVIRONMENT_KEYS if key in run_environment},
            fixture_version=run_environment["ARD_TEST_FIXTURE_VERSION"],
            seed=run_environment["ARD_TEST_SEED"],
        )
        printable = " ".join(command)
        if cacheable and not args.force and cache.has_pass(key):
            print("cached pass: " + printable)
            continue
        print(("would run: " if args.dry_run else "running: ") + printable)
        if args.dry_run:
            continue
        with (
            GPULock()
            if command_selects_marker(root=root, command=command, marker="gpu", environment=run_environment)
            else _NoopLock()
        ):
            completed = subprocess.run(command, cwd=root, env=run_environment)
        # Excluding T4/T5 can legitimately leave a changed scientific-only
        # file with no selected items.  It is not a failed non-scientific gate.
        status = "passed" if completed.returncode in ({0, 5} if args.non_scientific else {0}) else "failed"
        if cacheable and completed.returncode != 5:
            cache.append(CacheRecord(key, command, status))
        if status != "passed":
            return 1
    return 0


class _NoopLock:
    def __enter__(self) -> _NoopLock:
        return self

    def __exit__(self, *_: object) -> None:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
