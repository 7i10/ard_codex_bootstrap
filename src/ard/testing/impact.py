"""Deterministic path-to-test selection for the lightweight verification gate."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import PurePosixPath


@dataclass(frozen=True)
class ImpactSelection:
    tests: tuple[str, ...]
    tiers: tuple[str, ...]


RULES: tuple[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("external.lock.yaml",),
        (
            "tests/unit/test_external_management.py",
            "tests/regression/test_m2_upstream_oracle.py",
            "tests/regression/test_trades_upstream_differential.py",
        ),
        ("T0", "T1", "T2"),
    ),
    (
        ("teachers.lock.yaml", "scripts/bootstrap_teacher.py", "scripts/verify_teacher.py"),
        ("tests/unit/test_models_teacher.py", "tests/unit/test_external_management.py", "tests/unit/test_tracking.py"),
        ("T0", "T1"),
    ),
    (
        ("configs/teachers/",),
        (
            "tests/unit/test_models_teacher.py",
            "tests/unit/test_external_management.py",
            "tests/unit/test_tracking.py",
            "tests/unit/test_config.py",
        ),
        ("T0", "T1"),
    ),
    (
        ("scripts/bootstrap_external.py", "scripts/verify_external.py", "scripts/external_common.py"),
        ("tests/unit/test_external_management.py",),
        ("T0", "T1"),
    ),
    (
        ("scripts/verify.py", "src/ard/testing/", "Makefile", ".gitignore"),
        ("tests/unit/test_verify_gate.py",),
        ("T0", "T1"),
    ),
    (
        ("src/ard/config/",),
        (
            "tests/unit/test_config.py",
            "tests/regression/test_m3_student_aware.py",
            "tests/regression/test_m3_distributed.py",
            "tests/regression/test_m3_runtime_efficiency.py",
            "tests/regression/test_trades_upstream_differential.py",
            "tests/integration/test_synthetic_training.py",
        ),
        ("T0", "T2", "T3"),
    ),
    (
        ("src/ard/data/",),
        (
            "tests/unit/test_data.py",
            "tests/integration/test_checkpoint_resume.py",
            "tests/integration/test_synthetic_training.py",
        ),
        ("T1", "T3"),
    ),
    (
        ("src/ard/models/",),
        (
            "tests/unit/test_models_teacher.py",
            "tests/unit/test_pgd.py",
            "tests/integration/test_checkpoint_resume.py",
            "tests/integration/test_synthetic_training.py",
        ),
        ("T1", "T2", "T3"),
    ),
    (
        ("src/ard/protocols/", "src/ard/schedules/"),
        (
            "tests/unit/test_protocols_schedules.py",
            "tests/unit/test_config.py",
            "tests/unit/test_tracking.py",
            "tests/unit/test_evaluation.py",
            "tests/integration/test_checkpoint_resume.py",
            "tests/integration/test_tracking_evaluation.py",
        ),
        ("T0", "T1", "T3"),
    ),
    (
        ("src/ard/attacks/",),
        (
            "tests/unit/test_pgd.py",
            "tests/regression/test_m2_baselines.py",
            "tests/regression/test_m2_upstream_oracle.py",
            "tests/regression/test_m3_runtime_efficiency.py",
            "tests/regression/test_trades_upstream_differential.py",
            "tests/integration/test_m2_method_switch.py",
            "tests/integration/test_checkpoint_resume.py",
            "tests/integration/test_synthetic_training.py",
            "tests/smoke/test_gpu_pgd.py",
        ),
        ("T2", "T3"),
    ),
    (
        ("src/ard/objectives/",),
        (
            "tests/unit/test_pgd.py",
            "tests/regression/test_m2_baselines.py",
            "tests/regression/test_m2_upstream_oracle.py",
            "tests/regression/test_trades_upstream_differential.py",
            "tests/regression/test_m3_student_aware.py",
            "tests/regression/test_m3_distributed.py",
            "tests/regression/test_m3_runtime_efficiency.py",
            "tests/integration/test_m2_method_switch.py",
            "tests/integration/test_checkpoint_resume.py",
            "tests/integration/test_synthetic_training.py",
            "tests/smoke/test_training_smoke.py",
        ),
        ("T2", "T3"),
    ),
    (
        ("src/ard/signals/", "src/ard/policies/", "src/ard/state/"),
        (
            "tests/regression/test_m2_baselines.py",
            "tests/regression/test_m2_upstream_oracle.py",
            "tests/regression/test_m3_student_aware.py",
            "tests/regression/test_m3_distributed.py",
            "tests/regression/test_m3_runtime_efficiency.py",
            "tests/integration/test_m2_method_switch.py",
            "tests/integration/test_checkpoint_resume.py",
        ),
        ("T1", "T2", "T3"),
    ),
    (
        ("src/ard/engine/",),
        (
            "tests/unit/test_imports.py",
            "tests/regression/test_m2_baselines.py",
            "tests/regression/test_m3_student_aware.py",
            "tests/regression/test_m3_distributed.py",
            "tests/regression/test_m3_runtime_efficiency.py",
            "tests/integration/test_m2_method_switch.py",
            "tests/integration/test_checkpoint_resume.py",
            "tests/integration/test_synthetic_training.py",
            "tests/smoke/test_training_smoke.py",
        ),
        ("T0", "T3"),
    ),
    (
        ("src/ard/cli/train.py",),
        (
            "tests/unit/test_imports.py",
            "tests/unit/test_distributed.py",
            "tests/unit/test_tracking.py",
            "tests/regression/test_m2_baselines.py",
            "tests/regression/test_m3_student_aware.py",
            "tests/regression/test_m3_distributed.py",
            "tests/regression/test_m3_runtime_efficiency.py",
            "tests/regression/test_m4_distributed.py",
            "tests/integration/test_m2_method_switch.py",
            "tests/integration/test_checkpoint_resume.py",
            "tests/integration/test_synthetic_training.py",
            "tests/integration/test_tracking_evaluation.py",
            "tests/smoke/test_training_smoke.py",
        ),
        ("T0", "T1", "T3"),
    ),
    (
        ("src/ard/tracking/", "scripts/sync_wandb.py"),
        (
            "tests/unit/test_tracking.py",
            "tests/integration/test_tracking_evaluation.py",
            "tests/regression/test_m3_runtime_efficiency.py",
            "tests/regression/test_m4_distributed.py",
        ),
        ("T1", "T3"),
    ),
    (
        ("src/ard/evaluation/", "src/ard/analysis/", "src/ard/cli/evaluate.py"),
        ("tests/unit/test_evaluation.py", "tests/integration/test_tracking_evaluation.py"),
        ("T1", "T3"),
    ),
    (
        ("configs/",),
        (
            "tests/unit/test_config.py",
            "tests/regression/test_m2_baselines.py",
            "tests/regression/test_trades_upstream_differential.py",
            "tests/integration/test_m2_method_switch.py",
            "tests/integration/test_synthetic_training.py",
        ),
        ("T0", "T2", "T3"),
    ),
    (
        ("pyproject.toml", "requirements/"),
        (
            "tests/unit/test_verify_gate.py",
            "tests/unit/test_config.py",
            "tests/unit/test_data.py",
            "tests/unit/test_models_teacher.py",
            "tests/unit/test_pgd.py",
            "tests/integration/test_checkpoint_resume.py",
            "tests/integration/test_synthetic_training.py",
            "tests/unit/test_tracking.py",
            "tests/unit/test_evaluation.py",
            "tests/integration/test_tracking_evaluation.py",
            "tests/regression/test_m4_distributed.py",
            "tests/smoke/test_gpu_pgd.py",
            "tests/smoke/test_training_smoke.py",
        ),
        ("T0", "T1", "T2", "T3"),
    ),
    (("docs/",), (), ("T0",)),
)


def _matches(path: str, pattern: str) -> bool:
    return path.startswith(pattern) if pattern.endswith("/") else path == pattern


def select(paths: Iterable[str], available_tests: Iterable[str]) -> ImpactSelection:
    """Select focused tests, falling back conservatively for unknown code paths."""
    changed = tuple(sorted({PurePosixPath(path).as_posix() for path in paths}))
    available = set(available_tests)
    tests: set[str] = set()
    tiers: set[str] = set()
    has_unknown = False
    for path in changed:
        path_known = False
        for patterns, candidates, rule_tiers in RULES:
            if any(_matches(path, pattern) for pattern in patterns):
                matched_tests = {test for test in candidates if test in available}
                # A mapped path is only narrow when its mapped tests exist; docs-only rules are intentional.
                path_known = path_known or not candidates or bool(matched_tests)
                tests.update(matched_tests)
                tiers.update(rule_tiers)
        if path.startswith("tests/"):
            if path in available and path.endswith(".py"):
                path_known = True
                tests.add(path)
                tiers.update(("T0", "T1"))
            elif path.startswith("tests/smoke/"):
                # Smoke-only helpers cannot affect unit or scientific tests.
                path_known = True
                tests.update(test for test in available if test.startswith("tests/smoke/"))
                tiers.add("T3")
            else:
                # conftest.py, helpers, and fixtures may affect every collected test.
                has_unknown = True
        if not path_known:
            has_unknown = True
    if has_unknown:
        # One unknown path makes the complete change set unknown, even if other paths map narrowly.
        tests.update(available)
        tiers.update(("T0", "T1"))
    return ImpactSelection(tuple(sorted(tests)), tuple(sorted(tiers)))
