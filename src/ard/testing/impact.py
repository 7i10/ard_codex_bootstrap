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
        (
            "src/ard/analysis/ert_stage_a_calibration.py",
            "src/ard/cli/ert_stage_a_calibration.py",
            "configs/analysis/ert_stage_a_calibration_v1.yaml",
        ),
        ("tests/unit/test_ert_stage_a_calibration.py",),
        ("T0", "T1", "T2"),
    ),
    (
        ("src/ard/analysis/ert_stage_a_runtime.py", "src/ard/cli/ert_stage_a_runtime.py"),
        ("tests/unit/test_ert_stage_a_runtime.py",),
        ("T0", "T1", "T2", "T3"),
    ),
    (
        ("src/ard/analysis/ert_stage_a_endpoint.py", "src/ard/cli/ert_stage_a_endpoint.py"),
        ("tests/unit/test_ert_stage_a_endpoint.py",),
        ("T0", "T1", "T2", "T3"),
    ),
    (
        ("src/ard/analysis/ert_stage_a_report.py", "src/ard/cli/ert_stage_a_report.py"),
        ("tests/unit/test_ert_stage_a_report.py",),
        ("T0", "T1"),
    ),
    (
        ("src/ard/analysis/ert_stage_a_endpoint.py", "src/ard/cli/ert_stage_a_endpoint.py"),
        ("tests/unit/test_ert_stage_a_endpoint.py",),
        ("T0", "T1", "T2", "T3"),
    ),
    (
        (
            "src/ard/analysis/ert_online_routing_proxy.py",
            "src/ard/cli/ert_online_routing_proxy.py",
            "configs/analysis/ert_online_routing_proxy_v1.yaml",
            "configs/analysis/ert_online_routing_proxy_v2.yaml",
        ),
        ("tests/unit/test_ert_online_routing_proxy.py",),
        ("T0", "T1"),
    ),
    (
        (
            "src/ard/analysis/ert_state_overlay.py",
            "src/ard/cli/ert_state_overlay.py",
            "configs/analysis/ert_state_overlay_v1.yaml",
        ),
        ("tests/unit/test_ert_state_overlay.py",),
        ("T0", "T1"),
    ),
    (
        (
            "src/ard/campaign/",
            "scripts/campaign/",
            "configs/campaigns/",
            "tools/internal/legacy_campaign/",
        ),
        (),
        ("T0",),
    ),
    (
        (
            "src/ard/analysis/ffnr_strong_diagnostics.py",
            "src/ard/cli/ffnr_strong_diagnostics.py",
            "configs/analysis/ffnr_strong_diagnostics_v1.yaml",
            "configs/analysis/ffnr_strong_replay_chen_l2_dense.yaml",
            "configs/analysis/ffnr_strong_replay_chen_l4_dense.yaml",
        ),
        ("tests/unit/test_ffnr_strong_diagnostics.py",),
        ("T0", "T1", "T2"),
    ),
    (
        ("scripts/render_ffnr_review.py",),
        ("tests/unit/test_render_ffnr_review.py",),
        ("T0", "T1"),
    ),
    (
        ("scripts/analyze_ffnr_human_review.py",),
        ("tests/unit/test_analyze_ffnr_human_review.py",),
        ("T0", "T1"),
    ),
    (
        ("scripts/analyze_ffnr_next_evidence.py",),
        ("tests/unit/test_ffnr_next_evidence.py",),
        ("T0", "T1", "T2"),
    ),
    (
        (
            "src/ard/config/teacher_audit.py",
            "src/ard/evaluation/teacher_audit.py",
            "src/ard/cli/audit_teacher.py",
            "configs/audit/",
        ),
        ("tests/unit/test_teacher_accuracy_audit.py",),
        ("T0", "T1", "T2"),
    ),
    (
        ("external.lock.yaml",),
        (
            "tests/unit/test_external_management.py",
            "tests/unit/test_models_teacher.py",
            "tests/unit/test_tracking.py",
            "tests/regression/test_m2_upstream_oracle.py",
            "tests/regression/test_trades_upstream_differential.py",
        ),
        ("T0", "T1", "T2"),
    ),
    (
        (
            "teachers.lock.yaml",
            "scripts/bootstrap_teacher.py",
            "scripts/verify_teacher.py",
            "scripts/acquire_robustbench_teachers.py",
            "scripts/audit_robustbench_teacher.py",
        ),
        (
            "tests/unit/test_models_teacher.py",
            "tests/unit/test_external_management.py",
            "tests/unit/test_tracking.py",
            "tests/unit/test_teacher_acquisition.py",
        ),
        ("T0", "T1", "T2"),
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
        (
            "scripts/run_saad_upstream.py",
            "scripts/saad_runtime_bootstrap.py",
            "configs/upstream/",
            "requirements/saad-upstream-runtime.lock",
        ),
        ("tests/unit/test_run_saad_upstream.py",),
        ("T0", "T1"),
    ),
    (
        ("scripts/verify.py", "src/ard/testing/", "Makefile", ".gitignore"),
        ("tests/unit/test_verify_gate.py",),
        ("T0", "T1"),
    ),
    (
        (
            "src/ard/analysis/rslad_signal_replay.py",
            "src/ard/cli/rslad_signal_replay.py",
        ),
        ("tests/unit/test_rslad_signal_replay.py",),
        ("T0", "T1"),
    ),
    (
        (
            "src/ard/analysis/ffnr_forecasting.py",
            "src/ard/cli/ffnr_forecasting.py",
            "configs/analysis/ffnr_forecasting_v1.yaml",
        ),
        ("tests/unit/test_ffnr_forecasting.py",),
        ("T0", "T1"),
    ),
    (
        (
            "src/ard/analysis/ffnr_strong_replay.py",
            "src/ard/cli/ffnr_strong_replay.py",
            "configs/analysis/ffnr_strong_replay_",
        ),
        ("tests/unit/test_ffnr_strong_replay.py",),
        ("T0", "T1", "T2"),
    ),
    (
        (
            "src/ard/analysis/ffnr_strong_point.py",
            "src/ard/cli/ffnr_strong_point.py",
            "configs/analysis/ffnr_strong_point_v1.yaml",
        ),
        ("tests/unit/test_ffnr_strong_point.py",),
        ("T0", "T1", "T2"),
    ),
    (
        (
            "src/ard/analysis/epoch_metrics.py",
            "src/ard/analysis/wandb_history.py",
            "scripts/analyze_wandb_ro.py",
            "scripts/tag_wandb_runs.py",
        ),
        (
            "tests/unit/test_epoch_metrics.py",
            "tests/unit/test_wandb_history.py",
            "tests/integration/test_tracking_evaluation.py",
        ),
        ("T0", "T1"),
    ),
    (
        (
            "src/ard/analysis/logging_only_state.py",
            "src/ard/cli/logging_only_state.py",
        ),
        ("tests/unit/test_logging_only_state.py",),
        ("T0", "T1"),
    ),
    (
        (
            "src/ard/analysis/logging_only_prediction.py",
            "src/ard/cli/logging_only_prediction.py",
        ),
        ("tests/unit/test_logging_only_prediction.py",),
        ("T0", "T1"),
    ),
    (
        ("src/ard/analysis/history_cohort.py",),
        ("tests/unit/test_history_cohort.py",),
        ("T0", "T1"),
    ),
    (
        (
            "src/ard/analysis/history_screen.py",
            "src/ard/cli/history_screen.py",
        ),
        ("tests/unit/test_history_screen.py", "tests/unit/test_history_cohort.py"),
        ("T0", "T1"),
    ),
    (
        (
            "src/ard/analysis/history_early.py",
            "src/ard/cli/history_early.py",
        ),
        ("tests/unit/test_history_early.py", "tests/unit/test_history_cohort.py"),
        ("T0", "T1"),
    ),
    (
        ("src/ard/analysis/history_online_state.py", "src/ard/cli/history_online_state.py"),
        ("tests/unit/test_history_online_state.py",),
        ("T0", "T1"),
    ),
    (
        ("src/ard/analysis/history_bootstrap.py", "src/ard/cli/history_bootstrap.py"),
        ("tests/unit/test_history_bootstrap.py",),
        ("T0", "T1"),
    ),
    (
        (
            "src/ard/analysis/pre39_prescriptive.py",
            "src/ard/cli/pre39_prescriptive.py",
        ),
        ("tests/unit/test_pre39_prescriptive.py",),
        ("T0", "T1"),
    ),
    (
        ("src/ard/analysis/rescue_harm.py", "src/ard/cli/rescue_harm.py"),
        ("tests/unit/test_rescue_harm.py",),
        ("T0", "T1"),
    ),
    (
        ("src/ard/analysis/treatment_utility.py", "src/ard/cli/treatment_utility.py"),
        ("tests/unit/test_treatment_utility.py",),
        ("T0", "T1"),
    ),
    (
        ("src/ard/analysis/pre39_online_state.py", "src/ard/cli/pre39_online_state.py"),
        ("tests/unit/test_pre39_online_state.py",),
        ("T0", "T1"),
    ),
    (
        ("src/ard/analysis/pre39_online_confirm.py", "src/ard/cli/pre39_online_confirm.py"),
        ("tests/unit/test_pre39_online_confirm.py",),
        ("T0", "T1"),
    ),
    (
        ("src/ard/analysis/schedule_control_fork.py", "src/ard/cli/schedule_control_fork.py"),
        ("tests/unit/test_schedule_control_fork.py", "tests/unit/test_protocols_schedules.py"),
        ("T0", "T1", "T2"),
    ),
    (
        (
            "src/ard/analysis/prescriptive_v3.py",
            "src/ard/cli/prescriptive_v3.py",
        ),
        ("tests/unit/test_prescriptive_v3.py", "tests/unit/test_schedule_control_fork.py"),
        ("T0", "T1", "T2"),
    ),
    (
        ("tools/internal/schedule_control/",),
        ("tests/unit/test_schedule_control_inputs.py",),
        ("T0", "T1", "T2"),
    ),
    (
        (
            "src/ard/analysis/h4a_taxonomy.py",
            "src/ard/cli/h4a_taxonomy.py",
        ),
        ("tests/unit/test_h4a_taxonomy.py",),
        ("T0", "T1"),
    ),
    (
        (
            "src/ard/analysis/intervention_selector.py",
            "src/ard/analysis/intervention_fork.py",
            "src/ard/cli/intervention_selector.py",
        ),
        (
            "tests/unit/test_intervention_selector.py",
            "tests/unit/test_intervention_selector_cli.py",
            "tests/unit/test_intervention_fork.py",
        ),
        ("T0", "T1", "T2"),
    ),
    (
        (
            "src/ard/analysis/history_routing_v2.py",
            "src/ard/cli/history_routing_v2.py",
            "src/ard/targets/",
            "tools/internal/history_routing_v2/",
        ),
        (
            "tests/unit/test_history_routing_v2.py",
            "tests/unit/test_history_routing_v2_inputs.py",
            "tests/unit/test_intervention_fork.py",
            "tests/regression/test_m2_baselines.py",
            "tests/regression/test_m3_student_aware.py",
        ),
        ("T0", "T1", "T2"),
    ),
    (
        ("src/ard/cli/status.py",),
        ("tests/unit/test_status.py", "tests/unit/test_tracking.py"),
        ("T0", "T1"),
    ),
    (
        (
            "src/ard/analysis/signal_audit.py",
            "src/ard/analysis/teacher_risk_replay.py",
            "src/ard/analysis/__init__.py",
            "src/ard/cli/signal_audit.py",
            "src/ard/cli/replay_teacher_risk.py",
        ),
        (
            "tests/unit/test_signal_audit.py",
            "tests/unit/test_teacher_risk_replay.py",
        ),
        ("T0", "T1"),
    ),
    (
        ("src/ard/analysis/frozen_oracle.py", "src/ard/cli/build_frozen_oracle.py"),
        ("tests/unit/test_frozen_oracle.py",),
        ("T0", "T1", "T2"),
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
            "tests/unit/test_frozen_oracle.py",
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
            "tests/unit/test_distributed.py",
            "tests/unit/test_pilot_observability.py",
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
        (
            "src/ard/evaluation/",
            "src/ard/analysis/aggregate.py",
            "src/ard/analysis/sample_stats.py",
            "src/ard/cli/evaluate.py",
        ),
        ("tests/unit/test_evaluation.py", "tests/integration/test_tracking_evaluation.py"),
        ("T1", "T3"),
    ),
    (
        (
            "configs/audit/",
            "configs/experiments/",
            "configs/pilot/",
            "configs/production/",
            "configs/protocols/",
            "configs/scientific/",
            "configs/teachers/",
        ),
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
        ("pyproject.toml", "requirements/constraints.txt"),
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
