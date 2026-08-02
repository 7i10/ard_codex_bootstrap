from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from ard.analysis import wandb_history
from scripts import analyze_wandb_ro
from scripts.tag_wandb_runs import tag_registered_runs

pytestmark = pytest.mark.unit


class _Run:
    def __init__(self, run_id: str, rows: list[dict[str, object]]) -> None:
        self.id, self.summary, self._rows, self.history_calls = run_id, {"best_pgd_accuracy": 0.6}, rows, 0
        self.artifact_calls = 0
        self.state = "finished"
        self.history_kwargs: dict[str, object] | None = None
        self.tags: tuple[str, ...] = ("existing",)
        self.updated = 0

    def logged_artifacts(self):
        self.artifact_calls += 1
        return ()

    def history(self, **kwargs):
        self.history_calls += 1
        self.history_kwargs = kwargs
        return self._rows

    def update(self) -> None:
        self.updated += 1


def _cohort(path: Path, run_id: str = "run-1") -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "entity": "e",
                "project": "p",
                "metric": "val_pgd_accuracy",
                "expected_epochs": 3,
                "runs": [{"run_id": run_id}],
            }
        ),
        encoding="utf-8",
    )


def test_explicit_history_cohort_cache_skips_second_api_request(tmp_path: Path) -> None:
    cohort = tmp_path / "cohort.yaml"
    _cohort(cohort)
    rows = [{"epoch": epoch, "val_pgd_accuracy": 0.5, "val_clean_accuracy": 0.7} for epoch in range(3)]
    run = _Run("run-1", rows)
    result = wandb_history.analyze_cohort(
        cohort_path=cohort, cache_dir=tmp_path / "cache", fetch_run=lambda _: run, expected_epochs=3
    )
    assert result["cached"] is False and run.history_calls == 1
    assert run.history_kwargs is not None
    assert run.history_kwargs["keys"] == ["epoch", "val_pgd_accuracy", "val_clean_accuracy"]
    assert run.history_kwargs["x_axis"] == "epoch"
    cached = wandb_history.analyze_cohort(
        cohort_path=cohort,
        cache_dir=tmp_path / "cache",
        fetch_run=lambda _: (_ for _ in ()).throw(AssertionError("cache must not call W&B")),
        expected_epochs=3,
    )
    assert cached["cached"] is True
    forced = wandb_history.analyze_cohort(
        cohort_path=cohort,
        cache_dir=tmp_path / "cache",
        fetch_run=lambda _: run,
        expected_epochs=3,
        force=True,
    )
    assert forced["cached"] is False and run.history_calls == 2


def test_cache_fingerprint_includes_epoch_metrics_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cohort_bytes = b"cohort"
    before = wandb_history._cache_path(tmp_path, cohort_bytes=cohort_bytes, expected_epochs=3)
    monkeypatch.setattr(wandb_history, "_epoch_metrics_contract_fingerprint", lambda: "changed-contract")
    after = wandb_history._cache_path(tmp_path, cohort_bytes=cohort_bytes, expected_epochs=3)
    assert after != before


def test_cohort_schema_requires_active_metric_and_epoch_contract(tmp_path: Path) -> None:
    cohort = tmp_path / "cohort.yaml"
    cohort.write_text(yaml.safe_dump({"expected_epochs": 3, "runs": [{"run_id": "run-1"}]}), encoding="utf-8")
    with pytest.raises(wandb_history.WandbHistoryError, match="metric"):
        wandb_history.load_cohort(cohort)
    cohort.write_text(
        yaml.safe_dump({"metric": "val_pgd_accuracy", "expected_epochs": 1, "runs": [{"run_id": "run-1"}]}),
        encoding="utf-8",
    )
    with pytest.raises(wandb_history.WandbHistoryError, match="expected_epochs"):
        wandb_history.load_cohort(cohort)


def test_legacy_history_rejects_duplicate_or_incomplete_epoch_coverage(tmp_path: Path) -> None:
    cohort = tmp_path / "cohort.yaml"
    _cohort(cohort)
    run = _Run("run-1", [{"epoch": 0}, {"epoch": 0}, {"epoch": 2}])
    with pytest.raises(wandb_history.WandbHistoryError, match="duplicate"):
        wandb_history.analyze_cohort(
            cohort_path=cohort, cache_dir=tmp_path / "cache", fetch_run=lambda _: run, expected_epochs=3
        )


def test_explicit_legacy_source_skips_artifact_listing(tmp_path: Path) -> None:
    cohort = tmp_path / "cohort.yaml"
    cohort.write_text(
        yaml.safe_dump(
            {
                "metric": "val_pgd_accuracy",
                "expected_epochs": 3,
                "trajectory_source": "legacy_history",
                "runs": [{"run_id": "run-1"}],
            }
        ),
        encoding="utf-8",
    )
    run = _Run("run-1", [{"epoch": epoch, "val_pgd_accuracy": 0.5, "val_clean_accuracy": 0.7} for epoch in range(3)])
    result = wandb_history.analyze_cohort(
        cohort_path=cohort, cache_dir=tmp_path / "cache", fetch_run=lambda _: run, expected_epochs=3
    )
    assert run.artifact_calls == 0
    assert result["runs"][0]["trajectory_source"] == "legacy_history_exact_coverage"


def test_explicit_artifact_source_rejects_legacy_fallback(tmp_path: Path) -> None:
    cohort = tmp_path / "cohort.yaml"
    cohort.write_text(
        yaml.safe_dump(
            {
                "metric": "val_pgd_accuracy",
                "expected_epochs": 3,
                "runs": [{"run_id": "run-1", "trajectory_source": "epoch_metrics_artifact"}],
            }
        ),
        encoding="utf-8",
    )
    run = _Run("run-1", [{"epoch": epoch, "val_pgd_accuracy": 0.5, "val_clean_accuracy": 0.7} for epoch in range(3)])
    with pytest.raises(wandb_history.WandbHistoryError, match="requires an epoch-metrics artifact"):
        wandb_history.analyze_cohort(
            cohort_path=cohort, cache_dir=tmp_path / "cache", fetch_run=lambda _: run, expected_epochs=3
        )
    assert run.history_calls == 0


def test_cli_cache_hit_does_not_initialize_wandb_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    cohort = tmp_path / "cohort.yaml"
    _cohort(cohort)
    cache_dir = tmp_path / "cache"
    run = _Run("run-1", [{"epoch": epoch, "val_pgd_accuracy": 0.5, "val_clean_accuracy": 0.7} for epoch in range(3)])
    wandb_history.analyze_cohort(cohort_path=cohort, cache_dir=cache_dir, fetch_run=lambda _: run, expected_epochs=3)

    class _ForbiddenApi:
        def __init__(self) -> None:
            raise AssertionError("valid history cache must not initialize W&B API")

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Api=_ForbiddenApi))
    assert analyze_wandb_ro.main(["--cohort", str(cohort), "--cache-dir", str(cache_dir), "--epochs", "3"]) == 0
    assert json.loads(capsys.readouterr().out)["cached"] is True


def test_cli_rejects_epoch_override_that_disagrees_with_cohort(tmp_path: Path) -> None:
    cohort = tmp_path / "cohort.yaml"
    _cohort(cohort)
    with pytest.raises(wandb_history.WandbHistoryError, match="--epochs must equal"):
        analyze_wandb_ro.main(["--cohort", str(cohort), "--epochs", "2"])


def test_mixed_full_and_continuation_cohort_validates_each_explicit_epoch_range(tmp_path: Path) -> None:
    cohort = tmp_path / "cohort.yaml"
    cohort.write_text(
        yaml.safe_dump(
            {
                "entity": "e",
                "project": "p",
                "metric": "val_pgd_accuracy",
                "expected_epochs": 200,
                "runs": [
                    {"run_id": "full"},
                    {"run_id": "continuation", "epoch_start": 100, "epoch_end": 199},
                ],
            }
        ),
        encoding="utf-8",
    )
    runs = {
        run_id: _Run(
            run_id,
            [{"epoch": epoch, "val_pgd_accuracy": epoch / 199, "val_clean_accuracy": 0.7} for epoch in epochs],
        )
        for run_id, epochs in (("full", range(200)), ("continuation", range(100, 200)))
    }
    result = wandb_history.analyze_cohort(
        cohort_path=cohort, cache_dir=tmp_path / "cache", fetch_run=runs.__getitem__, expected_epochs=200
    )
    full, continued = result["runs"]
    assert full["trajectory"]["trajectory_requested_epoch_start"] == 0
    assert full["trajectory"]["trajectory_epoch_count"] == 200
    trajectory = continued["trajectory"]
    assert trajectory["trajectory_first_epoch"] == 100
    assert trajectory["trajectory_last_epoch"] == 199
    assert trajectory["trajectory_epoch_count"] == 100
    assert trajectory["trajectory_complete"] is True
    assert trajectory["val_pgd_best_epoch"] == 199
    assert trajectory["val_pgd_robust_overfit_gap"] == 0
    assert trajectory["val_pgd_normalized_auc_requested_range"] == pytest.approx(149.5 / 199)
    assert trajectory["val_pgd_mean_epoch_100_199"] == pytest.approx(sum(range(100, 200)) / 100 / 199)
    assert trajectory["val_pgd_slope_epoch_120_199"] == pytest.approx(1 / 199)


def test_continuation_coverage_error_names_run_range_and_count(tmp_path: Path) -> None:
    cohort = tmp_path / "cohort.yaml"
    cohort.write_text(
        yaml.safe_dump(
            {
                "metric": "val_pgd_accuracy",
                "expected_epochs": 200,
                "runs": [{"run_id": "continued", "epoch_start": 100, "epoch_end": 199}],
            }
        ),
        encoding="utf-8",
    )
    run = _Run(
        "continued",
        [{"epoch": epoch, "val_pgd_accuracy": 0.5, "val_clean_accuracy": 0.7} for epoch in range(100, 199)],
    )
    with pytest.raises(wandb_history.WandbHistoryError, match=r"continued range 100\.\.199.*count=99"):
        wandb_history.analyze_cohort(
            cohort_path=cohort, cache_dir=tmp_path / "cache", fetch_run=lambda _: run, expected_epochs=200
        )


def test_unfinished_runs_are_never_cached(tmp_path: Path) -> None:
    cohort = tmp_path / "cohort.yaml"
    _cohort(cohort)
    run = _Run("run-1", [])
    run.state = "running"
    with pytest.raises(wandb_history.WandbHistoryError, match="unfinished"):
        wandb_history.analyze_cohort(
            cohort_path=cohort, cache_dir=tmp_path / "cache", fetch_run=lambda _: run, expected_epochs=3
        )
    assert not (tmp_path / "cache").exists()


def test_nested_wandb_summary_values_are_excluded_from_json_cache_snapshot(tmp_path: Path) -> None:
    cohort = tmp_path / "cohort.yaml"
    _cohort(cohort)
    rows = [{"epoch": epoch, "val_pgd_accuracy": 0.5, "val_clean_accuracy": 0.7} for epoch in range(3)]
    run = _Run("run-1", rows)
    run.summary = {"best_pgd_accuracy": 0.6, "nested": {"wandb": "SummarySubDict-like"}, "_step": 3}
    result = wandb_history.analyze_cohort(
        cohort_path=cohort, cache_dir=tmp_path / "cache", fetch_run=lambda _: run, expected_epochs=3
    )
    snapshot = result["runs"][0]["summary_first"]
    assert snapshot == {"_step": 3, "best_pgd_accuracy": 0.6}
    assert result["run_fingerprints"][0]["summary_last_step"] == 3


def test_explicit_tagger_preserves_tags_and_second_apply_is_noop(tmp_path: Path) -> None:
    registry = tmp_path / "tags.yaml"
    registry.write_text(
        yaml.safe_dump({"runs": [{"run_id": "run-1", "tags": ["pilot", "teacher:chen"]}]}), encoding="utf-8"
    )
    run = _Run("run-1", [])
    dry = tag_registered_runs(registry_path=registry, fetch_run=lambda _: run)
    assert dry[0]["changed"] is True and dry[0]["applied"] is False and run.updated == 0
    applied = tag_registered_runs(registry_path=registry, fetch_run=lambda _: run, apply=True)
    assert applied[0]["applied"] is True and set(run.tags) == {"existing", "pilot", "teacher:chen"}
    repeated = tag_registered_runs(registry_path=registry, fetch_run=lambda _: run, apply=True)
    assert repeated[0]["changed"] is False and run.updated == 1
