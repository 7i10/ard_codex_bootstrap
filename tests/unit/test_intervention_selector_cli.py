"""Argument wiring tests for the frozen intervention selector CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ard.cli import intervention_selector

pytestmark = pytest.mark.unit


def test_selector_cli_wires_all_inputs_and_prints_sorted_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    captured: dict[str, object] = {}

    def fake_build(*, files, output_dir):
        captured["files"] = files
        captured["output_dir"] = output_dir
        return {
            "random_mask": output_dir / "random-mask.json",
            "bundle": output_dir / "selector-bundle.json",
            "history_mask": output_dir / "history-mask.json",
        }

    monkeypatch.setattr(intervention_selector, "build_selector_bundle", fake_build)
    paths = [tmp_path / name for name in ("feature", "outcome", "report", "lineage", "l3-feature", "l3-lineage")]
    output = tmp_path / "new-output"
    assert (
        intervention_selector.main(
            [
                "--seed0-feature-panel",
                str(paths[0]),
                "--seed0-outcome-panel",
                str(paths[1]),
                "--seed0-report",
                str(paths[2]),
                "--seed0-lineage",
                str(paths[3]),
                "--l3-feature-panel",
                str(paths[4]),
                "--l3-lineage",
                str(paths[5]),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    files = captured["files"]
    assert files.seed0_feature_panel == paths[0].resolve()
    assert files.seed0_outcome_panel == paths[1].resolve()
    assert files.seed0_report == paths[2].resolve()
    assert files.seed0_lineage == paths[3].resolve()
    assert files.l3_feature_panel == paths[4].resolve()
    assert files.l3_lineage == paths[5].resolve()
    assert captured["output_dir"] == output.resolve()
    assert json.loads(capsys.readouterr().out) == {
        "bundle": str((output / "selector-bundle.json").resolve()),
        "history_mask": str((output / "history-mask.json").resolve()),
        "random_mask": str((output / "random-mask.json").resolve()),
    }


def test_selector_cli_requires_all_six_inputs_and_output() -> None:
    with pytest.raises(SystemExit):
        intervention_selector.main([])
