from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ard.analysis.ert_confirmatory_report import ConfirmatoryReportError, _validate_fork, _validate_rows_against


def test_confirmatory_fork_binds_parent_arm_and_horizon_artifact(tmp_path: Path) -> None:
    checkpoint = tmp_path / "epoch-84.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "fork_lineage": {"parent_epoch": 79, "parent_checkpoint_sha256": "parent", "arm": "T1WCONF"},
                "artifacts": [{"sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "aliases": ["epoch-84"]}],
            }
        ),
        encoding="utf-8",
    )
    result = _validate_fork(manifest, parent_sha="parent", arm="T1WCONF", checkpoint=checkpoint)
    assert result["checkpoint_sha256"] == hashlib.sha256(b"checkpoint").hexdigest()


def test_confirmatory_report_rejects_class_mismatch() -> None:
    with pytest.raises(ConfirmatoryReportError, match="class mismatch"):
        _validate_rows_against({1: {"true_label": 0}}, {1: {"true_label": 1}}, "fixture")
