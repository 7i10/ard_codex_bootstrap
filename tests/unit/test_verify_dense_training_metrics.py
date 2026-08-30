from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from verify_dense_training_metrics import DenseMetricsError, validate_dense_rows  # noqa: E402

pytestmark = pytest.mark.unit


def rows(n: int = 3) -> list[dict[str, object]]:
    return [{"epoch": i, "metric": 0.5 + i / 100, "source_sha": "abc"} for i in range(n)]


def test_complete_dense_rows_pass() -> None:
    result = validate_dense_rows(
        rows(), expected_end=2, required_fields=("epoch", "metric"), identity={"source_sha": "abc"}
    )
    assert result["valid"] is True and result["row_count"] == 3


@pytest.mark.parametrize(
    "bad", [[rows()[0], rows()[2]], rows() + [rows()[1]], [{**r, "metric": float("nan")} for r in rows()]]
)
def test_missing_duplicate_or_nonfinite_rows_fail(bad) -> None:
    with pytest.raises(DenseMetricsError):
        validate_dense_rows(bad, expected_end=2, required_fields=("epoch", "metric"), identity={"source_sha": "abc"})


def test_identity_drift_fails() -> None:
    bad = rows()
    bad[1]["source_sha"] = "different"
    with pytest.raises(DenseMetricsError, match="identity"):
        validate_dense_rows(bad, expected_end=2, required_fields=("epoch", "metric"), identity={"source_sha": "abc"})


def test_cli_returns_nonzero_for_duplicate(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [rows()[0], rows()[0], rows()[2]]) + "\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/verify_dense_training_metrics.py"),
            str(path),
            "--expected-end",
            "2",
            "--required-field",
            "metric",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
