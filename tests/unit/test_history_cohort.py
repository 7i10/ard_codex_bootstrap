from __future__ import annotations

import json
from pathlib import Path

import pytest

from ard.analysis.history_cohort import HistoryCohortError, bind_reports_to_cohort, load_cohort_inventory

pytestmark = pytest.mark.t1


def _inventory(path: Path) -> dict[str, dict[str, object]]:
    runs = {
        "L1": {
            "run_id": "l1",
            "config_hash": "1" * 64,
            "scientific_git_sha": "a" * 40,
            "seed": 1,
            "teacher_registry_id": "bartoldson2024_adversarial_wrn94_16",
        },
        "L2": {
            "run_id": "l2",
            "config_hash": "2" * 64,
            "scientific_git_sha": "b" * 40,
            "seed": 1,
            "teacher_registry_id": "chen2021_ltd_wrn34_10",
        },
        "L3": {
            "run_id": "l3",
            "config_hash": "3" * 64,
            "scientific_git_sha": "c" * 40,
            "seed": 2,
            "teacher_registry_id": "bartoldson2024_adversarial_wrn94_16",
        },
        "L4": {
            "run_id": "l4",
            "config_hash": "4" * 64,
            "scientific_git_sha": "d" * 40,
            "seed": 2,
            "teacher_registry_id": "chen2021_ltd_wrn34_10",
        },
    }
    path.write_text(json.dumps({"schema_version": 1, "contract": "h5_confirmatory_cohort_inventory_v1", "runs": runs}))
    return runs


def test_cohort_rejects_duplicate_swapped_seed_and_teacher_identities(tmp_path: Path) -> None:
    path = tmp_path / "cohort.json"
    expected = _inventory(path)
    parsed, _ = load_cohort_inventory(path)
    reports = {label: {"input_identity": value} for label, value in parsed.items()}
    bind_reports_to_cohort(inventory=parsed, reports=reports)
    for field, value in (("run_id", "l1"), ("seed", 1), ("teacher_registry_id", "chen2021_ltd_wrn34_10")):
        raw = json.loads(path.read_text())
        raw["runs"]["L3"][field] = value
        path.write_text(json.dumps(raw))
        with pytest.raises(HistoryCohortError):
            load_cohort_inventory(path)
        _inventory(path)
    swapped = {label: {"input_identity": dict(value)} for label, value in parsed.items()}
    swapped["L1"]["input_identity"] = dict(expected["L3"])
    with pytest.raises(HistoryCohortError, match="L1"):
        bind_reports_to_cohort(inventory=parsed, reports=swapped)
