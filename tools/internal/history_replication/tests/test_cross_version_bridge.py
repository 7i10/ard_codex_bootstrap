from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from tools.internal.history_replication.bridge.cross_version_bridge import (
    _SOURCE_HASH_KEYS,
    EMISSION_SCHEMA_VERSION,
    ObservationBridgeError,
    compare,
    observation_kwargs_for,
)

OLD_SHA = "a" * 40
NEW_SHA = "b" * 40


class _LegacyTrainer:
    def __init__(self, *, observe_teacher_signals: bool = False) -> None: ...


class _ProfileTrainer:
    def __init__(self, *, observation_profile: str = "off") -> None: ...


class _AmbiguousTrainer:
    def __init__(self, *, observe_teacher_signals: bool = False, observation_profile: str = "off") -> None: ...


class _MissingTrainer:
    def __init__(self) -> None: ...


def test_observation_api_detection_is_explicit_and_fail_closed() -> None:
    assert observation_kwargs_for(_LegacyTrainer) == ("observe_teacher_signals", {"observe_teacher_signals": True})
    assert observation_kwargs_for(_ProfileTrainer) == (
        "observation_profile",
        {"observation_profile": "teacher_response"},
    )
    with pytest.raises(ObservationBridgeError, match="exactly one"):
        observation_kwargs_for(_AmbiguousTrainer)
    with pytest.raises(ObservationBridgeError, match="exactly one"):
        observation_kwargs_for(_MissingTrainer)


def _record() -> dict[str, object]:
    return {
        "margin_ema": 0.25,
        "seen": 2,
        "robust_correct_count": 2,
        "previous_robust_correct": True,
        "forgetting_count": 0,
        "last_update": 2,
        "last_margin": 0.25,
        "true_label": 0,
        "teacher_clean_entropy": 0.5,
        "teacher_clean_true_probability": 0.8,
        "teacher_clean_max_wrong_probability": 0.1,
        "teacher_clean_prediction": 0,
        "teacher_clean_correct": True,
        "teacher_adversarial_entropy": 0.6,
        "teacher_adversarial_true_probability": 0.7,
        "teacher_adversarial_max_wrong_probability": 0.2,
        "teacher_adversarial_prediction": 0,
        "teacher_adversarial_correct": True,
        "first_robustly_learned_epoch": 0,
        "current_correct_streak": 2,
        "longest_correct_streak": 2,
        "margin_mean": 0.25,
        "margin_m2": 0.0,
        "margin_time_sum": 4.0,
        "margin_time_squared_sum": 10.0,
        "margin_time_margin_sum": 1.0,
        "history_statistics_complete": True,
        "teacher_clean_to_adversarial_margin_response": -0.1,
        "teacher_clean_to_adversarial_js_response": 0.01,
    }


def _emission(*, api: str, sha: str, delta: float = 0.0) -> dict[str, Any]:
    records = {sample_id: _record() for sample_id in ("0", "1", "2", "6", "7")}
    records["0"]["margin_ema"] = 0.25 + delta
    return {
        "bridge_schema_version": EMISSION_SCHEMA_VERSION,
        "metadata": {
            "git_sha": sha,
            "command": ["python", "scripts/cross_version_observation_bridge.py"],
            "source_hashes": {key: "c" * 64 for key in _SOURCE_HASH_KEYS},
            "trainer_observation_api": api,
            "device": "cuda:0",
            "fixture": "synthetic-cifar8-rslad-pgd1-random-start-epochs2",
            "runtime_module_paths": {"trainer": "ard/engine/trainer.py"},
            "environment": {"torch": "test"},
        },
        "checkpoint": {
            "format_version": 1,
            "epoch": 1,
            "epoch_boundary": "end",
            "model": {"weight": torch.tensor([1.0])},
            "optimizer": {"state": {}, "param_groups": []},
            "scheduler": {"last_epoch": 1},
            "scaler": None,
            "rng": [
                {
                    "python": (3, 4),
                    "torch_cpu": torch.tensor([1], dtype=torch.uint8),
                    "torch_cuda": None,
                    "numpy": None,
                }
            ],
            "sampler_epoch": [1],
            "sampler_state": [{}],
            "sample_state": {
                "format_version": 3,
                "ema_decay": 0.9,
                "records": records,
                "pending": [],
                "next_order": 0,
            },
            "global_step": 4,
            "best_metric": 0.5,
            "selection_metadata": {"selected_epoch": 1},
            "tracker_run_id": f"run-{api}",
            "config_hash": f"config-{api}",
            "world_size": 1,
        },
        "telemetry": {"teacher_adversarial_forward_calls": 4.0},
    }


def _write(path: Path, emission: dict[str, Any]) -> None:
    torch.save(emission, path)


def _compare(left: Path, right: Path, attestation: Path) -> dict[str, Any]:
    return compare(
        left,
        right,
        attestation=attestation,
        left_expected_git_sha=OLD_SHA,
        right_expected_git_sha=NEW_SHA,
    )


def test_compare_allows_only_identity_and_forward_telemetry(tmp_path: Path) -> None:
    left, right, attestation = tmp_path / "old.pt", tmp_path / "new.pt", tmp_path / "attestation.json"
    old = _emission(api="observe_teacher_signals", sha=OLD_SHA)
    new = _emission(api="observation_profile", sha=NEW_SHA)
    new["metadata"]["source_hashes"] = {key: "d" * 64 for key in _SOURCE_HASH_KEYS}
    new["telemetry"] = {"teacher_clean_forward_calls": 4.0, "teacher_adversarial_forward_calls": 4.0}
    _write(left, old)
    _write(right, new)
    result = _compare(left, right, attestation)
    assert result["result"] == "pass"
    assert result["left"]["sample_state_record_count"] == 5
    assert attestation.is_file()


def test_compare_rejects_wrong_sha_api_or_non_distinct_emission(tmp_path: Path) -> None:
    left, right = tmp_path / "old.pt", tmp_path / "new.pt"
    _write(left, _emission(api="observe_teacher_signals", sha=OLD_SHA))
    _write(right, _emission(api="observation_profile", sha=NEW_SHA))
    with pytest.raises(ObservationBridgeError, match="expected runtime SHA"):
        compare(
            left,
            right,
            attestation=tmp_path / "wrong-sha.json",
            left_expected_git_sha=NEW_SHA,
            right_expected_git_sha=OLD_SHA,
        )
    reversed_api = _emission(api="observation_profile", sha=OLD_SHA)
    _write(left, reversed_api)
    with pytest.raises(ObservationBridgeError, match="legacy observation API"):
        _compare(left, right, tmp_path / "wrong-api.json")
    with pytest.raises(ObservationBridgeError, match="distinct files"):
        _compare(right, right, tmp_path / "same-file.json")


def test_compare_rejects_optimization_or_sample_state_difference(tmp_path: Path) -> None:
    left, right = tmp_path / "old.pt", tmp_path / "new.pt"
    _write(left, _emission(api="observe_teacher_signals", sha=OLD_SHA))
    _write(right, _emission(api="observation_profile", sha=NEW_SHA, delta=0.1))
    with pytest.raises(ObservationBridgeError, match="checkpoint.sample_state"):
        _compare(left, right, tmp_path / "state.json")

    _write(right, _emission(api="observation_profile", sha=NEW_SHA))
    changed = torch.load(right, weights_only=False)
    changed["checkpoint"]["global_step"] = 5
    _write(right, changed)
    with pytest.raises(ObservationBridgeError, match="checkpoint.global_step"):
        _compare(left, right, tmp_path / "optimization.json")


def test_compare_rejects_dropped_record_or_record_field(tmp_path: Path) -> None:
    left, right = tmp_path / "old.pt", tmp_path / "new.pt"
    _write(left, _emission(api="observe_teacher_signals", sha=OLD_SHA))
    missing_record = _emission(api="observation_profile", sha=NEW_SHA)
    del missing_record["checkpoint"]["sample_state"]["records"]["7"]
    _write(right, missing_record)
    with pytest.raises(ObservationBridgeError, match="stable sample IDs"):
        _compare(left, right, tmp_path / "record.json")
    missing_field = _emission(api="observation_profile", sha=NEW_SHA)
    del missing_field["checkpoint"]["sample_state"]["records"]["0"]["margin_ema"]
    _write(right, missing_field)
    with pytest.raises(ObservationBridgeError, match="record fields"):
        _compare(left, right, tmp_path / "field.json")


def test_wrapper_rejects_preloaded_ard_module(monkeypatch: pytest.MonkeyPatch) -> None:
    script = Path(__file__).resolve().parents[1] / "bridge" / "run.py"
    specification = importlib.util.spec_from_file_location("bridge_wrapper_for_test", script)
    assert specification is not None and specification.loader is not None
    wrapper = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(wrapper)
    monkeypatch.setitem(sys.modules, "ard.already_loaded", object())
    with pytest.raises(SystemExit, match="already imported"):
        wrapper._load_main(Path(__file__).resolve().parents[2])
