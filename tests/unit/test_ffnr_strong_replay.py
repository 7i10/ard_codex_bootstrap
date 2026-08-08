from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ard.analysis import ffnr_strong_replay
from ard.analysis.ffnr_strong_replay import (
    StrongReplayError,
    StrongReplayResult,
    build_checkpoint_inventory_document,
    classification_primitives,
    jensen_shannon,
    load_cached_checkpoint,
    load_checkpoint_inventory_document,
    parse_replay_config,
    replay_checkpoint_rows,
    select_explicit_checkpoints,
    selection_attack_from_training,
    write_checkpoint_cache,
    write_checkpoint_inventory,
)
from ard.analysis.signal_audit import CheckpointInventory
from ard.config.schema import AttackConfig
from ard.data import IndexedBatch

pytestmark = pytest.mark.t2


def _checkpoint(epoch: int, *, sha: str) -> CheckpointInventory:
    return CheckpointInventory(
        "run",
        "model",
        ("last",),
        epoch,
        "/tmp/fake.pt",
        sha,
        epoch,
        True,
        3,
        "a" * 64,
        "b" * 40,
    )


def _selection() -> AttackConfig:
    return AttackConfig(
        epsilon="8/255",
        step_size="2/255",
        steps=20,
        random_start=True,
        loss="ce",
        student_mode="eval",
        teacher_mode="eval",
    )


def _models() -> tuple[nn.Module, nn.Module]:
    torch.manual_seed(5)
    student = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 3))
    teacher = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 3))
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return student, teacher


def test_selection_attack_rejects_any_primary_contract_drift() -> None:
    selection_attack_from_training(SimpleNamespace(method=SimpleNamespace(selection_attack=_selection())))
    drift = _selection().model_copy(update={"steps": 10})
    with pytest.raises(StrongReplayError, match="exact pixel"):
        selection_attack_from_training(SimpleNamespace(method=SimpleNamespace(selection_attack=drift)))


def test_classification_js_and_sparse_ids_are_exact() -> None:
    logits = torch.tensor([[4.0, 1.0, 0.0], [0.0, 2.0, 3.0]])
    primitives = classification_primitives(logits, torch.tensor([0, 2]))
    assert primitives["correct"].tolist() == [True, True]
    assert primitives["probability_margin"].tolist()[0] > 0
    assert primitives["logit_margin"].tolist() == pytest.approx([3.0, 1.0])
    probabilities = primitives["probabilities"]
    assert jensen_shannon(probabilities, probabilities).tolist() == pytest.approx([0.0, 0.0], abs=1e-7)


def test_epoch_selection_is_explicit_and_fail_closed() -> None:
    one = _checkpoint(189, sha="1" * 64)
    two = _checkpoint(194, sha="2" * 64)
    assert select_explicit_checkpoints((one, two), run_id="run", epochs=(189, 194)) == (one, two)
    with pytest.raises(StrongReplayError, match="exactly one"):
        select_explicit_checkpoints((one, two), run_id="run", epochs=(199,))
    with pytest.raises(StrongReplayError, match="sorted unique"):
        select_explicit_checkpoints((one, two), run_id="run", epochs=(194, 189))


def test_parse_launcher_config_requires_frozen_schema() -> None:
    valid = {
        "schema_version": 1,
        "contract": "ffnr_strong_replay_ce_pgd20_v1",
        "run_id": "run",
        "manifest": "manifest.json",
        "checkpoint_inventory": "inventory.json",
        "semantic_role": "feature",
        "epochs": [189],
        "train_expected_count": 3,
        "replay_batch_size": 2,
        "attack_seed": 7,
        "replay_device_type": "cpu",
    }
    assert parse_replay_config(valid)["epochs"] == [189]
    with pytest.raises(StrongReplayError, match="keys"):
        parse_replay_config({**valid, "extra": True})


def test_reused_inventory_does_not_rehash_unselected_checkpoint_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"run_id": "run", "config_hash": "a" * 64, "git": {"sha": "b" * 40}}))
    selected_path, unselected_path = tmp_path / "selected.pt", tmp_path / "unselected.pt"
    selected_path.write_bytes(b"selected")
    unselected_path.write_bytes(b"unselected")
    selected = CheckpointInventory(
        "run", "model", ("last",), 189, str(selected_path), "1" * 64, 189, True, 3, "a" * 64, "b" * 40
    )
    unselected = CheckpointInventory(
        "run", "model", ("last",), 194, str(unselected_path), "2" * 64, 194, True, 3, "a" * 64, "b" * 40
    )
    monkeypatch.setattr(ffnr_strong_replay, "inventory_run_bundle", lambda _: (selected, unselected))
    digest = ffnr_strong_replay.sha256_file(manifest)
    document = build_checkpoint_inventory_document(manifest_path=manifest, run_id="run")
    inventory_path = write_checkpoint_inventory(path=tmp_path / "inventory.json", document=document)
    calls: list[Path] = []

    def only_manifest(path: Path) -> str:
        calls.append(path)
        if path == manifest:
            return digest
        raise AssertionError(f"unexpected checkpoint rehash: {path}")

    monkeypatch.setattr(ffnr_strong_replay, "sha256_file", only_manifest)
    loaded = load_checkpoint_inventory_document(path=inventory_path, manifest_path=manifest, run_id="run")
    assert loaded == (selected, unselected)
    assert calls == [manifest]


def test_cache_requires_exact_identity_and_preserves_strong_schema(tmp_path: Path) -> None:
    identity = {
        "checkpoint": {"sha256": "4" * 64, "epoch": 189},
        "contract": "fixture",
        "expected_sample_count": 1,
        "attack_seed_base": 7,
        "attack_identity": {"epsilon_value": 8 / 255},
    }
    row = {
        "namespace": "train",
        "sample_id": 903,
        "class_id": 1,
        "epoch": 189,
        "observation_schema_version": 1,
        "student_robust_correct": True,
        "student_adversarial_probability_margin": 0.2,
        "student_adversarial_logit_margin": 1.0,
        "student_adversarial_ce": 0.3,
        "student_clean_probability_margin": 0.4,
        "student_clean_logit_margin": 2.0,
        "student_clean_correct": True,
        "student_clean_to_adversarial_prediction_flip": False,
        "student_clean_to_adversarial_true_probability_delta": -0.2,
        "student_clean_to_adversarial_probability_margin_delta": -0.2,
        "student_clean_to_adversarial_logit_margin_delta": -1.0,
        "teacher_clean_probabilities": [0.8, 0.1, 0.1],
        "teacher_adversarial_probabilities": [0.7, 0.2, 0.1],
        "teacher_clean_adversarial_js": 0.01,
    }
    result = StrongReplayResult(189, "4" * 64, 7, 8 / 255, (row,))
    assert write_checkpoint_cache(cache_dir=tmp_path, identity=identity, result=result) == result
    assert load_cached_checkpoint(cache_dir=tmp_path, identity=identity) == result
    metadata_path = next(tmp_path.glob("*.json"))
    metadata = json.loads(metadata_path.read_text())
    metadata["payload_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(StrongReplayError, match="payload hash mismatch"):
        load_cached_checkpoint(cache_dir=tmp_path, identity=identity)
    metadata_path.unlink()
    with pytest.raises(StrongReplayError, match="partially present"):
        load_cached_checkpoint(cache_dir=tmp_path, identity=identity)


def test_replay_emits_strong_primitives_and_never_populates_teacher_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    student, teacher = _models()
    checkpoint = _checkpoint(189, sha="3" * 64)
    training = SimpleNamespace(method=SimpleNamespace(selection_attack=_selection()))
    monkeypatch.setattr(ffnr_strong_replay, "load_historical_student", lambda *_args, **_kwargs: (student, {}))
    batches = (
        IndexedBatch(images=torch.rand(2, 3, 4, 4), labels=torch.tensor([1, 2]), sample_ids=torch.tensor([903, 17])),
    )
    first = replay_checkpoint_rows(
        checkpoint=checkpoint,
        training_config=training,
        teacher=teacher,
        loader=batches,
        device=torch.device("cpu"),
        attack_seed_base=41,
    )
    second = replay_checkpoint_rows(
        checkpoint=checkpoint,
        training_config=training,
        teacher=teacher,
        loader=batches,
        device=torch.device("cpu"),
        attack_seed_base=41,
    )
    assert first.rows == second.rows
    assert [row["sample_id"] for row in first.rows] == [903, 17]
    assert first.max_abs_delta <= 8 / 255 + 1e-7
    for row in first.rows:
        assert row["student_adversarial_ce"] >= 0
        assert -1 <= row["student_adversarial_probability_margin"] <= 1
        assert len(row["teacher_clean_probabilities"]) == 3
        assert row["teacher_clean_adversarial_js"] >= 0
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in teacher.parameters())
