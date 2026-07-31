from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from ard.analysis.logging_only_state import LoggingOnlyStateError, logging_only_state_analysis
from ard.cli.logging_only_state import main
from ard.signals import teacher_confidence_primitives
from ard.state import SampleStateStore

pytestmark = pytest.mark.t1


def _state(*, update: int, epoch: int, margins: list[float], correct: list[bool]) -> SampleStateStore:
    labels = torch.tensor([0, 1])
    valid = torch.tensor([True, True])
    clean = teacher_confidence_primitives(torch.tensor([[3.0, 1.0], [0.0, 2.0]]), labels, valid)
    adversarial = teacher_confidence_primitives(torch.tensor([[1.0, 2.0], [0.5, 1.5]]), labels, valid)
    store = SampleStateStore()
    store.record_pending(
        sample_ids=torch.tensor([7, 3]),
        margins=torch.tensor(margins),
        robust_correct=torch.tensor(correct),
        valid_mask=valid,
        update=update,
        epoch=epoch,
        labels=labels,
        teacher_clean=clean,
        teacher_adversarial=adversarial,
        teacher_clean_to_adversarial_margin_response=torch.tensor([-0.5, -0.25]),
        teacher_clean_to_adversarial_js_response=torch.tensor([0.1, 0.2]),
    )
    store.merge_pending([store.pending_state()])
    return store


def _checkpoint(path: Path, *, epoch: int, store: SampleStateStore) -> None:
    for record in store.records.values():
        record.seen = epoch + 1
    torch.save(
        {
            "epoch": epoch,
            "config_hash": "a" * 64,
            "tracker_run_id": "logging-only",
            "world_size": 1,
            "sample_state": store.state_dict(),
        },
        path,
    )


def _analysis_provenance() -> dict[str, object]:
    source_files = {"analysis": "1" * 64, "cli": "2" * 64, "sample_store": "3" * 64}
    aggregate = hashlib.sha256(
        json.dumps(source_files, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    return {"git_sha": "c" * 40, "dirty": False, "source_files": source_files, "source_sha256": aggregate}


def _manifest(path: Path, *, anchor: Path, final: Path, status: str = "completed") -> Path:
    path.write_text(
        json.dumps(
            {
                "run_id": "logging-only",
                "config_hash": "a" * 64,
                "world_size": 1,
                "status": status,
                "git": {"sha": "b" * 40, "dirty": False},
                "artifacts": [
                    {
                        "name": "model-logging-only-last",
                        "type": "model",
                        "aliases": ["last"],
                        "local_path": "artifacts/anchor",
                        "sha256": hashlib.sha256(anchor.read_bytes()).hexdigest(),
                    },
                    {
                        "name": "model-logging-only-last",
                        "type": "model",
                        "aliases": ["last"],
                        "local_path": "artifacts/final",
                        "sha256": hashlib.sha256(final.read_bytes()).hexdigest(),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (path.parent / "completion.json").write_text(
        json.dumps({"status": status, "output_dir": str(path.parent.parent)}),
        encoding="utf-8",
    )
    return path


def test_logging_only_state_analysis_hash_binds_exact_anchors_and_exports_stable_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    anchor_store = _state(update=10, epoch=99, margins=[0.4, -0.2], correct=[True, False])
    final_store = _state(update=10, epoch=99, margins=[0.4, -0.2], correct=[True, False])
    labels = torch.tensor([0, 1])
    valid = torch.tensor([True, True])
    clean = teacher_confidence_primitives(torch.tensor([[3.0, 1.0], [0.0, 2.0]]), labels, valid)
    adversarial = teacher_confidence_primitives(torch.tensor([[1.0, 2.0], [0.5, 1.5]]), labels, valid)
    final_store.record_pending(
        sample_ids=torch.tensor([7, 3]),
        margins=torch.tensor([-0.4, 0.2]),
        robust_correct=torch.tensor([False, True]),
        valid_mask=valid,
        update=20,
        epoch=199,
        labels=labels,
        teacher_clean=clean,
        teacher_adversarial=adversarial,
        teacher_clean_to_adversarial_margin_response=torch.tensor([-0.5, -0.25]),
        teacher_clean_to_adversarial_js_response=torch.tensor([0.1, 0.2]),
    )
    final_store.merge_pending([final_store.pending_state()])
    anchor, final = tmp_path / "epoch99.pt", tmp_path / "epoch199.pt"
    _checkpoint(anchor, epoch=99, store=anchor_store)
    _checkpoint(final, epoch=199, store=final_store)

    manifest = _manifest(tmp_path / "manifest.json", anchor=anchor, final=final)
    analysis = logging_only_state_analysis(
        anchor_checkpoint=anchor,
        final_checkpoint=final,
        expected_count=2,
        run_bundle_manifest=manifest,
        analysis_provenance=_analysis_provenance(),
    )
    assert analysis.identity["contract"] == "logging_only_exact_state_anchor99_final199_v1"
    assert analysis.identity["anchor"]["checkpoint_sha256"] != analysis.identity["final"]["checkpoint_sha256"]
    assert analysis.identity["scientific_git_sha"] == "b" * 40
    assert [row["sample_id"] for row in analysis.rows] == [3, 7]
    row = analysis.rows[1]
    assert row["anchor_margin_ema"] == pytest.approx(0.4)
    assert row["anchor_teacher_clean_to_adversarial_js_response"] == pytest.approx(0.1)
    assert row["subsequent_forgetting_increment"] == 1
    assert row["future_online_forgetting"] == 1
    assert row["final_robust_error"] == 1

    output = tmp_path / "matrix.json"
    monkeypatch.setattr(
        "ard.analysis.logging_only_state._tracked_clean_analysis_provenance",
        _analysis_provenance,
    )
    assert (
        main(
            [
                "--anchor-checkpoint",
                str(anchor),
                "--final-checkpoint",
                str(final),
                "--expected-count",
                "2",
                "--run-bundle-manifest",
                str(manifest),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    emitted = json.loads(output.read_text(encoding="utf-8"))
    assert emitted["identity"] == analysis.identity
    assert emitted["rows"] == list(analysis.rows)


def test_logging_only_state_analysis_rejects_nonexact_anchor_epoch(tmp_path: Path) -> None:
    state = _state(update=1, epoch=99, margins=[0.1, -0.1], correct=[True, False])
    anchor, final = tmp_path / "wrong.pt", tmp_path / "epoch199.pt"
    _checkpoint(anchor, epoch=98, store=state)
    _checkpoint(final, epoch=199, store=state)
    with pytest.raises(LoggingOnlyStateError, match="exactly 99"):
        logging_only_state_analysis(
            anchor_checkpoint=anchor,
            final_checkpoint=final,
            expected_count=2,
            run_bundle_manifest=tmp_path / "unused.json",
            analysis_provenance=_analysis_provenance(),
        )


def test_logging_only_state_analysis_rejects_wrong_record_count(tmp_path: Path) -> None:
    state = _state(update=1, epoch=99, margins=[0.1, -0.1], correct=[True, False])
    anchor, final = tmp_path / "epoch99.pt", tmp_path / "epoch199.pt"
    _checkpoint(anchor, epoch=99, store=state)
    _checkpoint(final, epoch=199, store=state)
    with pytest.raises(LoggingOnlyStateError, match="expected exactly 45000"):
        logging_only_state_analysis(
            anchor_checkpoint=anchor,
            final_checkpoint=final,
            run_bundle_manifest=tmp_path / "unused.json",
            analysis_provenance=_analysis_provenance(),
        )


def test_logging_only_state_analysis_rejects_legacy_migrated_history(tmp_path: Path) -> None:
    state = _state(update=1, epoch=99, margins=[0.1, -0.1], correct=[True, False])
    for record in state.records.values():
        record.history_statistics_complete = False
    anchor, final = tmp_path / "epoch99.pt", tmp_path / "epoch199.pt"
    _checkpoint(anchor, epoch=99, store=state)
    _checkpoint(final, epoch=199, store=state)
    with pytest.raises(LoggingOnlyStateError, match="incomplete legacy state"):
        logging_only_state_analysis(
            anchor_checkpoint=anchor,
            final_checkpoint=final,
            expected_count=2,
            run_bundle_manifest=tmp_path / "unused.json",
            analysis_provenance=_analysis_provenance(),
        )


def test_logging_only_state_analysis_rejects_missing_epoch_observation(tmp_path: Path) -> None:
    anchor_store = _state(update=1, epoch=99, margins=[0.1, -0.1], correct=[True, False])
    final_store = _state(update=2, epoch=199, margins=[0.2, -0.2], correct=[True, False])
    anchor, final = tmp_path / "epoch99.pt", tmp_path / "epoch199.pt"
    _checkpoint(anchor, epoch=99, store=anchor_store)
    _checkpoint(final, epoch=199, store=final_store)
    payload = torch.load(anchor, map_location="cpu", weights_only=False)
    payload["sample_state"]["records"]["7"]["seen"] = 99
    torch.save(payload, anchor)
    manifest = _manifest(tmp_path / "manifest.json", anchor=anchor, final=final)
    with pytest.raises(LoggingOnlyStateError, match="every protocol epoch"):
        logging_only_state_analysis(
            anchor_checkpoint=anchor,
            final_checkpoint=final,
            expected_count=2,
            run_bundle_manifest=manifest,
            analysis_provenance=_analysis_provenance(),
        )


def test_logging_only_state_analysis_rejects_checkpoint_not_in_manifest(tmp_path: Path) -> None:
    state = _state(update=1, epoch=99, margins=[0.1, -0.1], correct=[True, False])
    anchor, final = tmp_path / "epoch99.pt", tmp_path / "epoch199.pt"
    _checkpoint(anchor, epoch=99, store=state)
    _checkpoint(final, epoch=199, store=state)
    manifest = _manifest(tmp_path / "manifest.json", anchor=anchor, final=final)
    payload = torch.load(final, map_location="cpu", weights_only=False)
    payload["tampered_after_artifact_publication"] = True
    torch.save(payload, final)
    with pytest.raises(LoggingOnlyStateError, match="not one exact last-model artifact"):
        logging_only_state_analysis(
            anchor_checkpoint=anchor,
            final_checkpoint=final,
            expected_count=2,
            run_bundle_manifest=manifest,
            analysis_provenance=_analysis_provenance(),
        )


@pytest.mark.parametrize("status", ["running", "failed"])
def test_logging_only_state_analysis_rejects_nonterminal_manifest(tmp_path: Path, status: str) -> None:
    state = _state(update=1, epoch=99, margins=[0.1, -0.1], correct=[True, False])
    anchor, final = tmp_path / "epoch99.pt", tmp_path / "epoch199.pt"
    _checkpoint(anchor, epoch=99, store=state)
    _checkpoint(final, epoch=199, store=state)
    manifest = _manifest(tmp_path / "manifest.json", anchor=anchor, final=final, status=status)
    with pytest.raises(LoggingOnlyStateError, match="not terminal-success"):
        logging_only_state_analysis(
            anchor_checkpoint=anchor,
            final_checkpoint=final,
            expected_count=2,
            run_bundle_manifest=manifest,
            analysis_provenance=_analysis_provenance(),
        )


def test_logging_only_state_cli_requires_run_bundle_manifest() -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--anchor-checkpoint",
                "anchor.pt",
                "--final-checkpoint",
                "final.pt",
                "--expected-count",
                "2",
                "--output",
                "output.json",
            ]
        )
