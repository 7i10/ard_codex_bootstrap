from __future__ import annotations

import hashlib
import shutil
from types import SimpleNamespace

import pytest
import torch

from ard.analysis.history_early import _mid as h5_early_midrank
from ard.analysis.history_routing_v2 import Q, _random_ids, _route_selection
from ard.cli.train import _attach_history_routing_v2_input_artifacts
from ard.objectives import RSLADObjective
from ard.targets import TrueLabelMixTeacherTargetPolicy, UniformSofteningTeacherTargetPolicy
from ard.tracking import LocalTracker, TrackingError


def test_true_label_target_mix_is_selected_only_detached_and_preserves_clean_branch() -> None:
    labels = torch.tensor([0, 2], dtype=torch.long)
    teacher = torch.tensor([[2.0, -0.5, 0.3], [-0.1, 1.7, 0.2]], dtype=torch.float64, requires_grad=True)
    risk = torch.tensor([1.0, 0.0], dtype=torch.float64, requires_grad=True)
    target = TrueLabelMixTeacherTargetPolicy()(teacher_logits=teacher, risk=risk, temperature=2.0, labels=labels)
    expected_teacher = torch.softmax(teacher.detach() / 2.0, dim=1)
    expected = expected_teacher.clone()
    expected[0] = 0.5 * expected_teacher[0] + 0.5 * torch.nn.functional.one_hot(labels[0], 3).to(expected)
    torch.testing.assert_close(target.probabilities, expected, rtol=0, atol=0)
    torch.testing.assert_close(target.probabilities[1], expected_teacher[1], rtol=0, atol=0)
    assert target.rho.tolist() == [0.5, 0.0]
    assert not target.probabilities.requires_grad and not target.rho.requires_grad

    student_adv = torch.tensor([[0.2, -0.3, 0.6], [0.1, 0.4, -0.5]], dtype=torch.float64, requires_grad=True)
    student_clean = torch.tensor([[0.1, 0.5, -0.2], [0.4, -0.2, 0.2]], dtype=torch.float64, requires_grad=True)
    objective = RSLADObjective(temperature=2.0)
    baseline = objective(
        student_logits=student_adv, clean_student_logits=student_clean, teacher_logits=teacher.detach(), labels=labels
    )
    mixed = objective(
        student_logits=student_adv,
        clean_student_logits=student_clean,
        teacher_logits=teacher.detach(),
        labels=labels,
        adversarial_target_probabilities=target.probabilities,
    )
    assert baseline.clean_kd is not None and mixed.clean_kd is not None
    torch.testing.assert_close(mixed.clean_kd, baseline.clean_kd, rtol=0, atol=0)
    assert mixed.adversarial_kd is not None and baseline.adversarial_kd is not None
    torch.testing.assert_close(mixed.adversarial_kd[1], baseline.adversarial_kd[1], rtol=0, atol=1e-15)
    mixed.total.sum().backward()
    assert teacher.grad is None and risk.grad is None


def test_uniform_target_does_not_implicitly_consume_labels() -> None:
    with torch.no_grad():
        teacher = torch.randn(2, 3)
    try:
        UniformSofteningTeacherTargetPolicy(rho_max=0.5)(
            teacher_logits=teacher,
            risk=torch.zeros(2),
            temperature=1.0,
            labels=torch.zeros(2, dtype=torch.long),
        )
    except ValueError as exc:
        assert "does not consume labels" in str(exc)
    else:  # pragma: no cover - assertion with a useful failure instead of a silent compatibility drift.
        raise AssertionError("uniform target policy unexpectedly consumed labels")


def _state(*, correct: bool) -> dict[str, object]:
    records: dict[str, object] = {}
    for sample_id in range(20):
        records[str(sample_id)] = {
            "previous_robust_correct": correct if sample_id < 10 else not correct,
            "robust_correct_count": sample_id % 5,
            "seen": 40,
            "margin_ema": float(sample_id - 10) / 10,
            "true_label": sample_id % 2,
        }
    return {"records": records}


def test_epoch39_selector_uses_no_future_data_tie_breaks_and_matches_class_state_count_random() -> None:
    state = _state(correct=True)
    selected, eligible_labels, metadata = _route_selection(state, route="peak_failure")
    assert metadata["anchor_epoch"] == 39 and metadata["anchor_robust_correct"] is True
    assert metadata["selected_count"] == int(Q * metadata["eligible_count"])
    assert selected == sorted(selected)
    # A future-looking field must not affect the selection at all.
    for record in state["records"].values():
        assert isinstance(record, dict)
        record["future_forgetting"] = True
    repeated, labels_again, repeated_metadata = _route_selection(state, route="peak_failure")
    assert repeated == selected and labels_again == eligible_labels and repeated_metadata == metadata
    counts = {str(label): sum(eligible_labels[sample_id] == label for sample_id in selected) for label in {0, 1}}
    random_ids, seed = _random_ids(
        eligible_labels=eligible_labels,
        selected_counts=counts,
        parent_sha="a" * 64,
        route="peak_failure",
    )
    rerun, repeated_seed = _random_ids(
        eligible_labels=eligible_labels,
        selected_counts=counts,
        parent_sha="a" * 64,
        route="peak_failure",
    )
    assert random_ids == rerun and seed == repeated_seed
    assert {
        str(label): sum(eligible_labels[sample_id] == label for sample_id in random_ids) for label in {0, 1}
    } == counts
    wrong, _, wrong_metadata = _route_selection(state, route="non_recovery")
    assert set(selected).isdisjoint(wrong)
    assert wrong_metadata["anchor_robust_correct"] is False


def test_epoch39_selector_matches_h5_early_global_feature_rank_domain() -> None:
    state = _state(correct=True)
    records = state["records"]
    assert isinstance(records, dict)
    all_records = {int(sample_id): record for sample_id, record in records.items()}
    frequency = h5_early_midrank(
        {
            sample_id: 1 - int(record["robust_correct_count"]) / int(record["seen"])
            for sample_id, record in all_records.items()
        }
    )
    margin = h5_early_midrank({sample_id: -float(record["margin_ema"]) for sample_id, record in all_records.items()})
    eligible = [sample_id for sample_id, record in all_records.items() if record["previous_robust_correct"] is True]
    expected = sorted(
        sorted(eligible, key=lambda sample_id: (-(frequency[sample_id] + margin[sample_id]) / 2, sample_id))[
            : int(Q * len(eligible))
        ]
    )
    selected, _, metadata = _route_selection(state, route="peak_failure")
    assert selected == expected
    assert metadata["rank_population"] == "all_train_sample_state_records_before_anchor_correctness_route"


def test_history_routing_v2_resume_artifact_semantics(tmp_path) -> None:
    mask, bundle = tmp_path / "mask.json", tmp_path / "bundle.json"
    mask.write_text("{}", encoding="utf-8")
    bundle.write_text("{}", encoding="utf-8")
    config = SimpleNamespace(
        intervention=SimpleNamespace(
            arm="PF_TA",
            parent=SimpleNamespace(epoch=39),
            mask=SimpleNamespace(path=mask, sha256=hashlib.sha256(mask.read_bytes()).hexdigest()),
            selector_bundle_path=bundle,
            selector_bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
        )
    )

    class Tracker(LocalTracker):
        def __init__(self, root):
            self.run_id = "v2-run"
            self.bundle_dir = root / "run-bundle"
            self.bundle_dir.mkdir(parents=True)
            self.manifest = {"artifacts": []}

        def log_artifact(self, path, *, name, artifact_type, aliases=()):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            local = self.bundle_dir / "artifacts" / name / digest
            local.mkdir(parents=True)
            shutil.copy2(path, local / path.name)
            self.manifest["artifacts"].append(
                {
                    "name": name,
                    "type": artifact_type,
                    "aliases": list(aliases),
                    "sha256": digest,
                    "local_path": str(local.relative_to(self.bundle_dir)),
                }
            )

    lineage = {
        "kind": "history_routing_v2_intervention_v1",
        "arm": "PF_TA",
        "parent_epoch": 39,
        "child_tracker_run_id": "v2-run",
    }
    initial, later = tmp_path / "initial.pt", tmp_path / "later.pt"
    torch.save({"epoch": 39, "tracker_run_id": "v2-run", "fork_lineage": lineage}, initial)
    torch.save({"epoch": 40, "tracker_run_id": "v2-run", "fork_lineage": lineage}, later)
    tracker = Tracker(tmp_path / "initial")

    def rank_zero(active, *, phase, action):
        assert phase == "history-routing v2 input artifacts"
        action(active)

    _attach_history_routing_v2_input_artifacts(tracker=tracker, config=config, resume=initial, coordinator=rank_zero)
    assert [(entry["name"], entry["sha256"]) for entry in tracker.manifest["artifacts"]] == [
        ("history-routing-v2-mask-v2-run-pf_ta", config.intervention.mask.sha256),
        ("history-routing-v2-selector-v2-run", config.intervention.selector_bundle_sha256),
    ]
    _attach_history_routing_v2_input_artifacts(tracker=tracker, config=config, resume=initial, coordinator=rank_zero)
    assert len(tracker.manifest["artifacts"]) == 2
    _attach_history_routing_v2_input_artifacts(tracker=tracker, config=config, resume=later, coordinator=rank_zero)
    _attach_history_routing_v2_input_artifacts(
        tracker=tracker, config=config, resume=initial, coordinator=lambda *_args, **_kwargs: None
    )
    assert len(tracker.manifest["artifacts"]) == 2
    tracker.manifest["artifacts"].append({**tracker.manifest["artifacts"][0], "sha256": "0" * 64})
    with pytest.raises(TrackingError, match="conflicting local analysis-input"):
        _attach_history_routing_v2_input_artifacts(
            tracker=tracker, config=config, resume=initial, coordinator=rank_zero
        )
    for damaged in (Tracker(tmp_path / "absent"), Tracker(tmp_path / "mismatched")):
        if damaged.bundle_dir.parent.name == "mismatched":
            damaged.manifest["artifacts"] = [{**entry, "sha256": "0" * 64} for entry in tracker.manifest["artifacts"]]
        with pytest.raises(TrackingError, match="(lacks exact|conflicting) local analysis-input"):
            _attach_history_routing_v2_input_artifacts(
                tracker=damaged, config=config, resume=later, coordinator=rank_zero
            )
