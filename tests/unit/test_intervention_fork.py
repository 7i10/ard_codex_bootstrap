from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from ard.analysis.intervention_fork import (
    InterventionForkError,
    build_parent_artifact_attestation,
    create_intervention_forks,
)
from ard.config import ExperimentConfig
from ard.config.loader import resolved_config_dict
from ard.engine import config_digest
from ard.engine.checkpoint import REQUIRED_KEYS, capture_rng_state
from ard.objectives import RSLADObjective
from ard.policies import FixedMaskError, RSLADBaselinePolicy, load_fixed_intervention_mask, selected_ids_sha256
from ard.policies.base import PolicyContext
from ard.state import SampleRecord
from ard.targets import UniformSofteningTeacherTargetPolicy

pytestmark = pytest.mark.t2


def _state_equal(left: object, right: object) -> bool:
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_state_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_state_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _mask(
    path: Path, *, ids: list[int], labels: dict[int, int], provenance: dict[str, object], classes: int = 3
) -> dict[str, object]:
    provenance = {
        "selector_spec_path": None,
        "reference_history_selector_spec_sha256": None,
        **provenance,
    }
    sorted_ids = tuple(sorted(ids))
    counts: dict[str, int] = {}
    for sample_id in sorted_ids:
        class_id = str(labels[sample_id])
        counts[class_id] = counts.get(class_id, 0) + 1
    payload = {
        "schema_version": 1,
        "namespace": "train",
        "num_classes": classes,
        "selected_ids": list(sorted_ids),
        "selected_ids_sha256": selected_ids_sha256(sorted_ids),
        "selected_count": len(sorted_ids),
        "selected_class_counts": counts,
        "provenance": provenance,
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "selected_ids_sha256": payload["selected_ids_sha256"],
        "selected_count": payload["selected_count"],
        "selected_class_counts": counts,
        "provenance": provenance,
    }


def test_fixed_mask_rejects_non_train_ids_and_preserves_selected_class_budget(tmp_path: Path) -> None:
    labels = {0: 0, 1: 1, 2: 1, 3: 2}
    provenance = {
        "source": "seed0_bartoldson_frozen_predictor",
        "approved_selector_spec_sha256": "a" * 64,
        "parent_checkpoint_sha256": "b" * 64,
        "parent_sample_state_sha256": "c" * 64,
        "random_seed": None,
        "generator": None,
        "generator_version": None,
        "reference_history_mask_sha256": None,
        "reference_selected_count": None,
        "reference_selected_class_counts": None,
        "selector_spec_path": None,
        "reference_history_selector_spec_sha256": None,
    }
    config = _mask(tmp_path / "mask.json", ids=[1, 2], labels=labels, provenance=provenance)
    lookup = load_fixed_intervention_mask(
        Path(str(config["path"])),
        expected_sha256=str(config["sha256"]),
        expected_selected_ids_sha256=str(config["selected_ids_sha256"]),
        expected_selected_count=2,
        expected_class_counts={"1": 2},
        expected_provenance=provenance,
        train_labels=labels,
        num_classes=3,
    )
    assert torch.equal(
        lookup.values(torch.tensor([0, 1, 2, 3]), device=torch.device("cpu"), dtype=torch.float64),
        torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float64),
    )
    bad = json.loads(Path(str(config["path"])).read_text(encoding="utf-8"))
    bad["selected_ids"] = [1, 9]
    bad["selected_ids_sha256"] = selected_ids_sha256((1, 9))
    Path(str(config["path"])).write_text(json.dumps(bad, sort_keys=True), encoding="utf-8")
    with pytest.raises(FixedMaskError, match="SHA-256"):
        load_fixed_intervention_mask(
            Path(str(config["path"])),
            expected_sha256=str(config["sha256"]),
            expected_selected_ids_sha256=str(config["selected_ids_sha256"]),
            expected_selected_count=2,
            expected_class_counts={"1": 2},
            expected_provenance=provenance,
            train_labels=labels,
            num_classes=3,
        )


def test_registered_downweight_scales_only_adversarial_kd_and_softening_is_adv_only() -> None:
    labels = torch.tensor([0, 2])
    teacher = torch.tensor([[2.0, -0.5, 0.3], [-0.1, 1.7, 0.2]], dtype=torch.float64)
    base_adv = torch.tensor([[0.2, -0.3, 0.6], [0.1, 0.4, -0.5]], dtype=torch.float64, requires_grad=True)
    base_clean = torch.tensor([[0.1, 0.5, -0.2], [0.4, -0.2, 0.2]], dtype=torch.float64, requires_grad=True)
    objective = RSLADObjective(temperature=2.0)
    baseline = objective(
        student_logits=base_adv, clean_student_logits=base_clean, teacher_logits=teacher, labels=labels
    )
    weights = RSLADBaselinePolicy().compute(
        {}, context=PolicyContext(torch.tensor([True, True]), lambda value: value), num_classes=3
    )
    weighted = baseline.apply_policy(weights)
    downweighted = weighted.scale_adversarial_kd(torch.tensor([0.5, 1.0], dtype=torch.float64), coefficient=5.0 / 6.0)
    assert torch.equal(downweighted.hard, weighted.hard)
    assert downweighted.clean_kd is not None and weighted.clean_kd is not None
    assert torch.equal(downweighted.clean_kd, weighted.clean_kd)
    assert downweighted.adversarial_kd is not None and weighted.adversarial_kd is not None
    expected = weighted.kd - (5.0 / 6.0) * 0.5 * weighted.adversarial_kd * torch.tensor([1.0, 0.0])
    torch.testing.assert_close(downweighted.kd, expected, rtol=0, atol=1e-15)
    clean_gradient = torch.autograd.grad(downweighted.kd.sum(), base_clean, retain_graph=True)[0]
    baseline_clean_gradient = torch.autograd.grad(weighted.kd.sum(), base_clean, retain_graph=True)[0]
    torch.testing.assert_close(clean_gradient, baseline_clean_gradient, rtol=0, atol=1e-15)

    softened_target = UniformSofteningTeacherTargetPolicy(rho_max=0.5)(
        teacher_logits=teacher, risk=torch.tensor([1.0, 0.0], dtype=torch.float64), temperature=2.0
    )
    soft_adv = base_adv.detach().clone().requires_grad_()
    soft_clean = base_clean.detach().clone().requires_grad_()
    softened = objective(
        student_logits=soft_adv,
        clean_student_logits=soft_clean,
        teacher_logits=teacher,
        labels=labels,
        adversarial_target_probabilities=softened_target.probabilities,
    )
    assert softened.clean_kd is not None and baseline.clean_kd is not None
    torch.testing.assert_close(softened.clean_kd, baseline.clean_kd, rtol=0, atol=0)
    assert softened.adversarial_kd is not None and baseline.adversarial_kd is not None
    assert not torch.equal(softened.adversarial_kd[0], baseline.adversarial_kd[0])
    torch.testing.assert_close(softened.adversarial_kd[1], baseline.adversarial_kd[1], rtol=0, atol=1e-15)


def _parent_config(tmp_path: Path) -> dict[str, object]:
    return {
        "schema_version": 2,
        "protocol": {"id": "controlled_cifar10_r18_v1"},
        "tier": "dev",
        "seeds": {
            "split": 20260722,
            "model_init": 0,
            "data_order": 0,
            "augmentation": 0,
            "train_attack": 0,
            "evaluation_attack": 0,
            "qualitative_panel": 0,
        },
        "dataset": {"name": "cifar10", "root": str(tmp_path / "cifar"), "num_classes": 10},
        "student": {
            "architecture": "saad_resnet18_cifar_v1",
            "num_classes": 10,
            "normalization": {"profile": "cifar10_raw_identity"},
        },
        "teacher": {
            "source": "robustbench",
            "architecture": "robustbench_wide_resnet",
            "num_classes": 10,
            "normalization": {"profile": "robustbench_cifar10_bartoldson_embedded"},
            "preprocessing_owner": "model_embedded",
            "checkpoint": str(tmp_path / "teacher.pt"),
            "checkpoint_sha256": "e" * 64,
            "registry_id": "bartoldson2024_adversarial_wrn94_16",
        },
        "method": {
            "id": "rslad",
            "version": 1,
            "attack": {"loss": "kl", "kl_target": "teacher_clean"},
            "selection_attack": {"loss": "ce", "steps": 20},
        },
        "optimizer": {"id": "sgd", "learning_rate": 0.1, "momentum": 0.9, "weight_decay": 0.0005, "nesterov": False},
        "scheduler": {"id": "multistep", "milestones": [100, 150], "gamma": 0.1, "step_at": "epoch_end"},
        "training": {"epochs": 200, "per_rank_batch_size": 128, "global_batch_size": 128, "validation_fraction": 0.1},
        "observation": {"profile": "teacher_response"},
        "output_dir": str(tmp_path / "parent"),
        "intervention": None,
    }


def _complete_sample_state() -> dict[str, object]:
    record = asdict(
        SampleRecord(
            0.0,
            100,
            0,
            False,
            0,
            0,
            true_label=0,
            teacher_clean_entropy=0.1,
            teacher_clean_true_probability=0.8,
            teacher_clean_max_wrong_probability=0.1,
            teacher_clean_prediction=0,
            teacher_clean_correct=True,
            teacher_adversarial_entropy=0.2,
            teacher_adversarial_true_probability=0.7,
            teacher_adversarial_max_wrong_probability=0.2,
            teacher_adversarial_prediction=0,
            teacher_adversarial_correct=True,
            teacher_clean_to_adversarial_margin_response=-0.2,
            teacher_clean_to_adversarial_js_response=0.01,
            history_statistics_complete=True,
        )
    )
    return {
        "format_version": 3,
        "ema_decay": 0.9,
        "records": {str(index): copy.deepcopy(record) for index in range(45000)},
        "pending": [],
        "next_order": 0,
    }


def _write_fork_inputs(
    tmp_path: Path,
) -> tuple[Path, Path, Path, dict[str, object], dict[str, object], dict[str, object]]:
    sample_state = _complete_sample_state()
    rows = [[index, 0] for index in range(45000)]
    ids_hash = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    partition = tmp_path / "train-partition.json"
    partition.write_text(
        json.dumps({"schema_version": 1, "namespace": "train", "ids_labels": rows, "ids_labels_sha256": ids_hash}),
        encoding="utf-8",
    )
    parent_fields = {
        "sample_state_sha256": hashlib.sha256(
            json.dumps(sample_state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
        "train_partition_manifest": str(partition),
        "train_partition_manifest_sha256": hashlib.sha256(partition.read_bytes()).hexdigest(),
        "train_partition_ids_labels_sha256": ids_hash,
    }
    parent = ExperimentConfig.model_validate(_parent_config(tmp_path))
    raw = resolved_config_dict(parent)
    config_path = tmp_path / "parent-resolved.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    raw_hash = config_digest(raw)
    checkpoint = tmp_path / "parent.pt"
    payload = {
        "format_version": 1,
        "epoch": 99,
        "epoch_boundary": "end",
        "model": {"weight": torch.tensor([1.0])},
        "optimizer": {"state": {}, "param_groups": []},
        "scheduler": {},
        "scaler": None,
        "rng": [{**capture_rng_state(), "torch_cuda": [torch.tensor([0], dtype=torch.uint8)]}],
        "sampler_epoch": [99],
        "sampler_state": [{"epoch": 99, "seed": 0, "rank": 0, "world_size": 1, "shuffle": True}],
        "sample_state": sample_state,
        "global_step": 35_200,
        "best_metric": 0.7,
        "selection_metadata": {"metric": "val_pgd_accuracy", "selected_epoch": 42},
        "tracker_run_id": "parent-run",
        "config_hash": raw_hash,
        "world_size": 1,
    }
    assert REQUIRED_KEYS.issubset(payload)
    torch.save(payload, checkpoint)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "config_hash": raw_hash,
                "run_id": "parent-run",
                "git": {"sha": "a" * 40},
                "teacher": {"checkpoint_sha256": "e" * 64},
            }
        ),
        encoding="utf-8",
    )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "artifact": {
                    "name": "model-parent-run-last",
                    "version": "v19",
                    "digest": "d" * 32,
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                }
            }
        ),
        encoding="utf-8",
    )
    attestation = tmp_path / "attestation.json"
    attestation.write_text(
        json.dumps(
            build_parent_artifact_attestation(
                parent_manifest=manifest, artifact_inventory=inventory, checkpoint=checkpoint
            ),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    parent_fields["artifact_attestation"] = str(attestation)
    parent_fields["artifact_attestation_sha256"] = hashlib.sha256(attestation.read_bytes()).hexdigest()
    parent_fields["artifact_inventory"] = str(inventory)
    parent_fields["artifact_inventory_sha256"] = hashlib.sha256(inventory.read_bytes()).hexdigest()
    return checkpoint, config_path, manifest, raw, payload, parent_fields


def test_common_state_fork_preserves_parent_state_resets_best_and_records_lineage(tmp_path: Path) -> None:
    checkpoint, parent_config, manifest, raw, parent_payload, parent_fields = _write_fork_inputs(tmp_path)
    labels = {0: 0, 1: 0}
    parent_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    parent_hash = config_digest(raw)
    selector_spec = tmp_path / "selector.json"
    selector_spec.write_text(
        json.dumps(
            {
                "confirmatory_design_sha256": "a0a7fe0e70fcc8aaf519440012900c7bd8e6db92a8f0143d06892fca1146dd38",
                "predictor_spec_sha256": "d653d9ef08cfa94976a0e3279166b47543d16f3eaadb69810769470b77838c12",
                "seed0_report_sha256": "d44ee166f8866b77067ebd07757d394a060242c9cf1cdc5d4513f127897981f8",
                "seed0_lineage_sha256": "9b6ea091dc9ed4ff81bb579bf05d6650ac8e6d4ab6104981c446f29069e4a64e",
                "anchor_epoch": 99,
                "input_namespace": "train_sample_state_only",
                "coefficients_sha256": "a" * 64,
                "preprocessing_sha256": "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    history_provenance = {
        "source": "seed0_bartoldson_frozen_predictor",
        "approved_selector_spec_sha256": hashlib.sha256(selector_spec.read_bytes()).hexdigest(),
        "selector_spec_path": str(selector_spec),
        "parent_checkpoint_sha256": parent_sha,
        "parent_sample_state_sha256": parent_fields["sample_state_sha256"],
        "random_seed": None,
        "generator": None,
        "generator_version": None,
        "reference_history_mask_sha256": None,
        "reference_selected_count": None,
        "reference_selected_class_counts": None,
    }
    history_mask = _mask(tmp_path / "history.json", ids=[0], labels=labels, provenance=history_provenance, classes=10)
    random_mask = _mask(
        tmp_path / "random.json",
        ids=[1],
        labels=labels,
        provenance={
            "source": "class_matched_random",
            "approved_selector_spec_sha256": None,
            "parent_checkpoint_sha256": parent_sha,
            "parent_sample_state_sha256": parent_fields["sample_state_sha256"],
            "random_seed": 17,
            "generator": "numpy_pcg64",
            "generator_version": "1",
            "reference_history_mask_sha256": history_mask["sha256"],
            "reference_selected_count": 1,
            "reference_selected_class_counts": {"0": 1},
            "reference_history_selector_spec_sha256": hashlib.sha256(selector_spec.read_bytes()).hexdigest(),
        },
        classes=10,
    )
    arm_paths: list[Path] = []
    for arm, selector, kind, mask in (
        ("C", "none", "ordinary_rslad", None),
        ("HS", "student_history", "uniform_target_softening", history_mask),
        ("RS", "class_matched_random", "uniform_target_softening", random_mask),
        ("HD", "student_history", "adversarial_kd_downweight", history_mask),
        ("RD", "class_matched_random", "adversarial_kd_downweight", random_mask),
    ):
        child = copy.deepcopy(raw)
        child["output_dir"] = str(tmp_path / "screen" / arm)
        child["intervention"] = {
            "arm": arm,
            "selector": selector,
            "kind": kind,
            "parent": {
                "checkpoint_sha256": parent_sha,
                "raw_config_sha256": parent_hash,
                "git_sha": "a" * 40,
                "epoch": 99,
                "world_size": 1,
                "teacher_checkpoint_sha256": "e" * 64,
                "sample_state_records": 45000,
                **parent_fields,
            },
            "mask": mask,
        }
        path = tmp_path / f"{arm}.yaml"
        import yaml

        path.write_text(yaml.safe_dump(child), encoding="utf-8")
        arm_paths.append(path)
    created = create_intervention_forks(
        parent_checkpoint=checkpoint,
        parent_resolved_config=parent_config,
        parent_manifest=manifest,
        arm_config_paths=arm_paths,
        root=Path.cwd(),
        git_state_collector=lambda _root: {"sha": "b" * 40, "dirty": False},
    )
    assert set(created) == {"C", "HS", "RS", "HD", "RD"}
    child = torch.load(created["HD"], map_location="cpu", weights_only=False)
    for key in (
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "rng",
        "sampler_epoch",
        "sampler_state",
        "sample_state",
        "global_step",
    ):
        assert _state_equal(child[key], parent_payload[key]), key
    assert child["best_metric"] == float("-inf")
    assert child["selection_metadata"]["scope"] == "post_fork_best"
    assert child["fork_lineage"]["parent_checkpoint_sha256"] == parent_sha
    assert child["fork_lineage"]["parent_best_metric"] == parent_payload["best_metric"]
    assert child["fork_lineage"]["parent_selection_metadata"] == parent_payload["selection_metadata"]
    assert child["fork_lineage"]["parent_artifact_attestation_sha256"] == parent_fields["artifact_attestation_sha256"]
    assert child["fork_lineage"]["parent_wandb_checkpoint_artifact"] == {
        "name": "model-parent-run-last",
        "version": "v19",
        "digest": "d" * 32,
        "checkpoint_sha256": parent_sha,
    }
    screen = json.loads((created["HD"].parent.parent / "screen-complete.json").read_text(encoding="utf-8"))
    assert screen["status"] == "complete" and {entry["arm"] for entry in screen["arms"]} == {
        "C",
        "HS",
        "RS",
        "HD",
        "RD",
    }
    assert not (created["HD"].parent / "best.pt").exists()

    # A byte drift is rejected directly by the bound inventory SHA.  Restoring
    # those exact bytes but rebinding the arm configs to an alternate path must
    # still fail: the immutable attestation binds the original inventory path.
    inventory = Path(str(parent_fields["artifact_inventory"]))
    original_inventory = inventory.read_bytes()
    inventory.write_text(
        json.dumps(
            {
                "artifact": {
                    "name": "model-parent-run-last",
                    "version": "v19",
                    "digest": "0" * 32,
                    "checkpoint_sha256": parent_sha,
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(InterventionForkError, match="inventory bytes"):
        create_intervention_forks(
            parent_checkpoint=checkpoint,
            parent_resolved_config=parent_config,
            parent_manifest=manifest,
            arm_config_paths=arm_paths,
            root=Path.cwd(),
            git_state_collector=lambda _root: {"sha": "b" * 40, "dirty": False},
        )
    inventory.write_bytes(original_inventory)
    alternate_inventory = tmp_path / "inventory-relocated.json"
    alternate_inventory.write_bytes(original_inventory)
    import yaml

    for arm_path in arm_paths:
        arm_raw = yaml.safe_load(arm_path.read_text(encoding="utf-8"))
        arm_raw["intervention"]["parent"]["artifact_inventory"] = str(alternate_inventory)
        arm_raw["intervention"]["parent"]["artifact_inventory_sha256"] = hashlib.sha256(
            alternate_inventory.read_bytes()
        ).hexdigest()
        arm_path.write_text(yaml.safe_dump(arm_raw), encoding="utf-8")
    with pytest.raises(InterventionForkError, match="attestation does not bind"):
        create_intervention_forks(
            parent_checkpoint=checkpoint,
            parent_resolved_config=parent_config,
            parent_manifest=manifest,
            arm_config_paths=arm_paths,
            root=Path.cwd(),
            git_state_collector=lambda _root: {"sha": "b" * 40, "dirty": False},
        )


def test_common_state_fork_rejects_parent_hash_drift_before_creating_outputs(tmp_path: Path) -> None:
    checkpoint, parent_config, manifest, raw, _, parent_fields = _write_fork_inputs(tmp_path)
    child = copy.deepcopy(raw)
    child["output_dir"] = str(tmp_path / "screen" / "C")
    child["intervention"] = {
        "arm": "C",
        "selector": "none",
        "kind": "ordinary_rslad",
        "parent": {
            "checkpoint_sha256": "0" * 64,
            "raw_config_sha256": config_digest(raw),
            "git_sha": "a" * 40,
            "epoch": 99,
            "world_size": 1,
            "teacher_checkpoint_sha256": "e" * 64,
            "sample_state_records": 45000,
            **parent_fields,
        },
    }
    path = tmp_path / "C.yaml"
    import yaml

    path.write_text(yaml.safe_dump(child), encoding="utf-8")
    with pytest.raises(InterventionForkError, match="exactly the registered C/HS/RS/HD/RD"):
        create_intervention_forks(
            parent_checkpoint=checkpoint,
            parent_resolved_config=parent_config,
            parent_manifest=manifest,
            arm_config_paths=[path],
            root=Path.cwd(),
        )
    assert not (tmp_path / "arm").exists()
