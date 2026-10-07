"""ImageNet teacher profiles and the distillation config contract (plan 0103 Phase 2, batch D)."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import pytest
import torch
from pydantic import ValidationError

from ard.config.distillation import IMAGENET_TEACHER_REGISTRY_IDS, DistillationConfig
from ard.config.schema import ExperimentConfig, NormalizationConfig, TeacherConfig
from ard.models import imagenet_teacher_registry as registry
from ard.models.imagenet_teacher_registry import (
    IMAGENET_TEACHER_SPECS,
    ImageNetTeacherRegistryError,
    build_imagenet_teacher_architecture,
    load_imagenet_teacher_network,
    read_verified_teacher_state,
    validate_imagenet_teacher_config,
)
from ard.models.teacher import TeacherAdapter, build_teacher

EXTERNAL = Path.home() / "workspace-local" / "external-checkpoints"
REAL_CHECKPOINTS = {
    "salman2020_resnet50_linf_eps4": EXTERNAL / "madrylab" / "resnet50_linf_eps4.0.ckpt",
    "singh2023_convnext_t_convstem": EXTERNAL
    / "singh2023_revisiting_at"
    / "convnext_t_cvst"
    / "convnext_tiny_cvst_robust.pt",
    "singh2023_vit_s_convstem": EXTERNAL / "singh2023_revisiting_at" / "vit_s_cvst" / "vit_s_cvst_robust.pt",
    "singh2023_convnext_b_convstem": EXTERNAL
    / "singh2023_revisiting_at"
    / "convnext_b_cvst"
    / "convnext_b_cvst_robust.pt",
    "ard_mobilenetv4_conv_medium_phase1_random": EXTERNAL
    / "ard_own"
    / "plan0103-phase1-mobilenetv4-conv-medium-random-v1"
    / "last.pt",
}


def test_registry_ids_match_schema_literal() -> None:
    assert set(IMAGENET_TEACHER_SPECS) == IMAGENET_TEACHER_REGISTRY_IDS
    for spec in IMAGENET_TEACHER_SPECS.values():
        assert spec.threat() == {"norm": "linf", "epsilon": "4/255", "input_domain": "pixel_0_1"}
        assert len(spec.checkpoint_sha256) == 64
        spec.normalization()  # every profile is a valid NormalizationConfig


@pytest.mark.parametrize("registry_id", sorted(IMAGENET_TEACHER_SPECS))
def test_architecture_parameter_counts(registry_id: str) -> None:
    spec = IMAGENET_TEACHER_SPECS[registry_id]
    model = build_imagenet_teacher_architecture(spec.architecture)
    assert sum(parameter.numel() for parameter in model.parameters()) == spec.expected_parameter_count
    with torch.no_grad():
        assert model.eval()(torch.rand(1, 3, 224, 224)).shape == (1, 1000)


def _spec_with(registry_id: str, path: Path, **changes: Any) -> registry.ImageNetTeacherSpec:
    import dataclasses

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return dataclasses.replace(IMAGENET_TEACHER_SPECS[registry_id], checkpoint_sha256=digest, **changes)


def _fake_singh(
    model: torch.nn.Module, *, normalize: tuple[list[float], list[float]] | None
) -> dict[str, torch.Tensor]:
    state = {f"base_model.{key}": value for key, value in model.state_dict().items()}
    if normalize is not None:
        state = {f"base_model.model.{key[len('base_model.') :]}": value for key, value in state.items()}
        state["base_model.normalize.mean"] = torch.tensor(normalize[0]).view(1, 3, 1, 1)
        state["base_model.normalize.std"] = torch.tensor(normalize[1]).view(1, 3, 1, 1)
    return state


def test_singh_raw_identity_round_trip_and_tamper(tmp_path: Path) -> None:
    model = build_imagenet_teacher_architecture("convnext_tiny_convstem_imagenet")
    path = tmp_path / "convnext_tiny_cvst_robust.pt"
    torch.save(_fake_singh(model, normalize=None), path)
    spec = _spec_with("singh2023_convnext_t_convstem", path)
    loaded = load_imagenet_teacher_network(spec, path)
    sample = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        assert torch.equal(loaded.eval()(sample), model.eval()(sample))
    # A normalizer embedded where the profile says raw pixels is refused.
    torch.save(_fake_singh(model, normalize=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])), path)
    with pytest.raises(ImageNetTeacherRegistryError, match="hash mismatch"):
        read_verified_teacher_state(spec, path)
    with pytest.raises(ImageNetTeacherRegistryError, match="embeds"):
        read_verified_teacher_state(_spec_with("singh2023_convnext_t_convstem", path), path)


def test_singh_embedded_normalizer_must_equal_profile_exactly(tmp_path: Path) -> None:
    model = build_imagenet_teacher_architecture("vit_s_convstem_imagenet")
    spec_values = IMAGENET_TEACHER_SPECS["singh2023_vit_s_convstem"]
    assert spec_values.custom_mean is not None and spec_values.custom_std is not None
    path = tmp_path / "vit_s_cvst_robust.pt"
    torch.save(_fake_singh(model, normalize=(list(spec_values.custom_mean), list(spec_values.custom_std))), path)
    state = read_verified_teacher_state(_spec_with("singh2023_vit_s_convstem", path), path)
    assert "normalize.mean" not in state and "cls_token" in state
    # The textbook ImageNet mean differs from the checkpoint's by <5e-5: still refused.
    torch.save(_fake_singh(model, normalize=([0.485, 0.456, 0.406], list(spec_values.custom_std))), path)
    with pytest.raises(ImageNetTeacherRegistryError, match="differs from the profile"):
        read_verified_teacher_state(_spec_with("singh2023_vit_s_convstem", path), path)


def test_madrylab_format_checks_normalizer_and_attacker_copy(tmp_path: Path) -> None:
    model = build_imagenet_teacher_architecture("resnet50_imagenet")
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    raw: dict[str, torch.Tensor] = {}
    for prefix in ("module.normalizer.", "module.attacker.normalize."):
        raw[f"{prefix}new_mean"], raw[f"{prefix}new_std"] = mean, std
    for key, value in model.state_dict().items():
        raw[f"module.model.{key}"] = value
        raw[f"module.attacker.model.{key}"] = value.clone()
    path = tmp_path / "resnet50_linf_eps4.0.ckpt"
    torch.save({"model": raw, "epoch": 90}, path)
    loaded = load_imagenet_teacher_network(_spec_with("salman2020_resnet50_linf_eps4", path), path)
    assert torch.equal(loaded.conv1.weight, model.conv1.weight)
    tampered = copy.copy(raw)
    tampered["module.attacker.model.fc.bias"] = raw["module.model.fc.bias"] + 1
    torch.save({"model": tampered}, path)
    with pytest.raises(ImageNetTeacherRegistryError, match="attacker copy"):
        read_verified_teacher_state(_spec_with("salman2020_resnet50_linf_eps4", path), path)


def test_ard_training_checkpoint_format(tmp_path: Path) -> None:
    from ard.models.registry import PixelModel

    net = build_imagenet_teacher_architecture("mobilenetv4_conv_medium_imagenet")
    wrapped = PixelModel(net, NormalizationConfig(profile="imagenet_standard"))
    path = tmp_path / "last.pt"
    torch.save({"model": wrapped.state_dict(), "epoch_boundary": "end"}, path)
    state = read_verified_teacher_state(_spec_with("ard_mobilenetv4_conv_medium_phase1_random", path), path)
    assert set(state) == set(net.state_dict())
    torch.save({"model": wrapped.state_dict(), "epoch_boundary": "mid"}, path)
    with pytest.raises(ImageNetTeacherRegistryError, match="epoch-boundary"):
        read_verified_teacher_state(_spec_with("ard_mobilenetv4_conv_medium_phase1_random", path), path)


def _teacher_config(profile_id: str, path: Path, **changes: Any) -> TeacherConfig:
    spec = IMAGENET_TEACHER_SPECS[profile_id]
    registry_id = profile_id
    payload: dict[str, Any] = {
        "source": "imagenet_registry",
        "registry_id": registry_id,
        "architecture": spec.architecture,
        "num_classes": 1000,
        "normalization": spec.normalization().model_dump(),
        "checkpoint": path,
        "checkpoint_sha256": spec.checkpoint_sha256,
        "threat_epsilon": "4/255",
    }
    payload.update(changes)
    return TeacherConfig.model_validate(payload)


def test_teacher_config_schema_rules(tmp_path: Path) -> None:
    path = tmp_path / "convnext_tiny_cvst_robust.pt"
    config = _teacher_config("singh2023_convnext_t_convstem", path)
    assert validate_imagenet_teacher_config(config).registry_id == "singh2023_convnext_t_convstem"
    with pytest.raises(ValidationError, match="4/255"):
        _teacher_config("singh2023_convnext_t_convstem", path, threat_epsilon="8/255")
    with pytest.raises(ValidationError, match="ImageNet registry ID"):
        _teacher_config("singh2023_convnext_t_convstem", path, registry_id="chen2021_ltd_wrn34_10")
    with pytest.raises(ValidationError, match="imagenet_registry"):
        TeacherConfig.model_validate({**config.model_dump(), "source": "robustbench", "threat_epsilon": "8/255"})
    wrong_norm = _teacher_config("singh2023_convnext_t_convstem", path, normalization={"profile": "imagenet_standard"})
    with pytest.raises(ImageNetTeacherRegistryError, match="normalization"):
        validate_imagenet_teacher_config(wrong_norm)
    wrong_name = _teacher_config("singh2023_convnext_t_convstem", tmp_path / "other.pt")
    with pytest.raises(ImageNetTeacherRegistryError, match="checkpoint filename"):
        validate_imagenet_teacher_config(wrong_name)


def test_build_teacher_is_frozen_eval_and_digest_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = build_imagenet_teacher_architecture("convnext_tiny_convstem_imagenet")
    path = tmp_path / "convnext_tiny_cvst_robust.pt"
    torch.save(_fake_singh(model, normalize=None), path)
    spec = _spec_with("singh2023_convnext_t_convstem", path)
    monkeypatch.setitem(registry.IMAGENET_TEACHER_SPECS, spec.registry_id, spec)  # type: ignore[index]
    config = _teacher_config(spec.registry_id, path, checkpoint_sha256=spec.checkpoint_sha256)
    teacher = build_teacher(config, tier="production")
    assert isinstance(teacher, TeacherAdapter)
    assert not teacher.training and all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    teacher.train(True)
    assert not teacher.training and not teacher.model.training
    assert teacher.metadata.threat_model == {"norm": "linf", "epsilon": "4/255", "input_domain": "pixel_0_1"}
    # raw-identity profile: the adapter applies no shift/scale.
    pixels = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        assert torch.equal(teacher(pixels), model.eval()(pixels))
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ImageNetTeacherRegistryError, match="hash mismatch"):
        build_teacher(config, tier="production")


@pytest.mark.parametrize("registry_id", sorted(REAL_CHECKPOINTS))
def test_real_checkpoints_strict_load_when_present(registry_id: str) -> None:
    path = REAL_CHECKPOINTS[registry_id]
    if not path.is_file():
        pytest.skip(f"real checkpoint not on this host: {path}")
    spec = IMAGENET_TEACHER_SPECS[registry_id]
    assert path.stat().st_size == spec.checkpoint_bytes
    load_imagenet_teacher_network(spec, path)


# --------------------------------------------------------------------------
# Distillation config contract.
# --------------------------------------------------------------------------


def experiment_payload(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    spec = IMAGENET_TEACHER_SPECS["salman2020_resnet50_linf_eps4"]
    attack = {"loss": "kl", "kl_target": "teacher_clean", "epsilon": "4/255", "step_size": "8/765", "steps": 1}
    payload: dict[str, Any] = {
        "schema_version": 2,
        "protocol": {"id": "imagenet_stage0_dev_v1"},
        "tier": "dev",
        "seeds": {
            "split": 1,
            "model_init": 2,
            "data_order": 3,
            "augmentation": 4,
            "train_attack": 5,
            "evaluation_attack": 0,
            "qualitative_panel": 6,
        },
        "dataset": {"name": "imagenet", "root": str(tmp_path), "num_classes": 4, "image_size": 16},
        "student": {"architecture": "fixture_cnn", "num_classes": 4, "normalization": {"profile": "imagenet_standard"}},
        "teacher": {
            "source": "imagenet_registry",
            "registry_id": spec.registry_id,
            "architecture": spec.architecture,
            "num_classes": 4,
            "normalization": {"profile": "imagenet_standard"},
            "checkpoint": str(tmp_path / spec.checkpoint_filename),
            "checkpoint_sha256": spec.checkpoint_sha256,
            "threat_epsilon": "4/255",
        },
        "method": {"id": "rslad", "version": 1, "attack": attack},
        "optimizer": {"id": "sgd", "learning_rate": 0.1, "momentum": 0.9, "weight_decay": 0.0, "nesterov": False},
        "scheduler": {"id": "identity", "milestones": [], "gamma": 1.0, "step_at": "epoch_end"},
        "training": {"epochs": 2, "per_rank_batch_size": 4, "global_batch_size": 4, "validation_fraction": 0.25},
        "distillation": {"target_source": "online_teacher"},
    }
    for key, value in overrides.items():
        payload[key] = value
    return payload


def test_distillation_block_rules(tmp_path: Path) -> None:
    config = ExperimentConfig.model_validate(experiment_payload(tmp_path))
    assert config.distillation == DistillationConfig(target_source="online_teacher")
    with pytest.raises(ValidationError, match="explicit distillation block"):
        ExperimentConfig.model_validate({**experiment_payload(tmp_path), "distillation": None})
    bank = {"path": str(tmp_path / "bank"), "manifest_sha256": "e" * 64, "top_k": 2}
    with pytest.raises(ValidationError, match="only valid with"):
        ExperimentConfig.model_validate(
            experiment_payload(tmp_path, distillation={"target_source": "online_teacher", "bank": bank})
        )
    banked = ExperimentConfig.model_validate(
        experiment_payload(tmp_path, distillation={"target_source": "soft_label_bank", "bank": bank})
    )
    assert banked.distillation is not None
    assert banked.distillation.protocol_identity(banked.teacher) == {
        "target_source": "soft_label_bank",
        "teacher_registry_id": "salman2020_resnet50_linf_eps4",
        "teacher_checkpoint_sha256": IMAGENET_TEACHER_SPECS["salman2020_resnet50_linf_eps4"].checkpoint_sha256,
        "bank_storage": "ard-soft-label-bank-v1/top_k_marginal_smoothing_v1/fp16",
        "bank_top_k": 2,
    }
    # Seeds of one arm (different per-seed bank digests) pool; K or source do not.
    other_seed_bank = {**bank, "manifest_sha256": "d" * 64}
    other = ExperimentConfig.model_validate(
        experiment_payload(tmp_path, distillation={"target_source": "soft_label_bank", "bank": other_seed_bank})
    )
    assert other.distillation is not None
    assert other.distillation.protocol_identity(other.teacher) == banked.distillation.protocol_identity(banked.teacher)
    assert other.distillation.run_lineage() != banked.distillation.run_lineage()
    from ard.config.distillation import BANK_STORAGE_IDENTITY
    from ard.distillation.soft_label_bank import BANK_FORMAT, STORAGE

    assert BANK_STORAGE_IDENTITY == f"{BANK_FORMAT}/{STORAGE}/fp16"
    heavy = experiment_payload(tmp_path, distillation={"target_source": "soft_label_bank", "bank": bank})
    heavy["dataset"] = {**heavy["dataset"], "imagenet_heavy_augmentation": True}
    with pytest.raises(ValidationError, match="heavy_augmentation"):
        ExperimentConfig.model_validate(heavy)
    hot = experiment_payload(tmp_path, distillation={"target_source": "soft_label_bank", "bank": bank})
    hot["method"] = {**hot["method"], "temperature": 2.0}
    with pytest.raises(ValidationError, match="temperature=1"):
        ExperimentConfig.model_validate(hot)
    observed = experiment_payload(tmp_path, observation={"profile": "teacher_response"})
    with pytest.raises(ValidationError, match="observation.profile=off"):
        ExperimentConfig.model_validate(observed)
    advt = experiment_payload(tmp_path)
    advt["method"] = {**advt["method"], "id": "rslad_advt"}
    assert ExperimentConfig.model_validate(advt).method.id == "rslad_advt"
    advt_no_block = {**advt, "distillation": None}
    advt_no_block["teacher"] = {**advt["teacher"]}
    with pytest.raises(ValidationError, match="distillation"):
        ExperimentConfig.model_validate(advt_no_block)


def test_distillation_serialized_only_when_set(tmp_path: Path) -> None:
    payload = experiment_payload(tmp_path)
    payload.pop("teacher")
    payload.pop("distillation")
    payload["method"] = {
        "id": "pgd_at",
        "version": 1,
        "attack": {"loss": "ce", "epsilon": "4/255", "step_size": "8/765", "steps": 1},
    }
    dumped = ExperimentConfig.model_validate(payload).model_dump(mode="json")
    assert "distillation" not in dumped
