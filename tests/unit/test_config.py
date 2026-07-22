from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ard.config import load_config, save_resolved_config
from ard.config.schema import MethodConfig, NormalizationConfig

pytestmark = pytest.mark.t0


def base_config() -> dict:
    return {
        "schema_version": 2,
        "protocol": {"id": "synthetic_smoke_v2"},
        "tier": "dev",
        "seeds": {
            "split": 0,
            "model_init": 0,
            "data_order": 0,
            "augmentation": 0,
            "train_attack": 0,
            "evaluation_attack": 0,
            "qualitative_panel": 0,
        },
        "dataset": {"name": "synthetic_cifar", "num_classes": 10, "num_samples": 4},
        "student": {"architecture": "fixture_cnn", "num_classes": 10},
        "method": {"id": "pgd_at", "version": 1, "attack": {"epsilon": "8/255", "step_size": "2/255", "steps": 1}},
        "optimizer": {
            "id": "sgd",
            "learning_rate": 0.01,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "nesterov": False,
        },
        "scheduler": {"id": "identity", "milestones": [], "gamma": 1.0, "step_at": "epoch_end"},
        "training": {"epochs": 1, "per_rank_batch_size": 2, "global_batch_size": 2},
        "output_dir": "${ARD_TEST_OUTPUT}",
    }


def write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_strict_config_env_override_and_resolved_rationals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "config.yaml"
    write_yaml(path, base_config())
    monkeypatch.setenv("ARD_TEST_OUTPUT", str(tmp_path / "output"))

    config = load_config(path, ["seeds.model_init=7", "method.attack.random_start=false"])

    assert config.seed == 7
    assert config.method.attack.epsilon == "8/255"
    assert config.method.attack.epsilon_value == pytest.approx(8 / 255)
    assert config.method.attack.step_size == "2/255"
    assert config.method.attack.step_size_value == pytest.approx(2 / 255)
    assert not config.method.attack.random_start
    assert config.method.selection_attack is not None
    assert config.method.selection_attack.epsilon == config.method.attack.epsilon
    assert config.method.selection_attack.steps == config.method.attack.steps
    assert config.method.selection_attack.student_mode == "eval"
    assert config.method.selection_attack.teacher_mode == "eval"
    resolved = tmp_path / "resolved.yaml"
    save_resolved_config(config, resolved)
    reloaded = load_config(resolved)
    assert reloaded == config


def test_unknown_keys_and_unresolved_environment_are_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = base_config()
    data["scientific_typo"] = True
    path = tmp_path / "bad.yaml"
    write_yaml(path, data)
    monkeypatch.setenv("ARD_TEST_OUTPUT", str(tmp_path))
    with pytest.raises(ValidationError, match="scientific_typo"):
        load_config(path)
    monkeypatch.delenv("ARD_TEST_OUTPUT")
    data.pop("scientific_typo")
    write_yaml(path, data)
    with pytest.raises(ValueError, match="ARD_TEST_OUTPUT"):
        load_config(path)


@pytest.mark.parametrize("schema_version", (None, 1))
def test_schema_v1_or_missing_is_actionably_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, schema_version: int | None
) -> None:
    data = base_config()
    if schema_version is None:
        data.pop("schema_version")
    else:
        data["schema_version"] = schema_version
    path = tmp_path / "schema.yaml"
    write_yaml(path, data)
    monkeypatch.setenv("ARD_TEST_OUTPUT", str(tmp_path / "output"))
    with pytest.raises(ValidationError, match="schema_version.*exactly 2"):
        load_config(path)


def test_resolved_quantity_mismatch_and_unknown_override_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = base_config()
    data["method"]["attack"]["epsilon_value"] = 0.5
    path = tmp_path / "bad.yaml"
    write_yaml(path, data)
    monkeypatch.setenv("ARD_TEST_OUTPUT", str(tmp_path))
    with pytest.raises(ValidationError, match="epsilon_value"):
        load_config(path)
    data["method"]["attack"].pop("epsilon_value")
    write_yaml(path, data)
    with pytest.raises(ValidationError, match="unknown"):
        load_config(path, ["method.attack.unknown=1"])


def test_selection_attack_rejects_training_modes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = base_config()
    data["method"]["selection_attack"] = {
        "epsilon": "8/255",
        "step_size": "2/255",
        "steps": 1,
        "student_mode": "train",
    }
    path = tmp_path / "bad-selection.yaml"
    write_yaml(path, data)
    monkeypatch.setenv("ARD_TEST_OUTPUT", str(tmp_path / "output"))
    with pytest.raises(ValidationError, match="selection attack must keep"):
        load_config(path)


@pytest.mark.parametrize("name", ("cifar10", "cifar100"))
def test_cifar_rejects_nonexistent_validation_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    data = base_config()
    data["dataset"] = {
        "name": name,
        "root": str(tmp_path / "data"),
        "split": "val",
        "num_classes": 10 if name == "cifar10" else 100,
    }
    data["student"] = {
        "architecture": "resnet18_cifar",
        "num_classes": data["dataset"]["num_classes"],
        "normalization": {"profile": "cifar10_standard" if name == "cifar10" else "cifar100_standard"},
    }
    path = tmp_path / f"{name}-val.yaml"
    write_yaml(path, data)
    monkeypatch.setenv("ARD_TEST_OUTPUT", str(tmp_path / "output"))

    with pytest.raises(ValidationError, match="no validation split alias"):
        load_config(path)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("epsilon", "4/255"),
        ("step_size", "1/255"),
        ("steps", 2),
        ("random_start", False),
        ("temperature", 2.0),
        ("temperature_squared", False),
    ),
)
def test_selection_attack_rejects_threat_model_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    data = base_config()
    selection = dict(data["method"]["attack"])
    selection[field] = value
    data["method"]["selection_attack"] = selection
    path = tmp_path / f"bad-selection-{field}.yaml"
    write_yaml(path, data)
    monkeypatch.setenv("ARD_TEST_OUTPUT", str(tmp_path / "output"))
    with pytest.raises(ValidationError, match="selection attack must match"):
        load_config(path)


def test_m4_production_requires_tracking_and_unknown_lineage_bypass_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = base_config()
    data.update(
        {
            "tier": "production",
            "dataset": {"name": "cifar10", "root": str(tmp_path), "num_classes": 10},
            "student": {
                "architecture": "resnet18_cifar",
                "num_classes": 10,
                "normalization": {"profile": "cifar10_standard"},
            },
        }
    )
    path = tmp_path / "production.yaml"
    write_yaml(path, data)
    monkeypatch.setenv("ARD_TEST_OUTPUT", str(tmp_path / "output"))
    with pytest.raises(ValidationError, match="production requires non-disabled tracking"):
        load_config(path)

    data["tracking_lineage_available"] = True
    write_yaml(path, data)
    with pytest.raises(ValidationError, match="tracking_lineage_available"):
        load_config(path)

    data.pop("tracking_lineage_available")
    data["tier"] = "repro"
    data["tracking"] = {"mode": "offline_sync", "project": "ard-test"}
    write_yaml(path, data)
    assert load_config(path).tier == "repro"


def test_real_dataset_normalization_is_required_and_adapter_profiles_are_independent(tmp_path: Path) -> None:
    base = base_config()
    base.update(
        {
            "dataset": {"name": "cifar10", "root": str(tmp_path), "num_classes": 10},
            "student": {"architecture": "resnet18_cifar", "num_classes": 10},
            "output_dir": str(tmp_path / "output"),
        }
    )
    path = tmp_path / "normalization.yaml"
    write_yaml(path, base)
    with pytest.raises(ValidationError, match="requires student normalization profile cifar10_standard"):
        load_config(path)

    base["student"]["normalization"] = {"profile": "cifar100_standard"}
    write_yaml(path, base)
    with pytest.raises(ValidationError, match="requires student normalization profile cifar10_standard"):
        load_config(path)

    base["student"]["normalization"] = {"profile": "cifar10_standard"}
    base["teacher"] = {
        "source": "fixture",
        "architecture": "fixture_cnn",
        "num_classes": 10,
        "preprocessing_owner": "teacher_adapter",
        "normalization": {"profile": "cifar100_standard"},
    }
    write_yaml(path, base)
    assert load_config(path).teacher.normalization.profile == "cifar100_standard"

    base["teacher"]["preprocessing_owner"] = "unsupported_owner"
    write_yaml(path, base)
    with pytest.raises(ValidationError, match="preprocessing_owner"):
        load_config(path)


def test_synthetic_identity_and_cifar100_provenance_are_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "synthetic.yaml"
    write_yaml(path, base_config())
    monkeypatch.setenv("ARD_TEST_OUTPUT", str(tmp_path / "output"))
    config = load_config(path)
    assert config.student.normalization.profile == "fixture_unit"
    assert config.student.normalization.mean == (0.0, 0.0, 0.0)
    assert NormalizationConfig(profile="cifar100_standard").provenance == (
        "CIFAR-100 repository profile; not claimed upstream-exact"
    )


def test_repository_experiment_configs_resolve() -> None:
    """Every checked-in experiment is a directly loadable config fixture."""
    config_dir = Path(__file__).resolve().parents[2] / "configs" / "experiments"
    paths = sorted(config_dir.glob("*.yaml"))
    assert paths
    configs = {path.stem: load_config(path) for path in paths}
    assert "synthetic_rslad_student" in configs
    assert "synthetic_rslad_joint" in configs


def test_every_repository_yaml_is_parseable() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    paths = sorted(config_dir.rglob("*.yaml"))
    assert paths
    for path in paths:
        assert yaml.safe_load(path.read_text(encoding="utf-8")) is not None, path


def test_method_fragments_validate() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs" / "methods"
    paths = sorted(config_dir.glob("*.yaml"))
    assert paths
    for path in paths:
        MethodConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def test_top_level_configs_resolve_under_controlled_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    teacher_checkpoint = tmp_path / "teacher.pt"
    teacher_checkpoint.write_bytes(b"fixture checkpoint")
    values = {
        "ARD_SEED": "7",
        "ARD_CIFAR10_ROOT": str(tmp_path / "cifar10"),
        "ARD_TEACHER_CHECKPOINT": str(teacher_checkpoint),
        "ARD_TEACHER_CHECKPOINT_SHA256": "a" * 64,
        "ARD_LEARNING_RATE": "0.1",
        "ARD_MOMENTUM": "0.9",
        "ARD_WEIGHT_DECAY": "0.0005",
        "ARD_TRAIN_EPOCHS": "1",
        "ARD_BATCH_SIZE": "2",
        "ARD_NUM_WORKERS": "0",
        "ARD_DEVICE": "cpu",
        "ARD_VALIDATION_FRACTION": "0.25",
        "ARD_OUTPUT_ROOT": str(tmp_path / "outputs"),
        "WANDB_ENTITY": "entity",
        "WANDB_PROJECT": "project",
        "WANDB_GROUP": "group",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    paths = sorted(
        [
            *(config_dir / "experiments").glob("*.yaml"),
            *(config_dir / "reproduction").glob("*.yaml"),
            *(config_dir / "production").glob("*.yaml"),
        ]
    )
    assert paths
    for path in paths:
        config = load_config(path)
        assert config.output_dir
