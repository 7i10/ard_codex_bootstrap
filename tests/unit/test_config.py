from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ard.campaign.schema import load_campaign
from ard.config import load_config, save_resolved_config
from ard.config.schema import AttackConfig, MethodConfig, NormalizationConfig
from ard.config.teacher_audit import load_teacher_audit_config

pytestmark = pytest.mark.t0


def _set_repository_config_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, per_rank: int = 64) -> None:
    values = {
        "ARD_CIFAR10_ROOT": str(tmp_path / "cifar"),
        "ARD_OUTPUT_ROOT": str(tmp_path / "outputs"),
        "ARD_JOB_OUTPUT_DIR": str(tmp_path / "job-output"),
        "ARD_RUN_ID": "config-test-run",
        "ARD_PER_RANK_BATCH_SIZE": str(per_rank),
        "ARD_NUM_WORKERS": "0",
        "ARD_DEVICE": "cpu",
        "ARD_SEED": "1",
        "WANDB_ENTITY": "entity",
        "WANDB_PROJECT": "single-teacher-ard",
        "WANDB_GROUP_CHEN": "chen-comparison",
        "WANDB_GROUP_BARTOLDSON": "bartoldson-comparison",
        "ARD_TEACHER_CHEN2021_LTD_WRN34_10_CHECKPOINT": str(tmp_path / "chen.pt"),
        "ARD_TEACHER_CHEN2021_LTD_WRN34_10_CHECKPOINT_SHA256": "a" * 64,
        "ARD_TEACHER_BARTOLDSON2024_ADVERSARIAL_WRN94_16_CHECKPOINT": str(tmp_path / "bart.pt"),
        "ARD_TEACHER_BARTOLDSON2024_ADVERSARIAL_WRN94_16_CHECKPOINT_SHA256": "b" * 64,
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)


def test_experiment_taxonomy_and_execution_profiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_repository_config_env(monkeypatch, tmp_path)
    assert not list(Path("configs/reproduction").glob("*.yaml"))
    audit = sorted(Path("configs/audit").glob("*.yaml"))
    pilot = sorted(Path("configs/pilot").glob("*.yaml"))
    production = sorted(Path("configs/production").glob("*.yaml"))
    assert len(audit) == 2 and len(pilot) == 2 and len(production) == 8
    assert {load_teacher_audit_config(path).teacher.registry_id for path in audit} == {
        "chen2021_ltd_wrn34_10",
        "bartoldson2024_adversarial_wrn94_16",
    }
    pilot_configs = [load_config(path) for path in pilot]
    production_configs = [load_config(path) for path in production]
    assert {config.training.epochs for config in pilot_configs} == {5}
    assert {config.training.epochs for config in production_configs} == {200}
    assert {config.tier for config in pilot_configs} == {"pilot"}
    assert {config.tier for config in production_configs} == {"production"}
    assert all(config.training.global_batch_size == 128 for config in pilot_configs + production_configs)
    assert all(config.training.per_rank_batch_size == 64 for config in pilot_configs + production_configs)
    assert all(config.training.batchnorm_mode == "local_per_rank" for config in pilot_configs + production_configs)
    assert {config.teacher.registry_id for config in production_configs} == {
        "chen2021_ltd_wrn34_10",
        "bartoldson2024_adversarial_wrn94_16",
    }
    groups_by_teacher: dict[str, set[str]] = {}
    for config in production_configs:
        groups_by_teacher.setdefault(config.teacher.registry_id or "", set()).add(config.tracking.group or "")
    assert {teacher: len(groups) for teacher, groups in groups_by_teacher.items()} == {
        "chen2021_ltd_wrn34_10": 1,
        "bartoldson2024_adversarial_wrn94_16": 1,
    }
    assert next(iter(groups_by_teacher["chen2021_ltd_wrn34_10"])) != next(
        iter(groups_by_teacher["bartoldson2024_adversarial_wrn94_16"])
    )
    assert all("method" not in config.tracking.group.lower() for config in production_configs)
    assert all("seed" not in config.tracking.group.lower() for config in production_configs)


def test_single_gpu_campaign_configs_are_explicit_and_resolve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_repository_config_env(monkeypatch, tmp_path)
    pilots = [load_config(path) for path in sorted(Path("configs/pilot/single_gpu").glob("*.yaml"))]
    production = [load_config(path) for path in sorted(Path("configs/production/single_gpu").glob("*.yaml"))]
    assert len(pilots) == 3
    assert sorted(config.training.epochs for config in pilots) == [1, 1, 3]
    assert len(production) == 8
    assert {config.method.id for config in production} == {
        "rslad",
        "rslad_entropy",
        "rslad_student",
        "rslad_joint",
    }
    assert {config.teacher.registry_id for config in production if config.teacher is not None} == {
        "chen2021_ltd_wrn34_10",
        "bartoldson2024_adversarial_wrn94_16",
    }
    for config in [*pilots, *production]:
        assert config.training.per_rank_batch_size == 128
        assert config.training.global_batch_size == 128
        assert config.training.batchnorm_mode == "local_per_rank"
        assert config.tracking.mode == "online"
        assert config.tracking.run_id == "config-test-run"
        assert config.output_dir == tmp_path / "job-output"


def test_single_gpu_campaign_crosswalk_and_scientific_protocol_are_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_repository_config_env(monkeypatch, tmp_path)
    production = load_campaign(Path("configs/campaigns/five_gpu_single_process_v1.yaml"))
    canonical_by_cell = {}
    for path in sorted(Path("configs/production").glob("*.yaml")):
        config = load_config(path)
        assert config.teacher is not None
        canonical_by_cell[(config.teacher.registry_id, config.method.id)] = config
    assert len(canonical_by_cell) == 8

    for job in production.jobs:
        monkeypatch.setenv("WANDB_ENTITY", job.wandb.entity)
        monkeypatch.setenv("WANDB_PROJECT", job.wandb.project)
        monkeypatch.setenv("WANDB_GROUP_CHEN", job.wandb.group)
        monkeypatch.setenv("WANDB_GROUP_BARTOLDSON", job.wandb.group)
        monkeypatch.setenv("ARD_RUN_ID", job.wandb.run_id)
        config = load_config(Path(job.config))
        assert config.teacher is not None
        assert (config.teacher.registry_id, config.method.id) == (job.teacher, job.method)
        assert config.tracking.group == job.wandb.group
        assert config.tracking.project == job.wandb.project
        assert config.tracking.entity == job.wandb.entity
        canonical = canonical_by_cell[(job.teacher, job.method)]
        assert config.dataset == canonical.dataset
        assert config.student == canonical.student
        assert config.teacher == canonical.teacher
        assert config.method == canonical.method
        assert config.optimizer == canonical.optimizer
        assert config.scheduler == canonical.scheduler
        assert config.seeds == canonical.seeds
        assert config.training.epochs == canonical.training.epochs == 200
        assert config.training.global_batch_size == canonical.training.global_batch_size == 128
        assert config.training.per_rank_batch_size == 128
        assert canonical.training.per_rank_batch_size == 64
        assert config.method.attack.identity() == canonical.method.attack.identity()
        assert config.method.selection_attack is not None
        assert canonical.method.selection_attack is not None
        assert config.method.selection_attack.identity() == canonical.method.selection_attack.identity()
        assert job.phases.train == (
            "{PYTHON}",
            "-m",
            "ard.cli.train",
            "--config",
            "{CONFIG_PATH}",
            "--output",
            "{JOB_OUTPUT_DIR}",
        )


def test_two_gpu_profile_can_be_resolved_as_one_gpu_batch_128(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_repository_config_env(monkeypatch, tmp_path, per_rank=128)
    config = load_config(Path("configs/production/cifar10_r18_rslad_chen2021_ltd_wrn34_10.yaml"))
    assert config.training.per_rank_batch_size == 128
    assert config.training.global_batch_size == 128


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


def test_attack_trace_is_resolved_observability_not_threat_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attack = AttackConfig(trace_step_losses=True)
    assert attack.trace_step_losses
    assert "trace_step_losses" not in attack.identity()
    assert len(attack.identity()) == 14
    data = base_config()
    data["method"]["attack"]["trace_step_losses"] = True
    path = tmp_path / "trace.yaml"
    write_yaml(path, data)
    monkeypatch.setenv("ARD_TEST_OUTPUT", str(tmp_path / "output"))
    config = load_config(path)
    resolved = tmp_path / "resolved.yaml"
    save_resolved_config(config, resolved)
    assert yaml.safe_load(resolved.read_text(encoding="utf-8"))["method"]["attack"]["trace_step_losses"] is True


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
        ("random_start", False),
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
            # This test swaps the synthetic fixture for real CIFAR inputs but
            # deliberately does not claim a runnable controlled protocol.
            "protocol": {"id": "saad_code_295121c_audit_v1"},
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


@pytest.mark.parametrize("diagnostics_mode", ("off", "summary"))
def test_production_requires_panel_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, diagnostics_mode: str
) -> None:
    data = base_config()
    data.update(
        {
            "protocol": {"id": "saad_code_295121c_audit_v1"},
            "tier": "production",
            "dataset": {"name": "cifar10", "root": str(tmp_path), "num_classes": 10},
            "student": {
                "architecture": "resnet18_cifar",
                "num_classes": 10,
                "normalization": {"profile": "cifar10_standard"},
            },
            "tracking": {
                "mode": "online",
                "project": "ard-test",
                "entity": "ard-test",
                "group": "m3-test",
                "diagnostics_mode": diagnostics_mode,
            },
        }
    )
    path = tmp_path / "production-panel.yaml"
    write_yaml(path, data)
    monkeypatch.setenv("ARD_TEST_OUTPUT", str(tmp_path / "output"))
    with pytest.raises(ValidationError, match="production requires panel diagnostics"):
        load_config(path)


def test_real_dataset_normalization_is_required_and_adapter_profiles_are_independent(tmp_path: Path) -> None:
    base = base_config()
    base.update(
        {
            "protocol": {"id": "saad_code_295121c_audit_v1"},
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
        "ARD_TEACHER_CHEN2021_LTD_WRN34_10_CHECKPOINT": str(teacher_checkpoint),
        "ARD_TEACHER_CHEN2021_LTD_WRN34_10_CHECKPOINT_SHA256": "a" * 64,
        "ARD_TEACHER_BARTOLDSON2024_ADVERSARIAL_WRN94_16_CHECKPOINT": str(teacher_checkpoint),
        "ARD_TEACHER_BARTOLDSON2024_ADVERSARIAL_WRN94_16_CHECKPOINT_SHA256": "b" * 64,
        "ARD_PER_RANK_BATCH_SIZE": "128",
        "ARD_NUM_WORKERS": "0",
        "ARD_DEVICE": "cpu",
        "ARD_OUTPUT_ROOT": str(tmp_path / "outputs"),
        "ARD_JOB_OUTPUT_DIR": str(tmp_path / "job-output"),
        "ARD_RUN_ID": "config-test-run",
        "WANDB_ENTITY": "entity",
        "WANDB_PROJECT": "project",
        "WANDB_GROUP_CHEN": "chen-group",
        "WANDB_GROUP_BARTOLDSON": "bartoldson-group",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    paths = sorted(
        [
            *(config_dir / "experiments").glob("*.yaml"),
            *(config_dir / "pilot").glob("*.yaml"),
            *(config_dir / "production").glob("*.yaml"),
        ]
    )
    assert paths
    for path in paths:
        config = load_config(path)
        assert config.output_dir
