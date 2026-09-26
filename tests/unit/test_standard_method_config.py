"""Plan 0103 option A: the ``standard`` (clean-training) method identity.

Schema refusal of contradictory ``standard`` configs, the new
``controlled_imagenet_stage02_clean_budget_v1`` protocol, and the two
ImageNet control configs being their PGD-AT counterparts with only method,
protocol.id and tracking.group changed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ard.cli.train import _build_method
from ard.config import load_config
from ard.config.schema import ExperimentConfig, MethodConfig
from ard.objectives import PGDATObjective
from ard.protocols import ensure_local_trainable
from ard.tracking.adapter import canonical_run_group

pytestmark = pytest.mark.t0

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "scientific"
SELECTION = {
    "loss": "ce",
    "epsilon": "4/255",
    "step_size": "8/765",
    "steps": 10,
    "random_start": True,
    "student_mode": "eval",
    "teacher_mode": "eval",
}


def _standard(**extra: Any) -> dict[str, Any]:
    return {"id": "standard", "version": 1, "selection_attack": dict(SELECTION), **extra}


def _experiment(method: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "protocol": {"id": "synthetic_smoke_v2"},
        "tier": "dev",
        "seeds": dict.fromkeys(
            (
                "split",
                "model_init",
                "data_order",
                "augmentation",
                "train_attack",
                "evaluation_attack",
                "qualitative_panel",
            ),
            3,
        ),
        "dataset": {"name": "synthetic_cifar", "num_samples": 8, "num_classes": 3, "image_size": 4},
        "student": {"architecture": "fixture_cnn", "num_classes": 3},
        "method": method,
        "optimizer": {"id": "sgd", "learning_rate": 0.01, "momentum": 0.9, "weight_decay": 0.0, "nesterov": False},
        "scheduler": {"id": "identity", "milestones": [], "gamma": 1.0, "step_at": "epoch_end"},
        "training": {"epochs": 1, "per_rank_batch_size": 2, "global_batch_size": 2, "device": "cpu"},
        **extra,
    }


def test_standard_has_no_training_attack_and_keeps_an_explicit_selection_attack() -> None:
    method = MethodConfig.model_validate(_standard())
    assert method.attack is None
    assert method.selection_attack is not None
    assert method.selection_attack.steps == 10 and method.selection_attack.loss == "ce"
    # The resolved form (``attack: null``) round-trips.
    dumped = method.model_dump(mode="json")
    assert dumped["attack"] is None
    assert MethodConfig.model_validate(dumped) == method


def test_standard_builds_the_pgd_at_ce_objective_with_label_smoothing() -> None:
    config = ExperimentConfig.model_validate(_experiment(_standard(label_smoothing=0.1)))
    objective, policy, sample_store, target_policy = _build_method(config)
    assert isinstance(objective, PGDATObjective) and objective.label_smoothing == 0.1
    assert policy is None and sample_store is None and target_policy is None


@pytest.mark.parametrize(
    ("method", "message"),
    [
        (_standard(attack={"loss": "ce", "epsilon": "4/255", "step_size": "8/765", "steps": 3}), "no training attack"),
        (_standard(attack={}), "no training attack"),
        ({"id": "standard", "version": 1}, "explicit method.selection_attack"),
        (_standard(selection_attack={**SELECTION, "loss": "kl", "kl_target": "student_clean"}), "hard-label CE"),
        (_standard(selection_attack={**SELECTION, "student_mode": "train"}), "eval mode"),
        (_standard(temperature=2.0), "does not use method.temperature"),
        (_standard(temperature_squared=False), "does not use method.temperature_squared"),
        (_standard(trades_beta=1.0), "does not use method.trades_beta"),
        (_standard(entropy_gamma=2.0), "does not use method.entropy_gamma"),
        (_standard(student_ema_decay=0.5), "does not use method.student_ema_decay"),
        (_standard(student_policy_warmup_epochs=2), "does not use method.student_policy_warmup_epochs"),
        (_standard(target_policy={"rho_max": 0.5}), "target_policy"),
        (
            _standard(adr={"ema_decay": 0.9, "temperature_high": 2.0, "temperature_low": 1.0, "lambda_low": 0.5}),
            "adr configuration",
        ),
        (_standard(oracle_mask=True), "oracle_mask"),
        (_standard(frozen_oracle_manifest="x.json", frozen_oracle_manifest_sha256="a" * 64), "frozen_oracle"),
    ],
)
def test_standard_refuses_fields_it_would_silently_ignore(method: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        MethodConfig.model_validate(method)


def test_other_methods_refuse_a_null_training_attack() -> None:
    with pytest.raises(ValidationError, match="requires a training attack"):
        MethodConfig.model_validate({"id": "pgd_at", "version": 1, "attack": None})
    # Omitting it keeps today's default attack exactly.
    assert MethodConfig.model_validate({"id": "pgd_at", "version": 1}).attack is not None


def test_standard_refuses_a_teacher_and_epsilon_warmup() -> None:
    teacher = {"source": "fixture", "architecture": "fixture_cnn", "num_classes": 3}
    with pytest.raises(ValidationError, match="without a teacher"):
        ExperimentConfig.model_validate(_experiment(_standard(), teacher=teacher))
    data = _experiment(_standard())
    data["training"]["epsilon_warmup_epochs"] = 1
    with pytest.raises(ValidationError, match="epsilon_warmup_epochs"):
        ExperimentConfig.model_validate(data)


def test_clean_budget_protocol_is_registered_and_restricted_to_standard() -> None:
    assert ensure_local_trainable("controlled_imagenet_stage02_clean_budget_v1").runnable_locally
    data = _experiment(_standard(), protocol={"id": "controlled_imagenet_stage02_clean_budget_v1"})
    assert ExperimentConfig.model_validate(data).method.id == "standard"
    data["method"] = {"id": "pgd_at", "version": 1}
    with pytest.raises(ValidationError, match="requires method standard"):
        ExperimentConfig.model_validate(data)


def test_standard_fails_closed_under_a_controlled_cifar_contract() -> None:
    data = _experiment(_standard(), protocol={"id": "controlled_cifar10_r18_v1"})
    with pytest.raises(ValidationError, match="defines no method without a training attack"):
        ExperimentConfig.model_validate(data)


def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for key, value in {
        "ARD_SEED": "7",
        "ARD_IMAGENET_ROOT": str(tmp_path / "imagenet"),
        "ARD_NUM_WORKERS": "0",
        "ARD_JOB_OUTPUT_DIR": str(tmp_path / "job-output"),
        "ARD_RUN_ID": "config-test-run",
        "WANDB_ENTITY": "entity",
        "WANDB_PROJECT": "project",
    }.items():
        monkeypatch.setenv(key, value)


@pytest.mark.parametrize("epochs", [50, 100])
def test_standard_configs_are_their_pgd_at_counterparts_except_method_protocol_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, epochs: int
) -> None:
    _env(monkeypatch, tmp_path)
    pgd_at_config = load_config(CONFIG_DIR / f"imagenet_mobilenetv4_pgd_at_random_init_{epochs}ep.yaml")
    standard_config = load_config(CONFIG_DIR / f"imagenet_mobilenetv4_standard_{epochs}ep.yaml")
    pgd_at = pgd_at_config.model_dump(mode="json")
    standard = standard_config.model_dump(mode="json")
    assert standard["method"]["id"] == "standard" and standard["method"]["attack"] is None
    assert standard["protocol"]["id"] == "controlled_imagenet_stage02_clean_budget_v1"
    assert standard["tracking"]["group"] == "imagenet-mobilenetv4-standard-clean-budget"
    assert standard["student"]["pretrained"] is False
    assert standard["training"]["epochs"] == epochs and standard["training"]["train_probe_size"] == 2000
    # Selection attack (PGD-10, 4/255) and evaluation attack unchanged.
    assert standard["method"]["selection_attack"] == pgd_at["method"]["selection_attack"]
    assert standard["method"]["selection_attack"]["steps"] == 10
    assert standard["evaluation"] == pgd_at["evaluation"]
    # Every other method field equals pgd_at's (defaults, label_smoothing 0).
    for payload in (pgd_at["method"], standard["method"]):
        payload["id"] = payload["attack"] = None
    assert pgd_at["method"] == standard["method"]
    for payload in (pgd_at, standard):
        payload["method"] = payload["protocol"] = None
        payload["tracking"] = {**payload["tracking"], "group": None}
    assert pgd_at == standard
    # Constructible on the real CLI path, and a teacherless production group.
    objective, _, _, _ = _build_method(standard_config)
    assert isinstance(objective, PGDATObjective)
    execution: dict[str, object] = {
        "world_size": 1,
        "per_rank_batch_size": 128,
        "global_batch_size": 128,
        "effective_global_batch_size": 128,
        "batchnorm_mode": "local_per_rank",
    }
    assert canonical_run_group(standard_config, training_execution=execution)


def test_standard_50_and_100_epoch_configs_differ_only_in_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _env(monkeypatch, tmp_path)
    short = load_config(CONFIG_DIR / "imagenet_mobilenetv4_standard_50ep.yaml").model_dump(mode="json")
    long = load_config(CONFIG_DIR / "imagenet_mobilenetv4_standard_100ep.yaml").model_dump(mode="json")
    assert short["scheduler"]["milestones"] == [25, 38] and long["scheduler"]["milestones"] == [50, 76]
    for payload in (short, long):
        payload["training"] = {**payload["training"], "epochs": None}
        payload["scheduler"] = {**payload["scheduler"], "milestones": None}
    assert short == long
