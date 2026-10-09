"""``training.cuda_graph`` (plan 0105): default off, byte-identical configs, fail-closed scope,
the stale-graph fingerprint, static-buffer admission, and an unchanged ``LinfPGD.generate``
around the new sync-free core.

The GPU parity and guard differentials live in
``tests/integration/test_cuda_graph_training_step.py``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from torch import nn
from torch.optim import SGD, Adam, AdamW

import ard.config.schema as schema_module
from ard.attacks import AttackRequest, LinfPGD
from ard.cli.evaluate import _throughput_protocol_identity
from ard.config import load_config
from ard.config.loader import _expand_environment, resolved_config_dict
from ard.config.schema import (
    CUDA_GRAPH_ARCHITECTURES,
    CUDA_GRAPH_IDENTITY_RECORDED_ARCHITECTURES,
    CUDA_GRAPH_TEACHER_ARCHITECTURES,
    AttackConfig,
    AwpConfig,
    ExperimentConfig,
    MixedBatchConfig,
    TrainingConfig,
    reject_throughput_options,
)
from ard.device_checks import require
from ard.distillation.soft_label_bank import SoftLabelBank, SoftLabelBankTeacher
from ard.distillation.trainer_hooks import DistillationTargetHooks
from ard.engine.checkpoint import config_digest
from ard.engine.cuda_graph import (
    CudaGraphStepState,
    module_fingerprint,
    momentum_buffers_ready,
    optimizer_fingerprint,
    optimizer_groups_fingerprint,
    optimizer_state_ready,
)
from ard.engine.mixed_batch import AuxiliaryBatchNorm
from ard.engine.trainer import Trainer
from ard.models.teacher import TeacherAdapter, TeacherMetadata
from ard.objectives import PGDATObjective, RSLADObjective
from ard.policies import RSLADBaselinePolicy

pytestmark = pytest.mark.t1

ROOT = Path(__file__).resolve().parents[2]
# The commit this change is based on: the pre-change schema and configs.
PRE_CHANGE_COMMIT = "4fd5ceb"
STAGE1 = ROOT / "configs" / "scientific" / "imagenet_mobilenetv4_twostage_stage1_112_pgd1_resized_s256.yaml"
# LinfPGD.generate outputs of the pre-change attack (PRE_CHANGE_COMMIT's
# src/ard/attacks/pgd.py) on the seeded cases below, as SHA-256 digests.
# Regenerate only from that file: ``python tests/unit/test_cuda_graph_config.py``.
GOLDEN = Path(__file__).with_name("fixtures") / "pgd_generate_golden_4fd5ceb.json"
_SGD_MNV4S = {"optimizer_id": "sgd", "student_architecture": "mobilenetv4_conv_small_imagenet"}


def _identity_keys(config: ExperimentConfig) -> dict[str, str]:
    return {"optimizer_id": config.optimizer.id, "student_architecture": config.student.architecture}


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:  # pragma: no cover - shallow clones
        pytest.skip(f"git {' '.join(args)} unavailable: {error}")


def _pre_change_module(relative: str, name: str, package: str) -> types.ModuleType:
    source = _git("show", f"{PRE_CHANGE_COMMIT}:{relative}")
    module = types.ModuleType(name)
    module.__package__ = package
    module.__file__ = f"{PRE_CHANGE_COMMIT}:{relative}"
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Every environment variable a repository config interpolates (as in test_imagenet_loader_speedups)."""
    teacher_checkpoint = tmp_path / "teacher.pt"
    teacher_checkpoint.write_bytes(b"fixture checkpoint")
    for key, value in {
        "ARD_TEACHER_CHEN2021_LTD_WRN34_10_CHECKPOINT": str(teacher_checkpoint),
        "ARD_TEACHER_CHEN2021_LTD_WRN34_10_CHECKPOINT_SHA256": "a" * 64,
        "ARD_TEACHER_BARTOLDSON2024_ADVERSARIAL_WRN94_16_CHECKPOINT": str(teacher_checkpoint),
        "ARD_TEACHER_BARTOLDSON2024_ADVERSARIAL_WRN94_16_CHECKPOINT_SHA256": "b" * 64,
        "ARD_PER_RANK_BATCH_SIZE": "128",
        "ARD_DEVICE": "cpu",
        "ARD_OUTPUT_ROOT": str(tmp_path / "outputs"),
        "ARD_STAGEWISE_SWITCH_EPOCH": "100",
        "ARD_STAGEWISE_LATE_POLICY": "crop_re",
        "WANDB_GROUP_STAGEWISE": "stagewise-group",
        "WANDB_GROUP_CHEN": "chen-group",
        "WANDB_GROUP_BARTOLDSON": "bartoldson-group",
        "WANDB_GROUP_BARTOLDSON_ORACLE": "bartoldson-oracle-group",
        "ARD_FROZEN_ORACLE_MANIFEST": str(tmp_path / "frozen-oracle.json"),
        "ARD_FROZEN_ORACLE_MANIFEST_SHA256": "c" * 64,
        "ARD_SEED": "7",
        "ARD_IMAGENET_ROOT": str(tmp_path / "imagenet"),
        "ARD_IMAGENET_TRAIN_ROOT": str(tmp_path / "imagenet_train_derived"),
        "ARD_IMAGENET100_ROOT": str(tmp_path / "imagenet100"),
        "ARD_CIFAR10_ROOT": str(tmp_path / "cifar10"),
        "ARD_NUM_WORKERS": "0",
        "ARD_JOB_OUTPUT_DIR": str(tmp_path / "job-output"),
        "ARD_RUN_ID": "config-test-run",
        "WANDB_ENTITY": "entity",
        "WANDB_PROJECT": "project",
        "ARD_STAGE1_CHECKPOINT": str(tmp_path / "stage1" / "last.pt"),
        "ARD_STAGE1_CHECKPOINT_SHA256": "d" * 64,
        "ARD_SOFT_LABEL_BANK_ROOT": str(tmp_path / "banks"),
        "ARD_SOFT_LABEL_BANK_SHA256_SALMAN_R50": "e" * 64,
        "ARD_SOFT_LABEL_BANK_SHA256_CONVNEXT_B_CVST": "f" * 64,
        "ARD_EXTERNAL_CHECKPOINT_ROOT": str(tmp_path / "external"),
    }.items():
        monkeypatch.setenv(key, value)


# =========================================================================== config


def test_default_off_and_unserialized() -> None:
    training = TrainingConfig(per_rank_batch_size=4, global_batch_size=4)
    assert training.cuda_graph is False
    assert "cuda_graph" not in json.loads(training.model_dump_json())


def test_configs_that_existed_before_the_change_serialize_exactly_as_under_the_pre_change_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every config as it was at PRE_CHANGE_COMMIT (content read from git, so later edits
    or new configs cannot break this) resolves and hashes byte-identically."""
    _env(monkeypatch, tmp_path)
    old_schema = _pre_change_module("src/ard/config/schema.py", "_ard_schema_pre_cuda_graph", "ard.config")
    listed = _git("ls-tree", "-r", "--name-only", PRE_CHANGE_COMMIT, "configs/").splitlines()
    paths = [
        path
        for path in listed
        if path.endswith(".yaml") and path.split("/")[1] in {"experiments", "pilot", "production", "scientific"}
    ]
    assert len(paths) > 20
    for path in paths:
        expanded = _expand_environment(yaml.safe_load(_git("show", f"{PRE_CHANGE_COMMIT}:{path}")))
        new = ExperimentConfig.model_validate(expanded)
        old = old_schema.ExperimentConfig.model_validate(expanded)
        assert resolved_config_dict(new) == json.loads(old.model_dump_json()), path
        assert config_digest(resolved_config_dict(new)) == config_digest(json.loads(old.model_dump_json()))


def _stage1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    _env(monkeypatch, tmp_path)
    return resolved_config_dict(load_config(STAGE1))


def _with(raw: dict[str, Any], **sections: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(raw)
    for section, values in sections.items():
        updated[section] = {**updated[section], **values}
    return updated


def test_stage1_config_accepts_cuda_graph_and_records_it_in_the_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _stage1(monkeypatch, tmp_path)
    enabled = ExperimentConfig.model_validate(_with(raw, training={"cuda_graph": True}))
    assert enabled.training.cuda_graph is True
    assert resolved_config_dict(enabled)["training"]["cuda_graph"] is True
    assert config_digest(resolved_config_dict(enabled)) != config_digest(raw)
    # A saved resolved config reloads to the same setting.
    assert ExperimentConfig.model_validate(resolved_config_dict(enabled)).training.cuda_graph is True


def test_allowlist_is_the_parity_tested_students() -> None:
    assert CUDA_GRAPH_ARCHITECTURES == {
        "mobilenetv4_conv_small_imagenet",
        "mobilenetv4_conv_medium_imagenet",
        "efficientnet_b0_imagenet",
        # 2026-10-09: the ConvNeXt-Atto / DeiT-Tiny family (SGD and AdamW parity-tested).
        "convnext_atto_imagenet",
        "convnext_atto_deep_narrow_imagenet",
        "convnext_atto_ols_imagenet",
        "convnext_atto_convstem_imagenet",
        "deit_tiny_imagenet",
        "deit_tiny_convstem_imagenet",
    }


@pytest.mark.parametrize(
    ("training", "reason"),
    [
        ({"device": "auto"}, "training.device=cuda"),
        ({"amp": True}, "training.amp=false"),
        ({"deterministic": False, "compile": True}, "training.compile=false"),
        ({"deterministic": False, "cudnn_benchmark": True}, "training.cudnn_benchmark=false"),
        ({"step_diagnostics": True}, "training.step_diagnostics=false"),
        ({"global_batch_size": 256}, "world size 1"),
        ({"epsilon_warmup_epochs": 5}, "training.epsilon_warmup_epochs unset"),
    ],
)
def test_training_scope_fails_closed(training: dict[str, Any], reason: str) -> None:
    base = {
        "epochs": 10,
        "per_rank_batch_size": 128,
        "global_batch_size": 128,
        "device": "cuda",
        "step_diagnostics": False,
    }
    TrainingConfig(**base, cuda_graph=True)
    with pytest.raises(ValueError, match="training.cuda_graph=true requires") as refused:
        TrainingConfig(**{**base, **training}, cuda_graph=True)
    assert reason in str(refused.value)


@pytest.mark.parametrize("deterministic", [True, False])
def test_weight_ema_and_label_smoothing_are_admitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, deterministic: bool
) -> None:
    """Plan 0103 Phase 2 batch A (2026-10-08): the plain weight EMA is updated inside the captured
    step and label smoothing is part of the captured objective; both are parity-tested in
    tests/integration/test_cuda_graph_training_step.py. The EMA stays refused with ADR."""
    base = {"per_rank_batch_size": 4, "global_batch_size": 4, "device": "cuda", "step_diagnostics": False}
    assert TrainingConfig(**base, deterministic=deterministic, weight_ema_decay=0.999, cuda_graph=True).cuda_graph
    raw = _stage1(monkeypatch, tmp_path)
    enabled = ExperimentConfig.model_validate(
        _with(
            raw,
            training={"deterministic": deterministic, "weight_ema_decay": 0.999, "cuda_graph": True},
            method={"label_smoothing": 0.1},
        )
    )
    assert enabled.training.weight_ema_decay == 0.999 and enabled.method.label_smoothing == 0.1
    # Deterministic: still out of the pooling identity (bitwise parity); otherwise recorded.
    assert _throughput_protocol_identity(enabled.training, **_identity_keys(enabled)) == (
        {} if deterministic else {"cuda_graph": True}
    )


@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize(
    "method",
    [{"mixed_batch": {"adversarial_fraction": 0.5, "adversarial_weight": 0.3}}, {"awp": {"gamma": 0.01}}],
)
def test_mixed_batch_and_awp_are_admitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: dict[str, Any], deterministic: bool
) -> None:
    """Human decision 2026-10-08: transcribed in the captured step and parity-tested (bitwise when
    deterministic, the one-step rule otherwise) in tests/integration/test_cuda_graph_training_step.py.
    AWP is admitted only in deterministic mode (its nondeterministic one-step check could not resolve it)."""
    raw = _stage1(monkeypatch, tmp_path)
    updated = _with(raw, method=method, training={"cuda_graph": True, "deterministic": deterministic})
    if "awp" in method and not deterministic:
        with pytest.raises(ValueError, match="method.awp only with training.deterministic=true"):
            ExperimentConfig.model_validate(updated)
        return
    enabled = ExperimentConfig.model_validate(updated)
    assert enabled.training.cuda_graph
    assert _throughput_protocol_identity(enabled.training, **_identity_keys(enabled)) == (
        {} if deterministic else {"cuda_graph": True}
    )


def test_split_batchnorm_stays_outside_the_graph_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _stage1(monkeypatch, tmp_path)
    method = {"mixed_batch": {"adversarial_fraction": 0.5, "adversarial_weight": 0.3, "split_batchnorm": True}}
    ExperimentConfig.model_validate(_with(raw, method=method))
    with pytest.raises(ValueError, match="training.cuda_graph=true requires") as refused:
        ExperimentConfig.model_validate(_with(raw, method=method, training={"cuda_graph": True}))
    assert "no method.mixed_batch.split_batchnorm" in str(refused.value)


_RSLAD_BANK = ROOT / "configs" / "scientific" / "imagenet_mobilenetv4_rslad_phase2_salman_r50_fkd.yaml"
_RSLAD_ADVT_BANK = ROOT / "configs" / "scientific" / "imagenet_mobilenetv4_rslad_advt_phase2_convnext_b_cvst_fkd.yaml"


def _phase2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: Path) -> dict[str, Any]:
    _env(monkeypatch, tmp_path)
    return resolved_config_dict(load_config(path))


@pytest.mark.parametrize("path", [_RSLAD_BANK, _RSLAD_ADVT_BANK])
@pytest.mark.parametrize("deterministic", [True, False])
def test_rslad_and_rslad_advt_distillation_are_admitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: Path, deterministic: bool
) -> None:
    """Human decision 2026-10-08: ImageNet RSLAD / RSLAD-advT (bank target; advT's teacher forward inside the
    step) with an allowlisted student and teacher."""
    raw = _phase2(monkeypatch, tmp_path, path)
    enabled = ExperimentConfig.model_validate(_with(raw, training={"cuda_graph": True, "deterministic": deterministic}))
    assert enabled.training.cuda_graph and enabled.method.id in {"rslad", "rslad_advt"}
    assert _throughput_protocol_identity(enabled.training, **_identity_keys(enabled)) == (
        {} if deterministic else {"cuda_graph": True}
    )
    online = _with(raw, training={"cuda_graph": True}, distillation={"target_source": "online_teacher", "bank": None})
    online["distillation"].pop("bank")
    if enabled.method.id == "rslad":
        assert ExperimentConfig.model_validate(online).distillation.target_source == "online_teacher"
    else:
        # Review of d2e82b2 (P2-1): online advT is not parity-tested, so it is refused with the graph.
        online_without_graph = copy.deepcopy(online)
        online_without_graph["training"].pop("cuda_graph")
        ExperimentConfig.model_validate(online_without_graph)
        with pytest.raises(ValueError, match="training.cuda_graph=true requires") as refused:
            ExperimentConfig.model_validate(online)
        assert "rslad_advt with distillation.target_source=soft_label_bank" in str(refused.value)


@pytest.mark.parametrize("target_source", ["soft_label_bank", "online_teacher"])
@pytest.mark.parametrize("where", ["method", "attack"])
def test_rslad_at_a_temperature_other_than_1_is_refused_with_the_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target_source: str, where: str
) -> None:
    """Review of d2e82b2 (P2-1): only temperature 1 is parity-tested (the bank requires it anyway)."""
    raw = _phase2(monkeypatch, tmp_path, _RSLAD_BANK)
    if target_source == "online_teacher":
        raw = _with(raw, distillation={"target_source": "online_teacher"})
        raw["distillation"].pop("bank")
    if where == "method":
        raw["method"]["temperature"] = 2.0
    else:
        raw["method"]["attack"]["temperature"] = 2.0
    graph = _with(raw, training={"cuda_graph": True})
    with pytest.raises(ValueError) as refused:
        ExperimentConfig.model_validate(graph)
    if target_source == "online_teacher":
        ExperimentConfig.model_validate(raw)  # valid without the graph
        assert "method.temperature=1 and method.attack.temperature=1 for rslad/rslad_advt" in str(refused.value)


@pytest.mark.parametrize(("path", "refused"), [(_RSLAD_BANK, False), (_RSLAD_ADVT_BANK, True)])
def test_a_teacher_inside_the_step_must_be_allowlisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: Path, refused: bool
) -> None:
    """Plain bank RSLAD never runs the teacher; advT (and an online target) runs it inside the step."""
    raw = _with(_phase2(monkeypatch, tmp_path, path), training={"cuda_graph": True})
    monkeypatch.setattr(schema_module, "CUDA_GRAPH_TEACHER_ARCHITECTURES", frozenset())
    if not refused:
        ExperimentConfig.model_validate(raw)
        online = _with(raw, distillation={"target_source": "online_teacher"})
        online["distillation"].pop("bank")
        raw = online
    with pytest.raises(ValueError, match="training.cuda_graph=true requires") as error:
        ExperimentConfig.model_validate(raw)
    assert "a parity-tested teacher.architecture" in str(error.value)


def test_allowlisted_teachers_are_the_phase2_imagenet_teachers() -> None:
    from ard.models.imagenet_teacher_registry import IMAGENET_TEACHER_SPECS

    assert CUDA_GRAPH_TEACHER_ARCHITECTURES == {spec.architecture for spec in IMAGENET_TEACHER_SPECS.values()}


def test_rslad_scope_still_refuses_a_sample_keyed_attack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _with(_phase2(monkeypatch, tmp_path, _RSLAD_BANK), training={"cuda_graph": True})
    keyed = copy.deepcopy(raw)
    keyed["method"]["attack"]["random_start_keying"] = "sample_keyed_v1"
    with pytest.raises(ValueError, match="random_start_keying=batch"):
        ExperimentConfig.model_validate(keyed)


def test_sgd_norm_bias_exclusion_is_admitted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two SGD parameter groups (weight decay 0 on ndim <= 1); parity-tested with the graph."""
    raw = _stage1(monkeypatch, tmp_path)
    enabled = ExperimentConfig.model_validate(
        _with(raw, optimizer={"exclude_norm_bias_from_weight_decay": True}, training={"cuda_graph": True})
    )
    assert enabled.optimizer.exclude_norm_bias_from_weight_decay


@pytest.mark.parametrize("training", [{"deterministic": False}])
def test_nondeterministic_mode_admits_cuda_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, training: dict[str, Any]
) -> None:
    """Human decision 2026-10-03: deterministic=false may use the graph (cuDNN benchmark may not);
    exact RNG streams and one-step equivalence (within 4x max(eager spread, FP32 rounding) of the nearest eager
    outcome) are tested in tests/integration/test_cuda_graph_training_step.py."""
    base = {"per_rank_batch_size": 4, "global_batch_size": 4, "device": "cuda", "step_diagnostics": False}
    assert TrainingConfig(**base, **training, cuda_graph=True).cuda_graph is True
    raw = _stage1(monkeypatch, tmp_path)
    enabled = ExperimentConfig.model_validate(_with(raw, training={**training, "cuda_graph": True}))
    assert enabled.training.deterministic is False and enabled.training.cuda_graph is True
    # Every other scope restriction still holds in nondeterministic mode.
    with pytest.raises(ValueError, match="training.cuda_graph=true requires") as refused:
        TrainingConfig(**base, **training, amp=True, cuda_graph=True)
    assert "training.amp=false" in str(refused.value)
    off_list = _with(
        raw, training={**training, "cuda_graph": True}, student={"architecture": "mobilenet_v3_small_imagenet"}
    )
    with pytest.raises(ValueError, match="a parity-tested student.architecture"):
        ExperimentConfig.model_validate(off_list)


@pytest.mark.parametrize(
    ("sections", "reason"),
    [
        ({"student": {"architecture": "mobilenet_v3_small_imagenet"}}, "a parity-tested student.architecture"),
        ({"attack": {"student_mode": "train"}}, "method.attack.student_mode=eval"),
        ({"attack": {"random_start_keying": "sample_keyed_v1"}}, "random_start_keying=batch"),
        ({"attack": {"trace_step_losses": True}}, "trace_step_losses=false"),
        ({"attack": {"epsilon": "0", "step_size": "1/255"}}, "0 < epsilon"),
        ({"attack": {"epsilon": "2/255", "step_size": "4/255"}}, "step_size <= epsilon"),
    ],
)
def test_experiment_scope_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sections: dict[str, Any], reason: str
) -> None:
    raw = _with(_stage1(monkeypatch, tmp_path), training={"cuda_graph": True})
    attack = sections.pop("attack", None)
    updated = _with(raw, **sections)
    if attack is not None:
        updated["method"]["attack"] = {**updated["method"]["attack"], **attack}
        for key in ("epsilon_value", "step_size_value"):
            if key.split("_value")[0] in attack:
                updated["method"]["attack"].pop(key, None)
        if "epsilon" in attack:
            # Selection and evaluation must keep the training threat model.
            updated["method"]["selection_step_size_independent"] = False
            for container, key in ((updated["method"], "selection_attack"), (updated["evaluation"], "attack")):
                container[key] = {**container[key], "epsilon": attack["epsilon"], "step_size": attack["step_size"]}
                container[key].pop("epsilon_value", None)
                container[key].pop("step_size_value", None)
    ExperimentConfig.model_validate(raw)
    try:
        without_graph = copy.deepcopy(updated)
        without_graph["training"].pop("cuda_graph")
        ExperimentConfig.model_validate(without_graph)
    except ValueError as error:  # pragma: no cover - the fixture edit itself must be valid
        pytest.fail(f"fixture edit is invalid even without cuda_graph: {error}")
    with pytest.raises(ValueError, match="training.cuda_graph=true requires") as refused:
        ExperimentConfig.model_validate(updated)
    assert reason in str(refused.value)


def test_runtimes_other_than_the_trainer_cli_refuse_cuda_graph() -> None:
    training = TrainingConfig(
        per_rank_batch_size=4, global_batch_size=4, device="cuda", step_diagnostics=False, cuda_graph=True
    )
    with pytest.raises(ValueError) as refused:
        reject_throughput_options(training, runtime="fixture")
    assert str(refused.value) == (
        "fixture does not implement training.step_diagnostics=false, training.cuda_graph=true; run it with "
        "training.step_diagnostics=true, training.cuda_graph=false (only ard.cli.train applies these options)"
    )


def test_cuda_graph_pooling_identity_depends_on_the_determinism_class() -> None:
    """Deterministic: out of the pooling identity, only because the scope checks above confine it to
    the parity-tested allowlist / eval-mode attack where it is bitwise equal to eager (plan 0105).
    Nondeterministic: recorded (equal only within FP32 rounding, not bitwise), so a graph run never pools
    silently with a nondeterministic eager run. (cudnn_benchmark, recorded when true, is refused with it.)"""
    base = {"per_rank_batch_size": 4, "global_batch_size": 4, "device": "cuda", "step_diagnostics": False}
    assert _throughput_protocol_identity(TrainingConfig(**base, cuda_graph=True), **_SGD_MNV4S) == {}
    assert _throughput_protocol_identity(TrainingConfig(**base), **_SGD_MNV4S) == {}
    nondeterministic = {**base, "deterministic": False}
    assert _throughput_protocol_identity(TrainingConfig(**nondeterministic, cuda_graph=True), **_SGD_MNV4S) == {
        "cuda_graph": True
    }
    assert _throughput_protocol_identity(TrainingConfig(**nondeterministic), **_SGD_MNV4S) == {}
    benchmark = {**nondeterministic, "cudnn_benchmark": True}
    assert _throughput_protocol_identity(TrainingConfig(**benchmark), **_SGD_MNV4S) == {"cudnn_benchmark": True}


def test_adamw_cuda_graph_is_recorded_in_both_determinism_classes() -> None:
    """2026-10-09: a cuda_graph AdamW run uses torch's capturable AdamW (device bias corrections), bitwise equal to
    its own eager steps but not to the capturable=False AdamW of an eager run, so it is recorded even when
    deterministic. SGD keeps its rule; without the flag nothing is recorded (existing identities unchanged)."""
    base = {"per_rank_batch_size": 4, "global_batch_size": 4, "device": "cuda", "step_diagnostics": False}
    for deterministic in (True, False):
        training = {**base, "deterministic": deterministic}
        graph = TrainingConfig(**training, cuda_graph=True)
        mnv4s = "mobilenetv4_conv_small_imagenet"
        assert _throughput_protocol_identity(graph, optimizer_id="adamw", student_architecture=mnv4s) == {
            "cuda_graph": True
        }
        assert (
            _throughput_protocol_identity(TrainingConfig(**training), optimizer_id="adamw", student_architecture=mnv4s)
            == {}
        )
        assert _throughput_protocol_identity(graph, optimizer_id="sgd", student_architecture=mnv4s) == (
            {} if deterministic else {"cuda_graph": True}
        )
        # Review of 1f17fa2 (P2-2): the ConvNeXt-Atto / DeiT-Tiny family is recorded with SGD too, until the
        # production-shape bitwise SGD check has passed for it; without the flag nothing is recorded.
        for architecture in CUDA_GRAPH_IDENTITY_RECORDED_ARCHITECTURES:
            assert _throughput_protocol_identity(graph, optimizer_id="sgd", student_architecture=architecture) == {
                "cuda_graph": True
            }
            assert (
                _throughput_protocol_identity(
                    TrainingConfig(**training), optimizer_id="sgd", student_architecture=architecture
                )
                == {}
            )
    assert _throughput_protocol_identity(TrainingConfig(**base, cuda_graph=True), **_SGD_MNV4S) == {}


_ADAMW_PHASE2_CONFIGS = sorted(
    path.name
    for path in (ROOT / "configs" / "scientific").glob("imagenet_*.yaml")
    if path.name.startswith(("imagenet_convnext_atto", "imagenet_deit_tiny"))
    and "optimizer: {id: adamw" in path.read_text()
)


@pytest.mark.parametrize(
    "name",
    [
        "imagenet_convnext_atto_pgd_at_phase2_cosine.yaml",
        "imagenet_convnext_atto_rslad_phase2_convnext_b_cvst_fkd.yaml",
        "imagenet_deit_tiny_pgd_at_phase2_mixed_kurakin.yaml",
    ],
)
def test_adamw_convnext_and_deit_configs_admit_cuda_graph_without_cudnn_benchmark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """The Phase 2 AdamW ConvNeXt-Atto / DeiT-Tiny configs (deterministic=false, cudnn_benchmark=true today) are
    admitted with cuda_graph once cudnn_benchmark is off -- and refused with it on."""
    assert name in _ADAMW_PHASE2_CONFIGS
    _env(monkeypatch, tmp_path)
    monkeypatch.setenv("ARD_PHASE2_ADAMW_LR_DEIT_TINY", "5e-4")
    raw = resolved_config_dict(load_config(ROOT / "configs" / "scientific" / name))
    assert raw["optimizer"]["id"] == "adamw" and raw["training"]["cudnn_benchmark"] is True
    with pytest.raises(ValueError, match="training.cudnn_benchmark=false"):
        ExperimentConfig.model_validate(_with(raw, training={"cuda_graph": True}))
    enabled = ExperimentConfig.model_validate(_with(raw, training={"cuda_graph": True, "cudnn_benchmark": False}))
    assert enabled.training.cuda_graph is True and enabled.optimizer.id == "adamw"


def test_adamw_is_built_capturable_exactly_with_cuda_graph(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ard.cli.train: without training.cuda_graph the AdamW call is the pre-change one (torch defaults, so
    capturable=False and host bias corrections: no existing run changes); with it, capturable=True."""
    from ard.cli.train import _build_optimizer

    _env(monkeypatch, tmp_path)
    raw = resolved_config_dict(
        load_config(ROOT / "configs" / "scientific" / "imagenet_convnext_atto_pgd_at_phase2_cosine.yaml")
    )
    model = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4), nn.Linear(4, 2))
    eager = _build_optimizer(ExperimentConfig.model_validate(raw), list(model.parameters()))
    reference = AdamW(
        [
            {"params": [p for p in model.parameters() if p.ndim > 1], "weight_decay": raw["optimizer"]["weight_decay"]},
            {"params": [p for p in model.parameters() if p.ndim <= 1], "weight_decay": 0.0},
        ],
        lr=raw["optimizer"]["learning_rate"],
        betas=(raw["optimizer"]["beta1"], raw["optimizer"]["beta2"]),
    )
    assert type(eager) is AdamW
    strip = lambda groups: [{k: v for k, v in g.items() if k != "params"} for g in groups]  # noqa: E731
    assert strip(eager.param_groups) == strip(reference.param_groups)
    assert all(group["capturable"] is False for group in eager.param_groups)
    graph_config = ExperimentConfig.model_validate(_with(raw, training={"cuda_graph": True, "cudnn_benchmark": False}))
    graph = _build_optimizer(graph_config, list(model.parameters()))
    assert all(group["capturable"] is True for group in graph.param_groups)
    assert strip(graph.param_groups) == [{**group, "capturable": True} for group in strip(reference.param_groups)]


# =========================================================================== trainer scope


def _trainer_kwargs(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    model = nn.Sequential(nn.Flatten(), nn.Linear(12, 3))
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
    attack = LinfPGD(AttackConfig(epsilon="4/255", step_size="4/255", steps=1))
    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": None,
        "scaler": None,
        "attack": attack,
        "selection_attack": attack,
        "objective": PGDATObjective(),
        "device": torch.device("cpu"),
        "output_dir": tmp_path,
        "config_hash": "c" * 64,
        "seed": 0,
        "step_diagnostics": False,
        "cuda_graph": True,
        "student_architecture": "mobilenetv4_conv_small_imagenet",
        **overrides,
    }


def _refusal(tmp_path: Path, **overrides: Any) -> str:
    with pytest.raises(ValueError, match="cuda_graph requires") as refused:
        Trainer(**_trainer_kwargs(tmp_path, **overrides))
    return str(refused.value)


class _PGDSubclass(LinfPGD):
    pass


def test_trainer_refuses_cuda_graph_outside_its_scope(tmp_path: Path) -> None:
    Trainer(**{**_trainer_kwargs(tmp_path), "cuda_graph": False})
    message = _refusal(tmp_path, step_diagnostics=True, weight_ema_decay=0.99)
    for reason in ("a CUDA device", "step_diagnostics=False"):
        assert reason in message
    # A plain weight EMA is in scope (plan 0103 Phase 2 batch A); since 2026-10-08 so are mixed batch
    # (without split BN) and AWP: only the CPU device is refused for them here.
    assert "EMA" not in message
    mixed = MixedBatchConfig(adversarial_fraction=0.5, adversarial_weight=0.3)
    assert _refusal(tmp_path, mixed_batch=mixed) == "cuda_graph requires a CUDA device"
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        assert _refusal(tmp_path, awp=AwpConfig()) == "cuda_graph requires a CUDA device"
        # AWP with the graph only under deterministic algorithms (nondeterministic one-step check unresolved).
        torch.use_deterministic_algorithms(False)
        assert "AWP only with deterministic algorithms" in _refusal(tmp_path, awp=AwpConfig())
    finally:
        torch.use_deterministic_algorithms(previous)
    # Only the CPU device is out of scope in the base fixture.
    assert "parity-tested student architecture" not in _refusal(tmp_path)
    assert "LinfPGD (not a subclass)" not in _refusal(tmp_path)
    keyed = LinfPGD(AttackConfig(epsilon="4/255", step_size="4/255", random_start_keying="sample_keyed_v1"))
    assert "an eval-mode, batch-keyed" in _refusal(tmp_path, attack=keyed)
    kl = LinfPGD(AttackConfig(epsilon="4/255", step_size="4/255", loss="kl", kl_target="teacher_clean"))
    assert "a CE training attack (PGD-AT)" in _refusal(tmp_path, attack=kl)


def test_trainer_refuses_split_batchnorm(tmp_path: Path) -> None:
    model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.Flatten(), nn.Linear(16, 3))
    mixed = MixedBatchConfig(adversarial_fraction=0.5, adversarial_weight=0.3, split_batchnorm=True)
    auxiliary = AuxiliaryBatchNorm(model)
    optimizer = SGD([*model.parameters(), *auxiliary.parameters()], lr=0.1, momentum=0.9)
    message = _refusal(tmp_path, model=model, optimizer=optimizer, mixed_batch=mixed, auxiliary_batchnorm=auxiliary)
    assert "no split BN" in message


def _fixture_teacher(architecture: str = "resnet50_imagenet") -> TeacherAdapter:
    metadata = TeacherMetadata(
        architecture=architecture,
        num_classes=3,
        normalization={"profile": "fixture_unit"},  # type: ignore[arg-type]
        checkpoint_sha256="0" * 64,
    )
    return TeacherAdapter(nn.Sequential(nn.Flatten(), nn.Linear(12, 3)), metadata)


def _bank_teacher(online: TeacherAdapter | None = None) -> SoftLabelBankTeacher:
    bank = SoftLabelBank(Path("in-memory"), {"top_k": 2, "num_classes": 3, "epoch_records": {}}, np.arange(4))
    return SoftLabelBankTeacher(bank, online_teacher=online)


def _rslad(tmp_path: Path, *, teacher: nn.Module, advt: bool = False, **overrides: Any) -> str:
    kl = LinfPGD(AttackConfig(epsilon="4/255", step_size="4/255", steps=1, loss="kl", kl_target="teacher_clean"))
    return _refusal(
        tmp_path,
        **{
            "attack": kl,
            "objective": RSLADObjective(),
            "policy": RSLADBaselinePolicy(),
            "teacher": teacher,
            "distillation_hooks": DistillationTargetHooks(adversarial_teacher_target=advt, temperature=1.0),
            **overrides,
        },
    )


def test_trainer_admits_rslad_from_a_bank_and_an_allowlisted_frozen_teacher(tmp_path: Path) -> None:
    only_device = "cuda_graph requires a CUDA device"
    assert _rslad(tmp_path, teacher=_bank_teacher()) == only_device
    assert _rslad(tmp_path, teacher=_bank_teacher(_fixture_teacher()), advt=True) == only_device
    assert _rslad(tmp_path, teacher=_fixture_teacher()) == only_device


def test_trainer_refuses_rslad_outside_its_scope(tmp_path: Path) -> None:
    in_step = "a frozen eval-mode teacher with a parity-tested architecture inside the step"
    assert in_step in _rslad(tmp_path, teacher=_fixture_teacher("fixture_cnn"))
    assert in_step in _rslad(tmp_path, teacher=_bank_teacher(_fixture_teacher("fixture_cnn")), advt=True)
    # A bank without advT never runs the teacher: any (even off-list) online teacher is irrelevant there.
    assert in_step not in _rslad(tmp_path, teacher=_bank_teacher(_fixture_teacher("fixture_cnn")))
    trainable = _fixture_teacher()
    nn.Module.train(trainable, True)  # bypass the adapter's eval-only override
    assert in_step in _rslad(tmp_path, teacher=trainable)
    assert "an advT online teacher" in _rslad(tmp_path, teacher=_bank_teacher(), advt=True)
    assert "RSLAD baseline policy" in _rslad(tmp_path, teacher=_bank_teacher(), policy=None)
    assert "RSLAD baseline policy" in _rslad(tmp_path, teacher=_bank_teacher(), distillation_hooks=None)
    ce = LinfPGD(AttackConfig(epsilon="4/255", step_size="4/255", steps=1))
    assert "KL-to-teacher-clean" in _rslad(tmp_path, teacher=_bank_teacher(), attack=ce)
    # PGD-AT keeps refusing a teacher.
    assert "no teacher (PGD-AT)" in _refusal(tmp_path, teacher=_fixture_teacher())
    # Review of d2e82b2 (P2-1): online advT and temperatures other than 1 are not parity-tested.
    assert "RSLAD-advT only from a soft-label bank" in _rslad(tmp_path, teacher=_fixture_teacher(), advt=True)
    temperature = "RSLAD at temperature 1"
    assert temperature in _rslad(tmp_path, teacher=_fixture_teacher(), objective=RSLADObjective(temperature=2.0))
    hot = LinfPGD(
        AttackConfig(epsilon="4/255", step_size="4/255", steps=1, loss="kl", kl_target="teacher_clean", temperature=2.0)
    )
    assert temperature in _rslad(tmp_path, teacher=_fixture_teacher(), attack=hot)
    hooks = DistillationTargetHooks(adversarial_teacher_target=False, temperature=2.0)
    assert temperature in _rslad(tmp_path, teacher=_bank_teacher(), distillation_hooks=hooks)
    assert temperature not in _rslad(tmp_path, teacher=_bank_teacher())


@pytest.mark.parametrize("architecture", [None, "fixture_cnn", "mobilenet_v3_small_imagenet"])
def test_trainer_refuses_an_architecture_off_the_allowlist(tmp_path: Path, architecture: str | None) -> None:
    assert "a parity-tested student architecture" in _refusal(tmp_path, student_architecture=architecture)


def test_trainer_refuses_a_train_mode_attack(tmp_path: Path) -> None:
    train_mode = LinfPGD(AttackConfig(epsilon="4/255", step_size="4/255", steps=1, student_mode="train"))
    assert "an eval-mode, batch-keyed" in _refusal(tmp_path, attack=train_mode)


def _adamw(model: nn.Module, **options: Any) -> AdamW:
    return AdamW(model.parameters(), lr=1e-3, **{"capturable": True, **options})


def test_trainer_admits_only_the_capturable_foreach_adamw(tmp_path: Path) -> None:
    model = nn.Sequential(nn.Flatten(), nn.Linear(12, 3))
    # In scope: only the CPU device is refused.
    assert _refusal(tmp_path, model=model, optimizer=_adamw(model)) == "cuda_graph requires a CUDA device"
    assert _refusal(tmp_path, model=model, optimizer=_adamw(model, foreach=True)) == (
        "cuda_graph requires a CUDA device"
    )
    for optimizer in (
        AdamW(model.parameters(), lr=1e-3),  # capturable=False: host step counter and bias corrections
        _adamw(model, foreach=False),
        _adamw(model, amsgrad=True),
        _adamw(model, maximize=True),
        Adam(model.parameters(), lr=1e-3, capturable=True),  # coupled weight decay: not the recipe
        torch.optim.RMSprop(model.parameters(), lr=1e-3),
    ):
        assert "a torch.optim.SGD optimizer or a torch.optim.AdamW with capturable=True" in _refusal(
            tmp_path, model=model, optimizer=optimizer
        ), optimizer


def test_trainer_refuses_a_linf_pgd_subclass(tmp_path: Path) -> None:
    subclass = _PGDSubclass(AttackConfig(epsilon="4/255", step_size="4/255", steps=1))
    assert "a LinfPGD (not a subclass) training attack" in _refusal(tmp_path, attack=subclass)


@pytest.mark.parametrize(
    ("enabled", "warn_only", "benchmark", "refusal"),
    [
        (True, False, False, None),
        (False, False, False, None),  # training.deterministic=false
        (False, True, False, None),  # warn_only is inert while the mode is off
        (True, True, False, "torch deterministic algorithms strictly on or off (not warn_only)"),
        (False, False, True, "cuDNN benchmark off"),
        (True, False, True, "cuDNN benchmark off"),
    ],
)
def test_trainer_determinism_scope(
    tmp_path: Path, enabled: bool, warn_only: bool, benchmark: bool, refusal: str | None
) -> None:
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.benchmark,
    )
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    torch.backends.cudnn.benchmark = benchmark
    try:
        message = _refusal(tmp_path)  # the CPU device is the base fixture's only other refusal
        if refusal is None:
            assert "torch deterministic algorithms" not in message and "cuDNN benchmark" not in message
        else:
            assert refusal in message
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
        torch.backends.cudnn.benchmark = previous[2]


# =========================================================================== static buffers


def _cpu_state() -> CudaGraphStepState:
    """A CudaGraphStepState on the CPU (bypassing its CUDA-only constructor) to test host logic."""
    state = object.__new__(CudaGraphStepState)
    for name, value in {
        "device": torch.device("cpu"),
        "images": None,
        "labels": None,
        "valid": None,
        "noise": None,
        "bank_index": None,
        "bank_prob": None,
        "bank_residual": None,
        "totals": torch.zeros(9, dtype=torch.float64),
        "mixed_totals": None,
        "awp_active": False,
        "graph": None,
        "fingerprint": None,
        "outputs": {},
        "eager_steps_this_epoch": 0,
        "captures": 0,
        "captures_this_epoch": 0,
        "replays_this_epoch": 0,
        "full_batches_this_epoch": 0,
        "deferred": [],
    }.items():
        setattr(state, name, value)
    return state


def test_static_buffers_admit_only_contiguous_batches_of_the_captured_layout() -> None:
    state = _cpu_state()
    images, labels = torch.rand(4, 3, 8, 8), torch.arange(4)
    assert state.matches(images, labels)
    assert state.matches(torch.rand(4, 3, 8, 8), torch.arange(4))
    # Non-contiguous (a channels-last view of the same shape) and partial batches run eagerly.
    channels_last = torch.rand(4, 3, 8, 8).to(memory_format=torch.channels_last)
    assert channels_last.shape == images.shape and not channels_last.is_contiguous()
    assert not state.matches(channels_last, labels)
    assert not state.matches(torch.rand(4, 8, 8, 3).permute(0, 3, 1, 2), labels)
    assert not state.matches(torch.rand(2, 3, 8, 8), torch.arange(2))
    assert not state.matches(images, torch.arange(8)[::2])
    assert state.full_batches_this_epoch == 2


def test_the_noise_buffer_holds_the_attacked_rows_only() -> None:
    state = _cpu_state()
    assert state.matches(torch.rand(4, 3, 8, 8), torch.arange(4), noise_rows=2)
    assert state.noise is not None and state.noise.shape == (2, 3, 8, 8)
    assert state.images is not None and state.images.shape == (4, 3, 8, 8)


def test_a_first_non_float_batch_never_allocates_the_buffers() -> None:
    state = _cpu_state()
    assert not state.matches(torch.zeros(4, 3, 8, 8, dtype=torch.uint8), torch.arange(4))
    assert state.images is None


def test_an_epoch_without_replays_fails_loudly() -> None:
    state = _cpu_state()
    state.full_batches_this_epoch = 2
    state.check_epoch_used_graph()  # too few full batches to require a replay
    state.full_batches_this_epoch, state.eager_steps_this_epoch = 3, 3
    with pytest.raises(RuntimeError, match="replayed no CUDA graph"):
        state.check_epoch_used_graph()
    state.replays_this_epoch = 1
    state.check_epoch_used_graph()


# =========================================================================== fingerprint


def _stepped_sgd(**group: Any) -> tuple[nn.Module, SGD]:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Linear(4, 2))
    optimizer = SGD(model.parameters(), **{"lr": 0.1, "momentum": 0.9, "weight_decay": 1e-4, **group})
    model(torch.randn(5, 3)).sum().backward()
    optimizer.step()
    return model, optimizer


@pytest.mark.parametrize(
    "mutate",
    [
        lambda model, opt: opt.param_groups[0].__setitem__("lr", 0.05),
        lambda model, opt: opt.param_groups[0].__setitem__("momentum", 0.8),
        lambda model, opt: opt.param_groups[0].__setitem__("weight_decay", 0.0),
        lambda model, opt: opt.param_groups[0].__setitem__("nesterov", True),
        lambda model, opt: opt.param_groups[0].__setitem__("dampening", 0.1),
        lambda model, opt: opt.load_state_dict(copy.deepcopy(opt.state_dict())),
        lambda model, opt: model.eval(),
        lambda model, opt: model[1].register_buffer("running_mean", torch.zeros(4)),
        lambda model, opt: opt.add_param_group({"params": [nn.Parameter(torch.zeros(1))]}),
    ],
)
def test_fingerprint_detects_every_change_a_replay_would_ignore(mutate: Any) -> None:
    model, optimizer = _stepped_sgd()
    before = optimizer_fingerprint(optimizer, model)
    assert optimizer_fingerprint(optimizer, model) == before
    mutate(model, optimizer)
    assert optimizer_fingerprint(optimizer, model) != before


def test_fingerprint_covers_the_ema_tensors_the_captured_step_writes() -> None:
    model, optimizer = _stepped_sgd()
    ema = copy.deepcopy(model)
    before = optimizer_fingerprint(optimizer, model, ema_model=ema)
    assert before != optimizer_fingerprint(optimizer, model)
    # In-place EMA updates (what a replay does) keep it; a new EMA tensor does not.
    with torch.no_grad():
        for value in ema.state_dict().values():
            value.mul_(1)
    assert optimizer_fingerprint(optimizer, model, ema_model=ema) == before
    ema[0].weight = nn.Parameter(ema[0].weight.detach().clone())
    assert optimizer_fingerprint(optimizer, model, ema_model=ema) != before


def test_fingerprint_ignores_in_place_value_updates_and_refuses_tensor_hyperparameters() -> None:
    model, optimizer = _stepped_sgd()
    before = optimizer_fingerprint(optimizer, model)
    # What a replay itself does: in-place updates of the same tensors.
    model(torch.randn(5, 3)).sum().backward()
    optimizer.step()
    optimizer.load_state_dict(optimizer.state_dict())  # same tensors, re-attached
    assert optimizer_fingerprint(optimizer, model) == before
    optimizer.param_groups[0]["lr"] = torch.tensor(0.1)
    with pytest.raises(RuntimeError, match="Python-scalar optimizer 'lr'"):
        optimizer_fingerprint(optimizer, model)


def test_fingerprint_covers_the_extra_inputs_of_the_2026_10_08_scope() -> None:
    model, optimizer = _stepped_sgd()
    teacher = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4)).eval()
    before = module_fingerprint(teacher)
    assert module_fingerprint(teacher) == before
    with torch.no_grad():
        teacher[0].weight.mul_(2)  # in place: same tensors, same fingerprint
    assert module_fingerprint(teacher) == before
    teacher[1].train()
    assert module_fingerprint(teacher) != before
    teacher[1].eval()
    teacher[0].weight = nn.Parameter(teacher[0].weight.detach().clone())
    assert module_fingerprint(teacher) != before
    proxy = SGD(nn.Linear(3, 2).parameters(), lr=0.01)
    proxy_before = optimizer_groups_fingerprint(proxy)
    proxy.param_groups[0]["lr"] = 0.02
    assert optimizer_groups_fingerprint(proxy) != proxy_before
    # The Trainer's extra tuple (static buffers, teacher, mixed k, AWP) is part of the compared fingerprint.
    assert optimizer_fingerprint(optimizer, model, extra=(1,)) != optimizer_fingerprint(optimizer, model, extra=(2,))


def _stepped_adamw() -> tuple[nn.Module, AdamW]:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4), nn.Linear(4, 2))
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    optimizer = AdamW([{"params": decay, "weight_decay": 0.05}, {"params": no_decay, "weight_decay": 0.0}], lr=1e-3)
    model(torch.randn(5, 3)).sum().backward()
    optimizer.step()
    return model, optimizer


def _replace_state(key: str) -> Any:
    def mutate(model: nn.Module, optimizer: AdamW) -> None:
        first = optimizer.param_groups[0]["params"][0]
        optimizer.state[first][key] = optimizer.state[first][key].clone()

    return mutate


@pytest.mark.parametrize(
    "mutate",
    [
        lambda model, opt: opt.param_groups[0].__setitem__("lr", 5e-4),
        lambda model, opt: opt.param_groups[1].__setitem__("lr", 5e-4),
        lambda model, opt: opt.param_groups[0].__setitem__("betas", (0.8, 0.999)),
        lambda model, opt: opt.param_groups[0].__setitem__("betas", (0.9, 0.99)),
        lambda model, opt: opt.param_groups[0].__setitem__("eps", 1e-6),
        lambda model, opt: opt.param_groups[0].__setitem__("weight_decay", 0.1),
        lambda model, opt: opt.param_groups[1].__setitem__("weight_decay", 0.05),
        lambda model, opt: opt.param_groups[0].__setitem__("amsgrad", True),
        _replace_state("exp_avg"),
        _replace_state("exp_avg_sq"),
        _replace_state("step"),
        lambda model, opt: opt.load_state_dict(copy.deepcopy(opt.state_dict())),
    ],
)
def test_fingerprint_covers_every_adamw_input_of_the_captured_step(mutate: Any) -> None:
    model, optimizer = _stepped_adamw()
    before = optimizer_fingerprint(optimizer, model)
    # What a replay does (in-place updates of the same tensors) keeps it.
    model(torch.randn(5, 3)).sum().backward()
    optimizer.step()
    assert optimizer_fingerprint(optimizer, model) == before
    mutate(model, optimizer)
    assert optimizer_fingerprint(optimizer, model) != before


def test_fingerprint_refuses_tensor_adamw_betas() -> None:
    model, optimizer = _stepped_adamw()
    optimizer.param_groups[0]["betas"] = (torch.tensor(0.9), 0.999)
    with pytest.raises(RuntimeError, match="Python-scalar optimizer 'betas'"):
        optimizer_fingerprint(optimizer, model)


def test_adamw_state_ready_only_after_the_first_update() -> None:
    model = nn.Linear(3, 2)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    assert not optimizer_state_ready(optimizer)
    model(torch.randn(4, 3)).sum().backward()
    optimizer.step()
    assert optimizer_state_ready(optimizer)
    # A step counter that does not live on the parameter's device (a non-capturable state) is not ready.
    first = next(iter(model.parameters()))
    optimizer.state[first]["step"] = torch.zeros((), device="meta")
    assert not optimizer_state_ready(optimizer)
    # SGD keeps its momentum-buffer rule.
    sgd = SGD(nn.Linear(3, 2).parameters(), lr=0.1, momentum=0.9)
    assert optimizer_state_ready(sgd) is momentum_buffers_ready(sgd) is False
    # Review of 1f17fa2 (P3-3): any other optimizer has no readiness rule and is refused, not guessed.
    for other in (Adam(nn.Linear(3, 2).parameters()), torch.optim.RMSprop(nn.Linear(3, 2).parameters())):
        with pytest.raises(TypeError, match="no state-readiness rule"):
            optimizer_state_ready(other)


def test_require_is_the_host_check_outside_a_capture() -> None:
    require(torch.tensor(True), "never raised")
    with pytest.raises(FloatingPointError, match="^bad value$"):
        require(torch.tensor(False), "bad value", FloatingPointError)
    with pytest.raises(ValueError, match="^bad value$"):
        require(torch.tensor(False), "bad value")


def test_momentum_buffers_ready_only_after_the_first_update() -> None:
    model = nn.Linear(3, 2)
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
    assert not momentum_buffers_ready(optimizer)
    model(torch.randn(4, 3)).sum().backward()
    optimizer.step()
    assert momentum_buffers_ready(optimizer)
    assert momentum_buffers_ready(SGD(nn.Linear(3, 2).parameters(), lr=0.1, momentum=0.0))


# =========================================================================== attack core


def _attack_cases() -> dict[str, tuple[AttackConfig, dict[str, Any]]]:
    return {
        "batch_random_start": (AttackConfig(epsilon="8/255", step_size="2/255", steps=3), {}),
        "no_random_start": (AttackConfig(epsilon="8/255", step_size="2/255", steps=3, random_start=False), {}),
        "trace_and_capture": (
            AttackConfig(epsilon="8/255", step_size="2/255", steps=3, trace_step_losses=True),
            {"capture_step": 2},
        ),
        "sample_keyed": (
            AttackConfig(epsilon="8/255", step_size="2/255", steps=2, random_start_keying="sample_keyed_v1"),
            {"source_ids": torch.arange(6), "epoch": 3, "attack_seed": 7},
        ),
        "overrides_with_zero_budget": (
            AttackConfig(epsilon="8/255", step_size="2/255", steps=2),
            {
                "epsilon_override": torch.tensor([0.0, 4 / 255, 8 / 255, 8 / 255, 2 / 255, 0.0]),
                "step_size_override": torch.tensor([0.0, 1 / 255, 2 / 255, 2 / 255, 1 / 255, 0.0]),
            },
        ),
        "kl_student_clean_train_mode": (
            AttackConfig(
                epsilon="8/255", step_size="2/255", steps=2, loss="kl", kl_target="student_clean", student_mode="train"
            ),
            {},
        ),
    }


def _tensor_digest(value: torch.Tensor | None) -> str | None:
    if value is None:
        return None
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(f"{tensor.dtype}|{tuple(tensor.shape)}|".encode() + tensor.numpy().tobytes()).hexdigest()


def generate_digests(pgd_module: Any) -> dict[str, dict[str, Any]]:
    """Digest every AttackResult field of ``pgd_module.LinfPGD.generate`` on the seeded cases."""
    digests = {}
    for name, (config, extra) in _attack_cases().items():
        torch.manual_seed(0)
        student = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(), nn.Flatten(), nn.Linear(4 * 16, 3)
        )
        images = torch.rand(6, 3, 4, 4, generator=torch.Generator().manual_seed(1))
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        student.train()
        request = AttackRequest(
            inputs=images, labels=labels, student=student, generator=torch.Generator().manual_seed(5), **extra
        )
        result = pgd_module.LinfPGD(config).generate(request)
        assert student.training  # mode restored
        digests[name] = {
            "adversarial": _tensor_digest(result.adversarial),
            "initial_delta": _tensor_digest(result.initial_delta),
            "step_losses": [float(loss).hex() for loss in result.step_losses],
            "max_abs_delta": float(result.max_abs_delta).hex(),
            "captured": _tensor_digest(result.captured_adversarial),
            "student_state": {key: _tensor_digest(value) for key, value in student.state_dict().items()},
        }
    return digests


def test_generate_is_bit_identical_to_the_pre_change_attack_golden_outputs() -> None:
    import ard.attacks.pgd as current

    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert expected["source"] == f"{PRE_CHANGE_COMMIT}:src/ard/attacks/pgd.py"
    assert generate_digests(current) == expected["digests"]


if __name__ == "__main__":  # pragma: no cover - fixture regeneration from the pre-change attack only
    old_pgd = _pre_change_module("src/ard/attacks/pgd.py", "ard.attacks._pgd_pre_cuda_graph", "ard.attacks")
    GOLDEN.parent.mkdir(exist_ok=True)
    payload = {
        "source": f"{PRE_CHANGE_COMMIT}:src/ard/attacks/pgd.py",
        "torch": torch.__version__,
        "digests": generate_digests(old_pgd),
    }
    GOLDEN.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {GOLDEN}")
