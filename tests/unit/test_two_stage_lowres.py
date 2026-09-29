"""Plan 0103 two-stage low-resolution PGD-AT (human-approved 2026-09-30).

* ``training.train_image_size`` changes only the output size of the
  training-partition views (training crop, in-training validation, train
  probe), never the crop parameters or the official evaluation dataset, and
  the resize is anti-aliased.
* ``training.init_checkpoint`` loads exactly a verified ``last.pt``'s student
  weights and refuses a bad digest, a non-final or foreign checkpoint, and
  ``student.pretrained``.
* Defaults are bit-identical: every existing config serializes (and so
  hashes) exactly as under the pre-change schema, and default data views equal
  the pre-change ``build_train_validation_views`` -- both loaded from git into
  this process, the same-process differential of ``test_step_sync_free_parity``.
* The two stage configs differ from the single-stage reference only in the
  intended fields.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from PIL import Image
from pydantic import ValidationError

from ard.attacks import AttackRequest, LinfPGD
from ard.cli.evaluate import _two_stage_protocol_identity
from ard.cli.train import _build_method
from ard.config import load_config
from ard.config.loader import _expand_environment, resolved_config_dict, save_resolved_config
from ard.config.schema import (
    AttackConfig,
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    reject_throughput_options,
)
from ard.data import build_dataset, build_train_probe_view, build_train_validation_views
from ard.data.datasets import EpochImageNetTransform, ImageNetEvalTransform
from ard.engine.checkpoint import REQUIRED_KEYS, config_digest, load_init_student_weights, read_init_checkpoint
from ard.models import build_student
from ard.protocols import ensure_local_trainable
from ard.tracking.adapter import canonical_run_group

pytestmark = pytest.mark.t1

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "scientific"
PROTOCOL = "controlled_imagenet_stage02_two_stage_lowres_v1"
# The commit this change is based on: the pre-change schema and data views.
PRE_CHANGE_COMMIT = "5427e3d"
STAGE_CONFIGS = {
    "imagenet_mobilenetv4_twostage_stage1_112_pgd1.yaml",
    "imagenet_mobilenetv4_twostage_stage2_224_pgd3_ft.yaml",
}
# Configs added after this module's pre-change commit that use later schema
# fields (loader speedup C's dataset.derived_from); their own byte-identity
# test is tests/unit/test_imagenet_loader_speedups.py.
LATER_CONFIGS = {"imagenet_mobilenetv4_twostage_stage1_112_pgd1_resized.yaml"}


def _pre_change_module(relative: str, name: str, package: str) -> types.ModuleType:
    try:
        source = subprocess.run(
            ["git", "show", f"{PRE_CHANGE_COMMIT}:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:  # pragma: no cover - shallow clones
        pytest.skip(f"pre-change {relative} at {PRE_CHANGE_COMMIT} unavailable from git: {error}")
    module = types.ModuleType(name)
    module.__package__ = package
    module.__file__ = f"{PRE_CHANGE_COMMIT}:{relative}"
    # pydantic resolves postponed annotations through sys.modules[__module__].
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    teacher_checkpoint = tmp_path / "teacher.pt"
    teacher_checkpoint.write_bytes(b"fixture checkpoint")
    for key, value in {
        # test_config.test_top_level_configs_resolve_under_controlled_environment's environment
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
    }.items():
        monkeypatch.setenv(key, value)


# --------------------------------------------------------------------------- defaults


def test_default_training_config_serializes_neither_new_field() -> None:
    training = TrainingConfig(per_rank_batch_size=4, global_batch_size=4)
    assert training.train_image_size is None and training.init_checkpoint is None
    dumped = json.loads(training.model_dump_json())
    assert "train_image_size" not in dumped and "init_checkpoint" not in dumped


def test_existing_configs_serialize_exactly_as_under_the_pre_change_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolved config (and therefore config hash, run ID and resume
    eligibility) of every pre-existing scientific/production config is
    byte-identical to the pre-change schema's."""
    _env(monkeypatch, tmp_path)
    old_schema = _pre_change_module("src/ard/config/schema.py", "_ard_schema_pre_two_stage", "ard.config")
    paths = [
        path
        for directory in ("experiments", "pilot", "production", "scientific")
        for path in sorted((ROOT / "configs" / directory).glob("*.yaml"))
        if path.name not in STAGE_CONFIGS | LATER_CONFIGS
    ]
    assert len(paths) > 20
    for path in paths:
        expanded = _expand_environment(yaml.safe_load(path.read_text(encoding="utf-8")))
        new = ExperimentConfig.model_validate(expanded)
        old = old_schema.ExperimentConfig.model_validate(expanded)
        assert resolved_config_dict(new) == json.loads(old.model_dump_json()), path.name
        assert config_digest(resolved_config_dict(new)) == config_digest(json.loads(old.model_dump_json()))


def _imagenet_layout(root: Path, *, size: tuple[int, int] = (48, 40), per_class: int = 3) -> None:
    for class_index, wnid in enumerate(("n001", "n002")):
        class_dir = root / "train" / wnid
        class_dir.mkdir(parents=True, exist_ok=True)
        for image_index in range(per_class):
            generator = torch.Generator().manual_seed(class_index * 10 + image_index)
            pixels = torch.rand(size[1], size[0], 3, generator=generator)
            Image.fromarray((pixels * 255).to(torch.uint8).numpy()).save(class_dir / f"{wnid}_{image_index}.JPEG")


def _views(module: Any, config: DatasetConfig, **extra: Any) -> tuple[Any, Any]:
    return module.build_train_validation_views(
        config, validation_fraction=0.34, split_seed=1, augmentation_seed=7, **extra
    )


def _materialize(view: Any, epochs: tuple[int, ...] = (0, 3)) -> list[tuple[torch.Tensor, int, int]]:
    items = []
    for epoch in epochs:
        view.set_epoch(epoch)
        items.extend(view[index] for index in range(len(view)))
    return items


def _assert_same(first: list[tuple[torch.Tensor, int, int]], second: list[tuple[torch.Tensor, int, int]]) -> None:
    assert len(first) == len(second)
    for (image_a, label_a, id_a), (image_b, label_b, id_b) in zip(first, second, strict=True):
        assert (label_a, id_a) == (label_b, id_b)
        assert torch.equal(image_a, image_b)


def test_default_views_are_bit_identical_to_the_pre_change_views(tmp_path: Path) -> None:
    old_data = _pre_change_module("src/ard/data/datasets.py", "ard.data._datasets_pre_two_stage", "ard.data")
    _imagenet_layout(tmp_path)
    config = DatasetConfig(name="imagenet", root=tmp_path, split="train", num_classes=2, image_size=32)
    old_train, old_validation = _views(old_data, config)
    for extra in ({}, {"train_image_size": None}):
        new_train, new_validation = _views(sys.modules["ard.data.datasets"], config, **extra)
        assert new_train.indices == old_train.indices and new_validation.indices == old_validation.indices
        _assert_same(_materialize(new_train), _materialize(old_train))
        _assert_same(_materialize(new_validation), _materialize(old_validation))


# --------------------------------------------------------------------------- train_image_size


def test_train_image_size_changes_only_the_output_size_of_the_training_partition_views(tmp_path: Path) -> None:
    _imagenet_layout(tmp_path)
    full = DatasetConfig(name="imagenet", root=tmp_path, split="train", num_classes=2, image_size=32)
    small = full.model_copy(update={"image_size": 16})
    train, validation = build_train_validation_views(
        full, validation_fraction=0.34, split_seed=1, augmentation_seed=7, train_image_size=16
    )
    # Same crop parameters and flips (drawn from the same keyed generator),
    # same split: exactly what a 16px dataset.image_size would produce.
    reference_train, reference_validation = build_train_validation_views(
        small, validation_fraction=0.34, split_seed=1, augmentation_seed=7
    )
    _assert_same(_materialize(train), _materialize(reference_train))
    _assert_same(_materialize(validation), _materialize(reference_validation))
    assert all(image.shape == (3, 16, 16) for image, _, _ in _materialize(train) + _materialize(validation))
    # The train probe reuses the validation view, so it is 16px as well.
    probe = build_train_probe_view(train, validation, size=2, seed=1)
    assert all(probe[index][0].shape == (3, 16, 16) for index in range(len(probe)))
    # The full-resolution views of the same config are unchanged by the option's existence.
    full_train, _ = build_train_validation_views(full, validation_fraction=0.34, split_seed=1, augmentation_seed=7)
    assert full_train[0][0].shape == (3, 32, 32)


def test_train_image_size_never_reaches_the_evaluation_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch, tmp_path)
    stage1 = load_config(CONFIG_DIR / "imagenet_mobilenetv4_twostage_stage1_112_pgd1.yaml")
    assert stage1.training.train_image_size == 112
    assert stage1.dataset.image_size == 224
    assert stage1.evaluation.dataset is not None and stage1.evaluation.dataset.image_size == 224
    # The evaluation split is built from evaluation.dataset alone.
    _imagenet_layout(tmp_path / "eval")
    (tmp_path / "eval" / "train").rename(tmp_path / "eval" / "val")
    evaluation = stage1.evaluation.dataset.model_copy(
        update={"root": tmp_path / "eval", "num_classes": 2, "image_size": 32, "content_sha256": None}
    )
    assert build_dataset(evaluation)[0][0].shape == (3, 32, 32)


def _checkerboard(size: int) -> Image.Image:
    pattern = (torch.arange(size)[:, None] + torch.arange(size)[None, :]) % 2
    pixels = (pattern * 255).to(torch.uint8)[..., None].expand(size, size, 3).contiguous()
    return Image.fromarray(pixels.numpy())


def test_reduced_resolution_resizes_are_anti_aliased() -> None:
    """A 1-pixel checkerboard averages to mid-grey under an anti-aliased
    downscale; a non-anti-aliased (point-sampling) resize keeps near-black and
    near-white pixels."""
    image = _checkerboard(256)
    outputs = [ImageNetEvalTransform(image_size=16)(image)]
    crop = EpochImageNetTransform(augmentation_seed=3, image_size=16)
    outputs.extend(crop(image, source_id=source_id) for source_id in range(8))
    for output in outputs:
        assert output.shape == (3, 16, 16)
        assert (output - 0.5).abs().max() < 0.1
    aliased = torch.nn.functional.interpolate(
        torch.as_tensor(list(image.getdata()), dtype=torch.float32)[:, 0].reshape(1, 1, 256, 256) / 255,
        size=(16, 16),
        mode="nearest",
    )
    assert (aliased - 0.5).abs().max() > 0.4


def _minimal(**training: Any) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "protocol": {"id": "synthetic_smoke_v2"},
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
            1,
        ),
        "dataset": {"name": "synthetic_cifar", "num_samples": 8, "num_classes": 2, "image_size": 8},
        "student": {"architecture": "fixture_cnn", "num_classes": 2},
        "method": {"id": "pgd_at", "version": 1},
        "optimizer": {"id": "sgd", "learning_rate": 0.01, "momentum": 0.9, "weight_decay": 0.0, "nesterov": False},
        "scheduler": {"id": "identity", "milestones": [], "gamma": 1.0, "step_at": "epoch_end"},
        "training": {"per_rank_batch_size": 2, "global_batch_size": 2, **training},
    }


def test_schema_refuses_ill_defined_two_stage_fields(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="train_image_size is only defined for the imagenet dataset"):
        ExperimentConfig.model_validate(_minimal(train_image_size=4))
    with pytest.raises(ValueError, match="only defined for the imagenet dataset"):
        build_train_validation_views(
            DatasetConfig(), validation_fraction=0.25, split_seed=0, augmentation_seed=0, train_image_size=4
        )
    imagenet = {"name": "imagenet", "root": str(tmp_path), "num_classes": 2, "image_size": 8}
    student = {"architecture": "fixture_cnn", "num_classes": 2, "normalization": {"profile": "imagenet_standard"}}
    with pytest.raises(ValidationError, match="equals dataset.image_size"):
        ExperimentConfig.model_validate({**_minimal(train_image_size=8), "dataset": imagenet, "student": student})
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        TrainingConfig(per_rank_batch_size=2, global_batch_size=2, train_image_size=0)
    for digest in ("A" * 64, "a" * 63, "not-a-digest"):
        with pytest.raises(ValidationError, match="lowercase 64-character"):
            TrainingConfig(
                per_rank_batch_size=2, global_batch_size=2, init_checkpoint={"path": "last.pt", "sha256": digest}
            )


def test_schema_refuses_init_checkpoint_with_pretrained(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch, tmp_path)
    with pytest.raises(ValidationError, match="cannot be combined with student.pretrained"):
        load_config(CONFIG_DIR / "imagenet_mobilenetv4_twostage_stage2_224_pgd3_ft.yaml", ["student.pretrained=true"])


def test_two_stage_protocol_requires_exactly_one_stage_field(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch, tmp_path)
    assert ensure_local_trainable(PROTOCOL).runnable_locally
    stage2 = CONFIG_DIR / "imagenet_mobilenetv4_twostage_stage2_224_pgd3_ft.yaml"
    with pytest.raises(ValidationError, match="requires exactly one of"):
        load_config(stage2, ["training.init_checkpoint=null"])
    with pytest.raises(ValidationError, match="requires exactly one of"):
        load_config(stage2, ["training.train_image_size=112"])


STAGE1 = CONFIG_DIR / "imagenet_mobilenetv4_twostage_stage1_112_pgd1.yaml"


def test_selection_step_guard_still_fires_without_the_explicit_stage1_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The selection-vs-training step-size equality guard is not weakened: a
    PGD-1 step-4/255 training attack with the reference's PGD-10 step-8/765
    selection attack is refused unless the explicit, protocol-scoped flag is
    set, and the flag exempts the step size only."""
    _env(monkeypatch, tmp_path)
    with pytest.raises(ValidationError, match="must match the training threat model: step_size"):
        load_config(STAGE1, ["method.selection_step_size_independent=false"])
    with pytest.raises(ValidationError, match="must match the training threat model: epsilon"):
        load_config(STAGE1, ["method.attack.epsilon=2/255"])
    with pytest.raises(ValidationError, match="requires an explicit method.selection_attack"):
        load_config(STAGE1, ["method.selection_attack=null"])
    with pytest.raises(ValidationError, match="only for the reduced-resolution stage 1"):
        load_config(
            STAGE1, ["protocol.id=controlled_imagenet_stage02_init_lr_grid_v1", "training.train_image_size=null"]
        )
    stage2 = CONFIG_DIR / "imagenet_mobilenetv4_twostage_stage2_224_pgd3_ft.yaml"
    with pytest.raises(ValidationError, match="only for the reduced-resolution stage 1"):
        load_config(stage2, ["method.selection_step_size_independent=true"])
    # Under the flag, the selection attack is pinned to the reference identity.
    for override in (
        "method.selection_attack.step_size=1/255",
        "method.selection_attack.steps=20",
    ):
        with pytest.raises(ValidationError, match="requires the reference selection attack exactly"):
            load_config(STAGE1, [override])
    method = load_config(STAGE1).method
    assert method.attack is not None and method.selection_attack is not None
    assert (method.attack.step_size, method.selection_attack.step_size) == ("4/255", "8/765")
    assert json.loads(method.model_dump_json())["selection_step_size_independent"] is True
    # Serialized only when true.
    reference = load_config(CONFIG_DIR / "imagenet_mobilenetv4_pgd_at_random_init_lr0025.yaml")
    assert "selection_step_size_independent" not in json.loads(reference.method.model_dump_json())


def test_runtimes_that_ignore_two_stage_fields_refuse_them() -> None:
    small = TrainingConfig(per_rank_batch_size=2, global_batch_size=2, train_image_size=112)
    with pytest.raises(ValueError, match=r"fixture does not implement training\.train_image_size=112"):
        reject_throughput_options(small, runtime="fixture")
    initialized = TrainingConfig(
        per_rank_batch_size=2, global_batch_size=2, init_checkpoint={"path": "last.pt", "sha256": "d" * 64}
    )
    with pytest.raises(
        ValueError, match=r"training\.init_checkpoint=<set>; run it with training\.init_checkpoint=null"
    ):
        reject_throughput_options(initialized, runtime="fixture")


def test_evaluation_protocol_identity_names_the_two_stage_fields_only_when_set() -> None:
    assert _two_stage_protocol_identity(TrainingConfig(per_rank_batch_size=2, global_batch_size=2)) == {}
    stage1 = TrainingConfig(per_rank_batch_size=2, global_batch_size=2, train_image_size=112)
    assert _two_stage_protocol_identity(stage1) == {"train_image_size": 112}
    stage2 = TrainingConfig(
        per_rank_batch_size=2, global_batch_size=2, init_checkpoint={"path": "/somewhere/last.pt", "sha256": "e" * 64}
    )
    # The digest is the identity; the host-specific path never is.
    assert _two_stage_protocol_identity(stage2) == {"init_checkpoint_sha256": "e" * 64}


# --------------------------------------------------------------------------- PGD-1 at step == eps


def test_pgd1_with_step_equal_to_epsilon_passes_the_unweakened_step_guard() -> None:
    config = AttackConfig(epsilon="4/255", step_size="4/255", steps=1, random_start=True)
    assert config.step_size_value == config.epsilon_value
    model = build_student(ModelConfig(architecture="fixture_cnn", num_classes=2), tier="dev")
    inputs = torch.rand(4, 3, 8, 8, generator=torch.Generator().manual_seed(0))
    labels = torch.tensor([0, 1, 0, 1])

    def request(**extra: Any) -> AttackRequest:
        return AttackRequest(
            student=model,
            inputs=inputs,
            labels=labels,
            generator=torch.Generator().manual_seed(1),
            **extra,
        )

    result = LinfPGD(config).generate(request())
    assert result.max_abs_delta <= 4 / 255 + 1e-7
    # The guard itself is unchanged: a step larger than epsilon is still refused.
    with pytest.raises(ValueError, match="cannot exceed epsilon"):
        LinfPGD(config).generate(request(epsilon_override=torch.full((4,), 2 / 255)))


# --------------------------------------------------------------------------- init_checkpoint


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _two_stage_config(tmp_path: Path, *, stage: int, **overrides: Any) -> ExperimentConfig:
    """A tiny, valid config of either stage (fixture student, 32px ImageNet)."""
    stage_field: dict[str, Any] = (
        {"train_image_size": 16}
        if stage == 1
        else {"init_checkpoint": {"path": str(tmp_path / "stage1" / "last.pt"), "sha256": "d" * 64}}
    )
    raw: dict[str, Any] = {
        "schema_version": 2,
        "protocol": {"id": PROTOCOL},
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
            1,
        ),
        "dataset": {"name": "imagenet", "root": str(tmp_path / "imagenet"), "num_classes": 2, "image_size": 32},
        "student": {"architecture": "fixture_cnn", "num_classes": 2, "normalization": {"profile": "imagenet_standard"}},
        "method": {"id": "pgd_at", "version": 1},
        "optimizer": {"id": "sgd", "learning_rate": 0.01, "momentum": 0.9, "weight_decay": 0.0, "nesterov": False},
        "scheduler": {"id": "identity", "milestones": [], "gamma": 1.0, "step_at": "epoch_end"},
        "training": {"epochs": 3, "per_rank_batch_size": 2, "global_batch_size": 2, **stage_field},
    }
    return ExperimentConfig.model_validate(_deep_update(raw, overrides))


def _source_run(
    tmp_path: Path, *, epoch: int = 2, source: dict[str, Any] | None = None, **payload_extra: Any
) -> tuple[Path, Any]:
    """A stage-1 run directory: a full resolved config next to its last.pt."""
    run = tmp_path / "stage1"
    run.mkdir(parents=True, exist_ok=True)
    save_resolved_config(_two_stage_config(tmp_path, stage=1, **(source or {})), run / "resolved_config.yaml")
    torch.manual_seed(5)
    model = build_student(ModelConfig(architecture="fixture_cnn", num_classes=2), tier="dev")
    payload: dict[str, Any] = dict.fromkeys(REQUIRED_KEYS)
    payload.update(
        {
            "format_version": 1,
            "epoch": epoch,
            "epoch_boundary": "end",
            "model": model.state_dict(),
            "optimizer": {"state": {"sentinel": 1}, "param_groups": []},
            "selection_metadata": {},
            "tracker_run_id": "stage1-run",
            "config_hash": config_digest(yaml.safe_load((run / "resolved_config.yaml").read_text(encoding="utf-8"))),
            "world_size": 1,
            **payload_extra,
        }
    )
    torch.save(payload, run / "last.pt")
    return run / "last.pt", model


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(tmp_path: Path, path: Path, *, expected_sha256: str | None = None) -> Any:
    return read_init_checkpoint(
        path,
        expected_sha256=_sha(path) if expected_sha256 is None else expected_sha256,
        target=_two_stage_config(tmp_path, stage=2),
    )


def test_init_checkpoint_loads_exactly_the_saved_student_weights_and_records_lineage(tmp_path: Path) -> None:
    path, source = _source_run(tmp_path)
    torch.manual_seed(99)
    target = build_student(ModelConfig(architecture="fixture_cnn", num_classes=2), tier="dev")
    assert any(not torch.equal(a, b) for a, b in zip(target.state_dict().values(), source.state_dict().values()))
    lineage = load_init_student_weights(
        target, path, expected_sha256=_sha(path), target=_two_stage_config(tmp_path, stage=2)
    )
    loaded = target.state_dict()
    assert loaded.keys() == source.state_dict().keys()
    for key, value in source.state_dict().items():
        assert torch.equal(loaded[key], value), key
    assert lineage == {
        "kind": "student_init_checkpoint_v1",
        "path": str(path),
        "sha256": _sha(path),
        "loaded_entry": "model",
        "strict": True,
        "source_tracker_run_id": "stage1-run",
        "source_config_hash": config_digest(yaml.safe_load((path.parent / "resolved_config.yaml").read_text())),
        "source_resolved_config_sha256": _sha(path.parent / "resolved_config.yaml"),
        "source_protocol_id": PROTOCOL,
        "source_epoch": 2,
        "source_epochs": 3,
        "source_world_size": 1,
        "source_train_image_size": 16,
    }


def test_init_checkpoint_refuses_a_missing_file_or_a_mismatched_digest(tmp_path: Path) -> None:
    path, _ = _source_run(tmp_path)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _read(tmp_path, tmp_path / "nowhere" / "last.pt", expected_sha256=_sha(path))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _read(tmp_path, path, expected_sha256="0" * 64)
    renamed = path.with_name("best.pt")
    renamed.write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="must be a run's last.pt"):
        _read(tmp_path, renamed)


@pytest.mark.parametrize(
    ("variant", "message"),
    [
        ({"epoch": 1}, "not its run's final epoch"),
        ({"config_hash": "f" * 64}, "config_hash does not match"),
        ({"epoch_boundary": "mid"}, "epoch-boundary"),
        ({"selection_metadata": {"selection_source": "ema"}}, "EMA-selected"),
    ],
)
def test_init_checkpoint_refuses_a_non_final_foreign_or_ema_checkpoint(
    tmp_path: Path, variant: dict[str, Any], message: str
) -> None:
    path, _ = _source_run(tmp_path, **variant)
    with pytest.raises(ValueError, match=message):
        _read(tmp_path, path)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ({"seeds": {"split": 2}}, "seeds.split"),
        ({"training": {"validation_fraction": 0.5}}, "training.validation_fraction"),
        ({"dataset": {"num_classes": 3}, "student": {"num_classes": 3}}, "dataset.num_classes"),
        ({"dataset": {"image_size": 64}}, "dataset.image_size"),
        ({"dataset": {"content_sha256": "a" * 64}}, "dataset.content_sha256"),
        # A single-stage run (no train_image_size) under another protocol.
        (
            {"protocol": {"id": "controlled_imagenet_stage02_init_lr_grid_v1"}, "training": {"train_image_size": None}},
            "must be a reduced-resolution stage 1",
        ),
        ({"protocol": {"id": "controlled_imagenet_stage02_init_lr_grid_v1"}}, "both runs must use protocol"),
        (
            {
                "student": {
                    "architecture": "mobilevit_s_imagenet",
                    "normalization": {"profile": "imagenet_raw_identity"},
                }
            },
            "student.normalization",
        ),
    ],
)
def test_init_checkpoint_refuses_a_source_run_incompatible_with_stage2(
    tmp_path: Path, source: dict[str, Any], message: str
) -> None:
    path, _ = _source_run(tmp_path, source=source)
    with pytest.raises(ValueError, match="incompatible with this stage-2 run") as refused:
        _read(tmp_path, path)
    assert message in str(refused.value)


def test_init_checkpoint_refuses_a_source_config_that_is_not_a_valid_experiment(tmp_path: Path) -> None:
    path, _ = _source_run(tmp_path)
    config_path = path.parent / "resolved_config.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["training"]["surprise"] = 1
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    payload = torch.load(path, weights_only=False)
    payload["config_hash"] = config_digest(raw)
    torch.save(payload, path)
    with pytest.raises(ValueError, match="not a valid config"):
        _read(tmp_path, path)


def test_init_checkpoint_refuses_a_missing_sibling_config_and_an_incomplete_payload(tmp_path: Path) -> None:
    path, _ = _source_run(tmp_path)
    (path.parent / "resolved_config.yaml").unlink()
    with pytest.raises(FileNotFoundError, match="sibling resolved_config.yaml"):
        _read(tmp_path, path)
    torch.save({"model": {}}, path)
    with pytest.raises(ValueError, match="incomplete"):
        _read(tmp_path, path)


def test_init_checkpoint_load_is_strict(tmp_path: Path) -> None:
    path, _ = _source_run(tmp_path)
    wider = build_student(ModelConfig(architecture="fixture_cnn", num_classes=3), tier="dev")
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_init_student_weights(wider, path, expected_sha256=_sha(path), target=_two_stage_config(tmp_path, stage=2))


def test_schema_restricts_init_checkpoint_to_the_two_stage_protocol(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="init_checkpoint is defined only for"):
        _two_stage_config(tmp_path, stage=2, protocol={"id": "controlled_imagenet_stage02_init_lr_grid_v1"})


# --------------------------------------------------------------------------- configs


def _dump(name: str) -> dict[str, Any]:
    return load_config(CONFIG_DIR / name).model_dump(mode="json")


def test_stage1_config_is_the_reference_except_resolution_attack_and_horizon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _env(monkeypatch, tmp_path)
    reference = _dump("imagenet_mobilenetv4_pgd_at_random_init_lr0025.yaml")
    stage1 = _dump("imagenet_mobilenetv4_twostage_stage1_112_pgd1.yaml")
    assert stage1["protocol"]["id"] == PROTOCOL
    assert stage1["tracking"]["group"] == "imagenet-mobilenetv4-twostage-lowres-stage1"
    assert stage1["training"]["train_image_size"] == 112 and "init_checkpoint" not in stage1["training"]
    assert stage1["training"]["epochs"] == 195
    assert stage1["method"]["selection_step_size_independent"] is True
    assert stage1["scheduler"]["milestones"] == [98, 148] and stage1["scheduler"]["warmup_epochs"] == 10
    attack = stage1["method"]["attack"]
    assert (attack["steps"], attack["step_size"], attack["epsilon"], attack["random_start"]) == (
        1,
        "4/255",
        "4/255",
        True,
    )
    assert attack["step_size_value"] == attack["epsilon_value"]
    # Selection and evaluation attacks, dataset (224) and evaluation dataset (224) unchanged.
    assert stage1["method"]["selection_attack"] == reference["method"]["selection_attack"]
    assert stage1["evaluation"] == reference["evaluation"] and stage1["dataset"] == reference["dataset"]
    for payload in (reference, stage1):
        payload["protocol"] = None
        payload["tracking"] = {**payload["tracking"], "group": None}
        payload["training"] = {**payload["training"], "epochs": None}
        payload["training"].pop("train_image_size", None)
        payload["scheduler"] = {**payload["scheduler"], "milestones": None}
        payload["method"].pop("selection_step_size_independent", None)
        payload["method"] = {
            **payload["method"],
            "attack": {**payload["method"]["attack"], "steps": None, "step_size": None, "step_size_value": None},
        }
    assert reference == stage1


def test_stage2_config_is_the_reference_except_init_lr_and_short_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _env(monkeypatch, tmp_path)
    reference = _dump("imagenet_mobilenetv4_pgd_at_random_init_lr0025.yaml")
    stage2 = _dump("imagenet_mobilenetv4_twostage_stage2_224_pgd3_ft.yaml")
    assert stage2["protocol"]["id"] == PROTOCOL
    assert stage2["tracking"]["group"] == "imagenet-mobilenetv4-twostage-lowres-stage2"
    assert stage2["training"]["init_checkpoint"] == {"path": str(tmp_path / "stage1" / "last.pt"), "sha256": "d" * 64}
    assert "train_image_size" not in stage2["training"] and stage2["student"]["pretrained"] is False
    assert stage2["training"]["epochs"] == 20 and stage2["optimizer"]["learning_rate"] == 0.0025
    assert stage2["scheduler"]["milestones"] == [10, 15] and stage2["scheduler"]["warmup_epochs"] == 2
    # The training attack is the reference's own 224px PGD-3 (step 8/765), exactly.
    assert stage2["method"] == reference["method"]
    for payload in (reference, stage2):
        payload["protocol"] = None
        payload["tracking"] = {**payload["tracking"], "group": None}
        payload["training"] = {**payload["training"], "epochs": None}
        payload["training"].pop("init_checkpoint", None)
        payload["optimizer"] = {**payload["optimizer"], "learning_rate": None}
        payload["scheduler"] = {**payload["scheduler"], "milestones": None, "warmup_epochs": None}
    assert reference == stage2


def test_stage_configs_are_constructible_on_the_cli_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch, tmp_path)
    execution: dict[str, object] = {
        "world_size": 1,
        "per_rank_batch_size": 128,
        "global_batch_size": 128,
        "effective_global_batch_size": 128,
        "batchnorm_mode": "local_per_rank",
    }
    for name in sorted(STAGE_CONFIGS):
        config = load_config(CONFIG_DIR / name)
        objective, _, _, _ = _build_method(config)
        assert objective is not None
        assert PROTOCOL in str(canonical_run_group(config, training_execution=execution))
