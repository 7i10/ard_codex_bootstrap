"""``training.selection_subset_size`` (plan 0103 Phase 2, human decision 2026-10-08):
default off, byte-identical existing configs, recorded in the hash and the evaluation
pooling identity, refused by runtimes that do not implement it, and a fixed, seeded,
class-stratified subset of the held-out split that leaves the training partition alone.

The trainer behaviour is in ``tests/integration/test_selection_subset_trainer.py`` and the
end-to-end CLI run in ``tests/integration/test_selection_subset_training.py``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch
import yaml

from ard.cli.evaluate import _selection_protocol_identity
from ard.config.loader import _expand_environment, resolved_config_dict
from ard.config.schema import ExperimentConfig, TrainingConfig, reject_throughput_options
from ard.data import (
    IndexedDataset,
    SyntheticCIFAR,
    build_selection_subset_view,
    selection_subset_ids,
    stratified_train_validation_split,
    train_probe_ids,
)
from ard.engine.checkpoint import config_digest

pytestmark = pytest.mark.t1

ROOT = Path(__file__).resolve().parents[2]
# The commit this change is based on: the pre-change schema and configs.
PRE_CHANGE_COMMIT = "922222a"


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:  # pragma: no cover - shallow clones
        pytest.skip(f"git {' '.join(args)} unavailable: {error}")


def _pre_change_schema() -> types.ModuleType:
    relative = "src/ard/config/schema.py"
    name = "_ard_schema_pre_selection_subset"
    module = types.ModuleType(name)
    module.__package__ = "ard.config"
    module.__file__ = f"{PRE_CHANGE_COMMIT}:{relative}"
    sys.modules[name] = module
    exec(compile(_git("show", f"{PRE_CHANGE_COMMIT}:{relative}"), module.__file__, "exec"), module.__dict__)
    return module


def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Every environment variable a repository config interpolates (as in test_cuda_graph_config)."""
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
    }.items():
        monkeypatch.setenv(key, value)


def _pre_change_configs() -> list[str]:
    listed = _git("ls-tree", "-r", "--name-only", PRE_CHANGE_COMMIT, "configs/").splitlines()
    return [
        path
        for path in listed
        if path.endswith(".yaml") and path.split("/")[1] in {"experiments", "pilot", "production", "scientific"}
    ]


# =========================================================================== config


def test_default_off_and_unserialized() -> None:
    training = TrainingConfig(per_rank_batch_size=4, global_batch_size=4)
    assert training.selection_subset_size is None
    assert "selection_subset_size" not in json.loads(training.model_dump_json())
    with pytest.raises(ValueError):
        TrainingConfig(per_rank_batch_size=4, global_batch_size=4, selection_subset_size=0)


def test_configs_that_existed_before_the_change_serialize_exactly_as_under_the_pre_change_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every config as it was at PRE_CHANGE_COMMIT (content read from git, so later edits
    or new configs cannot break this) resolves and hashes byte-identically."""
    _env(monkeypatch, tmp_path)
    old_schema = _pre_change_schema()
    paths = _pre_change_configs()
    assert len(paths) > 20
    for path in paths:
        expanded = _expand_environment(yaml.safe_load(_git("show", f"{PRE_CHANGE_COMMIT}:{path}")))
        new = ExperimentConfig.model_validate(expanded)
        old = old_schema.ExperimentConfig.model_validate(expanded)
        assert new.training.selection_subset_size is None, path
        assert resolved_config_dict(new) == json.loads(old.model_dump_json()), path
        assert config_digest(resolved_config_dict(new)) == config_digest(json.loads(old.model_dump_json()))


def test_setting_it_changes_the_hash_and_survives_a_resolved_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _env(monkeypatch, tmp_path)
    path = next(p for p in _pre_change_configs() if "imagenet" in p and "stage1" in p)
    raw = resolved_config_dict(
        ExperimentConfig.model_validate(
            _expand_environment(yaml.safe_load(_git("show", f"{PRE_CHANGE_COMMIT}:{path}")))
        )
    )
    enabled_raw = {**raw, "training": {**raw["training"], "selection_subset_size": 5000}}
    enabled = ExperimentConfig.model_validate(enabled_raw)
    assert resolved_config_dict(enabled)["training"]["selection_subset_size"] == 5000
    assert config_digest(resolved_config_dict(enabled)) != config_digest(raw)
    assert ExperimentConfig.model_validate(resolved_config_dict(enabled)).training.selection_subset_size == 5000


def test_runtimes_other_than_the_trainer_cli_refuse_it() -> None:
    training = TrainingConfig(per_rank_batch_size=4, global_batch_size=4, selection_subset_size=10)
    with pytest.raises(ValueError, match="fixture does not implement training.selection_subset_size=10"):
        reject_throughput_options(training, runtime="fixture")


def test_pooling_identity_records_it_only_when_set() -> None:
    """It changes which epoch is best.pt, so a subset-selected run never pools with a
    full-split-selected one; unset keeps every recorded identity byte-identical."""
    assert _selection_protocol_identity(TrainingConfig(per_rank_batch_size=4, global_batch_size=4)) == {}
    enabled = TrainingConfig(per_rank_batch_size=4, global_batch_size=4, selection_subset_size=5000)
    assert _selection_protocol_identity(enabled) == {"selection_subset_size": 5000}


# =========================================================================== subset


def _split(seed: int = 11) -> tuple[list[int], list[int], list[int]]:
    dataset = IndexedDataset(SyntheticCIFAR(size=400, num_classes=10, image_size=4, seed=3))
    train, validation = stratified_train_validation_split(dataset, validation_fraction=0.25, seed=seed)
    targets = list(dataset.dataset.targets)
    return train.indices, validation.indices, targets


def test_subset_is_fixed_sorted_stratified_and_inside_the_held_out_split() -> None:
    train, validation, targets = _split()
    first = selection_subset_ids(validation, targets, size=33, seed=11)
    again = selection_subset_ids(list(validation), list(targets), size=33, seed=11)
    assert first == again  # deterministic: same every epoch / run / seed sharing the split
    assert first == sorted(first) and len(set(first)) == 33
    assert set(first) <= set(validation)
    assert not set(first) & set(train)
    # 33 // 10 = 3 per class, three classes get one more from the leftovers.
    per_class = Counter(targets[i] for i in first)
    assert set(per_class) == set(range(10)) and min(per_class.values()) >= 3 and sum(per_class.values()) == 33
    assert selection_subset_ids(validation, targets, size=33, seed=12) != first


def test_subset_size_must_be_strictly_smaller_than_the_held_out_split() -> None:
    _, validation, targets = _split()
    with pytest.raises(ValueError, match="strictly smaller"):
        selection_subset_ids(validation, targets, size=len(validation), seed=11)
    with pytest.raises(ValueError, match="strictly smaller"):
        selection_subset_ids(validation, targets, size=0, seed=11)
    assert len(selection_subset_ids(validation, targets, size=len(validation) - 1, seed=11)) == len(validation) - 1


def test_view_shares_the_held_out_transform_and_leaves_the_train_partition_alone() -> None:
    dataset = IndexedDataset(SyntheticCIFAR(size=400, num_classes=10, image_size=4, seed=3))
    train, validation = stratified_train_validation_split(dataset, validation_fraction=0.25, seed=11)
    train_ids, validation_ids = list(train.indices), list(validation.indices)
    view = build_selection_subset_view(validation, size=20, seed=11)
    assert view.dataset is validation.dataset
    assert view.indices == selection_subset_ids(validation_ids, dataset.dataset.targets, size=20, seed=11)
    assert train.indices == train_ids and validation.indices == validation_ids
    image, label, source_id = view[0]
    assert source_id == view.indices[0] and label == dataset.dataset.targets[source_id]


def test_train_probe_ids_are_unchanged_by_the_shared_sampler() -> None:
    """train_probe_ids now delegates to the shared stratified sampler; it must return
    exactly what PRE_CHANGE_COMMIT's own function (read from git) returns."""
    source = _git("show", f"{PRE_CHANGE_COMMIT}:src/ard/data/datasets.py")
    snippet = source[source.index("def train_probe_ids(") : source.index("def build_train_probe_view(")]
    namespace: dict[str, object] = {"torch": torch, "defaultdict": defaultdict, "Sequence": Sequence}
    exec(compile(snippet, f"{PRE_CHANGE_COMMIT}:train_probe_ids", "exec"), namespace)
    old_probe_ids = namespace["train_probe_ids"]
    assert callable(old_probe_ids)
    train, _, targets = _split()
    for size, seed in ((10, 11), (37, 5), (1, 0), (len(train), 2)):
        assert train_probe_ids(train, targets, size=size, seed=seed) == old_probe_ids(
            train, targets, size=size, seed=seed
        )
    with pytest.raises(ValueError, match="train probe size"):
        train_probe_ids(train, targets, size=len(train) + 1, seed=11)
