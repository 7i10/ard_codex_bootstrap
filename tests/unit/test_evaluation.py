from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from ard.analysis import ParquetDependencyError, fixed_panel_ids, summarize_checkpoint_groups, write_sample_parquet
from ard.cli.evaluate import _evaluation_tracker_config, _validate_evaluation_tracking_identity
from ard.config.schema import ExperimentConfig, TrainingConfig, training_execution_identity, validate_global_batch_size
from ard.engine.checkpoint import REQUIRED_KEYS
from ard.evaluation.autoattack import run_autoattack
from ard.evaluation.saved_checkpoint import validate_checkpoint_lineage

pytestmark = pytest.mark.t1


def base() -> dict:
    return {
        "schema_version": 2,
        "protocol": {"id": "synthetic_smoke_v2"},
        "tier": "smoke",
        "seeds": {
            k: 0
            for k in (
                "split",
                "model_init",
                "data_order",
                "augmentation",
                "train_attack",
                "evaluation_attack",
                "qualitative_panel",
            )
        },
        "dataset": {"name": "synthetic_cifar", "num_samples": 4, "num_classes": 2, "split": "test"},
        "student": {"architecture": "fixture_cnn", "num_classes": 2},
        "method": {"id": "pgd_at", "version": 1, "attack": {"steps": 1}},
        "optimizer": {"id": "sgd", "learning_rate": 0.01, "momentum": 0.9, "weight_decay": 0.0, "nesterov": False},
        "scheduler": {"id": "identity", "milestones": [], "gamma": 1.0, "step_at": "epoch_end"},
        "training": {"epochs": 1, "per_rank_batch_size": 2, "global_batch_size": 2},
    }


def _tracked_repro_config() -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "schema_version": 2,
            "protocol": {"id": "controlled_cifar10_r18_v1"},
            "tier": "repro",
            "seeds": {
                "split": 20260722,
                "model_init": 0,
                "data_order": 0,
                "augmentation": 0,
                "train_attack": 0,
                "evaluation_attack": 0,
                "qualitative_panel": 0,
            },
            "dataset": {
                "name": "cifar10",
                "root": "data",
                "split": "train",
                "download": False,
                "num_classes": 10,
                "image_size": 32,
            },
            "student": {
                "architecture": "saad_resnet18_cifar_v1",
                "num_classes": 10,
                "preprocessing_owner": "student_adapter",
                "normalization": {"profile": "cifar10_raw_identity"},
            },
            "teacher": {
                "source": "checkpoint",
                "architecture": "fixture_cnn",
                "num_classes": 10,
                "checkpoint": "teacher.pt",
                "checkpoint_sha256": "a" * 64,
            },
            "method": {
                "id": "rslad",
                "version": 1,
                "attack": {
                    "loss": "kl",
                    "kl_target": "teacher_clean",
                    "temperature": 1.0,
                    "temperature_squared": True,
                    "epsilon": "8/255",
                    "step_size": "2/255",
                    "steps": 10,
                    "random_start": True,
                    "student_mode": "eval",
                    "teacher_mode": "eval",
                },
                "selection_attack": {
                    "loss": "ce",
                    "temperature": 1.0,
                    "temperature_squared": True,
                    "epsilon": "8/255",
                    "step_size": "2/255",
                    "steps": 20,
                    "random_start": True,
                    "student_mode": "eval",
                    "teacher_mode": "eval",
                },
            },
            "optimizer": {
                "id": "sgd",
                "learning_rate": 0.1,
                "momentum": 0.9,
                "weight_decay": 5e-4,
                "nesterov": False,
            },
            "scheduler": {"id": "multistep", "milestones": [100, 150], "gamma": 0.1, "step_at": "epoch_end"},
            "training": {
                "epochs": 200,
                "per_rank_batch_size": 128,
                "global_batch_size": 128,
                "deterministic": True,
                "validation_fraction": 0.1,
            },
            "tracking": {"mode": "offline_sync", "project": "project", "entity": "entity", "group": "group"},
            "evaluation": {"seed": 0},
        }
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("tier", "dev", "tier may not downgrade"),
        ("project", "other-project", "tracking project"),
        ("entity", "other-entity", "tracking entity"),
        ("group", "other-group", "tracking group"),
    ),
)
def test_evaluation_rejects_repro_tier_and_tracking_identity_downgrade(field: str, value: str, message: str) -> None:
    training = _tracked_repro_config()
    if field == "tier":
        evaluation = training.model_copy(update={"tier": value})
    else:
        evaluation = training.model_copy(update={"tracking": training.tracking.model_copy(update={field: value})})

    with pytest.raises(ValueError, match=message):
        _validate_evaluation_tracking_identity(evaluation, training)


def test_evaluation_protocol_id_must_match_resolved_training_config() -> None:
    training = _tracked_repro_config()
    evaluation = training.model_copy(
        update={"protocol": training.protocol.model_copy(update={"id": "saad_paper_reproduction_v1"})}
    )
    with pytest.raises(ValueError, match="protocol ID must match"):
        _validate_evaluation_tracking_identity(evaluation, training)


def test_evaluation_tracker_config_uses_training_identity_and_evaluation_only_fields(tmp_path: Path) -> None:
    training = _tracked_repro_config()
    evaluation = training.model_copy(
        update={
            "optimizer": training.optimizer.model_copy(update={"learning_rate": 0.025}),
            "training": training.training.model_copy(update={"num_workers": 3}),
            "evaluation": training.evaluation.model_copy(update={"seed": 91}),
            "output_dir": tmp_path / "caller-output",
        }
    )

    tracker_config = _evaluation_tracker_config(evaluation, training, output_dir=tmp_path / "evaluation-output")

    assert tracker_config.protocol == training.protocol
    assert tracker_config.seeds == training.seeds
    assert tracker_config.teacher == training.teacher
    assert tracker_config.method == training.method
    assert tracker_config.optimizer == training.optimizer
    assert tracker_config.training == training.training
    assert tracker_config.tracking == training.tracking
    assert tracker_config.evaluation == evaluation.evaluation
    assert tracker_config.output_dir == tmp_path / "evaluation-output"


def test_evaluation_attack_must_be_explicit_eval_ce_and_panel_is_stable() -> None:
    data = base()
    data["evaluation"] = {"attack": {"steps": 1, "student_mode": "train"}}
    with pytest.raises(ValueError, match="evaluation PGD must keep"):
        ExperimentConfig.model_validate(data)
    assert fixed_panel_ids([3, 1, 3, 7, 2], seed=4, size=3) == fixed_panel_ids([7, 2, 1, 3], seed=4, size=3)
    grouped = summarize_checkpoint_groups(
        [
            _evaluation_row(checkpoint="best", run_id="run-a", training_seed=1, teacher="teacher-a", robust=0.5),
            _evaluation_row(checkpoint="last", run_id="run-a", training_seed=1, teacher="teacher-a", robust=0.3),
        ],
        metric="robust",
    )
    assert set(grouped) == {"best", "last"}
    with pytest.raises(ValueError, match="mixed threat"):
        summarize_checkpoint_groups(
            [
                _evaluation_row(
                    checkpoint="best", run_id="run-a", training_seed=1, teacher="teacher-a", threat_hash="a"
                ),
                _evaluation_row(
                    checkpoint="last", run_id="run-a", training_seed=1, teacher="teacher-a", threat_hash="b"
                ),
            ],
            metric="robust",
        )


def test_global_batch_identity_is_exact_for_checkpoint_world_size() -> None:
    assert validate_global_batch_size(per_rank_batch_size=4, global_batch_size=8, world_size=2) == 8
    with pytest.raises(ValueError, match="global_batch_size"):
        validate_global_batch_size(per_rank_batch_size=4, global_batch_size=4, world_size=2)


def test_execution_identity_distinguishes_local_batchnorm_profiles_with_equal_global_batch() -> None:
    one_gpu = training_execution_identity(
        training=TrainingConfig(per_rank_batch_size=128, global_batch_size=128), world_size=1
    )
    two_gpu = training_execution_identity(
        training=TrainingConfig(per_rank_batch_size=64, global_batch_size=128), world_size=2
    )
    assert one_gpu["effective_global_batch_size"] == two_gpu["effective_global_batch_size"] == 128
    assert one_gpu["batchnorm_mode"] == two_gpu["batchnorm_mode"] == "local_per_rank"
    assert one_gpu != two_gpu


def test_parquet_never_falls_back_to_a_mislabeled_file(tmp_path: Path) -> None:
    path = tmp_path / "stats.parquet"
    if importlib.util.find_spec("pyarrow") is None:
        with pytest.raises(ParquetDependencyError, match="pyarrow"):
            write_sample_parquet([{"sample_id": 1, "clean_correct": True}], path)
        assert not path.exists()
    else:
        write_sample_parquet([{"sample_id": 1, "clean_correct": True}], path)
        assert path.read_bytes()[:4] == b"PAR1"


def test_autoattack_adapter_maps_linf_and_restores_eval_mode(tmp_path: Path) -> None:
    import torch
    from torch import nn

    received = {}
    instances = []

    class FakeAA:
        def __init__(self, model: nn.Module, **kwargs: object) -> None:
            received.update(kwargs)
            instances.append(self)

        def run_standard_evaluation(self, images: torch.Tensor, labels: torch.Tensor, bs: int) -> torch.Tensor:
            assert bs == 128
            return images

    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 2 * 2, 2))
    model.train()
    result = run_autoattack(
        model=model,
        images=torch.rand(2, 3, 2, 2),
        labels=torch.tensor([0, 1]),
        norm="linf",
        epsilon=8 / 255,
        seed=4,
        output_path=tmp_path / "aa.json",
        autoattack_cls=FakeAA,
    )
    assert (
        received["norm"] == "Linf"
        and result["version"] == "injected"
        and result["batch_size"] == 128
        and instances[0].seed == 4
        and model.training
    )


def test_autoattack_adapter_restores_mode_when_injected_adapter_raises(tmp_path: Path) -> None:
    import torch
    from torch import nn

    class FailingAA:
        def __init__(self, model: nn.Module, **kwargs: object) -> None:
            del model, kwargs

        def run_standard_evaluation(self, images: torch.Tensor, labels: torch.Tensor, bs: int) -> torch.Tensor:
            del images, labels, bs
            raise RuntimeError("injected AutoAttack failure")

    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 2 * 2, 2))
    model.train()
    with pytest.raises(RuntimeError, match="injected AutoAttack failure"):
        run_autoattack(
            model=model,
            images=torch.rand(2, 3, 2, 2),
            labels=torch.tensor([0, 1]),
            norm="linf",
            epsilon=8 / 255,
            seed=4,
            output_path=tmp_path / "aa.json",
            autoattack_cls=FailingAA,
        )
    assert model.training


def _lineage_payload(*, boundary: str = "end", world_size: object = 1) -> dict[str, object]:
    payload: dict[str, object] = {key: None for key in REQUIRED_KEYS}
    payload.update(
        {
            "model": {},
            "config_hash": "config-hash",
            "epoch_boundary": boundary,
            "world_size": world_size,
        }
    )
    return payload


def test_checkpoint_lineage_requires_complete_epoch_boundary_payload(tmp_path: Path) -> None:
    import torch

    missing = _lineage_payload()
    del missing["rng"]
    missing_path = tmp_path / "missing.pt"
    torch.save(missing, missing_path)
    with pytest.raises(ValueError, match="missing: rng"):
        validate_checkpoint_lineage(missing_path, expected_config_hash="config-hash")

    boundary_path = tmp_path / "mid-epoch.pt"
    torch.save(_lineage_payload(boundary="mid"), boundary_path)
    with pytest.raises(ValueError, match="epoch-boundary"):
        validate_checkpoint_lineage(boundary_path, expected_config_hash="config-hash")


@pytest.mark.parametrize("world_size", (0, -1, True, "1"))
def test_checkpoint_lineage_requires_valid_world_size(tmp_path: Path, world_size: object) -> None:
    import torch

    checkpoint = tmp_path / "bad-world-size.pt"
    torch.save(_lineage_payload(world_size=world_size), checkpoint)
    with pytest.raises(ValueError, match="world_size"):
        validate_checkpoint_lineage(checkpoint, expected_config_hash="config-hash")


def _evaluation_row(
    *,
    checkpoint: str,
    run_id: str,
    training_seed: int,
    teacher: str,
    dataset_identity: dict[str, str] | None = None,
    student_identity: dict[str, str] | None = None,
    method_identity: dict[str, str] | None = None,
    threat_hash: str = "threat-a",
    robust: float = 0.5,
    training_seeds: dict[str, int] | None = None,
) -> dict[str, object]:
    resolved_training_seeds = training_seeds or {
        "split": training_seed,
        "model_init": training_seed,
        "data_order": training_seed,
        "augmentation": training_seed,
        "train_attack": training_seed,
        "evaluation_attack": training_seed,
        "qualitative_panel": training_seed,
    }
    return {
        "checkpoint_alias": checkpoint,
        "checkpoint_filename": f"{checkpoint}.pt",
        "checkpoint_sha256": "a" * 64,
        "run_id": run_id,
        "train_run_id": run_id,
        "training_seed": training_seed,
        "training_seeds": resolved_training_seeds,
        "teacher_identity": {"checkpoint_sha256": teacher},
        "dataset_identity": dataset_identity or {"name": "cifar10", "split": "test"},
        "training_dataset_identity": {"name": "cifar10", "split": "train"},
        "student_identity": student_identity or {"architecture": "resnet18"},
        "method_identity": method_identity or {"name": "pgd_at"},
        "training_protocol_identity": {
            "id": "controlled_cifar10_r18_v1",
            "epochs": 1,
            "execution": {
                "world_size": 1,
                "per_rank_batch_size": 4,
                "global_batch_size": 4,
                "effective_global_batch_size": 4,
                "batchnorm_mode": "local_per_rank",
            },
        },
        "evaluation_protocol_identity": {"seed": 0, "loader_batch_size": 4},
        "threat_hash": threat_hash,
        "evaluation_seed": 0,
        "robust": robust,
    }


def test_checkpoint_aggregation_allows_multiple_seed_teacher_axes() -> None:
    rows = [
        _evaluation_row(checkpoint=checkpoint, run_id="run-a", training_seed=1, teacher="teacher-a")
        for checkpoint in ("best", "last")
    ] + [
        _evaluation_row(checkpoint=checkpoint, run_id="run-b", training_seed=1, teacher="teacher-a")
        for checkpoint in ("best", "last")
    ]

    grouped = summarize_checkpoint_groups(rows, metric="robust")

    assert {checkpoint: grouped[checkpoint]["count"] for checkpoint in ("best", "last")} == {"best": 2, "last": 2}


@pytest.mark.parametrize(
    ("field", "changed_value", "message"),
    [
        ("dataset_identity", {"name": "cifar100", "split": "test"}, "mixed experiment identities"),
        ("student_identity", {"architecture": "wide_resnet"}, "mixed experiment identities"),
        ("method_identity", {"name": "trades"}, "mixed experiment identities"),
        ("threat_hash", "threat-b", "mixed threat models"),
    ],
)
def test_checkpoint_aggregation_rejects_mixed_structured_identity(
    field: str, changed_value: object, message: str
) -> None:
    rows = [
        _evaluation_row(checkpoint=checkpoint, run_id="run-a", training_seed=1, teacher="teacher-a")
        for checkpoint in ("best", "last")
    ] + [
        _evaluation_row(checkpoint=checkpoint, run_id="run-b", training_seed=2, teacher="teacher-b")
        for checkpoint in ("best", "last")
    ]
    for row in rows[2:]:
        row[field] = changed_value

    with pytest.raises(ValueError, match=message):
        summarize_checkpoint_groups(rows, metric="robust")


def test_checkpoint_aggregation_requires_best_last_for_each_axis() -> None:
    rows = [
        _evaluation_row(checkpoint=checkpoint, run_id="run-a", training_seed=1, teacher="teacher-a")
        for checkpoint in ("best", "last")
    ] + [_evaluation_row(checkpoint="best", run_id="run-b", training_seed=2, teacher="teacher-b")]

    with pytest.raises(ValueError, match="exactly one best and one last"):
        summarize_checkpoint_groups(rows, metric="robust")


def test_checkpoint_aggregation_rejects_duplicate_checkpoint_for_one_train_run() -> None:
    rows = [
        _evaluation_row(checkpoint=checkpoint, run_id="run-a", training_seed=1, teacher="teacher-a")
        for checkpoint in ("best", "last", "best")
    ]

    with pytest.raises(ValueError, match="exactly one best and one last"):
        summarize_checkpoint_groups(rows, metric="robust")


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("evaluation_seed", 7),
        (
            "training_protocol_identity",
            {
                "id": "controlled_cifar10_r18_v1",
                "epochs": 2,
                "execution": {
                    "world_size": 1,
                    "per_rank_batch_size": 4,
                    "global_batch_size": 4,
                    "effective_global_batch_size": 4,
                    "batchnorm_mode": "local_per_rank",
                },
            },
        ),
    ),
)
def test_checkpoint_aggregation_rejects_mixed_evaluation_seed_or_training_protocol(field: str, value: object) -> None:
    rows = [
        _evaluation_row(checkpoint=checkpoint, run_id="run-a", training_seed=1, teacher="teacher-a")
        for checkpoint in ("best", "last")
    ] + [
        _evaluation_row(checkpoint=checkpoint, run_id="run-b", training_seed=2, teacher="teacher-b")
        for checkpoint in ("best", "last")
    ]
    for row in rows[2:]:
        row[field] = value

    with pytest.raises(ValueError, match="mixed experiment identities"):
        summarize_checkpoint_groups(rows, metric="robust")


def test_checkpoint_aggregation_rejects_rows_differing_only_in_protocol_id() -> None:
    rows = [
        _evaluation_row(checkpoint=checkpoint, run_id="run-a", training_seed=1, teacher="teacher-a")
        for checkpoint in ("best", "last")
    ] + [
        _evaluation_row(checkpoint=checkpoint, run_id="run-b", training_seed=2, teacher="teacher-b")
        for checkpoint in ("best", "last")
    ]
    for row in rows[2:]:
        protocol = row["training_protocol_identity"]
        assert isinstance(protocol, dict)
        row["training_protocol_identity"] = {**protocol, "id": "synthetic_smoke_v2"}

    with pytest.raises(ValueError, match="mixed experiment identities"):
        summarize_checkpoint_groups(rows, metric="robust")


def test_checkpoint_aggregation_allows_different_dataset_provenance_roots() -> None:
    rows = [
        _evaluation_row(checkpoint=checkpoint, run_id="run-a", training_seed=1, teacher="teacher-a")
        for checkpoint in ("best", "last")
    ] + [
        _evaluation_row(checkpoint=checkpoint, run_id="run-b", training_seed=2, teacher="teacher-b")
        for checkpoint in ("best", "last")
    ]
    for row, root in zip(rows, ("/datasets/a", "/datasets/a", "/datasets/b", "/datasets/b"), strict=True):
        row["dataset_provenance"] = {"root": root}

    assert set(summarize_checkpoint_groups(rows, metric="robust")) == {"best", "last"}


def test_checkpoint_aggregation_rejects_contradictory_metadata_for_one_train_run() -> None:
    changed_seeds = {
        "split": 2,
        "model_init": 1,
        "data_order": 1,
        "augmentation": 1,
        "train_attack": 1,
        "evaluation_attack": 1,
        "qualitative_panel": 1,
    }
    rows = [
        _evaluation_row(checkpoint="best", run_id="run-a", training_seed=1, teacher="teacher-a"),
        _evaluation_row(
            checkpoint="last",
            run_id="run-a",
            training_seed=1,
            training_seeds=changed_seeds,
            teacher="teacher-a",
        ),
    ]

    with pytest.raises(ValueError, match="contradictory metadata"):
        summarize_checkpoint_groups(rows, metric="robust")


def test_checkpoint_aggregation_requires_every_canonical_result_field() -> None:
    row = _evaluation_row(checkpoint="best", run_id="run-a", training_seed=1, teacher="teacher-a")
    del row["training_dataset_identity"]

    with pytest.raises(ValueError, match="training_dataset_identity"):
        summarize_checkpoint_groups([row], metric="robust")


@pytest.mark.parametrize(
    ("field", "value"),
    (
        (
            "training_protocol_identity",
            {
                "id": "controlled_cifar10_r18_v1",
                "epochs": 1,
                "execution": {
                    "world_size": 2,
                    "per_rank_batch_size": 2,
                    "global_batch_size": 4,
                    "effective_global_batch_size": 4,
                    "batchnorm_mode": "local_per_rank",
                },
            },
        ),
        ("dataset_identity", {"name": "cifar10", "split": "test", "content_fingerprint": "b" * 64}),
    ),
)
def test_checkpoint_aggregation_rejects_mixed_world_size_or_content_fingerprint(field: str, value: object) -> None:
    rows = [
        _evaluation_row(checkpoint=checkpoint, run_id="run-a", training_seed=1, teacher="teacher-a")
        for checkpoint in ("best", "last")
    ] + [
        _evaluation_row(checkpoint=checkpoint, run_id="run-b", training_seed=2, teacher="teacher-b")
        for checkpoint in ("best", "last")
    ]
    for row in rows[2:]:
        row[field] = value

    with pytest.raises(ValueError, match="mixed experiment identities"):
        summarize_checkpoint_groups(rows, metric="robust")


def test_checkpoint_aggregation_rejects_local_batchnorm_execution_profile_mixing() -> None:
    rows = [
        _evaluation_row(checkpoint=checkpoint, run_id="run-a", training_seed=1, teacher="teacher-a")
        for checkpoint in ("best", "last")
    ] + [
        _evaluation_row(checkpoint=checkpoint, run_id="run-b", training_seed=2, teacher="teacher-b")
        for checkpoint in ("best", "last")
    ]
    for row in rows[2:]:
        protocol = row["training_protocol_identity"]
        assert isinstance(protocol, dict)
        row["training_protocol_identity"] = {
            **protocol,
            "execution": {
                "world_size": 2,
                "per_rank_batch_size": 2,
                "global_batch_size": 4,
                "effective_global_batch_size": 4,
                "batchnorm_mode": "local_per_rank",
            },
        }
    with pytest.raises(ValueError, match="mixed experiment identities"):
        summarize_checkpoint_groups(rows, metric="robust")
