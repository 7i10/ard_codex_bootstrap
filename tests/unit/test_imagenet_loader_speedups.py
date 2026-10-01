"""Plan 0103 ImageNet loader speedups (human-approved 2026-09-30).

B -- ``training.jpeg_draft_decode``: the RandomResizedCrop training view
decodes at a reduced JPEG DCT scale. Same RNG draws, crop boxes and flips as
the default path; never an upscale caused by the reduction; bit-identical
whenever no reduction applies; validation views untouched.

C -- ``dataset.derived_from`` + ``scripts/build_resized_imagenet.py``: a
pre-resized derivative root with identical relative paths (so labels, source
IDs and the seeded split are identical), verified against its build manifest
at load time; stage 2 accepts a stage-1 source trained on a declared
derivative of its own dataset and nothing else.

Both default-off: every existing config serializes (and hashes) exactly as
under the pre-change schema, and default views equal the pre-change views.
"""

from __future__ import annotations

import hashlib
import importlib.util
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

from ard.cli.evaluate import _data_loading_protocol_identity, _dataset_identity
from ard.cli.train import _build_method
from ard.config import load_config
from ard.config.loader import _expand_environment, resolved_config_dict, save_resolved_config
from ard.config.schema import DatasetConfig, ExperimentConfig, ModelConfig, TrainingConfig, reject_throughput_options
from ard.data import build_train_validation_views
from ard.data.datasets import (
    DERIVED_DATASET_MANIFEST,
    EpochImageNetTransform,
    ImageNetDataset,
    build_raw_dataset,
)
from ard.engine.checkpoint import REQUIRED_KEYS, config_digest, read_init_checkpoint
from ard.models import build_student

pytestmark = pytest.mark.t1

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "scientific"
PROTOCOL = "controlled_imagenet_stage02_two_stage_lowres_v1"
# The commit this change is based on: the pre-change schema and data views.
PRE_CHANGE_COMMIT = "29caab2"
NEW_CONFIGS = {
    "imagenet_mobilenetv4_twostage_stage1_112_pgd1_resized_s160.yaml",
    "imagenet_mobilenetv4_twostage_stage1_112_pgd1_resized_s256.yaml",
}
REAL_IMAGENET_TRAIN = Path.home() / "workspace-local" / "datasets" / "imagenet" / "train"


def _load_builder() -> types.ModuleType:
    path = ROOT / "scripts" / "build_resized_imagenet.py"
    spec = importlib.util.spec_from_file_location("build_resized_imagenet", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered so multiprocessing can pickle the worker function by name.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load_builder()


def _pre_change_module(relative: str, name: str, package: str) -> types.ModuleType:
    try:
        source = subprocess.run(
            ["git", "show", f"{PRE_CHANGE_COMMIT}:{relative}"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:  # pragma: no cover - shallow clones
        pytest.skip(f"pre-change {relative} at {PRE_CHANGE_COMMIT} unavailable from git: {error}")
    module = types.ModuleType(name)
    module.__package__ = package
    module.__file__ = f"{PRE_CHANGE_COMMIT}:{relative}"
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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


def _smooth_image(width: int, height: int, seed: int) -> Image.Image:
    """Low-frequency content (JPEG-friendly; reduced-scale decoding is then close)."""
    generator = torch.Generator().manual_seed(seed)
    coarse = torch.rand(1, 3, max(2, height // 64), max(2, width // 64), generator=generator)
    pixels = torch.nn.functional.interpolate(coarse, size=(height, width), mode="bilinear", align_corners=False)[0]
    return Image.fromarray((pixels.permute(1, 2, 0) * 255).round().to(torch.uint8).numpy())


def _imagenet_layout(root: Path, sizes: list[tuple[int, int]], *, extra: bool = False) -> None:
    """Two classes; images of the given sizes (JPEG), optionally plus greyscale,
    CMYK and PNG-bytes-named-.JPEG oddities like real ImageNet has."""
    for class_index, wnid in enumerate(("n001", "n002")):
        class_dir = root / "train" / wnid
        class_dir.mkdir(parents=True, exist_ok=True)
        for image_index, (width, height) in enumerate(sizes):
            image = _smooth_image(width, height, class_index * 100 + image_index)
            image.save(class_dir / f"{wnid}_{image_index}.JPEG", quality=90)
        if extra:
            base = _smooth_image(700, 600, class_index + 50)
            base.convert("L").save(class_dir / f"{wnid}_grey.JPEG", quality=90)
            base.convert("CMYK").save(class_dir / f"{wnid}_cmyk.JPEG", quality=90)
            base.save(class_dir / f"{wnid}_png.JPEG", format="PNG")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_sha(root: Path) -> str:
    return str(ImageNetDataset(root, "train", image_size=32).content_identity["observed_sha256"])


# =========================================================================== defaults


def test_new_fields_default_off_and_unserialized() -> None:
    training = TrainingConfig(per_rank_batch_size=4, global_batch_size=4)
    assert training.jpeg_draft_decode is False
    assert "jpeg_draft_decode" not in json.loads(training.model_dump_json())
    dataset = DatasetConfig(name="imagenet", root=Path("/x"), num_classes=2)
    assert dataset.derived_from is None
    assert "derived_from" not in json.loads(dataset.model_dump_json())


def test_existing_configs_serialize_exactly_as_under_the_pre_change_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolved config and config hash of every pre-existing config is
    byte-identical to the pre-change schema's."""
    _env(monkeypatch, tmp_path)
    old_schema = _pre_change_module("src/ard/config/schema.py", "_ard_schema_pre_loader_speedups", "ard.config")
    # Configs as they were at PRE_CHANGE_COMMIT, content read from git, so configs added
    # later (which may use newer schema fields) cannot break this check.
    listed = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", PRE_CHANGE_COMMIT, "configs/"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    paths = [
        path
        for path in listed
        if path.endswith(".yaml") and path.split("/")[1] in {"experiments", "pilot", "production", "scientific"}
    ]
    assert len(paths) > 20
    for path in paths:
        text = subprocess.run(
            ["git", "show", f"{PRE_CHANGE_COMMIT}:{path}"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout
        expanded = _expand_environment(yaml.safe_load(text))
        new = ExperimentConfig.model_validate(expanded)
        old = old_schema.ExperimentConfig.model_validate(expanded)
        assert resolved_config_dict(new) == json.loads(old.model_dump_json()), path
        assert config_digest(resolved_config_dict(new)) == config_digest(json.loads(old.model_dump_json()))


def _views(module: Any, config: DatasetConfig, **extra: Any) -> tuple[Any, Any]:
    return module.build_train_validation_views(
        config, validation_fraction=0.34, split_seed=1, augmentation_seed=7, **extra
    )


def _materialize(view: Any, epochs: tuple[int, ...] = (0, 3)) -> list[tuple[torch.Tensor, int, int]]:
    items: list[tuple[torch.Tensor, int, int]] = []
    for epoch in epochs:
        view.set_epoch(epoch)
        items.extend(view[index] for index in range(len(view)))
    return items


def _assert_same(first: list[Any], second: list[Any]) -> None:
    assert len(first) == len(second)
    for (image_a, label_a, id_a), (image_b, label_b, id_b) in zip(first, second, strict=True):
        assert (label_a, id_a) == (label_b, id_b)
        assert torch.equal(image_a, image_b)


def test_default_views_are_bit_identical_to_the_pre_change_views(tmp_path: Path) -> None:
    old_data = _pre_change_module("src/ard/data/datasets.py", "ard.data._datasets_pre_loader_speedups", "ard.data")
    _imagenet_layout(tmp_path, [(300, 260), (90, 120), (520, 400)], extra=True)
    config = DatasetConfig(name="imagenet", root=tmp_path, split="train", num_classes=2, image_size=32)
    old_train, old_validation = _views(old_data, config, train_image_size=24)
    for extra in ({}, {"jpeg_draft_decode": False}):
        new_train, new_validation = _views(sys.modules["ard.data.datasets"], config, train_image_size=24, **extra)
        assert new_train.indices == old_train.indices and new_validation.indices == old_validation.indices
        _assert_same(_materialize(new_train), _materialize(old_train))
        _assert_same(_materialize(new_validation), _materialize(old_validation))


# =========================================================================== B: jpeg_draft_decode


@pytest.mark.parametrize(
    ("crop", "output", "scale"),
    [
        ((896, 896), 112, 8),
        ((895, 2000), 112, 4),
        ((448, 460), 112, 4),
        ((447, 447), 112, 2),
        ((224, 300), 112, 2),
        ((223, 300), 112, 1),
        ((100, 100), 112, 1),
    ],
)
def test_draft_scale_is_the_largest_reduction_that_keeps_a_downscale(
    crop: tuple[int, int], output: int, scale: int
) -> None:
    assert EpochImageNetTransform.draft_scale(crop[0], crop[1], output) == scale


def test_draft_plan_draws_exactly_the_default_crop_box_and_rng_stream() -> None:
    transform = EpochImageNetTransform(augmentation_seed=7, image_size=112, jpeg_draft_decode=True)
    for width, height in [(500, 375), (1600, 1200), (80, 60), (4000, 300), (333, 2500)]:
        for seed in range(40):
            default, draft = torch.Generator().manual_seed(seed), torch.Generator().manual_seed(seed)
            box = transform._crop_box_for_size(width, height, default)
            planned, scale = transform.draft_plan(width, height, draft)
            assert planned == box
            assert torch.equal(default.get_state(), draft.get_state())
            assert scale == EpochImageNetTransform.draft_scale(box[3], box[2], 112)


def _record_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    """Record every sampled crop box and the generator state after the flip draw."""
    calls: list[tuple[Any, ...]] = []
    original = EpochImageNetTransform._crop_box_for_size

    def recording(self: EpochImageNetTransform, width: int, height: int, generator: torch.Generator) -> Any:
        box = original(self, width, height, generator)
        calls.append((width, height, box))
        return box

    monkeypatch.setattr(EpochImageNetTransform, "_crop_box_for_size", recording)
    return calls


def test_draft_views_use_the_same_boxes_flips_and_split_as_the_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _imagenet_layout(tmp_path, [(1400, 1100), (900, 1300), (300, 260)], extra=True)
    config = DatasetConfig(name="imagenet", root=tmp_path, split="train", num_classes=2, image_size=64)
    calls = _record_calls(monkeypatch)
    default_train, default_validation = _views(sys.modules["ard.data.datasets"], config, train_image_size=32)
    draft_train, draft_validation = _views(
        sys.modules["ard.data.datasets"], config, train_image_size=32, jpeg_draft_decode=True
    )
    assert draft_train.indices == default_train.indices
    assert draft_validation.indices == default_validation.indices
    # Validation (and so probe) views: untouched, bit for bit.
    _assert_same(_materialize(draft_validation), _materialize(default_validation))
    calls.clear()
    default_items = _materialize(default_train, epochs=(0, 1, 5))
    default_calls = list(calls)
    calls.clear()
    draft_items = _materialize(draft_train, epochs=(0, 1, 5))
    assert calls == default_calls  # same image sizes and crop boxes, in the same order
    reduced = 0
    for (width, height, box), (default_image, label_a, id_a), (draft_image, label_b, id_b) in zip(
        default_calls, default_items, draft_items, strict=True
    ):
        assert (label_a, id_a) == (label_b, id_b)
        assert draft_image.shape == default_image.shape == (3, 32, 32)
        assert draft_image.dtype == default_image.dtype == torch.float32
        if EpochImageNetTransform.draft_scale(box[3], box[2], 32) == 1:
            assert torch.equal(draft_image, default_image)
        else:
            reduced += 1
            # Same flip: the unflipped comparison is far closer than the flipped one.
            same = (draft_image - default_image).abs().mean()
            flipped = (draft_image - default_image.flip(-1)).abs().mean()
            assert same < 2 / 255 and (flipped > same or torch.allclose(default_image, default_image.flip(-1)))
    assert reduced > 0


def test_draft_decode_never_upscales_and_handles_odd_files(tmp_path: Path) -> None:
    _imagenet_layout(tmp_path, [(1203, 997)], extra=True)
    transform = EpochImageNetTransform(augmentation_seed=3, image_size=40, jpeg_draft_decode=True)
    default = EpochImageNetTransform(augmentation_seed=3, image_size=40)
    for path in sorted((tmp_path / "train").rglob("*.JPEG")):
        for source_id in range(12):
            with Image.open(path) as image:
                width, height = image.size
                decoded = image.convert("RGB")
            box, scale = transform.draft_plan(width, height, torch.Generator().manual_seed(3 + 10_007 * source_id))
            if scale > 1:
                with Image.open(path) as image:
                    drafted = image.draft("RGB", (width // scale, height // scale))
                if drafted is not None:
                    extent = drafted[1]
                    # The mapped crop is still >= the output: a downscale, never an upscale.
                    assert box[3] * extent[2] / width >= 40 and box[2] * extent[3] / height >= 40
            out = transform(path, source_id=source_id)
            reference = default(decoded, source_id=source_id)
            assert out.shape == reference.shape == (3, 40, 40) and out.dtype == torch.float32
            assert torch.isfinite(out).all()
            if "png" in path.name or scale == 1:
                # draft is a no-op for a non-JPEG file; scale 1 is the default path itself.
                assert torch.equal(out, reference)


def test_draft_transform_requires_paths() -> None:
    transform = EpochImageNetTransform(augmentation_seed=0, image_size=8, jpeg_draft_decode=True)
    with pytest.raises(TypeError, match="requires the ImageNet train view to supply image paths"):
        transform(Image.new("RGB", (32, 32)), source_id=0)


def test_jpeg_draft_decode_is_imagenet_only_and_refused_by_other_runtimes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="jpeg_draft_decode is only defined for the imagenet dataset"):
        build_train_validation_views(
            DatasetConfig(name="synthetic_cifar", num_samples=8, num_classes=2),
            validation_fraction=0.5,
            split_seed=0,
            augmentation_seed=0,
            jpeg_draft_decode=True,
        )
    raw = _minimal_experiment(tmp_path)
    raw["protocol"] = {"id": "synthetic_smoke_v2"}
    raw["dataset"] = {"name": "synthetic_cifar", "num_samples": 8, "num_classes": 2}
    del raw["training"]["train_image_size"]
    raw["training"]["jpeg_draft_decode"] = True
    with pytest.raises(ValidationError, match="jpeg_draft_decode is only defined for the imagenet dataset"):
        ExperimentConfig.model_validate(raw)
    training = TrainingConfig(per_rank_batch_size=2, global_batch_size=2, jpeg_draft_decode=True)
    refused = r"training\.jpeg_draft_decode=true; run it with training\.jpeg_draft_decode=false"
    with pytest.raises(ValueError, match=refused):
        reject_throughput_options(training, runtime="fixture")


def test_jpeg_draft_decode_is_part_of_the_run_identity(tmp_path: Path) -> None:
    raw = _minimal_experiment(tmp_path)
    default = ExperimentConfig.model_validate(raw)
    raw["training"]["jpeg_draft_decode"] = True
    draft = ExperimentConfig.model_validate(raw)
    assert resolved_config_dict(draft)["training"]["jpeg_draft_decode"] is True
    assert config_digest(resolved_config_dict(draft)) != config_digest(resolved_config_dict(default))
    assert _data_loading_protocol_identity(default.training) == {}
    assert _data_loading_protocol_identity(draft.training) == {"jpeg_draft_decode": True}


@pytest.mark.skipif(not REAL_IMAGENET_TRAIN.is_dir(), reason="real ImageNet train root not present")
def test_draft_decode_fidelity_on_real_imagenet_jpegs(capsys: pytest.CaptureFixture[str]) -> None:
    """Fidelity number for the human (mean |draft - default| in 0-255 units over
    reduced-scale crops of real JPEGs at 112px); only finiteness is asserted."""
    paths = [path for wnid in sorted(REAL_IMAGENET_TRAIN.iterdir())[:4] for path in sorted(wnid.iterdir())[:10]]
    draft = EpochImageNetTransform(augmentation_seed=0, image_size=112, jpeg_draft_decode=True)
    default = EpochImageNetTransform(augmentation_seed=0, image_size=112)
    diffs: list[float] = []
    for source_id, path in enumerate(paths):
        with Image.open(path) as image:
            width, height = image.size
            decoded = image.convert("RGB")
        _, scale = draft.draft_plan(width, height, torch.Generator().manual_seed(10_007 * source_id))
        out, reference = draft(path, source_id=source_id), default(decoded, source_id=source_id)
        assert out.shape == reference.shape == (3, 112, 112)
        if scale > 1:
            diffs.append(float((out - reference).abs().mean()) * 255)
    assert diffs and all(torch.isfinite(torch.tensor(diffs)))
    with capsys.disabled():
        print(f"\n[fidelity] {len(diffs)}/{len(paths)} reduced crops; mean |diff| = {sum(diffs) / len(diffs):.3f}/255")


# =========================================================================== C: builder


def test_resized_dimensions_rule() -> None:
    assert builder.resized_dimensions(500, 375, 160) == (213, 160)
    assert builder.resized_dimensions(375, 500, 160) == (160, 213)
    assert builder.resized_dimensions(161, 1000, 160) == (160, 994)
    assert builder.resized_dimensions(160, 1000, 160) is None  # not larger: copied unchanged
    assert builder.resized_dimensions(100, 90, 160) is None


def _build(source: Path, out: Path, **overrides: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "source": source,
        "out": out,
        "short_side": 64,
        "quality": 95,
        "source_content_sha256": _manifest_sha(source),
        "workers": 2,
        "log": False,
    }
    arguments.update(overrides)
    return dict(builder.build(**arguments))


def test_builder_mirrors_layout_labels_and_split_and_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _imagenet_layout(source, [(300, 200), (150, 400), (60, 50), (64, 90)], extra=True)
    (source / "class_names.json").write_text('{"0": ["n001", "one"], "1": ["n002", "two"]}', encoding="utf-8")
    result = _build(source, tmp_path / "a")
    again = _build(source, tmp_path / "b")
    manifest_a = (tmp_path / "a" / DERIVED_DATASET_MANIFEST).read_bytes()
    assert manifest_a == (tmp_path / "b" / DERIVED_DATASET_MANIFEST).read_bytes()  # byte-deterministic
    assert result["manifest_sha256"] == again["manifest_sha256"] == hashlib.sha256(manifest_a).hexdigest()
    source_dataset = ImageNetDataset(source, "train", image_size=32)
    derived_dataset = ImageNetDataset(tmp_path / "a", "train", image_size=32)
    relative = [path.relative_to(source / "train") for path, _ in source_dataset.samples]
    assert [path.relative_to(tmp_path / "a" / "train") for path, _ in derived_dataset.samples] == relative
    assert derived_dataset.targets == source_dataset.targets
    assert result["derived_content_sha256"] == derived_dataset.content_identity["observed_sha256"]
    assert result["source_content_sha256"] == source_dataset.content_identity["observed_sha256"]
    assert (tmp_path / "a" / "class_names.json").read_bytes() == (source / "class_names.json").read_bytes()
    assert not (tmp_path / "a" / "val").exists()
    resized = copied = 0
    for path in relative:
        original, derived = source / "train" / path, tmp_path / "a" / "train" / path
        with Image.open(original) as before, Image.open(derived) as after:
            if min(before.size) > 64:
                resized += 1
                assert min(after.size) == 64 and after.mode == "RGB" and after.format == "JPEG"
                assert after.size == builder.resized_dimensions(*before.size, 64)
            else:
                copied += 1
                assert derived.read_bytes() == original.read_bytes()
    assert (resized, copied) == (result["resized_count"], result["copied_count"]) and resized and copied
    assert json.loads(manifest_a)["libjpeg_turbo_version"] == builder.features.version("libjpeg_turbo")
    # Same seeded train/validation partition over the same source IDs.
    source_config = DatasetConfig(name="imagenet", root=source, num_classes=2, image_size=32)
    derived_config = DatasetConfig.model_validate(
        {
            "name": "imagenet",
            "root": tmp_path / "a",
            "num_classes": 2,
            "image_size": 32,
            "content_sha256": result["derived_content_sha256"],
            "derived_from": _derived_from(result),
        }
    )
    source_views = _views(sys.modules["ard.data.datasets"], source_config)
    derived_views = _views(sys.modules["ard.data.datasets"], derived_config)
    assert [view.indices for view in source_views] == [view.indices for view in derived_views]


def test_builder_refuses_a_wrong_source_digest_or_a_populated_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _imagenet_layout(source, [(300, 200)])
    with pytest.raises(ValueError, match="source content_sha256 mismatch"):
        _build(source, tmp_path / "out", source_content_sha256="0" * 64)
    _build(source, tmp_path / "out")
    with pytest.raises(FileExistsError, match="already populated"):
        _build(source, tmp_path / "out")
    resumed = _build(source, tmp_path / "out", resume=True)
    assert resumed["derived_content_sha256"] == _manifest_sha(tmp_path / "out")
    # A resume must continue the same build: other parameters are refused.
    with pytest.raises(ValueError, match="different build parameters: jpeg_quality"):
        _build(source, tmp_path / "out", resume=True, quality=90)
    with pytest.raises(ValueError, match="different build parameters: short_side"):
        _build(source, tmp_path / "out", resume=True, short_side=100)
    (tmp_path / "out" / builder.BUILD_PARAMETERS).unlink()
    with pytest.raises(ValueError, match="cannot resume: no build_parameters.json"):
        _build(source, tmp_path / "out", resume=True)


# =========================================================================== C: identity + load-time checks


def _derived_from(result: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    return {
        "content_sha256": result["source_content_sha256"],
        "transform": "resize_short_side",
        "short_side": result["short_side"],
        "jpeg_quality": result["jpeg_quality"],
        "resample": "lanczos",
        "manifest_sha256": result["manifest_sha256"],
        **overrides,
    }


def test_derived_root_is_verified_against_its_own_digest_and_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _imagenet_layout(source, [(300, 200), (60, 50)])
    result = _build(source, tmp_path / "derived")

    def config(**overrides: Any) -> DatasetConfig:
        fields: dict[str, Any] = {
            "name": "imagenet",
            "root": tmp_path / "derived",
            "num_classes": 2,
            "image_size": 32,
            "content_sha256": result["derived_content_sha256"],
            "derived_from": _derived_from(result),
        }
        fields.update(overrides)
        return DatasetConfig.model_validate(fields)

    loaded = build_raw_dataset(config())
    assert isinstance(loaded, ImageNetDataset)
    derived_identity = loaded.content_identity["derived_from"]
    assert isinstance(derived_identity, dict) and derived_identity["verification"] == "manifest-matched"
    assert loaded.content_identity["verification"] == "computed-and-matched"
    with pytest.raises(ValueError, match="content_sha256 does not match"):
        build_raw_dataset(config(content_sha256="e" * 64))
    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        build_raw_dataset(config(derived_from=_derived_from(result, manifest_sha256="e" * 64)))
    manifest_path = tmp_path / "derived" / DERIVED_DATASET_MANIFEST
    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered["short_side"] = 65
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="disagrees with dataset.derived_from: short_side"):
        build_raw_dataset(config(derived_from=_derived_from(result, manifest_sha256=_sha(manifest_path))))
    manifest_path.unlink()
    with pytest.raises(FileNotFoundError, match="derived dataset manifest missing"):
        build_raw_dataset(config())
    # The views (training runtime path) go through the same verification.
    with pytest.raises(FileNotFoundError, match="derived dataset manifest missing"):
        _views(sys.modules["ard.data.datasets"], config())


def test_a_derived_root_cannot_be_loaded_as_original_data(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _imagenet_layout(source, [(300, 200), (60, 50)])
    result = _build(source, tmp_path / "derived")
    for digest in (result["derived_content_sha256"], None):
        undeclared = DatasetConfig(
            name="imagenet", root=tmp_path / "derived", num_classes=2, image_size=32, content_sha256=digest
        )
        with pytest.raises(ValueError, match="is a derived dataset root .* declare it with dataset.derived_from"):
            build_raw_dataset(undeclared)
        with pytest.raises(ValueError, match="declare it with dataset.derived_from"):
            _views(sys.modules["ard.data.datasets"], undeclared)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"name": "tiny_imagenet"}, "only defined for the imagenet train split"),
        ({"split": "val"}, "only defined for the imagenet train split"),
        ({"content_sha256": None}, "requires the derived root's own dataset.content_sha256"),
        ({"content_sha256": "a" * 64}, "must be the derived root's digest, not its source's"),
    ],
)
def test_schema_refuses_ill_defined_derived_datasets(overrides: dict[str, Any], message: str) -> None:
    fields: dict[str, Any] = {
        "name": "imagenet",
        "root": "/x",
        "num_classes": 2,
        "content_sha256": "b" * 64,
        "derived_from": {
            "content_sha256": "a" * 64,
            "transform": "resize_short_side",
            "short_side": 160,
            "jpeg_quality": 95,
            "resample": "lanczos",
            "manifest_sha256": "c" * 64,
        },
    }
    fields.update(overrides)
    with pytest.raises(ValidationError, match=message):
        DatasetConfig.model_validate(fields)


def test_derived_dataset_is_part_of_the_training_dataset_identity() -> None:
    plain = DatasetConfig(name="imagenet", root=Path("/x"), num_classes=2, content_sha256="a" * 64)
    assert "derived_from" not in _dataset_identity(plain)
    derived = DatasetConfig.model_validate(
        {
            "name": "imagenet",
            "root": "/y",
            "num_classes": 2,
            "content_sha256": "b" * 64,
            "derived_from": {
                "content_sha256": "a" * 64,
                "transform": "resize_short_side",
                "short_side": 160,
                "jpeg_quality": 95,
                "resample": "lanczos",
                "manifest_sha256": "c" * 64,
            },
        }
    )
    identity = _dataset_identity(derived)
    assert identity["content_fingerprint"] == "b" * 64
    declared = identity["derived_from"]
    assert isinstance(declared, dict)
    assert declared["short_side"] == 160 and declared["content_sha256"] == "a" * 64


# =========================================================================== C: init checkpoint rule


def _minimal_experiment(tmp_path: Path) -> dict[str, Any]:
    return {
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
        "training": {"epochs": 3, "per_rank_batch_size": 2, "global_batch_size": 2, "train_image_size": 16},
    }


ORIGINAL = "a" * 64
DERIVED = "b" * 64
DERIVATION = {
    "content_sha256": ORIGINAL,
    "transform": "resize_short_side",
    "short_side": 160,
    "jpeg_quality": 95,
    "resample": "lanczos",
    "manifest_sha256": "c" * 64,
}


def _stage(tmp_path: Path, *, stage: int, dataset: dict[str, Any]) -> ExperimentConfig:
    raw = _minimal_experiment(tmp_path)
    raw["dataset"].update(dataset)
    if stage == 2:
        del raw["training"]["train_image_size"]
        raw["training"]["init_checkpoint"] = {"path": str(tmp_path / "stage1" / "last.pt"), "sha256": "d" * 64}
    return ExperimentConfig.model_validate(raw)


def _source_run(tmp_path: Path, dataset: dict[str, Any]) -> Path:
    run = tmp_path / "stage1"
    run.mkdir(parents=True, exist_ok=True)
    save_resolved_config(_stage(tmp_path, stage=1, dataset=dataset), run / "resolved_config.yaml")
    payload: dict[str, Any] = dict.fromkeys(REQUIRED_KEYS)
    payload.update(
        {
            "format_version": 1,
            "epoch": 2,
            "epoch_boundary": "end",
            "model": build_student(ModelConfig(architecture="fixture_cnn", num_classes=2), tier="dev").state_dict(),
            "selection_metadata": {},
            "tracker_run_id": "stage1-run",
            "config_hash": config_digest(yaml.safe_load((run / "resolved_config.yaml").read_text(encoding="utf-8"))),
            "world_size": 1,
        }
    )
    torch.save(payload, run / "last.pt")
    return run / "last.pt"


def test_stage2_accepts_a_stage1_trained_on_a_declared_derivative_of_its_dataset(tmp_path: Path) -> None:
    path = _source_run(
        tmp_path, {"root": str(tmp_path / "derived"), "content_sha256": DERIVED, "derived_from": DERIVATION}
    )
    target = _stage(tmp_path, stage=2, dataset={"content_sha256": ORIGINAL})
    _, lineage = read_init_checkpoint(path, expected_sha256=_sha(path), target=target)
    assert lineage["source_dataset_content_sha256"] == DERIVED
    assert lineage["source_dataset_derived_from"] == DERIVATION
    assert lineage["source_jpeg_draft_decode"] is False


def test_stage2_lineage_records_the_source_draft_decode(tmp_path: Path) -> None:
    run = tmp_path / "stage1"
    path = _source_run(tmp_path, {"content_sha256": ORIGINAL})
    raw = yaml.safe_load((run / "resolved_config.yaml").read_text(encoding="utf-8"))
    raw["training"]["jpeg_draft_decode"] = True
    (run / "resolved_config.yaml").write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    payload = torch.load(path, weights_only=False)
    payload["config_hash"] = config_digest(raw)
    torch.save(payload, path)
    target = _stage(tmp_path, stage=2, dataset={"content_sha256": ORIGINAL})
    _, lineage = read_init_checkpoint(path, expected_sha256=_sha(path), target=target)
    assert lineage["source_jpeg_draft_decode"] is True and "source_dataset_derived_from" not in lineage


def test_stage2_on_the_same_dataset_records_no_derivation(tmp_path: Path) -> None:
    path = _source_run(tmp_path, {"content_sha256": ORIGINAL})
    target = _stage(tmp_path, stage=2, dataset={"content_sha256": ORIGINAL})
    _, lineage = read_init_checkpoint(path, expected_sha256=_sha(path), target=target)
    assert "source_dataset_derived_from" not in lineage and "source_dataset_content_sha256" not in lineage
    assert "source_jpeg_draft_decode" not in lineage


@pytest.mark.parametrize(
    ("source", "target", "message"),
    [
        # A derivative of some other dataset.
        (
            {"content_sha256": DERIVED, "derived_from": {**DERIVATION, "content_sha256": "e" * 64}},
            {"content_sha256": ORIGINAL},
            "dataset.content_sha256",
        ),
        # Stage 2 on a derivative that stage 1 did not use.
        (
            {"content_sha256": ORIGINAL},
            {"content_sha256": DERIVED, "derived_from": DERIVATION},
            "stage 2 must run on the original (non-derived) dataset",
        ),
        # Two different derivatives of the same original.
        (
            {"content_sha256": DERIVED, "derived_from": DERIVATION},
            {"content_sha256": "f" * 64, "derived_from": {**DERIVATION, "short_side": 256}},
            "stage 2 must run on the original (non-derived) dataset",
        ),
        # Stage 2 on the very same derived root as stage 1: still not original data.
        (
            {"content_sha256": DERIVED, "derived_from": DERIVATION},
            {"content_sha256": DERIVED, "derived_from": DERIVATION},
            "stage 2 must run on the original (non-derived) dataset",
        ),
        # Plain digest mismatch, no derivation declared (unchanged behaviour).
        ({"content_sha256": DERIVED}, {"content_sha256": ORIGINAL}, "dataset.content_sha256"),
        # The derivative is fine but another dataset field differs.
        (
            {"content_sha256": DERIVED, "derived_from": DERIVATION, "image_size": 64},
            {"content_sha256": ORIGINAL},
            "dataset.image_size",
        ),
    ],
)
def test_stage2_refuses_any_other_dataset_relation(
    tmp_path: Path, source: dict[str, Any], target: dict[str, Any], message: str
) -> None:
    path = _source_run(tmp_path, source)
    with pytest.raises(ValueError, match="incompatible with this stage-2 run") as refused:
        read_init_checkpoint(path, expected_sha256=_sha(path), target=_stage(tmp_path, stage=2, dataset=target))
    assert message in str(refused.value)


# =========================================================================== config


BUILT_DERIVATIVES = {
    160: (
        "f52345d89aa982871cb6bd3e642aed1b18a33778f1488ea0578c33c63058d8ba",
        "2bf01d09ed717eda0722a012fb86ad2b9fb22783c46d47355aac76f3db11c92a",
    ),
    256: (
        "65f729c7a34047a9063f95d2ac08e3701a8464203a7074e7216ea11321095ba0",
        "1b878f7a12c9def7bef00d9ebdd34d4136134c9e8c8772a05afdfce11bd07b0c",
    ),
}


@pytest.mark.parametrize("short_side", sorted(BUILT_DERIVATIVES))
def test_resized_stage1_config_is_stage1_except_the_declared_derived_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, short_side: int
) -> None:
    _env(monkeypatch, tmp_path)
    stage1 = load_config(CONFIG_DIR / "imagenet_mobilenetv4_twostage_stage1_112_pgd1.yaml").model_dump(mode="json")
    name = f"imagenet_mobilenetv4_twostage_stage1_112_pgd1_resized_s{short_side}.yaml"
    resized_config = load_config(CONFIG_DIR / name)
    resized = resized_config.model_dump(mode="json")
    derived = resized["dataset"]["derived_from"]
    assert derived["short_side"] == short_side and derived["jpeg_quality"] == 95
    assert (resized["dataset"]["content_sha256"], derived["manifest_sha256"]) == BUILT_DERIVATIVES[short_side]
    assert resized["tracking"]["group"] == f"imagenet-mobilenetv4-twostage-lowres-stage1-s{short_side}"
    assert resized["dataset"]["root"] == str(tmp_path / "imagenet_train_derived")
    assert derived["content_sha256"] == stage1["dataset"]["content_sha256"]
    assert resized["dataset"]["content_sha256"] != stage1["dataset"]["content_sha256"]
    assert derived["transform"] == "resize_short_side" and derived["resample"] == "lanczos"
    # The official evaluation dataset stays the original root at 224px.
    assert resized["evaluation"] == stage1["evaluation"]
    assert resized["evaluation"]["dataset"]["root"] == str(tmp_path / "imagenet")
    # S=256 additionally drops per-step diagnostic host syncs (same math, sync-free parity-tested),
    # human-approved 2026-09-30 because stage 1 at 112 px / batch 128 is kernel-launch bound.
    assert resized["training"].get("step_diagnostics", True) is (short_side != 256)
    resized["training"].pop("step_diagnostics", None)
    for payload in (stage1, resized):
        payload["dataset"] = {**payload["dataset"], "root": None, "content_sha256": None}
        payload["dataset"].pop("derived_from", None)
        payload["tracking"] = {**payload["tracking"], "group": None}
    assert stage1 == resized
    objective, _, _, _ = _build_method(resized_config)
    assert objective is not None
    # A stage 2 on the original dataset accepts this stage 1 as its source.
    stage2 = load_config(CONFIG_DIR / "imagenet_mobilenetv4_twostage_stage2_224_pgd3_ft.yaml")
    from ard.engine.checkpoint import _source_compatibility_errors

    assert _source_compatibility_errors(resized_config, stage2) == []


def _crop_upsample_fraction(
    width: int, height: int, *, output: int = 112, samples: int = 20_000
) -> tuple[float, float]:
    """Fraction of RandomResizedCrop boxes smaller than ``output`` px in at least
    one / in both dimensions (so upsampled to ``output``), for one image size."""
    transform = EpochImageNetTransform(augmentation_seed=0, image_size=output)
    generator = torch.Generator().manual_seed(0)
    either = both = 0
    for _ in range(samples):
        _, _, crop_height, crop_width = transform._crop_box_for_size(width, height, generator)
        either += crop_width < output or crop_height < output
        both += crop_width < output and crop_height < output
    return either / samples, both / samples


@pytest.mark.parametrize(
    ("short_side", "either", "both"),
    [(None, 0.002, 0.000), (256, 0.125, 0.065), (160, 0.498, 0.346)],
)
def test_crop_upsample_fraction_per_short_side(short_side: int | None, either: float, both: float) -> None:
    """The blur cost quoted in the resized configs' headers: a 500x375 image,
    as stored at the original size or by the builder at each short side."""
    size = (500, 375) if short_side is None else builder.resized_dimensions(500, 375, short_side)
    observed = _crop_upsample_fraction(*size)
    # Monte Carlo standard error at 20k draws is <= 0.0036; 0.01 is ~3 SE.
    assert observed == pytest.approx((either, both), abs=0.01)
