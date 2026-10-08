"""Plan 0103 Phase 2 batch C (human-approved 2026-10-08): IDBH for ImageNet.

``dataset.imagenet_augmentation: idbh_weak_nocropshift | idbh_weak |
idbh_strong`` appends IDBH (Li & Spratling, ICLR 2023) -- ColorShape('color')
and Random Erasing, plus a fraction-preserving CropShift adaptation for
idbh_weak / idbh_strong -- after the ImageNet RandomResizedCrop + flip. Every draw comes from
a generator keyed by (augmentation seed, epoch, source id), never from the
global RNG. Default "standard" is unserialized, so every existing config and
every default view is byte-identical to the pre-change code.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from PIL import Image
from pydantic import ValidationError

from ard.cli.evaluate import _augmentation_protocol_identity
from ard.config import load_config
from ard.config.loader import _expand_environment, resolved_config_dict
from ard.config.schema import DatasetConfig, ExperimentConfig
from ard.data import build_train_validation_views
from ard.data import datasets as data_module
from ard.data.datasets import (
    EpochCropReTransform,
    EpochCropshiftTransform,
    EpochIdbhWeakTransform,
    EpochImageNetTransform,
    EpochSourceTransform,
    idbh_cropshift_strength,
)
from ard.engine.checkpoint import config_digest

pytestmark = pytest.mark.t1

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "scientific"
# The commit this change is based on: the pre-change schema and data views.
PRE_CHANGE_COMMIT = "0c3cdd3"
BASELINE = "imagenet_mobilenetv4_pgd_at_random_init_lr0025_cg.yaml"


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
        # Phase 2 batches B/D configs.
        "ARD_EXTERNAL_CHECKPOINT_ROOT": str(tmp_path / "external"),
        "ARD_SOFT_LABEL_BANK_ROOT": str(tmp_path / "banks"),
        "ARD_SOFT_LABEL_BANK_SHA256_CONVNEXT_B_CVST": "e" * 64,
        "ARD_SOFT_LABEL_BANK_SHA256_CONVNEXT_T_CVST": "e" * 64,
        "ARD_SOFT_LABEL_BANK_SHA256_MNV4M_OWN": "e" * 64,
        "ARD_SOFT_LABEL_BANK_SHA256_SALMAN_R50": "e" * 64,
        "ARD_SOFT_LABEL_BANK_SHA256_VIT_S_CVST": "e" * 64,
        "ARD_PHASE2_ADAMW_LR_CONVNEXT_ATTO": "0.0005",
        "ARD_PHASE2_ADAMW_LR_DEIT_TINY": "0.0005",
    }.items():
        monkeypatch.setenv(key, value)


def _textured(width: int, height: int, seed: int) -> Image.Image:
    generator = torch.Generator().manual_seed(seed)
    pixels = torch.randint(0, 256, (height, width, 3), generator=generator, dtype=torch.uint8)
    return Image.fromarray(pixels.numpy(), mode="RGB")


def _imagenet_layout(root: Path) -> None:
    for class_index, wnid in enumerate(("n001", "n002", "n003")):
        class_dir = root / "train" / wnid
        class_dir.mkdir(parents=True, exist_ok=True)
        for image_index, size in enumerate(((120, 90), (90, 140), (200, 160))):
            _textured(*size, seed=class_index * 10 + image_index).save(class_dir / f"{wnid}_{image_index}.JPEG")


# =========================================================================== defaults unchanged


def test_default_is_standard_and_unserialized() -> None:
    dataset = DatasetConfig(name="imagenet", root=Path("/x"), num_classes=2)
    assert dataset.imagenet_augmentation == "standard"
    assert "imagenet_augmentation" not in json.loads(dataset.model_dump_json())
    assert EpochImageNetTransform(augmentation_seed=0, image_size=8).augmentation == "standard"


def test_existing_configs_serialize_exactly_as_under_the_pre_change_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolved config and config hash of every pre-existing config is
    byte-identical to the pre-change schema's."""
    _env(monkeypatch, tmp_path)
    old_schema = _pre_change_module("src/ard/config/schema.py", "_ard_schema_pre_imagenet_idbh", "ard.config")
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


def test_default_transforms_are_bit_identical_to_the_pre_change_code() -> None:
    """The ImageNet standard path and every CIFAR policy (whose CropShift was
    refactored to share ``_cropshift_translate``) reproduce the pre-change
    module's output exactly."""
    old = _pre_change_module("src/ard/data/datasets.py", "ard.data._datasets_pre_imagenet_idbh", "ard.data")
    images = [_textured(w, h, seed) for seed, (w, h) in enumerate(((96, 64), (64, 96), (50, 50), (300, 200)))]
    for epoch in (0, 3):
        for source_id, image in enumerate(images):
            for size in (32, 48):
                new_t = EpochImageNetTransform(augmentation_seed=11, image_size=size)
                old_t = old.EpochImageNetTransform(augmentation_seed=11, image_size=size)
                new_t.set_epoch(epoch)
                old_t.set_epoch(epoch)
                assert torch.equal(new_t(image, source_id=source_id * 101), old_t(image, source_id=source_id * 101))
    cifar = _textured(32, 32, seed=99)
    for cls in (EpochSourceTransform, EpochCropshiftTransform, EpochCropReTransform, EpochIdbhWeakTransform):
        new_t, old_t = cls(augmentation_seed=5), getattr(old, cls.__name__)(augmentation_seed=5)
        for epoch in (0, 2):
            new_t.set_epoch(epoch)
            old_t.set_epoch(epoch)
            for source_id in range(16):
                assert torch.equal(new_t(cifar, source_id=source_id), old_t(cifar, source_id=source_id)), cls


def test_default_views_are_bit_identical_to_the_pre_change_views(tmp_path: Path) -> None:
    old = _pre_change_module("src/ard/data/datasets.py", "ard.data._datasets_pre_imagenet_idbh_views", "ard.data")
    _imagenet_layout(tmp_path)
    config = DatasetConfig(name="imagenet", root=tmp_path, split="train", num_classes=3, image_size=32)
    new_train, _ = build_train_validation_views(config, validation_fraction=0.34, split_seed=1, augmentation_seed=7)
    old_train, _ = old.build_train_validation_views(
        config, validation_fraction=0.34, split_seed=1, augmentation_seed=7
    )
    for epoch in (0, 4):
        new_train.set_epoch(epoch)
        old_train.set_epoch(epoch)
        for index in range(len(new_train)):
            new_image, new_label, new_id = new_train[index]
            old_image, old_label, old_id = old_train[index]
            assert (new_label, new_id) == (old_label, old_id)
            assert torch.equal(new_image, old_image)


# =========================================================================== IDBH transform


@pytest.mark.parametrize("augmentation", ("idbh_weak_nocropshift", "idbh_weak", "idbh_strong"))
@pytest.mark.parametrize(("image_size", "source_size"), ((224, (500, 375)), (112, (300, 400)), (32, (40, 40))))
def test_idbh_shapes_dtype_and_range(augmentation: str, image_size: int, source_size: tuple[int, int]) -> None:
    transform = EpochImageNetTransform(augmentation_seed=3, image_size=image_size, augmentation=augmentation)
    image = _textured(*source_size, seed=1)
    for source_id in range(12):
        tensor = transform(image, source_id=source_id)
        assert tensor.shape == (3, image_size, image_size)
        assert tensor.dtype == torch.float32
        assert float(tensor.min()) >= 0.0 and float(tensor.max()) <= 1.0


@pytest.mark.parametrize("augmentation", ("idbh_weak_nocropshift", "idbh_weak", "idbh_strong"))
def test_idbh_is_keyed_by_seed_epoch_and_source_id_only(augmentation: str) -> None:
    image = _textured(160, 120, seed=2)

    def view(seed: int, epoch: int, source_id: int, global_seed: int) -> torch.Tensor:
        torch.manual_seed(global_seed)  # the global RNG must not matter
        transform = EpochImageNetTransform(augmentation_seed=seed, image_size=64, augmentation=augmentation)
        transform.set_epoch(epoch)
        return transform(image, source_id=source_id)

    reference = view(7, 2, 5, global_seed=0)
    # Same key, different global RNG state, fresh object: identical.
    assert torch.equal(reference, view(7, 2, 5, global_seed=12345))
    # Call order on one object does not matter either.
    transform = EpochImageNetTransform(augmentation_seed=7, image_size=64, augmentation=augmentation)
    transform.set_epoch(2)
    for other in (9, 1, 3):
        transform(image, source_id=other)
    assert torch.equal(reference, transform(image, source_id=5))
    # Global RNG state is untouched by the transform.
    torch.manual_seed(4)
    expected_next = torch.rand(())
    torch.manual_seed(4)
    transform(image, source_id=5)
    assert torch.equal(torch.rand(()), expected_next)
    # Any key component changes the view.
    assert not torch.equal(reference, view(8, 2, 5, global_seed=0))
    assert not torch.equal(reference, view(7, 3, 5, global_seed=0))
    assert not torch.equal(reference, view(7, 2, 6, global_seed=0))


def test_idbh_keeps_the_standard_crop_and_flip(monkeypatch: pytest.MonkeyPatch) -> None:
    """IDBH is appended after the RandomResizedCrop + flip, which consume the
    primary generator exactly as the standard path does: with the IDBH layers
    replaced by identity, the output equals the standard view."""
    image = _textured(200, 150, seed=3)
    standard = EpochImageNetTransform(augmentation_seed=1, image_size=48)
    idbh = EpochImageNetTransform(augmentation_seed=1, image_size=48, augmentation="idbh_weak")
    monkeypatch.setattr(idbh, "_idbh", lambda cropped, *, source_id: data_module._to_tensor(cropped))
    for epoch in (0, 5):
        standard.set_epoch(epoch)
        idbh.set_epoch(epoch)
        for source_id in range(10):
            assert torch.equal(standard(image, source_id=source_id), idbh(image, source_id=source_id))


def test_idbh_layers_follow_the_upstream_distribution(monkeypatch: pytest.MonkeyPatch) -> None:
    """CropShift (idbh_weak / idbh_strong only) level ~ U{0..10} scaled by
    size/32; ColorShape('color') via the shared upstream implementation; Random
    Erasing p=0.5 (weak, weak_nocropshift) / 1.0 (strong) with torchvision's
    default scale and ratio."""
    assert [idbh_cropshift_strength(level, 32) for level in range(11)] == list(range(11))
    assert idbh_cropshift_strength(10, 224) == 70
    assert idbh_cropshift_strength(10, 112) == 35
    assert idbh_cropshift_strength(1, 224) == 7
    assert idbh_cropshift_strength(0, 224) == 0

    strengths: list[int] = []
    colour_calls: list[int] = []
    erase_p: list[float] = []
    original_shift = data_module._cropshift_translate
    original_colour = data_module._idbh_color_with_generator
    original_erase = data_module._random_erase_with_generator

    def shift(image: Any, *, generator: torch.Generator, strength: int) -> Any:
        strengths.append(strength)
        return original_shift(image, generator=generator, strength=strength)

    def colour(image: Any, *, generator: torch.Generator) -> Any:
        colour_calls.append(1)
        return original_colour(image, generator=generator)

    def erase(tensor: torch.Tensor, *, generator: torch.Generator, p: float) -> torch.Tensor:
        erase_p.append(p)
        return original_erase(tensor, generator=generator, p=p)

    monkeypatch.setattr(data_module, "_cropshift_translate", shift)
    monkeypatch.setattr(data_module, "_idbh_color_with_generator", colour)
    monkeypatch.setattr(data_module, "_random_erase_with_generator", erase)
    image = _textured(300, 240, seed=4)
    for augmentation, p, cropshift in (
        ("idbh_weak_nocropshift", 0.5, False),
        ("idbh_weak", 0.5, True),
        ("idbh_strong", 1.0, True),
    ):
        strengths.clear()
        erase_p.clear()
        transform = EpochImageNetTransform(augmentation_seed=0, image_size=224, augmentation=augmentation)
        for source_id in range(200):
            transform(image, source_id=source_id)
        assert set(erase_p) == {p}
        if cropshift:
            assert set(strengths) <= {7 * level for level in range(11)}
            assert len(set(strengths)) == 11  # every level occurs in 200 draws
        else:
            assert strengths == []
    assert len(colour_calls) == 600


def test_nocropshift_equals_idbh_weak_without_its_cropshift_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    """The layers use independent keyed substreams, so idbh_weak_nocropshift is
    exactly idbh_weak with the CropShift layer removed: same crop/flip, same
    ColorShape op and magnitude, same erasing rectangle per sample."""
    image = _textured(220, 170, seed=6)
    nocrop = EpochImageNetTransform(augmentation_seed=4, image_size=64, augmentation="idbh_weak_nocropshift")
    weak = EpochImageNetTransform(augmentation_seed=4, image_size=64, augmentation="idbh_weak")
    for epoch in (0, 7):
        nocrop.set_epoch(epoch)
        weak.set_epoch(epoch)
        reference = [nocrop(image, source_id=source_id) for source_id in range(12)]
        with monkeypatch.context() as patch:
            patch.setattr(data_module, "_cropshift_translate", lambda image, *, generator, strength: image)
            assert all(torch.equal(weak(image, source_id=i), view) for i, view in enumerate(reference))
        # With its CropShift, idbh_weak differs for at least one sample (10/11 shift).
        assert any(not torch.equal(weak(image, source_id=i), view) for i, view in enumerate(reference))


def test_idbh_strong_erases_almost_every_view_and_weak_about_half(monkeypatch: pytest.MonkeyPatch) -> None:
    """Observed, not re-derived: on a constant grey image with CropShift and
    ColorShape replaced by identity, an erased rectangle is the only source of
    exact zeros. (Erasing gives up after 10 rejected rectangles, as
    torchvision's does, so p=1 erases slightly fewer than all views.)"""
    monkeypatch.setattr(data_module, "_cropshift_translate", lambda image, *, generator, strength: image)
    monkeypatch.setattr(data_module, "_idbh_color_with_generator", lambda image, *, generator: image)
    image = Image.new("RGB", (64, 64), (200, 200, 200))
    rates = {}
    trials = 400
    for augmentation in ("idbh_weak_nocropshift", "idbh_weak", "idbh_strong"):
        transform = EpochImageNetTransform(augmentation_seed=0, image_size=32, augmentation=augmentation)
        erased = sum(bool((transform(image, source_id=source_id) == 0).any()) for source_id in range(trials))
        rates[augmentation] = erased / trials
    assert rates["idbh_strong"] >= 0.95
    assert 0.4 < rates["idbh_weak"] < 0.6
    assert rates["idbh_weak_nocropshift"] == rates["idbh_weak"]  # same keyed erase substream


def test_idbh_composes_with_jpeg_draft_decode(tmp_path: Path) -> None:
    path = tmp_path / "x.JPEG"
    _textured(640, 480, seed=5).save(path, quality=90)
    transform = EpochImageNetTransform(
        augmentation_seed=2, image_size=64, augmentation="idbh_strong", jpeg_draft_decode=True
    )
    first = transform(path, source_id=3)
    assert first.shape == (3, 64, 64)
    assert torch.equal(first, transform(path, source_id=3))


def test_idbh_rejects_bad_combinations() -> None:
    with pytest.raises(ValueError, match="unsupported ImageNet augmentation"):
        EpochImageNetTransform(augmentation_seed=0, image_size=8, augmentation="idbh")
    with pytest.raises(ValueError, match="mutually exclusive"):
        EpochImageNetTransform(augmentation_seed=0, image_size=8, augmentation="idbh_weak", heavy_augmentation=True)


# =========================================================================== config / identity


def test_config_field_validation_and_wiring(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="only defined for the imagenet train split"):
        DatasetConfig(name="cifar10", imagenet_augmentation="idbh_weak")
    with pytest.raises(ValidationError, match="only defined for the imagenet train split"):
        DatasetConfig(name="imagenet", root=tmp_path, split="val", num_classes=3, imagenet_augmentation="idbh_weak")
    with pytest.raises(ValidationError, match="mutually exclusive"):
        DatasetConfig(
            name="imagenet",
            root=tmp_path,
            num_classes=3,
            imagenet_augmentation="idbh_weak",
            imagenet_heavy_augmentation=True,
        )
    with pytest.raises(ValidationError):
        DatasetConfig(name="imagenet", root=tmp_path, num_classes=3, imagenet_augmentation="idbh")
    _imagenet_layout(tmp_path)
    config = DatasetConfig(
        name="imagenet", root=tmp_path, split="train", num_classes=3, image_size=32, imagenet_augmentation="idbh_strong"
    )
    assert json.loads(config.model_dump_json())["imagenet_augmentation"] == "idbh_strong"
    train_view, validation_view = build_train_validation_views(
        config, validation_fraction=0.34, split_seed=1, augmentation_seed=7
    )
    assert train_view.dataset.transform.augmentation == "idbh_strong"
    # The validation view is never augmented.
    validation_view.set_epoch(0)
    first = validation_view[0][0]
    validation_view.set_epoch(9)
    assert torch.equal(first, validation_view[0][0])


def test_augmentation_protocol_identity_present_only_when_set(tmp_path: Path) -> None:
    standard = DatasetConfig(name="imagenet", root=tmp_path, num_classes=3)
    idbh = DatasetConfig(name="imagenet", root=tmp_path, num_classes=3, imagenet_augmentation="idbh_weak")
    assert _augmentation_protocol_identity(standard) == {}
    assert _augmentation_protocol_identity(idbh) == {"imagenet_augmentation": "idbh_weak"}


def _load(name: str) -> dict[str, Any]:
    return load_config(CONFIG_DIR / name).model_dump(mode="json")


@pytest.mark.parametrize(
    ("name", "augmentation", "epochs"),
    (
        ("imagenet_mobilenetv4_pgd_at_random_init_lr0025_idbh_weak_nocropshift_cg.yaml", "idbh_weak_nocropshift", 50),
        (
            "imagenet_mobilenetv4_pgd_at_random_init_lr0025_idbh_weak_nocropshift_100ep_cg.yaml",
            "idbh_weak_nocropshift",
            100,
        ),
        ("imagenet_mobilenetv4_pgd_at_random_init_lr0025_100ep_cg.yaml", "standard", 100),
    ),
)
def test_phase2_augmentation_configs_differ_from_the_baseline_only_where_declared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, augmentation: str, epochs: int
) -> None:
    """The 2x2 {standard, IDBH} x {50, 100 epochs}: only the augmentation,
    (for 100 epochs) epochs + milestones scaled to [50, 76] with the same
    10-epoch warmup, the Phase 2 selection subset (5000) and the tracking
    group differ from the baseline. The
    attack identity, optimizer, protocol id and cuda_graph are unchanged."""
    _env(monkeypatch, tmp_path)
    baseline, arm = _load(BASELINE), _load(name)
    assert arm["dataset"].get("imagenet_augmentation", "standard") == augmentation
    assert "imagenet_augmentation" not in baseline["dataset"]
    assert arm["training"]["epochs"] == epochs
    assert arm["training"]["cuda_graph"] is True
    # Plan 0103 Phase 2 template: per-epoch selection on a fixed 5000-image subset.
    assert arm["training"]["selection_subset_size"] == 5000
    assert "selection_subset_size" not in baseline["training"]
    assert arm["scheduler"]["milestones"] == ([25, 38] if epochs == 50 else [50, 76])
    assert arm["scheduler"]["warmup_epochs"] == 10
    assert arm["tracking"]["group"] != baseline["tracking"]["group"]
    assert augmentation.replace("_", "-") in arm["tracking"]["group"]  # variant named in the W&B group
    assert arm["protocol"] == baseline["protocol"]  # shared contract; identity carries the arm
    for payload in (baseline, arm):
        payload["dataset"].pop("imagenet_augmentation", None)
        payload["training"]["epochs"] = None
        payload["training"].pop("selection_subset_size", None)
        payload["scheduler"]["milestones"] = None
        payload["tracking"]["group"] = None
    assert arm == baseline


# =========================================================================== throughput


@pytest.mark.slow
def test_cpu_throughput_micro_benchmark_single_process(capsys: pytest.CaptureFixture[str]) -> None:
    """Single-process CPU img/s of the 224 px training transform on decoded
    ImageNet-sized PIL images (decode excluded: it is identical for every
    path). Reported, with only a loose floor so the test cannot flake."""
    images = [_textured(500, 375, seed=seed) for seed in range(8)]
    rates: dict[str, float] = {}
    for augmentation in ("standard", "idbh_weak_nocropshift", "idbh_weak", "idbh_strong"):
        transform = EpochImageNetTransform(augmentation_seed=0, image_size=224, augmentation=augmentation)
        for source_id in range(8):  # warm-up
            transform(images[source_id], source_id=source_id)
        count = 0
        start = time.perf_counter()
        while time.perf_counter() - start < 1.5:
            transform(images[count % len(images)], source_id=count)
            count += 1
        rates[augmentation] = count / (time.perf_counter() - start)
    with capsys.disabled():
        print("\nImageNet 224px train-transform CPU throughput (img/s, 1 process): " + json.dumps(
            {key: round(value, 1) for key, value in rates.items()}
        ))
    assert rates["idbh_weak_nocropshift"] > 0.2 * rates["standard"]
    assert rates["idbh_weak"] > 0.2 * rates["standard"]
    assert rates["idbh_strong"] > 0.2 * rates["standard"]
