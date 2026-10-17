"""Plan 0107 structural reparameterization end to end on CPU (human-approved 2026-10-18).

The checked-in ``imagenet_mobilenetv4_reparam_pgd_at_phase2_random_lr0025.yaml``, shrunk to a tiny on-disk
ImageNet fixture at 64 px and 2 classes (the real reparameterized MobileNetV4-S student), deterministic on CPU,
runs through the real ``ard.cli.train``, ``ard.cli.fold_reparam`` and ``ard.cli.evaluate``. (64 px, not 32: at
32 px the last feature maps are 1x1, so with batch 4 each BN sees 4 values per channel; after two steps the
eval-mode running statistics amplify activations past float32 range -- the plain MobileNetV4-S reaches 1e25
there too -- and the non-finite validation guard stops the run.)

* one epoch, an injected crash, then ``--resume`` to two epochs gives last.pt bitwise equal to an
  uninterrupted two-epoch run (training checkpoints hold the unfolded model and EMA);
* ``ard.cli.evaluate`` refuses the unfolded training checkpoints;
* ``ard.cli.fold_reparam`` writes folded best/last/best-ema checkpoints whose ``model`` and ``ema`` load
  strictly into the plain ``mobilenetv4_conv_small_imagenet`` student and match the unfolded outputs;
* ``ard.cli.evaluate`` on the folded directory evaluates the plain network (raw and EMA weights) and
  records the deployed architecture and the fold record;
* the fold refuses (and writes nothing) on a failed equivalence check, an already-folded input, a
  non-reparameterized student, a non-empty output directory and a dirty source tree without the opt-in flag;
  the manifest and every fold record carry the source Git SHA, its dirty flag and the torch / timm versions.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import timm
import torch
import yaml
from PIL import Image

from ard.cli.fold_reparam import fold_run
from ard.config.schema import DatasetConfig, ModelConfig
from ard.data import build_dataset
from ard.models import build_student
from ard.models.reparam import FOLD_CONTRACT, ReparamConvBnAct

pytestmark = pytest.mark.t3

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "scientific" / "imagenet_mobilenetv4_reparam_pgd_at_phase2_random_lr0025.yaml"
REPARAM = "mobilenetv4_conv_small_reparam_imagenet"
PLAIN = "mobilenetv4_conv_small_imagenet"

_CRASH_AFTER_FIRST_EPOCH = """
import sys
from ard.engine import Trainer

original_fit = Trainer.fit

def fit(self, *args, **kwargs):
    inner = kwargs.get("on_epoch_end")

    def on_epoch_end(metrics, improved):
        if inner is not None:
            inner(metrics, improved)
        raise RuntimeError("injected crash after the first epoch")

    kwargs["on_epoch_end"] = on_epoch_end
    return original_fit(self, *args, **kwargs)

Trainer.fit = fit
from ard.cli.train import main
raise SystemExit(main(sys.argv[1:]))
"""


def _imagenet_fixture(root: Path) -> None:
    for split in ("train", "val"):
        for class_index, wnid in enumerate(("n001", "n002")):
            class_dir = root / split / wnid
            class_dir.mkdir(parents=True)
            for image_index in range(6):
                generator = torch.Generator().manual_seed(
                    100 * class_index + image_index + (7 if split == "val" else 0)
                )
                pixels = torch.rand(40, 48, 3, generator=generator)
                Image.fromarray((pixels * 255).to(torch.uint8).numpy()).save(class_dir / f"{wnid}_{image_index}.JPEG")


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run(
        [sys.executable, *args], cwd=ROOT, env=environment, text=True, capture_output=True, check=False
    )


def _tiny(*, data_root: Path, output: Path, run_id: str) -> dict[str, Any]:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert raw["student"]["architecture"] == REPARAM
    for key, split in (("dataset", "train"), ("evaluation", "val")):
        block = raw[key] if key == "dataset" else raw[key]["dataset"]
        observed = build_dataset(
            DatasetConfig(name="imagenet", root=data_root, split=split, num_classes=2, image_size=64)
        ).content_identity
        assert observed is not None
        block.update(root=str(data_root), num_classes=2, image_size=64, content_sha256=observed["observed_sha256"])
    raw["tier"] = "smoke"
    raw["student"].update(num_classes=2)
    raw["seeds"] = dict.fromkeys(raw["seeds"], 3)
    raw["training"].update(
        epochs=2,
        per_rank_batch_size=4,
        global_batch_size=4,
        num_workers=0,
        device="cpu",
        deterministic=True,
        cudnn_benchmark=False,
        validation_fraction=0.5,
        train_probe_size=2,
        selection_subset_size=2,
    )
    raw["scheduler"].update(milestones=[1], warmup_epochs=1)
    raw["tracking"] = {
        "mode": "offline",
        "project": "ard-test",
        "run_id": run_id,
        "group": raw["tracking"]["group"],
        "panel_size": 2,
        "panel_interval_epochs": 1,
        "artifact_retention": "full",
    }
    raw["evaluation"].update(panel_size=2, autoattack_batch_size=4)
    raw["output_dir"] = str(output)
    return raw


def _config(tmp_path: Path, name: str, *, data_root: Path) -> tuple[Path, Path]:
    output = tmp_path / name
    config = tmp_path / f"{name}.yaml"
    config.write_text(
        yaml.safe_dump(_tiny(data_root=data_root, output=output, run_id=f"offline-{name}"), sort_keys=False)
    )
    return config, output


def _evaluate(checkpoint_dir: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return _run(
        [
            "-m",
            "ard.cli.evaluate",
            "--config",
            str(checkpoint_dir / "resolved_config.yaml"),
            "--checkpoint-dir",
            str(checkpoint_dir),
            *extra,
        ]
    )


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """One uninterrupted two-epoch training run: (ImageNet fixture root, training output directory)."""
    root = tmp_path_factory.mktemp("reparam-run")
    data_root = root / "imagenet"
    _imagenet_fixture(data_root)
    full_config, full_out = _config(root, "full", data_root=data_root)
    trained = _run(["-m", "ard.cli.train", "--config", str(full_config)])
    assert trained.returncode == 0, trained.stderr
    return data_root, full_out


def _copy_run(source: Path, destination: Path) -> Path:
    """Checkpoints and resolved config only, so a test cannot disturb the shared run or see its folds."""
    destination.mkdir(parents=True)
    for name in ("resolved_config.yaml", "last.pt", "best.pt", "best-ema.pt"):
        shutil.copy2(source / name, destination / name)
    return destination


PROVENANCE_KEYS = ("source_git_sha", "source_git_dirty", "torch_version", "timm_version")


def _assert_provenance(record: dict[str, Any]) -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", record["source_git_sha"])
    assert isinstance(record["source_git_dirty"], bool)
    assert record["torch_version"] == torch.__version__
    assert record["timm_version"] == timm.__version__


def test_reparam_trains_resumes_folds_and_evaluates_as_the_plain_network(
    tmp_path: Path, trained_run: tuple[Path, Path]
) -> None:
    data_root, full_out = trained_run
    resumed_config, resumed_out = _config(tmp_path, "resumed", data_root=data_root)

    # (a) Resume: crash after epoch 1, resume from last.pt -> bitwise the uninterrupted run (model and EMA).
    crashed = _run(["-c", _CRASH_AFTER_FIRST_EPOCH, "--config", str(resumed_config)])
    assert crashed.returncode != 0 and "injected crash after the first epoch" in crashed.stderr
    resumed = _run(
        [
            "-m",
            "ard.cli.train",
            "--config",
            str(resumed_out / "resolved_config.yaml"),
            "--resume",
            str(resumed_out / "last.pt"),
        ]
    )
    assert resumed.returncode == 0, resumed.stderr
    full_last = torch.load(full_out / "last.pt", map_location="cpu", weights_only=False)
    resumed_last = torch.load(resumed_out / "last.pt", map_location="cpu", weights_only=False)
    for key in ("model", "ema"):
        assert list(resumed_last[key]) == list(full_last[key])
        for name, value in full_last[key].items():
            assert torch.equal(resumed_last[key][name], value), (key, name)
    # Training checkpoints hold the unfolded (multi-branch) student.
    assert any(".branches.0.conv.weight" in name for name in full_last["model"])

    # (b) The evaluation CLI refuses the unfolded training checkpoints.
    refused = _evaluate(full_out)
    assert refused.returncode != 0 and "ard.cli.fold_reparam" in refused.stderr
    assert not (full_out / "evaluation").exists()

    # (c) Fold.
    folded_dir = full_out / "folded"
    # --allow-dirty-source: the test may run in a working tree with uncommitted changes (recorded as such).
    folded = _run(["-m", "ard.cli.fold_reparam", "--run-dir", str(full_out), "--allow-dirty-source"])
    assert folded.returncode == 0, folded.stderr
    assert (folded_dir / "resolved_config.yaml").read_bytes() == (full_out / "resolved_config.yaml").read_bytes()
    manifest = json.loads((folded_dir / "fold-manifest.json").read_text(encoding="utf-8"))
    assert manifest["contract"] == FOLD_CONTRACT
    assert manifest["deployed_architecture"] == PLAIN and manifest["training_architecture"] == REPARAM
    _assert_provenance(manifest)
    assert sorted(item["source_checkpoint_filename"] for item in manifest["checkpoints"]) == [
        "best-ema.pt",
        "best.pt",
        "last.pt",
    ]
    plain_config = ModelConfig(architecture=PLAIN, num_classes=2)
    for item in manifest["checkpoints"]:
        name = item["source_checkpoint_filename"]
        source = torch.load(full_out / name, map_location="cpu", weights_only=False)
        payload = torch.load(folded_dir / name, map_location="cpu", weights_only=False)
        assert item["source_checkpoint_sha256"] == hashlib.sha256((full_out / name).read_bytes()).hexdigest()
        assert item["folded_checkpoint_sha256"] == hashlib.sha256((folded_dir / name).read_bytes()).hexdigest()
        assert item["folded_weight_sets"] == ["model", "ema"]
        for key in ("model", "ema"):
            check = item["equivalence_check"][key]
            assert check["float64"]["relative"] <= 1e-5 and check["float32"]["relative"] <= 1e-4
            plain = build_student(plain_config, tier="smoke")
            plain.load_state_dict(payload[key], strict=True)
        # Every lineage key except the folded weight sets is unchanged.
        for key, value in source.items():
            if key not in ("model", "ema"):
                assert key in payload
        assert payload["config_hash"] == source["config_hash"]
        assert payload["tracker_run_id"] == source["tracker_run_id"]
        assert payload["selection_metadata"] == source["selection_metadata"]
        assert payload["reparameterization"]["deployed_architecture"] == PLAIN
        _assert_provenance(payload["reparameterization"])
        assert {key: item[key] for key in PROVENANCE_KEYS} == {key: manifest[key] for key in PROVENANCE_KEYS}
    # (P2-3 d) A non-empty output directory is refused.
    again = _run(["-m", "ard.cli.fold_reparam", "--run-dir", str(full_out), "--allow-dirty-source"])
    assert again.returncode != 0 and "refusing to overwrite" in again.stderr

    # (d) Evaluate the folded checkpoints as the plain network, raw and EMA weights.
    for weights in ("model", "ema"):
        evaluated = _evaluate(folded_dir, "--weights", weights)
        assert evaluated.returncode == 0, evaluated.stderr
        results_dir = folded_dir / ("evaluation" if weights == "model" else "evaluation-ema")
        results = json.loads((results_dir / "evaluation-results.json").read_text(encoding="utf-8"))
        expected_aliases = ["best", "last"] if weights == "model" else ["best-ema", "last"]
        assert sorted(item["checkpoint_alias"] for item in results) == expected_aliases
        for item in results:
            assert item["student"] == REPARAM
            assert item["deployed_architecture"] == PLAIN
            assert item["reparameterization"]["contract"] == FOLD_CONTRACT
            assert item["reparameterization"]["source_checkpoint_filename"] == item["checkpoint_filename"]
            assert item["weights"] == weights


# ----------------------------------------------------------------- fold CLI fail-closed paths (review P2-3)


def test_fold_refuses_when_the_equivalence_check_fails_and_writes_nothing(
    tmp_path: Path, trained_run: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fold that forgets the identity branch fails the float64 check before any file is written."""
    run = _copy_run(trained_run[1], tmp_path / "run")
    original = ReparamConvBnAct.folded_kernel_and_bias

    def without_identity(self: ReparamConvBnAct) -> tuple[torch.Tensor, torch.Tensor]:
        identity, self.identity = self.identity, None
        try:
            return original(self)
        finally:
            self.identity = identity

    monkeypatch.setattr(ReparamConvBnAct, "folded_kernel_and_bias", without_identity)
    with pytest.raises(ValueError, match="differ beyond tolerance"):
        fold_run(run, allow_dirty_source=True)
    assert not (run / "folded").exists()


def test_fold_refuses_an_already_folded_directory(tmp_path: Path, trained_run: tuple[Path, Path]) -> None:
    run = _copy_run(trained_run[1], tmp_path / "run")
    fold_run(run, allow_dirty_source=True)
    with pytest.raises(ValueError, match="already folded"):
        fold_run(run / "folded", tmp_path / "refolded", allow_dirty_source=True)
    assert not (tmp_path / "refolded").exists()


def test_fold_refuses_a_run_whose_student_is_not_reparameterized(
    tmp_path: Path, trained_run: tuple[Path, Path]
) -> None:
    run = _copy_run(trained_run[1], tmp_path / "run")
    config = run / "resolved_config.yaml"
    text = config.read_text(encoding="utf-8")
    assert text.count(REPARAM) == 1
    config.write_text(text.replace(REPARAM, PLAIN), encoding="utf-8")
    with pytest.raises(ValueError, match="not a reparameterized architecture"):
        fold_run(run, allow_dirty_source=True)
    assert not (run / "folded").exists()


def test_fold_refuses_a_non_empty_output_directory(tmp_path: Path, trained_run: tuple[Path, Path]) -> None:
    run = _copy_run(trained_run[1], tmp_path / "run")
    (run / "folded").mkdir()
    (run / "folded" / "stray.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        fold_run(run, allow_dirty_source=True)
    assert sorted(path.name for path in (run / "folded").iterdir()) == ["stray.txt"]


def test_fold_refuses_a_dirty_source_tree_unless_allowed(
    tmp_path: Path, trained_run: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dirty-tree refusal and the recorded flag, independent of the real working tree's state."""
    from ard.cli import fold_reparam

    real_run = subprocess.run

    def git_reports_dirty(command: list[str], *args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command[:1] == ["git"] and "status" in command:
            return subprocess.CompletedProcess(command, 0, stdout=" M src/ard/models/reparam.py\n", stderr="")
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(fold_reparam.subprocess, "run", git_reports_dirty)
    run = _copy_run(trained_run[1], tmp_path / "run")
    with pytest.raises(ValueError, match="uncommitted changes"):
        fold_run(run)
    assert not (run / "folded").exists()
    manifest = fold_run(run, allow_dirty_source=True)
    assert manifest["source_git_dirty"] is True
    assert all(item["source_git_dirty"] is True for item in manifest["checkpoints"])
