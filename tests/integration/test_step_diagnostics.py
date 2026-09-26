"""``training.step_diagnostics``: the post-step diagnostic forwards never touch training.

Two same-process differentials on the fixture methods of
``test_step_sync_free_parity`` (PGD-AT, RSLAD with a sample store and
teacher-response observation, ADR with its EMA-agreement/rectified-mass
accumulators):

* default (``step_diagnostics=True``) against the pre-change Trainer, loaded
  from ``PRE_CHANGE_COMMIT``'s ``src/ard/engine/trainer.py``: identical epoch
  rows and identical model/optimizer/EMA/RNG/sampler/sample/selection state
  in ``last.pt`` and ``best.pt``;
* ``step_diagnostics=False`` against the default: identical checkpoint state,
  the diagnostic-only metrics absent and every other row entry identical.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch
import yaml
from torch import nn

import tests.integration.test_step_sync_free_parity as parity
from ard.analysis import summarize_checkpoint_groups
from ard.cli import evaluate as evaluate_cli
from ard.cli import train as train_cli
from tests.integration.test_compile_trainer import _cli_config, _write_config

pytestmark = pytest.mark.t3

# Last commit before training.step_diagnostics existed.
PRE_CHANGE_COMMIT = "56a9e80"
DROPPED_ROW_KEYS = frozenset(
    {
        "train_clean_accuracy",
        "train_robust_accuracy_eval_mode",
        "train_robust_overtakes_clean",
        "train_ema_student_agreement",
    }
)
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _pre_change_trainer_module() -> ModuleType:
    """Load the Trainer as it was at ``PRE_CHANGE_COMMIT``.

    One-time proof that introducing the option left the default path
    bit-identical. It depends on git history (absent in an export, a shallow
    clone, or once shared ``ard`` modules drift far enough that the old file no
    longer imports); the lasting guard is
    ``test_disabling_step_diagnostics_changes_no_training_state``, which needs
    no history.
    """
    try:
        completed = subprocess.run(
            ["git", "show", f"{PRE_CHANGE_COMMIT}:src/ard/engine/trainer.py"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
    except OSError as exc:  # pragma: no cover - git not installed
        pytest.skip(f"git unavailable, cannot load the pre-change trainer from {PRE_CHANGE_COMMIT}: {exc}")
    if completed.returncode != 0:  # pragma: no cover - commit absent (export or shallow clone)
        pytest.skip(
            f"commit {PRE_CHANGE_COMMIT} (pre-step_diagnostics trainer) is not in this checkout's history: "
            f"{completed.stderr.strip()}"
        )
    source = completed.stdout
    assert "step_diagnostics" not in source
    name = "ard.engine._trainer_pre_step_diagnostics"
    spec = importlib.util.spec_from_loader(name, loader=None)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "ard.engine"
    sys.modules[name] = module
    try:
        exec(compile(source, f"<{PRE_CHANGE_COMMIT}:src/ard/engine/trainer.py>", "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module


def _batchnorm_dropout_student(*_: Any, **__: Any) -> nn.Module:
    """fixture_cnn has neither BatchNorm nor dropout; this student has both.

    A diagnostic forward that ran in train mode would move BatchNorm running
    statistics and draw dropout masks from the global RNG, so both would
    show up in the checkpoint digests below.
    """
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.Dropout(p=0.25),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 3),
    )


def _run(
    output: Path, method: str, *, step_diagnostics: bool | None
) -> tuple[list[dict[str, Any]], dict[str, str], int]:
    """Fit two epochs; return epoch rows, checkpoint-state digests and student forward count."""
    trainer = parity._trainer(output, method)
    if step_diagnostics is not None:
        trainer.step_diagnostics = step_diagnostics
    forwards = 0

    def count(*_: Any) -> None:
        nonlocal forwards
        forwards += 1

    handle = trainer.model.register_forward_hook(count)
    loader, validation_loader = parity._loaders()
    try:
        history = trainer.fit(loader, validation_loader=validation_loader, epochs=2)
    finally:
        handle.remove()
    rows = [{key: value for key, value in row.items() if not key.endswith(parity._TIMING_SUFFIXES)} for row in history]
    digests: dict[str, str] = {}
    for name in ("last.pt", "best.pt"):
        payload = torch.load(output / name, map_location="cpu", weights_only=False)
        for key in parity._STATE_KEYS:
            if key in payload:
                digest = hashlib.sha256()
                parity._canonical(payload[key], digest)
                digests[f"{name}:{key}"] = digest.hexdigest()
    return rows, digests, forwards


def _exact(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{key: float(value).hex() for key, value in sorted(row.items())} for row in rows]


@pytest.fixture
def deterministic_algorithms() -> Any:
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


@pytest.mark.parametrize("student", ["fixture_cnn", "batchnorm_dropout"])
@pytest.mark.parametrize("method", parity.METHODS)
def test_default_is_bit_identical_to_the_pre_change_trainer(
    tmp_path: Path, method: str, student: str, monkeypatch: pytest.MonkeyPatch, deterministic_algorithms: None
) -> None:
    if student == "batchnorm_dropout":
        monkeypatch.setattr(parity, "build_student", _batchnorm_dropout_student)
    with monkeypatch.context() as patch:
        patch.setattr(parity, "Trainer", _pre_change_trainer_module().Trainer)
        old = parity._fingerprint(tmp_path / "old", method, device=torch.device("cpu"), loaders=parity._loaders())
    new_trainer = parity._trainer(tmp_path / "probe", method)
    assert new_trainer.step_diagnostics is True
    new = parity._fingerprint(tmp_path / "new", method, device=torch.device("cpu"), loaders=parity._loaders())
    assert "last.pt:rng" in new and "best.pt:model" in new
    assert new == old


@pytest.mark.parametrize("student", ["fixture_cnn", "batchnorm_dropout"])
@pytest.mark.parametrize("method", parity.METHODS)
def test_disabling_step_diagnostics_changes_no_training_state(
    tmp_path: Path, method: str, student: str, monkeypatch: pytest.MonkeyPatch, deterministic_algorithms: None
) -> None:
    if student == "batchnorm_dropout":
        monkeypatch.setattr(parity, "build_student", _batchnorm_dropout_student)
    rows_on, state_on, forwards_on = _run(tmp_path / "on", method, step_diagnostics=True)
    rows_off, state_off, forwards_off = _run(tmp_path / "off", method, step_diagnostics=False)

    # Model, optimizer, EMA, every RNG stream, sampler, sample state and
    # best/last selection are untouched by the skipped forwards.
    assert "last.pt:rng" in state_on and "best.pt:model" in state_on
    assert state_off == state_on
    # Exactly the two post-step forwards per step (2 epochs x 3 batches) were skipped.
    assert forwards_on - forwards_off == 2 * 2 * 3

    expected_dropped = {"train_clean_accuracy", "train_robust_accuracy_eval_mode", "train_robust_overtakes_clean"}
    if method == "adr":
        expected_dropped.add("train_ema_student_agreement")
    for row_on, row_off in zip(rows_on, rows_off, strict=True):
        assert expected_dropped <= set(row_on)
        assert not (DROPPED_ROW_KEYS & set(row_off))
        assert set(row_on) - set(row_off) == expected_dropped
        assert list(row_off) == [key for key in row_on if key not in expected_dropped]
    kept = [{key: value for key, value in row.items() if key not in DROPPED_ROW_KEYS} for row in rows_on]
    assert _exact(rows_off) == _exact(kept)


def test_default_still_requires_the_diagnostic_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With the default, a train_epoch result lacking clean_accuracy is an error, never a silent drop."""
    trainer = parity._trainer(tmp_path / "missing", "pgd_at")
    real_train_epoch = trainer.train_epoch

    def without_clean_accuracy(*args: Any, **kwargs: Any) -> dict[str, float]:
        metrics = real_train_epoch(*args, **kwargs)
        del metrics["clean_accuracy"]
        return metrics

    monkeypatch.setattr(trainer, "train_epoch", without_clean_accuracy)
    loader, validation_loader = parity._loaders()
    with pytest.raises(KeyError, match="clean_accuracy"):
        trainer.fit(loader, validation_loader=validation_loader, epochs=1)


def test_train_cli_threads_step_diagnostics_into_rows_and_resolved_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    deterministic = torch.are_deterministic_algorithms_enabled()
    rows: dict[bool, list[dict[str, Any]]] = {}
    try:
        for enabled in (True, False):
            output = tmp_path / f"diagnostics-{enabled}"
            data = _cli_config(output)
            if not enabled:
                data["training"]["step_diagnostics"] = False
            config_path = _write_config(tmp_path, f"diagnostics-{enabled}", data)
            assert train_cli.main(["--config", str(config_path)]) == 0
            resolved = yaml.safe_load((output / "resolved_config.yaml").read_text(encoding="utf-8"))
            if enabled:
                # Default configs keep a byte-identical resolved config / hash.
                assert "step_diagnostics" not in resolved["training"]
            else:
                assert resolved["training"]["step_diagnostics"] is False
            lines = (output / "epoch-metrics.jsonl").read_text(encoding="utf-8").splitlines()
            rows[enabled] = [json.loads(line) for line in lines if line]
    finally:
        torch.use_deterministic_algorithms(deterministic)
    assert rows[True] and len(rows[True]) == len(rows[False])
    for row_on, row_off in zip(rows[True], rows[False], strict=True):
        assert {"train_clean_accuracy", "train_robust_accuracy_eval_mode", "train_robust_overtakes_clean"} <= set(
            row_on
        )
        assert not (DROPPED_ROW_KEYS & set(row_off))
        assert row_off["train_robust_accuracy"] == row_on["train_robust_accuracy"]
        assert row_off["val_pgd_accuracy"] == row_on["val_pgd_accuracy"]


def test_step_diagnostics_never_splits_the_evaluation_pooling_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Training-neutral, so an arm's True and False seeds must pool (unlike compile)."""
    monkeypatch.chdir(tmp_path)
    deterministic = torch.are_deterministic_algorithms_enabled()
    rows: dict[bool, list[dict[str, Any]]] = {}
    try:
        for model_init, enabled in ((8, True), (9, False)):
            name = f"pool-{enabled}"
            output = tmp_path / name
            data = _cli_config(output)
            data["seeds"]["model_init"] = model_init
            data["tracker_run_id"] = f"smoke-local-{name}"
            if not enabled:
                data["training"]["step_diagnostics"] = False
            config_path = _write_config(tmp_path, name, data)
            assert train_cli.main(["--config", str(config_path)]) == 0
            evaluation_output = tmp_path / f"{name}-evaluation"
            arguments = [
                "--config",
                str(config_path),
                "--checkpoint-dir",
                str(output),
                "--output",
                str(evaluation_output),
            ]
            assert evaluate_cli.main(arguments) == 0
            rows[enabled] = json.loads((evaluation_output / "evaluation-results.json").read_text(encoding="utf-8"))
    finally:
        torch.use_deterministic_algorithms(deterministic)
    identities = {
        enabled: {json.dumps(row["training_protocol_identity"], sort_keys=True) for row in result}
        for enabled, result in rows.items()
    }
    assert len(identities[True]) == 1
    assert identities[True] == identities[False]
    assert all("step_diagnostics" not in row["training_protocol_identity"] for row in rows[True] + rows[False])
    pooled = summarize_checkpoint_groups(rows[True] + rows[False], metric="pgd_accuracy")
    assert {alias: summary["count"] for alias, summary in pooled.items()} == {"best": 2, "last": 2}
