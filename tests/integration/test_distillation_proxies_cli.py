"""End-to-end ``ard.cli.distillation_proxies --contract v2`` on a CPU ImageNet-layout fixture.

Two fixture "Phase 1 runs" (resolved_config.yaml + last.pt); one is the student,
the other is passed with ``--student-teacher`` as a proxy-only teacher.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from ard.cli import distillation_proxies as cli
from ard.config.loader import resolved_config_dict
from ard.distillation.proxies import PROXIES_V2_CONTRACT
from ard.models import build_student

_SPEC = importlib.util.spec_from_file_location(
    "soft_label_bank_fixtures", Path(__file__).resolve().parents[1] / "unit" / "test_soft_label_bank.py"
)
assert _SPEC is not None and _SPEC.loader is not None
fixtures: Any = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fixtures)


def _run(root: Path, runs: Path, name: str, seed: int) -> Path:
    config = fixtures._config(root)
    payload = resolved_config_dict(config)
    payload["tracking"]["run_id"] = name
    directory = runs / name / "outputs" / "train"
    directory.mkdir(parents=True)
    (directory / "resolved_config.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")
    torch.manual_seed(seed)
    model = build_student(config.student, tier=config.tier)
    torch.save({"model": model.state_dict()}, directory / "last.pt")
    return directory / "last.pt"


def test_v2_cli_with_a_same_recipe_teacher(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "imagenet"
    fixtures._make_imagenet(root)
    runs = tmp_path / "runs"
    student = _run(root, runs, "student-run", seed=1)
    _run(root, runs, "teacher-run", seed=2)
    rslad_config = tmp_path / "rslad.yaml"
    rslad_config.write_text(
        "method: {id: rslad, version: 1, attack: {loss: kl, kl_target: teacher_clean, epsilon: '4/255', "
        "step_size: '8/765', steps: 2, random_start: true}}\n",
        encoding="utf-8",
    )
    out = tmp_path / "proxies"
    argv = [
        "--contract", "v2", "--attack", "ce", "--attack", "rslad", "--rslad-attack-config", str(rslad_config),
        "--student-checkpoint", str(student), "--student-teacher", f"own={runs / 'teacher-run'}",
        "--student-teacher", f"self={student}", "--output-dir", str(out), "--subset-size", "8",
        "--batch-size", "3", "--num-workers", "0", "--device", "cpu",
    ]  # fmt: skip
    assert cli.main(argv) == 0
    files = sorted(path.name for path in out.iterdir())
    assert files == [
        f"proxies-v2-student-run-last-{teacher}-{attack}.json"
        for teacher in ("own", "self")
        for attack in ("ce", "rslad")
    ]
    record = json.loads((out / "proxies-v2-student-run-last-own-rslad.json").read_text())
    assert record["contract"] == PROXIES_V2_CONTRACT and record["official_evaluation"] is False
    assert record["student_attack"]["name"] == "rslad"
    assert record["student_attack"]["identity"]["loss"] == "kl"
    assert record["student_attack"]["identity"]["kl_target"] == "teacher_clean"
    assert record["student_attack"]["provenance"]["config"] == str(rslad_config)
    assert record["teacher_own_attack"]["identity"]["loss"] == "ce"
    assert record["teacher"]["source"] == "phase1_run_checkpoint" and record["teacher"]["run_id"] == "teacher-run"
    assert record["self_pair"] is False and record["metrics"]["count"] == 8
    for key in ("exposed_vulnerability_fraction", "input_gradient_cosine_clean", "correct_and_reactive_rate"):
        assert key in record["metrics"]
    own_self = json.loads((out / "proxies-v2-student-run-last-self-ce.json").read_text())
    assert own_self["self_pair"] is True
    assert own_self["metrics"]["input_gradient_cosine_clean"] == pytest.approx(1.0, abs=1e-6)
    # Re-running skips every existing (pair, attack) file.
    capsys.readouterr()
    assert cli.main(argv) == 0
    assert capsys.readouterr().out.count("skip existing") == 4


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ([], "at least one --teacher or --student-teacher"),
        (["--student-teacher", "x=/a"], "require --contract v2"),
        (["--teacher", "a=/x", "--attack", "rslad"], "require --contract v2"),
        (["--contract", "v2", "--teacher", "a=/x", "--student-teacher", "a=/y"], "unique"),
        (["--contract", "v2", "--teacher", "a"], "NAME=PATH"),
    ],
)
def test_v2_cli_argument_guards(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extra: list[str], message: str
) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--student-checkpoint", str(tmp_path / "x.pt"), "--output-dir", str(tmp_path), *extra])
    assert message in capsys.readouterr().err


def test_run_directory_resolution(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "outputs" / "train").mkdir(parents=True)
    (run / "outputs" / "train" / "last.pt").write_bytes(b"x")
    assert cli.resolve_run_checkpoint(run) == run / "outputs" / "train" / "last.pt"
    assert cli.resolve_run_checkpoint(run / "outputs" / "train") == run / "outputs" / "train" / "last.pt"
    with pytest.raises(SystemExit, match="no last.pt"):
        cli.resolve_run_checkpoint(tmp_path / "missing")
