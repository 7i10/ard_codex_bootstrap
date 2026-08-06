"""Pure-Python contracts for the isolated full-SAAD operational launcher."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from scripts.run_saad_upstream import (
    BARTOLDSON_SHA256,
    CIFAR10_ARCHIVE_SHA256,
    CIFAR10_EXTRACTED_SHA256,
    GPUSampler,
    SAADConfig,
    SAADLaunchError,
    build_runtime_command,
    classify_smoke,
    launch_identity,
    load_config,
    observe_loss_chunk,
    read_available_pipe_bytes,
    runtime_environment,
    select_execution,
    stage_inputs,
    telemetry_summary,
    upstream_args,
    verify_inputs,
)

pytestmark = [pytest.mark.t1, pytest.mark.unit]


def _raw_config(tmp_path: Path) -> dict[str, object]:
    return {
        "version": 1,
        "runtime": {"python": str(tmp_path / "python"), "python_version": "3.11.15"},
        "inputs": {
            "dataset_root": str(tmp_path / "dataset"),
            "cifar10_archive_sha256": CIFAR10_ARCHIVE_SHA256,
            "teacher_checkpoint": str(tmp_path / "teacher.pt"),
            "teacher_checkpoint_sha256": BARTOLDSON_SHA256,
        },
        "output": {"directory": str(tmp_path / "fresh-output")},
        "smoke": {"batch_size": 16, "loss_events": 2},
        "gpu": {"physical_id": 0},
        "teacher_logit_contract": {
            "reference_torch": "2.11.0+cu128",
            "candidate_torch": "2.4.1+cu121",
            "fixed_input_count": 4,
            "atol": 0.0001,
            "rtol": 0,
            "require_argmax_equal": True,
            "measured_max_abs": 0.000080824,
            "measured_mean_abs": 0.000030991,
        },
        "protocol": {
            "method": "saad",
            "epochs": 200,
            "batch_size": 128,
            "seed": 0,
            "student": "RES-18",
            "teacher_name": "Bartoldson2024Adversarial_WRN-94-16",
            "dataset": "cifar10",
            "swa_epoch": 95,
            "beta": 0,
            "gamma": 1,
            "igdm_alpha": 1,
            "lambda_inner": 1,
            "nowand": 1,
            "lr": 0.1,
            "momentum": 0.9,
            "weight_decay": 0.0002,
            "milestones": [100, 150],
            "inner_steps": 10,
            "epsilon": "8/255",
            "step_size": "2/255",
            "entropy_multiplier": 5,
        },
    }


def _config(tmp_path: Path, raw: dict[str, object] | None = None) -> SAADConfig:
    path = tmp_path / "saad.yaml"
    path.write_text(yaml.safe_dump(raw or _raw_config(tmp_path)), encoding="utf-8")
    return load_config(path)


def test_checked_in_operational_config_is_strictly_parseable() -> None:
    config = load_config(Path(__file__).parents[2] / "configs" / "upstream" / "saad_bartoldson_seed0.yaml")
    assert config.smoke_batch_size == 128
    assert config.physical_gpu == 0
    assert str(config.runtime_python) == "/home/shunsukenaito/.conda/envs/saad-oracle-py311/bin/python"
    assert config.teacher_logit_contract["atol"] == 0.0001
    lock = (Path(__file__).parents[2] / "requirements" / "saad-upstream-runtime.lock").read_text(encoding="utf-8")
    for pin in ("Jinja2==3.1.6", "timm==1.0.9", "pandas==2.2.3", "gdown==5.1.0", "setuptools==75.3.0"):
        assert pin in lock


def test_operational_config_rejects_unknown_key_and_protocol_drift(tmp_path: Path) -> None:
    raw = _raw_config(tmp_path)
    raw["unexpected"] = True
    with pytest.raises(SAADLaunchError, match="unknown keys"):
        _config(tmp_path, raw)
    raw = _raw_config(tmp_path)
    protocol = raw["protocol"]
    assert isinstance(protocol, dict)
    protocol["epochs"] = 199
    with pytest.raises(SAADLaunchError, match="protocol drift"):
        _config(tmp_path, raw)


def test_full_command_is_frozen_and_smoke_batch_is_explicit(tmp_path: Path) -> None:
    config = _config(tmp_path)
    smoke = upstream_args(batch_size=config.smoke_batch_size)
    full = upstream_args(batch_size=128)
    assert "--batch" in smoke and smoke[smoke.index("--batch") + 1] == "16"
    assert full[full.index("--batch") + 1] == "128"
    assert "--teacher_name" in full
    assert full[full.index("--teacher_name") + 1] == "Bartoldson2024Adversarial_WRN-94-16"
    assert full[full.index("--depth") + 1] == "0"
    assert full[full.index("--widen_factor") + 1] == "0"
    with pytest.raises(SAADLaunchError, match="smoke batch"):
        upstream_args(batch_size=64)


def test_runtime_command_cannot_forward_arbitrary_upstream_arguments(tmp_path: Path) -> None:
    config = _config(tmp_path)
    command = build_runtime_command(
        config=config,
        stage=tmp_path / "stage",
        robustbench=tmp_path / "robustbench",
        saad=tmp_path / "saad",
        provenance=tmp_path / "provenance.json",
        batch_size=128,
    )
    expected_bootstrap = Path(__file__).parents[2] / "scripts" / "saad_runtime_bootstrap.py"
    assert command[:2] == [str(tmp_path / "python"), str(expected_bootstrap)]
    assert command.count("--") == 1
    assert "--runtime-lock" in command
    assert command[command.index("--saad-root") + 1] == str(tmp_path / "saad")
    assert command[command.index("--expected-python") + 1] == "3.11.15"
    assert command[-1] == "0"
    assert "--epochs" in command and command[command.index("--epochs") + 1] == "200"


def test_safe_staging_uses_symlinks_and_refuses_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source-saad"
    source.mkdir()
    (source / "saad.py").write_text("# source\n", encoding="utf-8")
    (source / "autoattack").mkdir()
    (source / "autoattack" / "__init__.py").write_text("AutoAttack = object\n", encoding="utf-8")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"teacher")
    config = SAADConfig(
        runtime_python=tmp_path / "python",
        dataset_root=dataset,
        teacher_checkpoint=teacher,
        output_dir=tmp_path / "fresh-output",
        smoke_batch_size=16,
        smoke_loss_events=2,
        physical_gpu=0,
        teacher_logit_contract={},
    )
    stage = stage_inputs(config=config, saad=source, output_dir=config.output_dir)
    assert (stage / "saad.py").is_symlink()
    assert (config.output_dir / "dataset").resolve() == dataset.resolve()
    staged_teacher = stage / "models" / "cifar10" / "Linf" / "Bartoldson2024Adversarial_WRN-94-16.pt"
    assert staged_teacher.resolve() == teacher.resolve()
    with pytest.raises(SAADLaunchError, match="overwrite"):
        stage_inputs(config=config, saad=source, output_dir=config.output_dir)


@pytest.mark.parametrize(
    ("terminated", "events", "requested", "returncode", "expected"),
    [
        (True, 2, 2, -15, "expected_smoke_termination"),
        (True, 1, 2, -15, "smoke_failure"),
        (False, 0, 2, 0, "unexpected_smoke_completion"),
        (False, 1, 2, 1, "smoke_failure"),
    ],
)
def test_smoke_state_contract(terminated: bool, events: int, requested: int, returncode: int, expected: str) -> None:
    assert (
        classify_smoke(
            terminated_by_supervisor=terminated, loss_events=events, requested_events=requested, returncode=returncode
        )
        == expected
    )


def test_nonfinite_loss_never_satisfies_smoke_contract() -> None:
    assert (
        classify_smoke(
            terminated_by_supervisor=True,
            loss_events=2,
            requested_events=2,
            returncode=-15,
            nonfinite_loss=True,
        )
        == "smoke_failure"
    )


def test_stdout_progress_loss_stream_handles_split_chunks_cr_and_nonfinite_once() -> None:
    buffer, values, invalid = observe_loss_chunk("", b"\r[epoch 1] loss: 1.")
    assert values == () and invalid == ()
    buffer, values, invalid = observe_loss_chunk(buffer, b"25 |\r[epoch 1] loss: 2.5 |")
    assert values == (1.25, 2.5) and invalid == ()
    buffer, values, invalid = observe_loss_chunk(buffer, b"\r[epoch 1] loss: nan |")
    assert values == () and invalid == ("nan",)
    _, values, invalid = observe_loss_chunk(buffer, b"")
    assert values == () and invalid == ()


def test_prompt_pipe_read_does_not_wait_for_buffer_fill() -> None:
    read_fd, write_fd = os.pipe()
    try:
        with os.fdopen(read_fd, "rb", closefd=True) as reader:
            os.write(write_fd, b"\rloss: 1.25 |")
            # The writer intentionally remains open and contributes far fewer
            # than 4096 bytes.  BufferedReader.read(4096) would block here.
            assert read_available_pipe_bytes(reader) == b"\rloss: 1.25 |"
    finally:
        os.close(write_fd)


def test_runtime_bootstrap_has_explicit_two_autoattack_origins() -> None:
    source = (Path(__file__).parents[2] / "scripts" / "saad_runtime_bootstrap.py").read_text(encoding="utf-8")
    assert "import autoattack.state as official_state" in source
    assert "verify_runtime_lock(args.runtime_lock)" in source
    assert "runtime Python drift" in source
    assert "direct-reference drift" in source
    assert "del sys.modules[name]" in source
    assert "import autoattack as saad_autoattack" in source
    assert "SAAD local autoattack.AutoAttack was not selected" in source


def test_runtime_environment_rejects_ambient_import_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "/unsafe/import")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    environment = runtime_environment(physical_gpu=0)
    assert "PYTHONPATH" not in environment
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert environment["PYTHONUNBUFFERED"] == "1"


def test_gpu_telemetry_samples_short_smokes_at_half_second_and_has_stable_schema() -> None:
    assert GPUSampler(physical_gpu=0).interval_seconds == 0.5
    telemetry = telemetry_summary(
        physical_gpu=0,
        samples=[{"memory_mib": 5226, "utilization_percent": 77, "temperature_c": 42}],
        errors=[],
    )
    assert telemetry == {
        "physical_gpu": 0,
        "samples": [{"memory_mib": 5226, "utilization_percent": 77, "temperature_c": 42}],
        "peak_memory_mib": 5226,
        "peak_utilization_percent": 77,
        "peak_temperature_c": 42,
        "errors": [],
    }


def test_launch_identity_hashes_command_sources_and_current_git_state() -> None:
    root = Path(__file__).parents[2]
    identity = launch_identity(
        config_path=root / "configs" / "upstream" / "saad_bartoldson_seed0.yaml", command=["x", "y"]
    )
    assert len(identity["command_sha256"]) == 64
    assert len(identity["ard_git"]["head"]) == 40
    assert set(identity["source_files"]) == {"launcher", "runtime_bootstrap", "config", "runtime_lock"}


def test_execute_requires_fresh_output_override_and_explicit_smoke_variant(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with pytest.raises(SAADLaunchError, match="--output-dir"):
        select_execution(config, mode="smoke", smoke_batch_size=16, output_dir=None)
    smoke, batch = select_execution(config, mode="smoke", smoke_batch_size=16, output_dir=tmp_path / "smoke-16")
    assert batch == 16 and smoke.output_dir == tmp_path / "smoke-16"
    full, full_batch = select_execution(config, mode="full", smoke_batch_size=None, output_dir=tmp_path / "full")
    assert full_batch == 128 and full.output_dir == tmp_path / "full"
    with pytest.raises(SAADLaunchError, match="invalid in full"):
        select_execution(config, mode="full", smoke_batch_size=128, output_dir=tmp_path / "bad")


def test_cifar_integrity_requires_every_extracted_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    config.dataset_root.mkdir()
    extracted = config.dataset_root / "cifar-10-batches-py"
    extracted.mkdir()
    archive = config.dataset_root / "cifar-10-python.tar.gz"
    archive.write_bytes(b"archive")
    for name in CIFAR10_EXTRACTED_SHA256:
        (extracted / name).write_bytes(name.encode())
    config.teacher_checkpoint.write_bytes(b"teacher")

    def fake_hash(path: Path) -> str:
        if path == archive:
            return CIFAR10_ARCHIVE_SHA256
        if path == config.teacher_checkpoint:
            return BARTOLDSON_SHA256
        return CIFAR10_EXTRACTED_SHA256[path.name]

    monkeypatch.setattr("scripts.run_saad_upstream._sha256", fake_hash)
    identity = verify_inputs(config)
    assert set(identity["cifar10_extracted"]) == set(CIFAR10_EXTRACTED_SHA256)
    (extracted / "data_batch_5").unlink()
    with pytest.raises(SAADLaunchError, match="data_batch_5"):
        verify_inputs(config)


def test_symlink_stage_origin_is_verified_against_source_root(tmp_path: Path) -> None:
    bootstrap_path = Path(__file__).parents[2] / "scripts" / "saad_runtime_bootstrap.py"
    spec = importlib.util.spec_from_file_location("saad_runtime_bootstrap_test", bootstrap_path)
    assert spec is not None and spec.loader is not None
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    source = tmp_path / "source-saad"
    package = source / "autoattack"
    package.mkdir(parents=True)
    init = package / "__init__.py"
    init.write_text("AutoAttack = object\n", encoding="utf-8")
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "autoattack").symlink_to(package, target_is_directory=True)
    module = ModuleType("autoattack")
    module.__file__ = str(stage / "autoattack" / "__init__.py")
    identity = bootstrap.module_identity(module)
    assert identity["path"] == str(init.resolve())
    assert bootstrap.under(identity["path"], str(source))
