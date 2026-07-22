from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.t0

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from verify import (  # noqa: E402
    CACHE_ENVIRONMENT_KEYS,
    VerificationError,
    build_test_environment,
    changed_paths,
    command_cacheable,
    exclude_scientific_markers,
    gate_relevant_paths,
)
from verify import main as verify_main  # noqa: E402

from ard.testing.cache import CacheRecord, PassCache, environment_identity, external_identity, fingerprint  # noqa: E402
from ard.testing.gpu_lock import GPULock  # noqa: E402
from ard.testing.impact import select  # noqa: E402


def test_unborn_repository_changed_paths_include_untracked_files(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.DEVNULL)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "new.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "ignored").mkdir()
    (tmp_path / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    (tmp_path / "ignored" / "not-selected.py").write_text("x = 1\n", encoding="utf-8")

    assert changed_paths(tmp_path) == (".gitignore", "src/new.py")


def test_impact_map_selects_marker_tiers_and_conservative_unknown_fallback() -> None:
    tests = (
        "tests/unit/test_external_management.py",
        "tests/unit/test_verify_gate.py",
        "tests/regression/test_trades_upstream_differential.py",
    )
    external = select(("scripts/bootstrap_external.py",), tests)
    assert external.tests == ("tests/unit/test_external_management.py",)
    assert external.tiers == ("T0", "T1")
    external_lock = select(("external.lock.yaml",), tests)
    assert external_lock.tests == (
        "tests/regression/test_trades_upstream_differential.py",
        "tests/unit/test_external_management.py",
    )
    assert external_lock.tiers == ("T0", "T1", "T2")
    unknown = select(("src/ard/new_shared.py",), tests)
    assert unknown.tests == tuple(sorted(tests))
    assert unknown.tiers == ("T0", "T1")
    mixed = select(("Makefile", "src/ard/attacks/pgd.py"), tests)
    assert mixed.tests == (
        "tests/regression/test_trades_upstream_differential.py",
        "tests/unit/test_verify_gate.py",
    )
    test_helper = select(("tests/conftest.py",), tests)
    assert test_helper.tests == tuple(sorted(tests))


def test_configs_changes_select_repository_config_resolution_test() -> None:
    available = (
        "tests/unit/test_config.py",
        "tests/unit/test_verify_gate.py",
        "tests/integration/test_synthetic_training.py",
    )
    selected = select(("configs/experiments/synthetic_rslad_joint.yaml",), available)
    assert "tests/unit/test_config.py" in selected.tests


def test_m1_attack_impact_selects_numerical_integration_and_gpu_smoke() -> None:
    tests = (
        "tests/unit/test_pgd.py",
        "tests/integration/test_checkpoint_resume.py",
        "tests/integration/test_synthetic_training.py",
        "tests/smoke/test_gpu_pgd.py",
        "tests/unit/test_config.py",
    )
    selected = select(("src/ard/attacks/pgd.py",), tests)
    assert selected.tests == tuple(sorted(tests[:4]))
    assert selected.tiers == ("T2", "T3")


@pytest.mark.parametrize(
    "changed_path",
    [
        "src/ard/config/schema.py",
        "src/ard/engine/trainer.py",
        "src/ard/objectives/rslad.py",
        "src/ard/cli/train.py",
    ],
)
def test_m3_scientific_paths_select_both_m3_regressions(changed_path: str) -> None:
    m3_tests = {
        "tests/regression/test_m3_student_aware.py",
        "tests/regression/test_m3_distributed.py",
        "tests/regression/test_m3_runtime_efficiency.py",
    }
    available = tuple(
        sorted(
            {
                *m3_tests,
                "tests/unit/test_config.py",
                "tests/unit/test_imports.py",
                "tests/unit/test_pgd.py",
                "tests/regression/test_m2_baselines.py",
                "tests/regression/test_m2_upstream_oracle.py",
                "tests/integration/test_m2_method_switch.py",
                "tests/integration/test_checkpoint_resume.py",
                "tests/integration/test_synthetic_training.py",
            }
        )
    )
    selected = select((changed_path,), available)
    assert m3_tests.issubset(selected.tests)


def test_train_cli_impact_selects_m4_tracking_resume_and_ddp() -> None:
    required = {
        "tests/unit/test_tracking.py",
        "tests/integration/test_checkpoint_resume.py",
        "tests/integration/test_tracking_evaluation.py",
        "tests/regression/test_m4_distributed.py",
        "tests/smoke/test_training_smoke.py",
    }
    available = tuple(sorted(required | {"tests/unit/test_imports.py", "tests/unit/test_distributed.py"}))
    selected = select(("src/ard/cli/train.py",), available)
    assert required.issubset(selected.tests)
    assert {"T0", "T1", "T3"}.issubset(selected.tiers)


def test_pyproject_and_constraints_impact_select_tracking_and_evaluation() -> None:
    required = {
        "tests/unit/test_tracking.py",
        "tests/unit/test_evaluation.py",
        "tests/integration/test_tracking_evaluation.py",
    }
    available = tuple(sorted(required | {"tests/unit/test_verify_gate.py"}))
    for changed in ("pyproject.toml", "requirements/constraints.txt"):
        assert required.issubset(select((changed,), available).tests)


def test_pyproject_impact_selects_the_bounded_training_smoke() -> None:
    available = ("tests/smoke/test_gpu_pgd.py", "tests/smoke/test_training_smoke.py")
    selected = select(("pyproject.toml",), available)
    assert selected.tests == available


def test_invalid_base_fails_closed_with_nonzero_cli_status(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / "tracked.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=tmp_path, check=True, stdout=subprocess.DEVNULL)

    with pytest.raises(VerificationError, match="does-not-exist"):
        changed_paths(tmp_path, "does-not-exist")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "verify.py"),
            "--changed",
            "--base",
            "does-not-exist",
            "--root",
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert "does-not-exist" in completed.stderr


def test_pass_cache_requires_matching_fingerprint_and_retains_latest_failure(tmp_path: Path) -> None:
    (tmp_path / "source.py").write_text("one\n", encoding="utf-8")
    command = (sys.executable, "-m", "pytest", "-q", "tests/unit/test_verify_gate.py")
    key_one = fingerprint(
        root=tmp_path, command=command, relevant_paths=("source.py",), environment={"PYTHONHASHSEED": "0"}
    )
    cache = PassCache(tmp_path / ".cache" / "results.jsonl")
    cache.append(CacheRecord(key_one, command, "passed"))
    assert cache.has_pass(key_one)
    (tmp_path / "source.py").write_text("two\n", encoding="utf-8")
    key_two = fingerprint(
        root=tmp_path, command=command, relevant_paths=("source.py",), environment={"PYTHONHASHSEED": "0"}
    )
    assert key_two != key_one
    assert not cache.has_pass(key_two)
    cache.append(CacheRecord(key_two, command, "failed"))
    assert cache.failed_commands() == (command,)
    cache.append(CacheRecord(key_two, command, "passed"))
    assert cache.failed_commands() == ()


def test_fingerprint_covers_exact_command_test_source_config_fixture_and_seed(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "configs").mkdir()
    test = tmp_path / "tests" / "test_fixture.py"
    source = tmp_path / "src" / "module.py"
    config = tmp_path / "configs" / "fixture.yaml"
    test.write_text("def test_fixture(): pass\n", encoding="utf-8")
    source.write_text("VALUE = 1\n", encoding="utf-8")
    config.write_text("seed: 7\n", encoding="utf-8")
    paths = ("tests/test_fixture.py", "src/module.py", "configs/fixture.yaml")
    command = (sys.executable, "-m", "pytest", "-q", "tests/test_fixture.py")
    baseline = fingerprint(root=tmp_path, command=command, relevant_paths=paths, fixture_version="v1", seed="7")
    assert baseline != fingerprint(
        root=tmp_path, command=(*command, "-x"), relevant_paths=paths, fixture_version="v1", seed="7"
    )
    assert baseline != fingerprint(root=tmp_path, command=command, relevant_paths=paths, fixture_version="v2", seed="7")
    assert baseline != fingerprint(root=tmp_path, command=command, relevant_paths=paths, fixture_version="v1", seed="8")
    for path in (test, source, config):
        path.write_text(path.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
        assert baseline != fingerprint(
            root=tmp_path, command=command, relevant_paths=paths, fixture_version="v1", seed="7"
        )
        path.write_text(path.read_text(encoding="utf-8").replace("# changed\n", ""), encoding="utf-8")


def test_external_identity_and_fingerprint_cover_every_locked_repository(tmp_path: Path) -> None:
    lock = tmp_path / "external.lock.yaml"
    lock.write_text(
        """version: 1
repositories:
  trades:
    url: https://example.invalid/TRADES.git
    commit: "2222222222222222222222222222222222222222"
  saad:
    url: https://example.invalid/saad.git
    commit: "1111111111111111111111111111111111111111"
""",
        encoding="utf-8",
    )
    command = (sys.executable, "-m", "pytest", "-q", "tests/test_runtime.py")
    before = fingerprint(root=tmp_path, command=command, relevant_paths=())
    identities = external_identity(tmp_path)
    assert tuple(identities) == ("saad", "trades")
    assert identities["trades"]["locked_commit"] == "2" * 40
    lock.write_text(lock.read_text(encoding="utf-8").replace("2222", "3333", 1), encoding="utf-8")
    assert before != fingerprint(root=tmp_path, command=command, relevant_paths=())


def test_environment_identity_canonicalizes_cuda_uuid_and_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    class _CUuuid:
        def __init__(self, value: str) -> None:
            self.value = value

        def __str__(self) -> str:
            return self.value

    active_uuid: list[_CUuuid | None] = [_CUuuid("GPU-one")]

    def properties(_: int) -> SimpleNamespace:
        return SimpleNamespace(name="Fake CUDA", major=8, minor=9, uuid=active_uuid[0])

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)

    identity = environment_identity()
    assert json.loads(json.dumps(identity))["gpus"][0] == {
        "index": 0,
        "name": "Fake CUDA",
        "capability": [8, 9],
        "uuid": "GPU-one",
    }

    command = (sys.executable, "-m", "pytest", "-q", "tests/test_runtime.py")
    first = fingerprint(root=tmp_path, command=command, relevant_paths=())
    active_uuid[0] = _CUuuid("GPU-two")
    second = fingerprint(root=tmp_path, command=command, relevant_paths=())
    assert first != second

    active_uuid[0] = None
    assert environment_identity()["gpus"] == [{"index": 0, "name": "Fake CUDA", "capability": [8, 9], "uuid": None}]


def test_test_environment_sets_explicit_cache_seed_and_fixture_version(tmp_path: Path) -> None:
    environment = build_test_environment(tmp_path)
    assert environment["PYTHONHASHSEED"] == "0"
    assert environment["ARD_TEST_SEED"] == "0"
    assert environment["ARD_TEST_FIXTURE_VERSION"] == "1"
    assert {"ARD_RUN_SAAD_ORACLE", "ARD_TRADES_SOURCE_EVIDENCE"} <= set(CACHE_ENVIRONMENT_KEYS)


def test_gpu_lock_uses_one_file_per_visible_physical_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ard.testing.gpu_lock.visible_gpu_identities", lambda: ("gpu-b", "gpu-a"))
    with GPULock(lock_dir=tmp_path):
        assert sorted(path.name for path in tmp_path.glob("ard-test-gpu-*.lock"))
        assert len(list(tmp_path.glob("ard-test-gpu-*.lock"))) == 2


def test_smoke_gate_acquires_the_outer_gpu_lock_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "tests" / "smoke").mkdir(parents=True)
    (tmp_path / "tests" / "smoke" / "test_gpu.py").write_text("def test_gpu(): pass\n", encoding="utf-8")
    events: list[str] = []

    class TrackingLock:
        def __enter__(self) -> TrackingLock:
            events.append("enter")
            return self

        def __exit__(self, *_: object) -> None:
            events.append("exit")

    def fake_run(command: tuple[str, ...], **_: object) -> subprocess.CompletedProcess[object]:
        events.append("run")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("verify.GPULock", TrackingLock)
    monkeypatch.setattr("verify.command_cacheable", lambda **_: False)
    monkeypatch.setattr("verify.command_selects_marker", lambda **_: True)
    monkeypatch.setattr("verify.subprocess.run", fake_run)
    monkeypatch.setattr(sys, "argv", ["verify.py", "--smoke", "--root", str(tmp_path)])

    assert verify_main() == 0
    assert events == ["enter", "run", "exit"]


def test_latest_failure_invalidates_same_fingerprint_pass(tmp_path: Path) -> None:
    cache = PassCache(tmp_path / "results.jsonl")
    command = (sys.executable, "-m", "pytest", "-q", "tests/unit/test_verify_gate.py")
    cache.append(CacheRecord("same-key", command, "passed"))
    cache.append(CacheRecord("same-key", command, "failed"))
    assert not cache.has_pass("same-key")


def test_broad_gate_fingerprint_changes_when_production_source_changes(tmp_path: Path) -> None:
    (tmp_path / "src" / "ard").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    source = tmp_path / "src" / "ard" / "production.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    test_file = tmp_path / "tests" / "test_example.py"
    test_file.write_text("def test_example():\n    assert True\n", encoding="utf-8")
    command = (sys.executable, "-m", "pytest", "-q", "-m", "t1")
    relevant = gate_relevant_paths(tmp_path)
    assert "src/ard/production.py" in relevant
    before = fingerprint(root=tmp_path, command=command, relevant_paths=relevant)
    cache = PassCache(tmp_path / "results.jsonl")
    cache.append(CacheRecord(before, command, "passed"))

    source.write_text("VALUE = 2\n", encoding="utf-8")
    after = fingerprint(root=tmp_path, command=command, relevant_paths=gate_relevant_paths(tmp_path))
    assert after != before
    assert not cache.has_pass(after)


def test_changed_t4_file_command_is_never_cacheable(tmp_path: Path) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    scientific = tests / "test_scientific.py"
    scientific.write_text(
        "import pytest\n\n@pytest.mark.t4\ndef test_scientific():\n    assert True\n",
        encoding="utf-8",
    )
    command = (sys.executable, "-m", "pytest", "-q", "tests/test_scientific.py")
    assert not command_cacheable(
        root=tmp_path, command=command, requested_tier=None, environment=build_test_environment(tmp_path)
    )


def test_changed_t4_file_pass_is_not_written_to_cache(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.DEVNULL)
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_scientific.py").write_text(
        "import pytest\n\n@pytest.mark.t4\ndef test_scientific():\n    assert True\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify.py"), "--changed", "--root", str(tmp_path)],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "running:" in completed.stdout
    assert not (tmp_path / ".cache" / "test-gate" / "results.jsonl").exists()


def test_non_scientific_marker_expression_excludes_t4_and_t5() -> None:
    command = (sys.executable, "-m", "pytest", "-q", "tests/test_scientific.py")
    assert exclude_scientific_markers(command)[-2:] == ("-m", "not t4 and not t5")
    existing_marker = (*command, "-m", "t3")
    assert "(t3) and (not t4 and not t5)" in exclude_scientific_markers(existing_marker)


def test_non_scientific_gate_skips_t4_without_caching_it(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.DEVNULL)
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_scientific.py").write_text(
        "import pytest\n\n@pytest.mark.t4\ndef test_scientific():\n    raise AssertionError('must not run')\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify.py"), "--changed", "--non-scientific", "--root", str(tmp_path)],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert not (tmp_path / ".cache" / "test-gate" / "results.jsonl").exists()
