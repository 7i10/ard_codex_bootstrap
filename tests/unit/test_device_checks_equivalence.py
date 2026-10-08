"""``ard.device_checks.require`` conversion (plan 0105, review of d2e82b2 P3-3): outside a CUDA-graph capture
every converted check accepts and rejects exactly the inputs it did before, with the same exception type and
message, and returns bit-identical values.

The pre-change functions are read from git (``PRE_CHANGE_COMMIT``, master before the conversion) and run side by
side with the current ones on the same good and bad inputs (CPU).
"""

from __future__ import annotations

import subprocess
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

from ard.distillation import soft_label_bank as new_bank
from ard.distillation import trainer_hooks as new_hooks
from ard.policies import base as new_policies
from ard.signals import teacher_entropy as new_entropy
from ard.targets import validation as new_validation

pytestmark = pytest.mark.t1

ROOT = Path(__file__).resolve().parents[2]
PRE_CHANGE_COMMIT = "d5bae49"


def _old_module(relative: str, name: str, *, preload: dict[str, Any] | None = None, drop: str | None = None) -> Any:
    try:
        source = subprocess.run(
            ["git", "show", f"{PRE_CHANGE_COMMIT}:{relative}"], cwd=ROOT, check=True, text=True, capture_output=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:  # pragma: no cover - shallow clone
        pytest.skip(f"pre-change source unavailable: {error}")
    if drop is not None:
        assert drop in source
        source = source.replace(drop, "")
    module = types.ModuleType(name)
    module.__file__ = f"{PRE_CHANGE_COMMIT}:{relative}"
    module.__dict__.update(preload or {})
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


@pytest.fixture(scope="module")
def old() -> dict[str, Any]:
    bank = _old_module("src/ard/distillation/soft_label_bank.py", "_old_soft_label_bank")
    hooks = _old_module(
        "src/ard/distillation/trainer_hooks.py",
        "_old_trainer_hooks",
        preload={
            "SoftLabelBankTeacher": bank.SoftLabelBankTeacher,
            "kl_rows": bank.kl_rows,
            "truncate_like_bank": bank.truncate_like_bank,
        },
        drop="from .soft_label_bank import SoftLabelBankTeacher, kl_rows, truncate_like_bank\n",
    )
    return {
        "bank": bank,
        "hooks": hooks,
        "policies": _old_module("src/ard/policies/base.py", "_old_policies_base"),
        "entropy": _old_module("src/ard/signals/teacher_entropy.py", "_old_teacher_entropy"),
        "validation": _old_module("src/ard/targets/validation.py", "_old_targets_validation"),
    }


def _outcome(call: Callable[[], Any]) -> tuple[str, Any]:
    """('ok', value bits) or (exception type name, message)."""
    try:
        value = call()
    except Exception as error:  # noqa: BLE001 - the type and message ARE the compared outcome
        return type(error).__name__, str(error)
    if isinstance(value, torch.Tensor):
        return "ok", (value.dtype, tuple(value.shape), value.contiguous().view(-1).tolist())
    if value is None:
        return "ok", None
    return "ok", tuple(
        None if tensor is None else tensor.tolist() for tensor in (value.hard_weight, value.kd_weight, value.joint_risk)
    )


def _rows(k: int, classes: int, *, residual: float, prob: list[list[float]] | None = None) -> tuple[Any, ...]:
    probabilities = torch.tensor(prob if prob is not None else [[0.5, 0.2][:k] + [0.1] * (k - 2)] * 2)
    index = torch.arange(k).repeat(2, 1)
    return index, probabilities[:, :k].half(), torch.full((2,), residual).half(), classes


_RECONSTRUCT = {
    "good_top_k": _rows(2, 5, residual=0.3),
    "good_full_k": _rows(2, 2, residual=0.0, prob=[[0.6, 0.4], [0.5, 0.5]]),
    "full_k_nonzero_residual": _rows(2, 2, residual=0.1, prob=[[0.6, 0.4], [0.5, 0.5]]),
    "nan_probability": _rows(2, 5, residual=0.3, prob=[[float("nan"), 0.2], [0.5, 0.2]]),
    "empty_row": _rows(2, 5, residual=0.0, prob=[[0.0, 0.0], [0.5, 0.2]]),
    "misaligned": (torch.arange(2).repeat(2, 1), torch.ones(2, 3).half(), torch.zeros(2).half(), 5),
}
_DISTRIBUTIONS = {
    "good": torch.tensor([[0.25, 0.75], [0.5, 0.5]]),
    "nan": torch.tensor([[float("nan"), 0.5], [0.5, 0.5]]),
    "inf": torch.tensor([[float("inf"), 0.5], [0.5, 0.5]]),
    "negative": torch.tensor([[-0.25, 1.25], [0.5, 0.5]]),
    "not_normalized": torch.tensor([[0.3, 0.3], [0.5, 0.5]]),
}
_ENTROPY = {
    "good": torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]),
    "nan": torch.tensor([[float("nan"), 2.0, 3.0], [0.0, 0.0, 0.0]]),
    "inf": torch.tensor([[float("inf"), 2.0, 3.0], [0.0, 0.0, 0.0]]),
}
_POLICY = {
    "good": (torch.zeros(3), torch.ones(3), None),
    "nan_hard": (torch.tensor([0.0, float("nan"), 0.0]), torch.ones(3), None),
    "inf_kd": (torch.zeros(3), torch.tensor([1.0, float("inf"), 1.0]), None),
    "nan_risk": (torch.zeros(3), torch.ones(3), torch.tensor([0.0, float("nan"), 0.0])),
    "risk_shape": (torch.zeros(3), torch.ones(3), torch.zeros(2)),
}
_ADVERSARIAL_TARGET = {
    "good": torch.tensor([[1.0, 2.0, 0.5], [0.0, -1.0, 3.0]]),
    "nan": torch.tensor([[float("nan"), 2.0, 0.5], [0.0, -1.0, 3.0]]),
}


@pytest.mark.parametrize("case", sorted(_RECONSTRUCT))
def test_reconstruct_probabilities_is_unchanged(old: dict[str, Any], case: str) -> None:
    index, prob, residual, classes = _RECONSTRUCT[case]
    expected = _outcome(lambda: old["bank"].reconstruct_probabilities(index, prob, residual, num_classes=classes))
    observed = _outcome(lambda: new_bank.reconstruct_probabilities(index, prob, residual, num_classes=classes))
    assert observed == expected
    assert (expected[0] == "ok") is case.startswith("good")


@pytest.mark.parametrize("case", sorted(_DISTRIBUTIONS))
def test_validate_probability_distribution_is_unchanged(old: dict[str, Any], case: str) -> None:
    value = _DISTRIBUTIONS[case]
    expected = _outcome(lambda: old["validation"].validate_probability_distribution(value))
    assert _outcome(lambda: new_validation.validate_probability_distribution(value)) == expected
    assert (expected[0] == "ok") is (case == "good")


@pytest.mark.parametrize("case", sorted(_ENTROPY))
def test_shannon_entropy_is_unchanged(old: dict[str, Any], case: str) -> None:
    value = _ENTROPY[case]
    expected = _outcome(lambda: old["entropy"].shannon_entropy(value))
    assert _outcome(lambda: new_entropy.shannon_entropy(value)) == expected
    assert (expected[0] == "ok") is (case == "good")


@pytest.mark.parametrize("case", sorted(_POLICY))
def test_policy_weights_are_unchanged(old: dict[str, Any], case: str) -> None:
    hard, kd, risk = _POLICY[case]
    expected = _outcome(lambda: old["policies"].PolicyWeights(hard_weight=hard, kd_weight=kd, joint_risk=risk))
    assert _outcome(lambda: new_policies.PolicyWeights(hard_weight=hard, kd_weight=kd, joint_risk=risk)) == expected
    assert (expected[0] == "ok") is (case == "good")


@pytest.mark.parametrize("bank_teacher", [False, True])
@pytest.mark.parametrize("case", sorted(_ADVERSARIAL_TARGET))
def test_adversarial_target_is_unchanged(old: dict[str, Any], case: str, bank_teacher: bool) -> None:
    """With a top-2 bank teacher the target goes through the bank's truncation (reconstruction checks too)."""
    logits = _ADVERSARIAL_TARGET[case]
    clean = torch.tensor([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]])
    labels, mask = torch.tensor([1, 2]), torch.ones(2)

    def target(hooks_module: Any, bank_module: Any) -> Any:
        hooks = hooks_module.DistillationTargetHooks(adversarial_teacher_target=True, temperature=1.0)
        hooks.reset_epoch(torch.device("cpu"))
        teacher = None
        if bank_teacher:
            bank = bank_module.SoftLabelBank(Path("x"), {"top_k": 2, "num_classes": 3, "epoch_records": {}}, None)
            teacher = bank_module.SoftLabelBankTeacher(bank)
        return hooks.adversarial_target(
            teacher=teacher, teacher_adversarial_logits=logits, teacher_clean_logits=clean, labels=labels, mask=mask
        )

    expected = _outcome(lambda: target(old["hooks"], old["bank"]))
    assert _outcome(lambda: target(new_hooks, new_bank)) == expected
    assert (expected[0] == "ok") is (case == "good")
