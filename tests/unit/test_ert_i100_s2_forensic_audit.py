from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ard.analysis.ert_i100_s2_longitudinal import (
    LongitudinalStateError,
    canonical_action_states,
    replay_canonical_state,
)
from ard.analysis.ert_i100_s2_secant_forensic import (
    central_difference,
    dynamic_pair_margin,
    scalar_secant_loss,
    secant_components,
)


def _row(sample_id: int, *, clean: bool, adv: bool, margin: float, teacher_adv: bool = True) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "student_clean_correct": clean,
        "student_ce20_adv_correct": adv,
        "student_ce20_adv_margin": margin,
        "teacher_ce20_adv_correct": teacher_adv,
        "teacher_ce20_adv_margin": margin + 0.1,
    }


def test_action_state_partition_gives_clean_wrong_precedence() -> None:
    rows = [
        _row(1, clean=False, adv=False, margin=-0.4),
        _row(2, clean=True, adv=False, margin=-0.2),
        _row(3, clean=True, adv=True, margin=0.001),
        _row(4, clean=True, adv=True, margin=0.8),
    ]
    states = canonical_action_states(rows)["state_by_id"]
    assert states[1]["branch"] == "Clean-Wrong"
    assert states[2]["branch"] == "S3-non-CW"
    assert states[3]["branch"] == "S2"
    assert states[4]["branch"] == "S1"


def test_secant_scalar_autograd_matches_central_difference_away_from_kinks() -> None:
    adv = torch.tensor([-0.40], dtype=torch.float64, requires_grad=True)
    clean = torch.tensor([-0.10], dtype=torch.float64, requires_grad=True)
    rho = torch.tensor([0.03125], dtype=torch.float64)
    teacher_distance = torch.tensor([0.6], dtype=torch.float64)
    loss = scalar_secant_loss(adv, clean, rho=rho, d_teacher=teacher_distance, epsilon=1e-12).sum()
    grad_adv, grad_clean = torch.autograd.grad(loss, (adv, clean))
    fd_adv = central_difference(
        lambda value: scalar_secant_loss(
            value, clean.detach(), rho=rho, d_teacher=teacher_distance, epsilon=1e-12
        ).sum(),
        adv.detach(),
        step=1e-6,
    )
    fd_clean = central_difference(
        lambda value: scalar_secant_loss(adv.detach(), value, rho=rho, d_teacher=teacher_distance, epsilon=1e-12).sum(),
        clean.detach(),
        step=1e-6,
    )
    assert torch.allclose(grad_adv, fd_adv, atol=1e-6, rtol=1e-5)
    assert torch.allclose(grad_clean, fd_clean, atol=1e-6, rtol=1e-5)


def test_secant_student_q_retains_graph_and_teacher_pair_gate_is_not_teacher_argmax() -> None:
    # Student rival is class 1.  Teacher is globally wrong to class 2 but is
    # still pair-positive relative to that Student-selected rival.
    student_logits = torch.tensor([[0.2, 0.8, 0.7]], requires_grad=True)
    teacher_logits = torch.tensor([[0.7, 0.4, 0.9]])
    labels = torch.tensor([0])
    student_adv, rival = dynamic_pair_margin(student_logits, labels)
    teacher_adv, _ = dynamic_pair_margin(teacher_logits, labels, rival)
    assert rival.tolist() == [1]
    assert teacher_logits.argmax(dim=1).tolist() == [2]
    assert teacher_adv.item() > 0.0
    values = secant_components(
        student_adv_margin=student_adv,
        student_clean_margin=torch.tensor([0.1], requires_grad=True),
        teacher_adv_margin=teacher_adv,
        teacher_clean_margin=torch.tensor([0.5]),
        rho=torch.tensor([0.03]),
        selected=torch.ones(1),
        epsilon=1e-12,
    )
    assert values["teacher_pair_gate"].item() == 1.0
    gradient = torch.autograd.grad(values["raw_loss"].sum(), student_logits, allow_unused=False)[0]
    assert torch.isfinite(gradient).all()


def test_replay_allows_orchestrator_metadata_but_not_scientific_overwrite(tmp_path: Path) -> None:
    # Exercise the early output guard without loading a checkpoint: an
    # orchestrator-side log directory is execution metadata, while a prior
    # scientific result must fail closed.
    output = tmp_path / "output"
    (output / "orchestration").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        # Passing invalid typed values reaches the checkpoint loader only after
        # the output guard has accepted the metadata-only directory.
        replay_canonical_state(  # type: ignore[arg-type]
            config_path=Path("missing.yaml"),
            checkpoint=Path("missing.pt"),
            expected_checkpoint_sha256="0" * 64,
            expected_epoch=104,
            output_dir=output,
            device=torch.device("cpu"),
        )
    (output / "state-replay.json").write_text("{}", encoding="utf-8")
    with pytest.raises(LongitudinalStateError):
        replay_canonical_state(  # type: ignore[arg-type]
            config_path=Path("missing.yaml"),
            checkpoint=Path("missing.pt"),
            expected_checkpoint_sha256="0" * 64,
            expected_epoch=104,
            output_dir=output,
            device=torch.device("cpu"),
        )
