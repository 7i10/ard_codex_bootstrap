from __future__ import annotations

from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.optim import SGD

from ard.attacks import AttackGenerator, AttackRequest, AttackResult
from ard.data import IndexedBatch
from ard.engine.distributed import initialize_from_env, teardown, wrap_ddp
from ard.engine.trainer import Trainer
from ard.objectives import RSLADObjective
from ard.policies import RSLADBaselinePolicy
from ard.tracking.diagnostics import TrainingDiagnostics


class IdentityAttack(AttackGenerator):
    def generate(self, request: AttackRequest) -> AttackResult:
        return AttackResult(
            adversarial=request.inputs.detach(),
            initial_delta=torch.zeros_like(request.inputs),
            step_losses=(),
            max_abs_delta=0.0,
        )


def main() -> None:
    device, initialized = initialize_from_env("cpu")
    assert initialized
    try:
        torch.manual_seed(17)
        student = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 4 * 4, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.Linear(512, 3, bias=False),
        ).to(device)
        wrapped = wrap_ddp(student, device)
        assert wrapped.broadcast_buffers
        teacher = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 3)).to(device)
        optimizer = SGD(wrapped.parameters(), lr=0.01)
        trainer = Trainer(
            model=wrapped,
            teacher=teacher,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            attack=IdentityAttack(),
            selection_attack=IdentityAttack(),
            objective=RSLADObjective(),
            policy=RSLADBaselinePolicy(),
            device=device,
            output_dir=Path("unused-ddp-bn-regression-output"),
            config_hash="ddp-bn-regression",
            seed=23,
            diagnostics=TrainingDiagnostics(panel_ids=(), mode="summary"),
        )
        rank = dist.get_rank()
        images = torch.full((4, 3, 4, 4), float(rank + 1) / 4, device=device)
        images[:, 0, 0, 0] += torch.arange(4, dtype=torch.float32, device=device) / 20
        batch = IndexedBatch(
            images=images,
            labels=torch.tensor([0, 1, 2, 0], device=device),
            sample_ids=torch.arange(rank * 4, rank * 4 + 4, device=device),
        )
        before = wrapped.module[1].weight.detach().clone()
        metrics = trainer.train_epoch([batch])  # type: ignore[arg-type]

        assert wrapped.broadcast_buffers
        assert torch.isfinite(torch.tensor(metrics["loss"]))
        assert not torch.equal(before, wrapped.module[1].weight.detach())
        batchnorm = wrapped.module[2]
        assert isinstance(batchnorm, nn.BatchNorm1d)
        gathered = [torch.empty_like(batchnorm.running_mean) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, batchnorm.running_mean)
        # Trainer's post-backward clean forward resumes normal DDP buffer
        # synchronization, so rank-zero running statistics are shared again.
        assert torch.equal(gathered[0], gathered[1])
    finally:
        teardown()


if __name__ == "__main__":
    main()
