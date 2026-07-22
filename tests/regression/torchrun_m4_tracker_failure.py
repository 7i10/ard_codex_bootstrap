"""Two-rank failure oracle: every tracker phase must fail collectively."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch.distributed as dist

from ard.config.schema import ExperimentConfig
from ard.engine.distributed import initialize_from_env, teardown
from ard.tracking import NullTracker, coordinated_create_tracker, coordinated_tracker_action


def config(output: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "schema_version": 2,
            "protocol": {"id": "synthetic_smoke_v2"},
            "tier": "smoke",
            "seeds": {
                k: 0
                for k in (
                    "split",
                    "model_init",
                    "data_order",
                    "augmentation",
                    "train_attack",
                    "evaluation_attack",
                    "qualitative_panel",
                )
            },
            "dataset": {"name": "synthetic_cifar", "num_samples": 4, "num_classes": 2},
            "student": {"architecture": "fixture_cnn", "num_classes": 2},
            "method": {"id": "pgd_at", "version": 1, "attack": {"steps": 1}},
            "optimizer": {"id": "sgd", "learning_rate": 0.01, "momentum": 0.9, "weight_decay": 0.0, "nesterov": False},
            "scheduler": {"id": "identity", "milestones": [], "gamma": 1.0, "step_at": "epoch_end"},
            "training": {"epochs": 1, "per_rank_batch_size": 2, "global_batch_size": 4},
            "output_dir": str(output),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=("init", "metric", "artifact"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, initialized = initialize_from_env("cpu")
    assert initialized and dist.get_world_size() == 2
    entered_next_epoch = False
    try:
        message = ""
        if args.case == "init":
            import ard.tracking.adapter as adapter

            original = adapter.LocalTracker

            class FailingTracker(original):
                def __init__(self, *values: object, **kwargs: object) -> None:
                    raise RuntimeError("injected init failure")

            adapter.LocalTracker = FailingTracker
            try:
                coordinated_create_tracker(
                    config=config(args.output), output_dir=args.output, config_hash="x", root=Path.cwd()
                )
            except RuntimeError as exc:
                message = str(exc)
            finally:
                adapter.LocalTracker = original
        else:
            tracker = NullTracker("two-rank")
            phase = "tracker metric" if args.case == "metric" else "tracker artifact"
            try:
                coordinated_tracker_action(
                    tracker,
                    phase=phase,
                    action=lambda _: (_ for _ in ()).throw(RuntimeError(f"injected {args.case} failure")),
                )
            except RuntimeError as exc:
                message = str(exc)
        assert f"tracker {'init' if args.case == 'init' else args.case}" in message
        assert "rank-zero" in message and "RuntimeError" in message
        evidence: list[str | None] = [None, None]
        dist.all_gather_object(evidence, message)
        assert evidence[0] == evidence[1] == message
        # Recovery gather above is intentional evidence collection.  No rank
        # may advance to a subsequent epoch collective after the failed phase.
        assert not entered_next_epoch
    finally:
        teardown()


if __name__ == "__main__":
    main()
