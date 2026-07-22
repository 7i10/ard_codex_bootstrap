"""Two-rank terminal-resume preflight ownership and immutability oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist

from ard.cli import train as train_cli
from ard.engine.distributed import initialize_from_env, run_rank_zero_value, teardown


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=("success", "failure"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, initialized = initialize_from_env("cpu")
    assert initialized and dist.get_world_size() == 2
    rank = dist.get_rank()
    try:
        if rank == 0:
            bundle = args.output / "run-bundle"
            bundle.mkdir(parents=True)
            (bundle / "manifest.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
            torch.save({"epoch": 1 if args.case == "success" else 0}, args.output / "last.pt")
            before = _tree_bytes(args.output)
        else:
            before = None
            original_load = train_cli.torch.load

            def forbid_nonzero_load(*values: object, **kwargs: object) -> object:
                del values, kwargs
                raise AssertionError("terminal resume manifest/checkpoint must be read only by rank zero")

            train_cli.torch.load = forbid_nonzero_load  # type: ignore[assignment]
        dist.barrier()
        message = ""
        try:
            terminal = run_rank_zero_value(
                lambda: train_cli._terminal_resume_requested(
                    output_dir=args.output, resume=args.output / "last.pt", epochs=2
                ),
                phase="terminal resume preflight",
            )
            assert args.case == "success" and terminal is True
            message = "success"
        except RuntimeError as exc:
            assert args.case == "failure"
            assert "rank-zero terminal resume preflight failed (TrackingError)" in str(exc)
            message = str(exc)
        finally:
            if rank != 0:
                train_cli.torch.load = original_load  # type: ignore[assignment]
        outcomes: list[str | None] = [None, None]
        dist.all_gather_object(outcomes, message)
        assert outcomes[0] == outcomes[1] == message
        if rank == 0:
            assert _tree_bytes(args.output) == before
    finally:
        teardown()


if __name__ == "__main__":
    main()
