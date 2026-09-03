#!/usr/bin/env python3
"""Replay current CE-PGD20 Student/Teacher states for saved dynamic-BDD checkpoints."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import torch

from ard.analysis.ert_i100_s2_dynamic_bdd_state import replay_train_states


def _checkpoint(value: str) -> tuple[int, Path]:
    epoch_text, separator, path_text = value.partition("=")
    if not separator or not epoch_text.isdigit() or not path_text:
        raise argparse.ArgumentTypeError("checkpoint must be EPOCH=/absolute/path")
    path = Path(path_text)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("checkpoint path must be absolute")
    return int(epoch_text), path


def _checkpoint_sha256(value: str) -> tuple[int, str]:
    epoch_text, separator, digest = value.partition("=")
    if not separator or not epoch_text.isdigit() or len(digest) != 64:
        raise argparse.ArgumentTypeError("checkpoint SHA-256 must be EPOCH=<64 lowercase hex characters>")
    if any(character not in "0123456789abcdef" for character in digest):
        raise argparse.ArgumentTypeError("checkpoint SHA-256 must be EPOCH=<64 lowercase hex characters>")
    return int(epoch_text), digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=_checkpoint, action="append", required=True)
    parser.add_argument("--checkpoint-sha256", type=_checkpoint_sha256, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--expected-teacher-sha256", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if actual != args.expected_source_sha:
        raise SystemExit(f"source SHA mismatch: expected {args.expected_source_sha}, got {actual}")
    if not args.config.is_file():
        raise SystemExit(f"config is missing: {args.config}")
    if not args.output.is_absolute():
        raise SystemExit("output must be an absolute path")
    checkpoint_sha256 = dict(args.checkpoint_sha256)
    if len(checkpoint_sha256) != len(args.checkpoint_sha256):
        raise SystemExit("duplicate checkpoint SHA-256 epoch")
    expected_epochs = {epoch for epoch, _ in args.checkpoint}
    if set(checkpoint_sha256) != expected_epochs:
        raise SystemExit("checkpoint SHA-256 epochs must exactly match --checkpoint epochs")
    seen: set[int] = set()
    for epoch, checkpoint in args.checkpoint:
        if epoch in seen:
            raise SystemExit(f"duplicate checkpoint epoch: {epoch}")
        seen.add(epoch)
        if not checkpoint.is_file():
            raise SystemExit(f"checkpoint is missing: {checkpoint}")
        replay_train_states(
            config_path=args.config,
            checkpoint=checkpoint,
            output_dir=args.output / f"e{epoch}",
            device=torch.device(args.device),
            expected_epoch=epoch,
            expected_checkpoint_sha256=checkpoint_sha256[epoch],
            expected_teacher_sha256=args.expected_teacher_sha256,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
