#!/usr/bin/env python3
"""CLI-shape adapter between the production launch gate and ``ard.cli.train``.

Plan 0100 (docs/plans/0100-imagenet-stage01-r18-mobilenetv3-adr.md) is the
first campaign to launch a plain, non-forked ``ard.cli.train`` job through
``.agents/skills/production-launch-gate/scripts/launch_gate.py``.  The gate's
``_replace_epoch_bound`` unconditionally rewrites a literal ``--epochs``
token in every training job's command to the campaign's exclusive epoch
bound (``scientific_final_epoch + 1``) -- see ``references/campaign-spec.md``.
``ard.cli.train`` itself has no ``--epochs`` flag; epoch count is a config
field, settable only via the ``training.epochs=N`` dot-path override
(``overrides`` in ``ard.cli.train --help``). Every prior campaign that used
this gate wrapped its own bespoke launcher script (e.g.
``scripts/run_ert_i100_online_state_s2.py``) for the same reason.

This wrapper adds no scientific behavior: ``--epochs`` is forwarded verbatim
as ``training.epochs=<value>``, which every one of this plan's four frozen
configs already sets to the same value (50) -- so for the real campaign jobs
this override is confirmatory, not a change. It exists purely so the gate's
mechanical command-rewriting contract has a literal ``--epochs`` token to
rewrite, for a plain from-pretrained-init, non-forked, teacher-free training
job that otherwise needs no wrapper at all.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ard.cli.train import main as train_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adapt ard.cli.train to the launch gate's --epochs rewriting contract."
    )
    parser.add_argument("--config", required=True, type=Path, help="Forwarded to ard.cli.train --config.")
    parser.add_argument(
        "--epochs",
        required=True,
        type=int,
        help="Exclusive epoch upper bound (the gate rewrites this to scientific_final_epoch + 1). "
        "Forwarded as the training.epochs=<value> dot-path override.",
    )
    parser.add_argument("--output", type=Path, help="Forwarded to ard.cli.train --output.")
    parser.add_argument("--resume", type=Path, help="Forwarded to ard.cli.train --resume.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Forwarded to ard.cli.train --dry-run (used by the launch gate canary)."
    )
    parser.add_argument("overrides", nargs="*", help="Additional dot-path overrides, forwarded verbatim.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    forwarded: list[str] = ["--config", str(args.config)]
    if args.output is not None:
        forwarded += ["--output", str(args.output)]
    if args.resume is not None:
        forwarded += ["--resume", str(args.resume)]
    if args.dry_run:
        forwarded.append("--dry-run")
    forwarded.append(f"training.epochs={args.epochs}")
    forwarded.extend(args.overrides)
    return train_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
