"""Inject rank-aware sample-statistics assertions into the real train CLI."""

from __future__ import annotations

import os
import sys

import ard.cli.train as train_cli

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
original_write = train_cli.write_sample_parquet


def guarded_write(rows, path):
    if local_rank != 0:
        raise AssertionError("non-zero rank invoked sample Parquet writer")
    if os.environ.get("ARD_INJECT_SAMPLE_STATS_FAILURE") == "1":
        raise OSError("injected sample statistics failure")
    return original_write(rows, path)


train_cli.write_sample_parquet = guarded_write
try:
    raise SystemExit(train_cli.main())
except RuntimeError as exc:
    print(f"ARD_SAMPLE_STATS_FAILURE_RANK={local_rank}: {exc}", file=sys.stderr, flush=True)
    raise
