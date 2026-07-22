"""Test-only torchrun entry point that delays rank 1 before invoking the real CLI."""

from __future__ import annotations

import os
import time

from ard.cli.train import main

if os.environ.get("LOCAL_RANK") == "1":
    time.sleep(0.25)

raise SystemExit(main())
