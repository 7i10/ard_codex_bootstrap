"""Test-only two-rank CUDA entry point; it delegates to the production CLI."""

from ard.cli.train import main

raise SystemExit(main())
