# Legacy campaign tooling

This directory archives the historical Hamster/Ferret campaign, recovery, and
watchdog implementation. It is unsupported operational tooling, not a
scientific method and not a runtime dependency of the public `ard` package.

The archived files are intentionally not installed or imported by `ard`. Their
tests are excluded from the default pytest suite. Use the exact historical Git
SHA recorded by the campaign handoff when recovery of an old campaign is
required; do not use this directory to launch new paper experiments.

The archive preserves the original scripts, campaign specifications, tests, and
handoff documents for provenance and incident recovery only.
