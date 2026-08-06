# Full-SAAD allocator-only compatibility retry

## Status

- Owner: main thread; one Terra implementation owner
- Host: Hamster GPU 0 only; Ferret is forbidden
- Base SHA: `ff2c931c32fc90efeff29d4240fcde08804cb5a4`
- Current milestone: complete; preregistered memory-headroom No-Go
- Last updated: 2026-08-06

## Question

Can the exact upstream Bartoldson/full-SAAD batch-128 step fit one RTX 4090
when PyTorch fragmentation is reduced with the single preregistered allocator
setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`?

This is an operational compatibility diagnosis.  It does not change batch
size, BatchNorm domain, optimizer, objective, attack, schedule, data, seed,
teacher or student.  It cannot rescue the run by changing any of those values.

## Frozen execution

- Reuse the exact Plan-0024 upstream, RobustBench, teacher, dataset and Python
  runtime identities.
- Add exactly one environment difference:
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Run a fresh batch-16 smoke for two finite losses, then a fresh batch-128
  smoke for ten finite losses.  Both use physical Hamster GPU 0.
- Record the allocator string, source/command/runtime/input hashes, clean Git
  SHA, import origins, loss quota, elapsed time and 0.5-second GPU telemetry.
- No Ferret, multi-GPU, microbatch, source patch, AMP, gradient checkpointing,
  batch reduction or heavy run is permitted in M0/M1.

## Decision gate

Allocator retry is **Go** only when both smokes have:

- `expected_smoke_termination`, only finite losses and exact requested quota;
- identical frozen scientific/input/import identity apart from batch size;
- no telemetry errors and batch-128 peak memory at most `22,500 MiB`
  (at least about 2 GiB physical headroom on the 24,564-MiB device);
- batch-128 completes ten steps without upward unbounded memory trend;
- no per-epoch test evaluation, checkpoint or W&B initialization.

Any OOM, peak above 22,500 MiB, non-finite loss, import drift or source drift is
terminal No-Go for exact full SAAD on the current Hamster hardware.  Do not try
a second allocator setting.

## Milestones

- [x] M0 -- add strict allocator identity, explicit 2/10-event smoke selection
  and fail-closed full mode; focused tests and immutable commit.
- [x] M1 -- ran batch-16/2 and batch-128/10 bounded smokes on Hamster GPU 0;
  both were finite, but batch-128 failed the frozen memory-headroom gate.
- [x] M2 -- correctly skipped after M1 No-Go.  The scientific review's P1
  heavy-launch safeguards (hash-bound smoke/logit/VRAM evidence, pre-process
  immutable launch manifest and crash-resilient terminal evidence) remain
  mandatory if different hardware later authorizes a heavy launch.
- [x] M3 -- closed without a heavy run after the preregistered M1 No-Go; no
  first epoch, per-epoch test evaluation, PGD-20, checkpoint or W&B run exists.

## Tests

- Config rejects any allocator other than the exact frozen string.
- Child environment and manifest contain the allocator identity.
- Smoke quota is explicitly restricted to 2--10; full mode rejects it.
- Until M2 is implemented and valid evidence is supplied, full execution fails
  before output staging/process creation.
- Reuse all Plan-0024 focused integrity/import/stream/telemetry tests.

## Completion

The plan completes either with a safely launched exact heavy run after every
gate, or with a documented allocator No-Go and no heavy process.  A lower-batch
upstream feasibility run or clean-room controlled approximation requires a new
scientific plan and cannot be relabeled as this oracle.

## Progress log

- 2026-08-06: M0 fixes the allocator to
  `expandable_segments:True`, records it in the child environment/manifest,
  adds an execute-only 2--10 loss quota, and makes full execution fail closed
  before checkout/staging/process creation.  Focused verification passed:
  `pytest -q tests/unit/test_run_saad_upstream.py` (`26 passed`), Ruff check and
  format check, `git diff --check`, and impact-selected T0/T1 (`26 passed`).
- 2026-08-06: at clean Git `02d1921`, allocator batch-16/2 passed in 8.71
  seconds with peak 4,966 MiB, utilization 86% and temperature 39 C.  Exact
  batch-128/10 then passed ten finite full-SAAD updates in 22.70 seconds with
  no telemetry error and flat final memory, but peak memory was 23,730 of
  24,564 MiB (only about 834 MiB physical headroom).  This exceeds the frozen
  22,500-MiB ceiling, so M1 is terminal No-Go even though fragmentation no
  longer caused an immediate OOM.  The threshold was not relaxed after seeing
  the result, no second allocator was tried and no heavy process was started.
