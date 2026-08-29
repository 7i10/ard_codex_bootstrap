# 0062 — ERT/RSLAD runtime performance audit v2

## Status

- Owner: root
- Branch / base SHA: `master` / `5c87e705925ef1bb2e6600ee01e33034d0cd63da`
- Current milestone: M4 complete for the safe Hamster scope; Ferret controlled benchmark deferred
- Last updated: 2026-08-29

## Goal

Measure the current RSLAD runtime bottlenecks, explain the Hamster/Ferret
throughput gap, and evaluate only semantics-preserving bounded runtime
optimizations. Preserve the active Ferret GPU2 production run and stop before
any host-wide benchmark that could contaminate it.

## Non-goals

- No change to attack, objective, optimizer, scheduler, augmentation, batch,
  precision, or scientific checkpoint semantics.
- No interference with the active Ferret GPU2 production run.
- No full training, official test, AutoAttack, or new scientific method.
- No automatic adoption of `torch.compile` or any runtime candidate without
  parity and human review.

## Existing state

- Repository HEAD is `5c87e70`; active timing production uses pinned source
  `8083f9c5df9b46a3a02399fbf293ceee6db85083` in a separate runtime checkout.
- Ferret GPU2 currently runs the I75 seed-1 continuation. Ferret GPU0/GPU1
  are idle, but host-wide stress benchmarks remain prohibited until it exits.
- Hamster has no active training/evaluation process in the initial snapshot and
  may run a bounded benchmark on an isolated GPU.
- Historical production rows show approximately 620–630 images/s on Hamster
  and 360–370 images/s on Ferret under comparable single-GPU RSLAD workloads.

## Scientific contracts affected

This is a runtime-only audit. Threat model, pixel domain, PGD identity,
RSLAD loss, teacher/student semantics, augmentation RNG, sampler order,
checkpoint/resume state, and W&B scientific identity must remain unchanged.
Any adopted change requires explicit host-local parity evidence.

## Decisions

- Use one initial process/GPU snapshot only; do not actively poll the Ferret
  production run.
- Record Ferret controlled benchmark as deferred while GPU2 is active.
- Run only bounded Hamster benchmarks while Hamster is confirmed idle.
- Treat per-batch CUDA scalar extraction (`float(...cpu())` in training totals)
  and PGD `max_abs_delta` extraction as synchronization candidates to measure,
  not remove speculatively.
- Do not add production runtime options until a focused benchmark and parity
  harness demonstrate a reproducible benefit.

## Milestones

- [x] M0 — Initial reconciliation and protected-production snapshot.
- [x] M1 — Read-only hardware, environment, historical throughput, and source
  audit; record deferred Ferret benchmark.
- [x] M2 — Bounded Hamster baseline/compile/low-risk benchmark harness and
  machine reports.
- [x] M3 — Host-local parity checks and candidate acceptance/rejection.
- [x] M4 — Final runtime/scheduling report and optional review-ready runtime
  change (no automatic adoption).

### M1 files and acceptance

- Files: this plan; `docs/ERT_RSLAD_RUNTIME_PERFORMANCE_AUDIT.md`; host-audit
  artifact.
- Tests: static source audit and JSON/schema validation.
- Acceptance: active production is untouched; host topology, software stack,
  filesystem, and historical throughput are recorded with exact source IDs.
- Commit boundary: one audit/report commit after M1 evidence is stable.

### M2 files and acceptance

- Files: `scripts/analysis/ert_rslad_runtime_benchmark.py` and benchmark JSON
  artifacts.
- Tests: script help/import; bounded Hamster run; no W&B/model artifact upload.
- Acceptance: eager baseline and supported compile candidates report cold and
  steady-state timing; segment totals are finite and reproducible enough for
  comparison.

### M3 files and acceptance

- Files: parity artifact and focused regression tests only if a code defect is
  confirmed.
- Acceptance: candidate is retained only with strict functional/short-run
  parity; no candidate is promoted solely on speed.

## Agent and review budget

No subagent is needed: this is one bounded audit with one owning writer. Use
one consolidated scientific/runtime review only after evidence and reports are
stable; do not spawn per-candidate reviewers.

## Test plan

- Cached: existing changed-test gate for unaffected source.
- New: benchmark script `--help`, Python import/compile, JSON parse, and
  read-only static checks.
- GPU-bound: bounded Hamster benchmark only while idle.
- Deferred: Ferret controlled benchmark until production GPU2 exits.

## Risks and mitigations

- Production contamination: protect Ferret GPU2 and avoid host-wide Ferret
  load; use one initial snapshot only.
- Scientific drift: benchmark outside production output and never reuse its
  checkpoint or metrics.
- Compile divergence: require functional and short-trajectory parity; default
  is reject/hold for human review.
- Disk pressure: Hamster root has only about 23 GiB available; keep benchmark
  output small and never write model/run-bundle artifacts.
- Host confounds: report CPU/NUMA/PCIe/GPU state and compare host-local
  candidates rather than asserting cross-host bitwise equality.

## Progress log

- 2026-08-29: Reconciled HEAD `5c87e70`; confirmed Ferret GPU2 active and
  protected, Ferret GPU0/GPU1 idle, Hamster processes idle at snapshot.
- 2026-08-29: Recorded host difference: Hamster one socket/52 CPUs and Ferret
  two sockets/104 CPUs; both RTX 4090 with driver 595.84, CUDA 12.8,
  PyTorch 2.11.0; Hamster root filesystem is nearly full.
- 2026-08-29: Hamster eager profile measured 681.5 img/s with four workers;
  workers=0 measured 539.8 img/s. Pinning/non-blocking and eight workers did
  not improve throughput. All four Student-only compile modes were slower
  (−21.2% to −22.8%) and failed strict one-step parity.
- 2026-08-29: Repeated the bounded screen with the real I50 seed-1
  `last.pt` (SHA `e517354323db0ba1097607939ce360efe20c522419dee15cc13b897a0b70db99`).
  The current-config eager result was 679.1 img/s; pinning remained below 1%
  and compile remained about 21–23% slower. The harness explicitly measures
  the core model/attack path and does not claim to include diagnostics-panel
  or per-sample history overhead.
- 2026-08-29: Added hash-bound host, profile, low-risk, compile, parity,
  scheduling, and final-decision artifacts plus the human-facing audit report.
  No production runtime change was adopted; Ferret remains deferred while GPU2
  is active.

## Completion report

M1–M4 are complete for the permitted scope. Ferret benchmark remains
explicitly deferred until the active GPU2 production run is complete; no
completion polling is performed by this agent. The next Ferret measurement, if
needed, should reuse the committed benchmark scripts and fixed contracts.
