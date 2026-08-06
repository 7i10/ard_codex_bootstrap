# Isolated full-SAAD baseline readiness

## Status

- Owner: main thread; one Terra implementation owner; one consolidated
  scientific review after live smoke evidence is stable
- Host: Hamster only; Ferret is explicitly out of scope
- Base SHA: `246976064ad1213a16c054b0921f68ae3d914aeb`
- Current milestone: complete through mandatory gate failure
- Last updated: 2026-08-06

## Goal

Prepare and, only after bounded smoke gates pass, launch the pinned upstream
full-SAAD Bartoldson seed-0 oracle on Hamster.  Preserve the upstream checkout
as a clean, license-isolated execution oracle.  Record enough runtime, input,
import and command identity to distinguish this result from the controlled
local protocol and from a paper-number reproduction claim.

## Non-goals

- Do not edit, vendor or clean-room port `.external/saad`.
- Do not hide the RobustBench/AutoAttack namespace collision by silently
  selecting a different final AutoAttack implementation.
- Do not add full SAAD to `src/ard/` or duplicate the local trainer.
- Do not change upstream batch size, optimizer, threat model, schedule, SWA or
  evaluation when launching the heavy oracle.
- Do not start Chen, PGD-AT, TRADES, multi-seed, official local evaluation or
  any Ferret job in this milestone.
- Do not treat upstream test-each-epoch, final-SWA-only output as equivalent to
  local best/last or untouched-test evaluation.

## Frozen identity

- SAAD: `HongsinLee/saad` at
  `295121c5d2eed827b5b2d6aa42307de809bdfada`; license file absent.
- RobustBench: `RobustBench/robustbench` at
  `78fcc9e48a07a861268f295a777b975f25155964`.
- Teacher: `Bartoldson2024Adversarial_WRN-94-16`, checkpoint SHA-256
  `56bbad8ad748df86e67c24dba4f59a9e7d285e583251460b2ed154017a18cb0b`.
- Dataset: local torchvision CIFAR-10 bytes; downloads are forbidden.
- Heavy command: one GPU, batch 128, 200 epochs, seed 0, ResNet-18, SWA 95,
  `beta=0`, `gamma=1`, `igdm_alpha=1`, `lambda_inner=1`, W&B disabled.
- Effective upstream constants recorded in the manifest include SGD LR `0.1`,
  momentum `0.9`, weight decay `2e-4`, milestones `100/150`, inner PGD-10
  `8/255` with step `2/255`, per-epoch PGD-20, and entropy multiplier `5`.

The compatibility bootstrap must preload the pinned RobustBench package from
outside the SAAD checkout, prove the official `autoattack.state` origin, then
execute `saad.py` with SAAD's own `autoattack.AutoAttack` as its final evaluator.
Both module origins and source hashes are recorded.  Teacher logits on a fixed
input must match the existing strict adapter before a heavy launch.

## Milestones

- [x] M0 -- implement strict operational config, runtime lock, artifact
  staging, import provenance, immutable manifest/logs and expected-smoke
  termination.  Do not touch the scientific core.
- [x] M1 -- provision the separate pinned runtime, pass dependency/import and
  teacher-logit preflight, then run one batch-16 forward smoke and one exact
  batch-128 multi-batch smoke on Hamster.
- [x] M2 -- measured exact batch-128 VRAM; the mandatory gate failed and the
  Bartoldson full-SAAD seed-0 heavy run was correctly not launched.
- [x] M3 -- not applicable after the mandatory M2 failure; there is no heavy
  run whose first epoch or PGD-20 can be verified.

## Bounded smoke contract

The smoke uses the exact upstream `saad.py`, real CIFAR bytes, real teacher and
real student.  A supervisor observes finite upstream `loss:` progress for the
configured 2--10 training batches and terminates the whole process group before
epoch evaluation.  This is recorded as `expected_smoke_termination`, never as
completed training.  Batch 16 checks dependency/forward behavior; batch 128 is
the heavy-command VRAM/timing gate.  No download, checkpoint publication,
test-set metric or W&B run is permitted during smoke.

## Heavy-run launch gate

All conditions are mandatory:

- both external checkouts match remote/SHA/clean state;
- runtime Python/package/import origins match the lock;
- CIFAR and teacher bytes match their declared hashes and staging uses only
  non-escaping symlinks in a fresh output directory;
- fixed-input teacher logits match the local strict adapter in FP32 at
  `rtol=0`, `atol=1e-4`, with identical argmax;
- batch 16 and batch 128 smoke each observe multiple finite losses;
- batch 128 does not OOM and measured peak VRAM leaves a safe margin;
- command/source/runtime/input hashes are unchanged between smoke and launch;
- both Hamster GPUs are otherwise idle, disk/temperature are acceptable and a
  `1.5x` measured-time window is available;
- the manifest explicitly records that upstream has no resume, best/last or
  separate evaluation and that final AA failure is not training success.

OOM does not authorize automatic batch reduction or multi-GPU conversion.

## Test plan

- Focused T1: strict schema/unknown-key rejection; external, runtime, dataset
  and teacher drift; symlink escape; output overwrite; exact command; import
  provenance; expected smoke termination versus failure/completion.
- Existing T2: SAAD KL/entropy/gradient regression and pinned teacher adapter
  contracts; add only a focused bridge regression if live evidence exposes a
  real compatibility bug.
- Live GPU: batch-16 once, then batch-128 multi-batch once on Hamster.
- Run `scripts/verify.py --changed --non-scientific` once and one consolidated
  scientific review after live evidence.  Re-review only an actual P0/P1 fix.

## Risks

- The SAAD checkout has no license file; no source, patch or checkpoint enters
  the public repository.
- The compatibility bridge may change import identity.  Any change to teacher
  logits or final AutoAttack origin blocks the heavy run.
- Upstream reads the test set every epoch and has no resume.  An interrupted
  run restarts from zero and cannot support an untouched-test claim.
- The full run may finish training but fail final CW/AA.  Preserve this as a
  partial failure, not an exit-zero result.
- Upstream `download=True` must not cause network access; pre-existing verified
  CIFAR bytes are required.

## Completion condition

This plan completes when either (a) the exact heavy oracle is safely launched
and first-epoch training plus PGD-20 are finite, or (b) a frozen gate fails and
the blocker is documented without changing the protocol.  Only an exit-zero
200-epoch result later authorizes the separate Chen/controlled-baseline plan.

## Progress log

- 2026-08-06: Plan 0023 closed the tested sample-wise proposal branch as
  predictable but not actionable.  Both Hamster RTX 4090 GPUs were idle at
  this plan's preflight.  Ferret use was explicitly declined by the user.
- 2026-08-06: read-only upstream audit confirmed the exact source/runtime
  behavior, missing SAAD license file, local AutoAttack namespace collision,
  full-test evaluation after even one epoch, final-SWA-only checkpoint and
  absence of resume/best/last.  These are launch constraints, not bugs to hide.
- 2026-08-06: the upstream README's Python 3.8 claim was rejected for this
  pinned dependency combination.  A clean Python 3.8.20 / PyTorch 2.4.1
  environment failed while importing RobustBench `78fcc9e` because that source
  uses `list[...]` runtime annotations.  No source patch was made.  The runtime
  was therefore fixed to Python 3.11.15, PyTorch 2.4.1+cu121, torchvision
  0.19.1+cu121 and setuptools 75.3.0; this combination imports the exact pinned
  RobustBench and official AutoAttack `state` module successfully on Hamster.
- 2026-08-06: cross-runtime fixed-input Bartoldson comparison between the
  upstream runtime and the established ARD runtime gave maximum absolute logit
  difference `8.0824e-05`, mean `3.0991e-05`, identical argmax on all four
  deterministic inputs and maximum absolute logit `2.8449`.  Bit equality is
  not a valid gate across PyTorch 2.4.1 and 2.11.  The frozen bridge gate is
  therefore `rtol=0`, `atol=1e-4` plus identical argmax; both runtime versions
  and the measured difference must be recorded.
- 2026-08-06: M0 added a strict one-GPU launcher, import-order bootstrap,
  runtime lock, full CIFAR/teacher integrity checks, immutable source/command
  identity, physical-GPU telemetry and bounded expected-termination smoke.
  Focused verification passed: `pytest -q tests/unit/test_run_saad_upstream.py
  tests/unit/test_verify_gate.py` (`50 passed`) and
  `scripts/verify.py --changed --non-scientific` (`17 + 33 passed`).  A
  readiness review found that upstream progress is printed to stdout rather
  than stderr; the supervisor was corrected to observe both unbuffered streams
  before any live run, with chunked/CR/non-finite parsing regressions.
- 2026-08-06: the first batch-16 live attempt reached the exact pinned runtime,
  real CIFAR bytes and real Bartoldson teacher and emitted 15 finite losses,
  but was conservatively classified `smoke_failure`: Python's buffered
  `read(4096)` delayed delivery until 15 progress records had accumulated,
  exceeding the preregistered 10-record ceiling.  Peak physical-GPU memory was
  5,226 MiB, utilization 77% and temperature 40 C; import provenance and all
  input/source hashes matched.  This was a supervisor I/O defect, not a model
  failure.  The reader now uses prompt pipe bytes (`os.read`) and its focused
  regression passes (`18 passed`); a fresh batch-16 smoke is required.
- 2026-08-06: fresh batch-16 at Git `13ca475` passed with exactly two finite
  losses and `expected_smoke_termination` in 8.81 seconds.  All import,
  source, command and input identities matched a clean Git tree.  The 5-second
  telemetry cadence missed the short GPU-active window (622 MiB/0% reported),
  while the longer failed attempt had measured 5,226 MiB/77%.  The cadence was
  therefore tightened to 0.5 seconds with a schema/peak regression; batch-16
  is rerun once for trustworthy telemetry before batch-128.
- 2026-08-06: fixed-telemetry batch-16 at clean Git `ff2c931` passed with two
  finite losses and `expected_smoke_termination` in 8.55 seconds.  Peak GPU-0
  memory was 5,226 MiB, utilization 71%, temperature 41 C, with 17 telemetry
  samples and no telemetry error.  The subsequent exact batch-128 smoke failed
  before its first loss in the teacher input-gradient forward with
  `torch.OutOfMemoryError`; peak physical memory was 24,080/24,564 MiB,
  utilization 100% and temperature 43 C.  Runtime/import/source/input lineage
  remained clean and fixed at `ff2c931`.  By the frozen launch gate, this blocks
  M2 and the 200-epoch job: no automatic batch reduction, two-GPU conversion or
  heavy launch was performed.  The failed process did not reach per-epoch test
  evaluation, save a checkpoint or initialize W&B.
- 2026-08-06: the consolidated scientific review found no defect in the smoke
  source/input/import lineage and confirmed the batch-128 OOM as a P0 launch
  blocker.  It also found two P1 safeguards required before any future heavy
  authorization: machine-enforced hash-bound smoke/logit/VRAM gates, and a
  pre-process immutable launch manifest with crash-resilient terminalization.
  Those are not reasons to weaken this failed gate.  A single preregistered
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` retry was judged an
  admissible next diagnosis because it changes allocator behavior, not batch,
  BN, loss, attack, schedule or data; it is handled separately in Plan 0025.
