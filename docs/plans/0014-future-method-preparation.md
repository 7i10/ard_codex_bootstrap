# Future-method preparation

## Status

- Owner: main thread
- Branch / base SHA: `master` / `bbbe42d`
- Current milestone: complete
- Last updated: 2026-07-31

## Goal

Prepare method-independent infrastructure while the Ferret analyses run:

- a loss-identical RSLAD logging-only path that stores primitive student and
  teacher signals without using them in the objective;
- canonical controlled-protocol PGD-AT and TRADES configurations;
- bounded CPU/GPU parity and launch preflights, without starting a long run.

## Non-goals

- Do not start 200-epoch logging-only, PGD-AT, TRADES, or full-SAAD training.
- Do not select a v2 gate, risk formula, threshold, or intervention before the
  common-trajectory and frozen-oracle results.
- Do not add a schema default that changes historical resolved-config hashes.

## Existing state

- `rslad` has no `SampleStateStore`; student-aware methods already store
  margin EMA, correctness frequency, and forgetting.
- Training diagnostics observe teacher entropy but periodic RSLAD checkpoints
  do not retain teacher primitives.
- `controlled_cifar10_r18_v1` currently hard-codes the RSLAD KL inner attack,
  so real-data PGD-AT/TRADES configs cannot satisfy both method and protocol
  validation.
- The unrelated untracked
  `docs/ARD_RESEARCH_ISSEUES_AND_PROPOSALS.md` remains outside this work.

## Scientific contracts affected

- Logging-only and RSLAD must have exact model, optimizer, scheduler, loss,
  gradient, RNG, and attack parity under the same seed.
- Observations use pre-update detached FP32 student logits from the actual
  student-crafted adversarial input.
- Teacher clean and adversarial primitives are computed with frozen parameters
  and no input or parameter gradient.
- Stable IDs, DDP padding masks, epoch-boundary merge, checkpoint/resume, and
  rank-zero artifact ownership remain unchanged.
- PGD-AT uses CE inner/outer; TRADES uses student-clean KL inner and clean
  CE plus configured KL outer; selection remains independent CE PGD-20.

## Decisions

- Add the explicit method ID `rslad_logging_only`, rather than a new optional
  schema field. This preserves historical schema-v2 config hashes and makes
  the observation identity explicit.
- Store primitive values, not a proposed risk: student margin/current EMA,
  correctness frequency/forgetting, teacher entropy, true-class probability,
  maximum wrong-class probability, prediction, and correctness on clean and
  student-adversarial inputs. Wrong-confidence and future gates are derived
  offline.
- Keep signal, gate, combination, and intervention as separate future
  components. No v2 policy is implemented in this preparation.
- Make the controlled protocol select a method-family-specific training attack
  contract while retaining one common optimizer/schedule/evaluation contract.

## Milestones

- [x] M0: logging-only state and runtime composition.
- [x] M1: exact bounded parity, serialization, resume, and DDP-mask tests.
- [x] M2: canonical PGD-AT/TRADES configs and config/protocol tests.
- [x] M3: Hamster bounded CUDA parity/pilot preflight.
- [x] M4: impact-selected tests, one consolidated scientific review, docs,
  cohesive commit, and push.

## Agent and review budget

The main thread is the sole writer because the delta crosses config, trainer,
state, and tests. No new planning or mechanical agent is needed. Apply one
consolidated scientific review after focused evidence is stable; review a fix
delta only for an actual P0/P1.

## Test plan

- T1/T2: teacher primitive formula/range, state v1 migration/v2 round-trip,
  logging-only method validation, PGD-AT/TRADES protocol resolution.
- T2/T3: identical bounded RSLAD versus logging-only model/optimizer/scheduler/
  RNG state, with sample state intentionally different and complete.
- Conditional CUDA: a few bounded batches only; no CIFAR production run.
- Run `scripts/verify.py --changed` after focused tests and use cached passes.

## Risks and mitigations

- Extra teacher forwards could alter optimization or RNG. Reuse the existing
  detached teacher targets/diagnostic forward and require exact checkpoint
  parity excluding observation state and config identity.
- Expanding sample state could break historical resume. Accept format v1 and
  migrate missing teacher primitives to `None`.
- A hard wrong-confidence indicator at threshold zero means any teacher
  misclassification, not necessarily strong confidence. Store primitives and
  defer thresholds.
- Adding production configs could disturb the fixed eight-cell taxonomy. Keep
  independent baseline/logging-only templates under `configs/scientific/`.

## Progress log

- 2026-07-31: fixed the method-independent observation boundary and rejected a
  new defaulted schema field because it would drift historical config hashes.
- 2026-07-31: implemented `rslad_logging_only`, format-v2 primitive sample
  state with v1 migration, method-specific controlled attack contracts, and
  scientific templates. Focused CPU commands reported 128 and 135 passes.
  A one-epoch synthetic logging CLI stored all 10 train IDs and complete
  clean/adv teacher fields. Hamster CUDA exact parity reported 1 pass; PGD-AT
  and TRADES 16-sample CUDA smokes both exited 0. Their one-epoch real-data
  pilot identities also passed dry-run without creating W&B runs.
- 2026-07-31: full-SAAD preflight verified commit
  `295121c5d2eed827b5b2d6aa42307de809bdfada`, then failed before training
  because the upstream-local AutoAttack lacks `autoattack.state` required by
  the pinned RobustBench environment. This remains an explicit dependency
  blocker; no upstream source or environment was modified.
- 2026-07-31: non-isolated `scripts/verify.py --changed` completed all
  20 selected T0--T3 commands. The initial isolated attempt hit only the known
  localhost-socket restriction in three Gloo tests; `--lf` outside the sandbox
  reported `3 passed, 3 deselected`, then the exact selected command reported
  `6 passed`. Affected Ruff, mypy, and diff checks passed. The one consolidated
  scientific review returned no P0/P1; no review retry was used.

## Completion report

Implemented an explicit loss-identical `rslad_logging_only` runtime without
adding a defaulted schema field, primitive clean/adversarial teacher
confidence state, v1-to-v2 sample-state migration, enriched final scalar
Parquet, method-specific controlled attack contracts, teacherless baseline
tracking, and checked-in scientific templates for logging-only, PGD-AT, and
TRADES. CPU, synthetic CLI, Hamster CUDA parity, baseline CUDA smoke,
configuration dry-run, impact-selected T0--T3, Ruff, mypy, and diff checks all
passed as recorded above. No real-data long run, W&B run, full SAAD, or
AutoAttack was started. Full SAAD remains dependency-blocked before training.
