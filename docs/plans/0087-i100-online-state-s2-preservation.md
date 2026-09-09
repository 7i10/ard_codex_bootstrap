# 0087 — I100 Online-State S2×T1 Preservation Screen

## Status

- Owner: Codex
- Base SHA: `fcbebc6`
- Current milestone: complete; results recorded (push deferred to the user)
- Last updated: 2026-09-05

## Goal

Run the preregistered two-development-seed online-state screen from the exact
I100 epoch-99 parents.  Compare only I100 control, Online-State Pair-Margin
Preservation (OS-PMP), and Online-State Detached Boundary-Distance Preservation
(OS-DBDP).  The intervention may act only on the current pre-update
Online-S2×T1 branch during epochs 101–114.

## Frozen scientific contract

- Parents: dev-1 `360910a8…7630835`; dev-2 `bb0c7c1a…f7aaf7`.
- Teacher SHA: `fc398a48…c383983` and frozen parameters.
- Training attack: registered sample-keyed KL-PGD10 identity
  `97a41870…9623d4d`.
- Held-out endpoint: registered CE-PGD20 identity
  `70811016…dcc4f2` at e104/e109/e114; e114 is primary.
- Shared baseline e100 prefix once per seed.  Freeze seed-specific positive
  global-margin q10 thresholds from e100 observations before all six forks.
- Reuse the registered pair-margin and detached-boundary-distance coefficients
  only: `0.05380932585058825` and `31.649566509850324`.
- No Clean-Wrong or S3 action, S-BDP, threshold/coefficient tuning, e115–199,
  new seed, official test, or AutoAttack.

## Milestones

- [x] M0: reconcile source/lineage/inputs and classify the runtime as
  FAST_EXISTING_RUNTIME or FULL_NEW_INTEGRATION.
- [x] M1: implement only the minimal online S2×T1 router if existing runtime
  cannot express the exact same-step contract; add fail-closed assertions and
  focused tests.
- [x] M2: materialize two e100 common prefixes, derive and freeze q10
  thresholds, and verify shared-prefix parity.  Both prefixes and both
  seed-local threshold freezes completed in the `attempt11` run of campaign
  `ert-i100-online-state-s2-v1`; the aggregator re-verified the e100 prefix
  and threshold SHA-256 bindings for both seeds.
- [x] M3: freeze the source and an immutable DAG manifest; run static and
  exact scientific smoke coverage for the three scientific branches and needed
  host execution classes.  Frozen scientific source `bcb09a7`; frozen manifest
  `5c9220f3ff246f156d7bd075eac2035e21663c358bd33297722a492d21c5130c`.
- [x] M4: execute the two prefixes, six e101–114 continuations, e104/e109/e114
  endpoints, e114 train audit, collection, aggregation, and reporting via the
  detached DAG.  Completed, but **not as one clean DAG**: the legs were
  produced across `attempt11` (prefixes, thresholds, D-BDP arms),
  `recovery14` (control and PMP arms), `recovery15` (endpoints),
  `recovery16` (canonical e114 replays) and `recovery17`/`recovery18`
  (aggregation).  See caveats (b) and (d) in the completion report.
- [x] M5: write report/artifact, verify, and commit.  The **consolidated
  scientific review was NOT produced** for these results (caveat (e)), and
  **push is deferred to the user**.

## Required runtime invariants

- State is pre-update and derived from the baseline I100 adversarial example;
  the hard router/rival/gate is detached.
- Branch priority is Clean-Wrong, S3-non-Clean-Wrong, Online-S2, Online-S1.
- Treatment is exactly current Online-S2×T1; an assertion rejects every other
  active sample.
- Action pair uses the Student’s detached current strongest non-true rival and
  reuses that pair for Teacher action margins.  Router global margins remain
  distinct from action pair margins.
- Full-batch mean only; no selected-count normalization; no extra PGD; DBDP
  uses first-order input gradients and a detached Student denominator.

## Operational contract

- Use Hamster GPU0/GPU1 only.  Ferret GPU0 preflight passed, but the current
  immutable gate cannot SHA-bind fresh external outputs before collection;
  keeping the whole parent→child→endpoint chain local avoids weakening the
  lineage contract for this first online-state runtime.
- The one initial manifest includes prefixes, threshold artifact, forks,
  endpoints, e114 train audit, collection, aggregation, review bundle, and
  report nodes.
- Launch only through the production launch gate and immutable multi-GPU DAG;
  stable jobs are not actively polled by Codex.
- Manifest records work unit, host/GPU UUID, estimated work, transfer cost,
  parent/source/config/attack lineage, W&B metrics-only metadata, and all
  completion conditions.

## Launch blockers

- Any mismatch in parent, Teacher, attack, calibration artifact, source,
  state semantics, reduction, or shared-prefix parity.
- Online branch/action cannot be proven pre-update and detached.
- An extra loss can be active outside Online-S2×T1.
- Host profile, GPU reservation, required data/Teacher paths, or output
  ownership fails preflight.

## Progress log

- 2026-09-04: user authorized execution after the control-plane optimization
  milestone.  Began exact-prompt and current-runtime reconciliation.
- 2026-09-04: implemented the public online-state runtime, source/attack and
  threshold lineage bindings, separate state/action persistence telemetry,
  gradient boundary telemetry, and bounded public-runtime canary.
- 2026-09-04: consolidated scientific review found and resolved six P1
  issues (including public-canary `SampleRef` handling); final verdict was
  APPROVED pending frozen-source public canary.
- 2026-09-04: first immutable-manifest preflight correctly rejected a generic
  gate assumption that every training node must share the campaign's e114
  bound.  Added identity-bound job-local epoch contracts so the shared e100
  prefix remains exactly one epoch while children remain exactly e101–114;
  no GPU job was launched from the rejected manifest.
- 2026-09-04: bounded canary static validation rejected an incomplete
  `static_cli` entry before any GPU reservation.  Bound the check to the
  representative prefix job and added a manifest-schema regression test;
  re-freeze is required because this changes the production source SHA.
- 2026-09-04: the first bounded runtime canary reached 17 active OS-PMP
  examples, then correctly stopped because Trainer-computed boundary
  telemetry was omitted from both epoch metrics and the checkpointed online
  router state.  Bound those already-computed finite diagnostics to the
  router lineage and callback/W&B metrics, added a regression, and require a
  fresh immutable manifest plus canary.  No production prefix or child arm
  was launched from this rejected canary.
- 2026-09-04: the next manifest's static public-CLI probe correctly exposed
  that gate static commands execute outside the job environment and therefore
  lacked `PYTHONPATH=src`.  The manifest now binds that import root explicitly
  through `/usr/bin/env`; this is an execution-wrapper repair only, so a fresh
  source freeze and canary are required before any GPU reservation.
- 2026-09-04: a fresh bounded canary then exposed a pre-existing online W&B
  run-ID collision across isolated canary attempts.  The experiment contract
  already requires disabled/offline smoke tracking, so canary-only runtime
  paths now bind the production parent to `offline_sync` (W&B offline) while
  production remains online.  A regression covers this mode conversion;
  re-freeze and rerun the bounded canary before launch.
- 2026-09-04: attempt8 completed both exact e100 common prefixes, then both
  seed-local threshold nodes stopped before any child arm because atomic
  attempt-output promotion left the checkpoint's provenance path in its
  staging namespace.  The promoted canonical e100 Parquet bytes match the
  checkpoint-sealed SHA for both seeds.  Repair only the SHA-verified local
  materialization used for threshold freeze; do not alter scientific inputs,
  thresholds, or the branch contract.

- 2026-09-05: re-aggregated the finished campaign with the committed aggregator
  at `66a223d` into runtime campaign root
  `runs/ert-i100-online-state-s2-v1-recovery18`, reproducing `recovery17`'s
  numbers leaf-for-leaf (6869 identical leaves, 104 added provenance leaves,
  only `campaign_root` changed).  Imported the report and hash-bound JSON
  record and closed the plan; no replay or training bytes were changed.

## Completion report

Sources: frozen scientific source `bcb09a73814e7788026b287309d148b7791ccf75`
(all bytes are bound to it); aggregation code
`66a223daf73f65df4d7dae5dda39ac305f1ded82` (a later clean commit that only
repairs the aggregator).  Frozen manifest
`5c9220f3ff246f156d7bd075eac2035e21663c358bd33297722a492d21c5130c`.
Parents: dev-1 `360910a8a886cf904b206c9381cdf6eaa3e71d6150c0998224c7ab4307630835`;
dev-2 `bb0c7c1ace81fd3df1b85660af265b91b1cefd6e91f3ce5d035b0d0c94f7aaf7`.
Calibration artifact `37bf0a0e1aa6ff12951f1c05f59f6df55700be0e28291c6925670d7b6cb56840`.
Frozen q10 thresholds: dev-1 Student `0.140155` / Teacher `0.174019`
(`0e45334c…b8c4c40a`); dev-2 `0.138205` / `0.177792` (`ccc83541…0f7ba26f`).

Decision — e114 held-out CE-PGD20 robust Δ, dev-1 / dev-2:

| comparison | dev-1 | dev-2 | label |
| --- | ---: | ---: | --- |
| PMP − Control | +0.14 pp | +0.20 pp | SUPPORTED |
| DBDP − Control | +0.14 pp | +0.06 pp | SUPPORTED |
| DBDP − PMP | +0.00 pp | −0.14 pp | NOT_SUPPORTED |

Overall `ONLINE_S2_SUPPORTED_FOR_NEXT_STAGE`.  Stop clause: no automatic e199
extension, arm/threshold/coefficient change, fresh seed, official test, or
AutoAttack follows from this screen.

CAVEATS.  (a) Effects are 0.06–0.20 pp on 5,000 held-out samples where one
sample is 0.02 pp (3–10 samples), while RNG alone moves e114 held-out by
1–2 pp (`docs/archive/ard-distillation-2026/ERT_RESEARCH_STATUS_SUMMARY.md`); two development seeds,
directional only, below the noise floor.  (b) Producing the bytes spanned five
campaigns after `attempt11`'s arms were killed by the dirty-source guard:
`attempt11` (prefixes, thresholds, D-BDP), `recovery14` (control, PMP),
`recovery15` (endpoints), `recovery16` (canonical replays),
`recovery17`/`18` (aggregation); `docs/archive/ard-distillation-2026/ERT_I100_ONLINE_STATE_S2_LAUNCH_POSTMORTEM.md`
is stale, still saying attempt-11 is running.  (c) `recovery17` aggregated
through an undisclosed out-of-tree shim; the committed aggregator at `66a223d`
reproduces its numbers leaf-for-leaf in `recovery18`.  (d) `recovery12`–`17`
bypassed the production launch gate with hand-authored manifests.  (e) No
consolidated scientific review was run on these results.  (f) PMP dominates
D-BDP on cost: mean epoch time +1.2% vs +7.9% over Control.

Record: `docs/experiments/ert_rslad_i100_online_state_s2_preservation_v1.json`
(SHA-256 sidecar) + `docs/archive/ard-distillation-2026/ERT_RSLAD_I100_ONLINE_STATE_S2_PRESERVATION.md`.
