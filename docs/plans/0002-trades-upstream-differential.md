# Pinned TRADES Upstream Differential Audit

## Status

- Owner: primary agent with one Terra implementation owner and one consolidated Sol review
- Branch / base SHA: `master` / `3e309dd3b6bf232224810d207a06b3d43a61efe4`
- Current milestone: complete
- Last updated: 2026-07-22

## Goal

Pin the official TRADES repository as a second read-only external oracle, make external management and test fingerprints repository-aware, and add fixed-batch evidence that distinguishes the local TRADES contract from upstream behavior without claiming full-training parity.

## Non-goals

- CIFAR training, public-number reproduction, T4/T5, or full AutoAttack
- Importing `.external/trades` from production runtime
- Copying upstream implementation into `src/ard/`
- Silently changing the established local TRADES objective, attack defaults, scheduler, or checkpoint lifecycle
- Making a legal conclusion beyond recording the observed upstream license file and digest

## Existing state

- The implementation started from clean local commit `3e309dd`; external helpers now support one named repository or all locked repositories while preserving the SAAD default.
- Official TRADES is locally verified at `6e8e11b7c281371c2f027ffadfbaea80361f09de`; origin and worktree are clean, and the root MIT `LICENSE` digest is locked as `4b42e38a6899d82801eb6782fe161cccb5d3d685c8bcddc2b877ac9f87161a30`.
- Local `TRADESObjective` computes clean CE plus beta times `KL(clean || adversarial)` but detaches the clean target. The pinned upstream outer KL does not detach its clean branch, so equal scalar losses do not imply equal parameter gradients.
- Local PGD starts uniformly in the epsilon ball and immediately clamps to pixel `[0,1]`; upstream starts with `0.001 * Gaussian` noise before its iterative projection. Local defaults `8/255`, `2/255`, 10 steps differ from upstream CIFAR defaults `.031`, `.007`, 10 steps; beta 6 agrees.
- Fixed-batch differential coverage and a multi-repository external fingerprint are implemented; production TRADES semantics remain unchanged.

## Scientific contracts affected

- TRADES KL direction, clean-target detach behavior, temperature and `T^2`
- Pixel-space attack initialization, epsilon projection, input clamp, inner eval mode, and outer train mode
- Loss reduction, beta scaling, parameter gradients, and one-step optimizer deltas
- Exact upstream SHA/license lineage and conditional-oracle test caching

## Decisions

- Pin `https://github.com/yaodongyu/TRADES.git` at the exact verified SHA; never follow branch HEAD implicitly.
- Keep the local runtime contract unchanged in this audit. Record target detach, random initialization, numeric defaults, schedule, architecture, and data-pipeline differences explicitly.
- Use clean-room fixed-batch formulas for mandatory CPU regressions. Use the pristine clone only for opt-in source/subprocess evidence because its legacy code unconditionally requires CUDA and an old PyTorch environment.
- A differential test may intentionally prove a documented gradient or trajectory mismatch; it must not relabel that mismatch as parity.
- Generalize bootstrap/verify with an explicit repository selector and a deterministic all-repository mode while retaining safe existing-checkout and atomic-clone behavior.
- Include every locked external repository's observed state in the pass-cache fingerprint so a TRADES checkout change invalidates relevant cached evidence.

## Milestones

- [x] M0 Pin and generalize external management
  - Files/modules: `external.lock.yaml`, `scripts/external_common.py`, `scripts/bootstrap_external.py`, `scripts/verify_external.py`, external-management units
  - Owner: Terra, because lock parsing and mutation safety are one coherent implementation surface
  - Tests: focused T0/T1 lock parsing, named/all selection, atomic clone, remote/SHA/dirty/license verification, backward-compatible SAAD handling
  - Acceptance: `.external/trades` is clean at the exact locked SHA; MIT file presence and SHA-256 evidence are recorded; both repositories verify
  - Rollback point: lock/scripts/test diff before numerical work
  - Commit boundary: included in the single milestone commit `test: pin and audit official TRADES baseline`
- [x] M1 Differential scientific evidence
  - Files/modules: TRADES regression/oracle tests and test-cache external identity
  - Owner: Terra; production `src/ard/objectives` and `src/ard/attacks` remain unchanged unless a reviewed defect requires a separate decision
  - Tests: fixed logits for loss and clean/adv gradients; deterministic attack initialization/projection/clamp/mode contracts; beta/reduction; aligned SGD delta; opt-in pinned-source evidence
  - Acceptance: tests show scalar agreement where contracts align and exact expected gradient/optimizer divergence caused by clean-target detach; initialization and default differences are asserted and documented
  - Rollback point: focused T2 regression diff
  - Commit boundary: included in the single milestone commit
- [x] M2 Documentation, verification, and review
  - Files/modules: upstream/reproduction/test docs, this ledger, any concise command help updates
  - Owner: one batched Luna synchronization after APIs freeze; primary agent owns evidence ledger
  - Tests: `scripts/verify.py --changed`, focused external and TRADES regression commands, lint only when selected; unchanged successful commands use cache
  - Acceptance: one consolidated `scientific_reviewer` finds no P0/P1 issue; exact commands, passes, skips/cache hits, and deferred legacy/T4/T5 work are recorded
  - Rollback point: reviewed full milestone diff
  - Commit boundary: local commit only; no push

## Agent and review budget

- One `research_planner` pass and one parallel `upstream_explorer` pass were requested. The planner delegated a bounded audit; its stalled parent pass was interrupted after the audit and independent explorer had supplied the needed evidence.
- One `terra_implementer` owns scripts, lock schema, cache identity, and focused regressions. No overlapping write agent runs concurrently.
- One `luna_mechanical_worker` receives only the final stable facts for a single documentation batch.
- One consolidated `scientific_reviewer` reviews the final execution path and evidence. A delta-only rereview is used only for a P0/P1 fix or new scientific evidence.

## Test plan

- Newly required T0/T1: multi-repository lock validation, named/all CLI behavior, atomic clone, dirty/mismatched checkout preservation, license digest matching, all-external cache identity.
- Newly required T2: upstream-vs-local loss, clean/adv gradients, beta/reduction, initialization, projection/clamp/mode trace, and one SGD optimizer delta on deterministic fixed batches.
- Opt-in upstream: verify pinned source structure and, only in a compatible legacy CUDA environment, execute its subprocess oracle. A dependency/environment skip is reported, not counted as parity.
- Cached: unchanged tests are skipped only when `scripts/verify.py` reports the identical fingerprint as a prior pass.
- Deferred: T3 is unnecessary unless runtime code changes; T4/T5, CIFAR training, full AutoAttack, and public-number reproduction remain unexecuted.

## Risks and mitigations

- Gradient semantics: explicitly compare both clean and adversarial branches; do not infer gradient parity from equal loss values.
- Attack trajectory: separate initialization from projection/clamp assertions and record upstream Gaussian versus local uniform sampling.
- Version incompatibility: keep mandatory oracle arithmetic clean-room and CPU-compatible; quarantine actual legacy import behind opt-in subprocess execution.
- License/lineage: record exact root license digest without vendoring source; verify origin, SHA, and clean status on every use.
- Cache staleness: fingerprint all external checkout identities, not only SAAD.
- Scope drift: do not change local threat model or training schedule under the label of differential testing.

## Progress log

- 2026-07-22: Started from clean commit `3e309dd`. Read repository protocols and `scientific-change-review`; ran one planning/audit pass plus an independent upstream mapping.
- 2026-07-22: Verified remotely that official TRADES `master` points to `6e8e11b7c281371c2f027ffadfbaea80361f09de`. Identified the non-detached upstream clean KL target and Gaussian attack initialization as the two material differential contracts.
- 2026-07-22: Generalized external bootstrap/verification and cache identity, cloned the exact TRADES SHA, manually identified the root MIT license and locked its digest, and verified both external repositories.
- 2026-07-22: Added CPU fixed-batch loss/gradient/SGD-delta and attack-contract regressions plus opt-in pinned-source evidence. The forced impact gate reported TRADES `4 passed`, external management `13 passed`, verify/cache `25 passed`, and one dependency-limited SAAD oracle skip.
- 2026-07-22: A focused rerun exposed a stale impact-test ordering expectation; the failing node alone was rerun until passing, then the final forced gate passed. The identical non-forced gate returned four `cached pass` entries; the SAAD entry represents its earlier dependency skip, not upstream parity.
- 2026-07-22: Consolidated scientific review approved with no P0/P1. Its P2 requests were resolved by explicitly documenting and source-checking upstream `ToTensor()`-only normalization and by finalizing this ledger.

## Completion report

Implemented the pinned TRADES oracle and multi-repository external tooling without changing production attack/objective semantics. Exact upstream and local loss scalars agree for the aligned fixed batch, while the regression preserves the intentional clean-target gradient and SGD-delta difference; attack initialization/default and normalization differences are explicit.

Executed evidence:

- `ARD_TRADES_SOURCE_EVIDENCE=1 ARD_RUN_SAAD_ORACLE=1 /home/shunsukenaito/.conda/envs/adv/bin/python scripts/verify.py --changed --force --non-scientific`: TRADES 4 passed, external 13 passed, verify/cache 25 passed; the dependency-incomplete SAAD oracle skipped.
- The same command without `--force`: four cached command passes and no pytest execution; the cached SAAD command retains its explicitly reported skip boundary.
- `/home/shunsukenaito/.conda/envs/adv/bin/python scripts/verify_external.py --all`: SAAD and TRADES remote/SHA/clean/license evidence passed.
- Targeted Ruff format/check, mypy for the two changed `src/ard/testing` modules, and `git diff --check`: passed.

No CIFAR/Tiny training, T4/T5, full AutoAttack, legacy TRADES runtime, or live W&B operation was run. The remaining scientific risk is full schedule/data/architecture reproduction, including the upstream no-normalization model boundary versus local adapter-owned normalization.
