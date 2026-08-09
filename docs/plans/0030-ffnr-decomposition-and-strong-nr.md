# FFNR decomposition and strong-NR diagnostics

## Status

- Owner: main thread (integration), Terra (single implementation owner)
- Branch / base SHA: `master` / `8716cd5`
- Current milestone: D3-D5 implementation
- Last updated: 2026-08-09

## Purpose

Explain the Chen CE-PGD20 forecasting result before any intervention.  This
milestone decomposes the teacher AUROC, partitions FF dynamics, and completes
current-wrong diagnostics without changing training, attacks, official-test
evaluation, or the existing strong-point report.

Observable completion means a clean, hash-bound diagnostic report for L2/L4,
including dense CE-PGD20 NR trajectories, has passed focused tests and one
consolidated scientific review.

## Existing state

- The immutable strong point report at source commit `6327fd7` already reports
  Chen teacher signed-dominance AUROC near `.99` for current-correct future
  failure, but not its conditional decomposition.
- L2 and L4 have strong CE-PGD20 feature/outcome replays for sparse epochs and
  exact online state for anchors 39/59/79.  The remaining dense replay epochs
  are independent and cache-addressable.
- Hamster GPUs 0/1 and Ferret GPUs 0/1/2 were read-only preflighted idle on
  2026-08-09.  Ferret remains an immutable execution host; source integration
  and reporting remain on Hamster.
- Two user-authored `docs/ARD_*.md` files are dirty and outside this milestone;
  they must not be staged or reverted.

## Scientific contracts affected

- No attack, objective, training, checkpoint-selection, or official evaluation
  contract changes.  This is train-split read-only analysis.
- CE-PGD20 identity, stable sample/class universe, checkpoint SHA, input source
  hashes, and online-versus-replay domain boundaries are fail-closed inputs.
- “Non-recovery” is reserved for the dense same-attack trajectory.  Sparse
  plateau behavior remains “current-wrong future failure.”

## Non-goals

- No intervention, new training, official CIFAR test, AutoAttack, bootstrap, or
  automatic v2 selection.
- No inference of label noise from logits.  Image/label review is blinded and
  remains a later human annotation step.

## Frozen endpoints and cohorts

- [x] Primary endpoint: wrong on at least two of the three plateau checkpoints
  (`majority`).  `two_thirds` is the same mask and is not duplicated.
- [x] Secondary endpoint: wrong on all three plateau checkpoints (`all`).
- [x] Runs remain L2/L4 (Chen ERT seeds 1/2); anchors remain 39/59/79.
- [x] FF eligibility is exact-online current-correct; current-wrong eligibility
  is exact-online current-wrong.  Strong replay correctness never redefines
  these deployable cohorts.
- [x] Official CIFAR test and AutoAttack are forbidden inputs.

## D3: teacher-signal decomposition

- [ ] Report teacher-adversarial correct/wrong counts and future-failure rates,
  Wilson intervals, risk difference, and risk ratio.
- [ ] Report signed teacher dominance distributions by teacher state/outcome.
- [ ] Report Spearman correlation with strong Student logit-margin risk and
  online Student history.
- [ ] Run deterministic class-stratified stable-ID five-fold OOF logistic
  comparisons: `M` vs `M+D`, `H` vs `H+D`, and `M+H` vs `M+H+D`.
- [ ] Fit transforms on training folds only and report AUROC, AUPRC, log-loss,
  Brier, and paired deltas per run/anchor/endpoint.

## D4: FF snapshot taxonomy

- [ ] Build disjoint online-snapshot classes from correctness available by the
  anchor: oscillating (at least two flips), transient-correct (final correct
  suffix one), stable-then-future-failure (final correct suffix at least two),
  and other/insufficient.  Epoch 39 is necessarily insufficient.
- [ ] Repeat separately in the strong-feature snapshot domain.  Never combine
  online and replay correctness into one transition sequence.
- [ ] Report C/C, C/W, W/C, W/W eligibility transitions, subtype counts,
  teacher/student distributions, cross-seed overlap, and online-vs-strong
  current-state discordance.

## D5: strong current-wrong and NR

- [ ] Report clean difficulty, robustness-specific attackability, strong-domain
  disagreement, Student clean-to-adversarial response, teacher clean/adv state,
  JS/entropy/dominance, and plateau pattern.
- [ ] Generate a deterministic class-balanced blinded candidate-ID/image panel
  for persistent-clean-wrong plus teacher-wrong cases and matched controls.
  Human image/label-quality annotation remains external to logits.
- [ ] Replay only missing CE-PGD20 epochs `39..199` every five epochs.  Split L2
  across two Hamster GPUs and L4 across three Ferret GPUs; reuse existing
  feature/outcome epochs and their hashes.
- [ ] Merge chunks fail-closed on attack, source, run/config, checkpoint,
  stable-ID/class, and epoch coverage.
- [ ] In the dense strong domain only, classify current-wrong samples as
  persistent-wrong, recovered-stable, or recovered-relapsed.  Do not interpolate
  between five-epoch observations.

## Verification and stopping rules

- [ ] Focused unit tests cover sparse joins, probability algebra, fold-local
  transforms, deterministic folds, FF partition completeness, online/strong
  domain separation, current-wrong and dense-NR partitions, lineage hashes,
  and blinded-panel leakage.
- [ ] One real missing-epoch replay is validated before the five-job launch only
  if the execution path differs from the already passed `e5cb442` replay.  A
  config-only epoch subset does not require a ceremonial repeat smoke.
- [ ] Compute point estimates before any bootstrap.  No intervention, new
  training, official test, or AutoAttack is authorized by this milestone.
- [ ] Run one consolidated scientific review after code, replay outputs, and
  report are stable; re-review only a P0/P1 fix delta.

## Milestones and commit boundaries

- [x] M0 — analysis/config/test implementation; focused CPU verification.
  Planned commit: `Implement FFNR decomposition diagnostics`.
- [ ] M1 — five-way dense replay, hash-verified collection, point report.
  Outputs remain ignored cache artifacts; no stochastic result is committed.
- [ ] M2 — consolidated scientific review and result documentation.
  Planned commit: `Document FFNR decomposition results`.

Rollback is the last pushed immutable commit.  Replay jobs run only from the
pushed M0 SHA, so reverting analysis code cannot mutate completed checkpoints.

## Agent and review budget

- The completed planner pass froze endpoints and cohort/domain semantics.
- One Terra owner implements all dependent code/config/tests in one turn.
- The main thread integrates, launches independent GPU jobs, and writes docs.
- One scientific reviewer runs only after code, artifacts, and report are
  stable.  A second review is allowed only for an actual P0/P1 delta.
- No Luna or bug investigator is needed unless a genuinely unexplained failure
  appears.

## Test plan

- New focused unit tests: joins, teacher probability algebra, deterministic
  fold-local OOF, FF and NR partitions, lineage, blinded manifest.
- Static gates: Ruff, mypy, config load, CLI help.
- Impact gate: `scripts/verify.py --changed` once after the focused tests pass.
- GPU gate: one real missing epoch only if the public execution path changes;
  config-only epoch chunks reuse the already verified replay path.
- Deliberately excluded: official test, AutoAttack, bootstrap, new training.

## Risks and mitigations

- Outcome leakage: online cohort membership and plateau outcome remain separate;
  transformations are fitted inside each OOF training fold.
- Domain mixing: online and CE-PGD20 correctness sequences receive separate
  taxonomy names and tables.
- Partial/corrupt replay: every chunk must match attack/source/run/checkpoint and
  exact stable-ID/class hashes; expected epoch coverage is exact.
- Random panel bias: selection is class-balanced and stable-ID-hash deterministic;
  its blinded manifest excludes diagnostic state.
- Runtime: reuse existing epoch caches and dispatch only missing epochs with
  longest-processing-time-first across five idle GPUs.

## Progress log

- 2026-08-09: froze D3-D5 contracts; verified two Hamster and three Ferret RTX
  4090 GPUs idle; assigned one consolidated implementation turn.
- 2026-08-09: implemented D3/D4/D5 diagnostics, exact 2+3 dense replay chunks,
  fold-local class-stratified OOF, same-domain dense NR, and bounded blinded
  CIFAR panel.  Focused verification: diagnostics `8 passed`, Ruff passed,
  mypy passed; L2/L4 missing-epoch sets are exact (25/27).
