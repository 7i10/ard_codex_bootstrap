# AutoAttack provenance and terminal campaign reconciliation

Status: in progress

## Outcome

Make future AutoAttack evaluations self-identifying, preserve the exact source
identity of the completed seed-zero evaluations without rewriting their
artifacts, and reconcile the two canonical campaign state stores from portable
terminal evidence so a controller pass is a no-op.

The campaign/runtime subsystem remains internal operational code. Public
simplification is a later, separate change after the completed experiment
lineage is frozen.

## Decisions

- Treat already-finished `evaluation-results.json` files and W&B runs as
  immutable. Record recovered AutoAttack identity in an additive, explicitly
  post-hoc host attestation; current installed bytes do not prove historical
  execution bytes on their own.
- Identify AutoAttack using distribution metadata, VCS direct URL/commit when
  present, and a deterministic hash over relative Python source paths and
  bytes. Never rely on `__version__ == "unknown"` alone.
- Pin the evaluation dependency separately from `.external` repositories;
  `.external` remains reserved for inspected upstream worktrees.
- Reconcile only explicitly named jobs from a versioned evidence document.
  Validate the campaign identity, job identity, expected required phases,
  successful exits, result/checkpoint digests, and evidence digest before one
  atomic batch mutation.
- Preserve the prior durable job record in reconciliation history. Reapplying
  the same evidence is an idempotent no-op; conflicting evidence fails.
- Extra post-hoc AutoAttack for a job whose campaign contract ends at PGD is
  evidence only and does not become a required phase.

## Progress

- [x] Read repository contracts and inspect the current AutoAttack adapter,
  campaign state machine, campaign spec, and Hamster state.
- [x] Recover the installed AutoAttack identity independently on Hamster and
  Ferret: upstream commit
  `a39220048b3c9f2cca9a4d3a54604793c68eca7e`, distribution `0.1`, and
  canonical path/NUL/bytes source hash
  `e74d6dab0e34faf840f1bdfe0f77e9ddcc5f753a7426cbaa54b11bf17f896487`.
  An earlier ad-hoc `cf245…` hash used an unspecified framing and is not the
  retained contract.
- [x] Implement structured AutoAttack provenance and dependency pinning.
- [x] Add focused evaluation provenance tests.
- [x] Implement portable terminal evidence validation/import with atomic,
  idempotent state mutation.
- [x] Add focused campaign reconciliation tests and CLI dry-run coverage.
- [x] Generate and validate the additive AutoAttack provenance amendment and
  real Hamster reconciliation dry runs.
- [ ] Generate and validate remaining reconciliation records from
  completed Hamster and Ferret artifacts.
- [ ] Import only the affected jobs into their canonical host state stores.
- [ ] Verify one controller pass per host launches nothing and reaches the
  scientific-review boundary.
- [x] Run `scripts/verify.py --changed`, record cached passes, and perform one
  consolidated scientific review of the stable delta.
- [x] Update experiment/dashboard and reproducibility documentation.
- [ ] Commit the cohesive change and push after verification.

## Changed areas

- `src/ard/evaluation/autoattack.py`
- evaluation dependency constraints and evaluation unit tests
- `src/ard/campaign/state.py` and a narrow internal import CLI
- campaign reconciliation tests
- additive provenance/evidence records under `docs/experiments/`
- experiment dashboard and reproducibility documentation

No training objective, attack strength, checkpoint, sample state, or completed
evaluation result is changed.

## Test selection

- T1: AutoAttack provenance metadata/source hashing and injected-adapter path.
- T1: terminal evidence schema, identity/phase/exit/digest rejection,
  idempotency, conflicting import rejection, and all-or-nothing multi-job
  mutation.
- T1/T2: existing evaluation and campaign state/worker tests selected by the
  impact gate.
- Operational verification: dry-run import, real import, state snapshot
  comparison, and one no-launch controller pass on each host.
- No GPU scientific evaluation is repeated; checkpoint bytes and completed
  result bytes are inputs to provenance validation, not new experiments.

## Risks

- Cross-host output paths are not portable. Evidence therefore records
  content digests and execution-host identity rather than requiring those
  paths to exist on the canonical state host.
- A stale controller could race reconciliation. Import requires the controller
  to be paused and uses the existing host lock; operational preflight checks
  process/lease state.
- Direct URL metadata can be absent in some installs. The deterministic source
  hash remains mandatory, so the evaluation is still exactly identifiable.
- Installed AutoAttack package metadata and bundled license text may disagree.
  Verify and record both without presenting either as a repository
  redistribution license.
- Retrofitting provenance into finished W&B artifacts would corrupt lineage.
  The amendment references immutable result digests instead.

## Completion conditions

- New AutoAttack results contain an exact source identity and cannot silently
  report only `unknown`.
- Every completed seed-zero AutoAttack result is covered by an immutable
  additive provenance amendment.
- Each canonical job state agrees with the required phases actually completed,
  including cross-host reassignments, with prior records archived.
- Re-import is a no-op, conflicting evidence fails, and a controller pass on
  each host launches no work.
- Selected tests pass from actual commands; T4/T5 are not rerun.
- The final commit contains no outputs, checkpoints, W&B data, caches,
  credentials, or `.external` files.

## Review log

- The single research-planning pass concluded after the plan stabilized. Its
  accepted findings are: fail closed on future AutoAttack source mismatch,
  label recovered campaign provenance as post-hoc, use compare-and-swap and
  dry-run-by-default for terminal evidence import, preserve Student post-hoc
  AutoAttack as auxiliary evidence, and do not resume controllers before
  import. Its public-release licensing recommendation remains a separate
  maintainer decision.
- The consolidated scientific review found no P0. Its initial P1 findings
  required strict quiescence, host-batch journaling, exact imported-record
  equality, complete Student auxiliary evidence, exact phase-dependent input
  hashes, and a non-forgeable post-hoc amendment binding. The final delta
  review found no remaining P0/P1 after both the amendment reference and the
  allowed `(execution_host, evaluation_results_sha256)` pairs were pinned.

## Verification log

- `PYTHONPATH=src ... pytest -q tests/unit/test_campaign_reassignment.py
  tests/unit/test_evaluation.py`: `44 passed`.
- Final focused reassignment delta: `8 passed`; Ruff passed.
- `scripts/verify.py --changed --non-scientific`: exit 0. The first complete
  pass ran every selected T0--T3 command. The final delta pass reran six
  integration/smoke commands and reported cached pass for eleven unchanged
  unit commands.
- Real read-only collection validated Chen/Entropy primary evidence and
  Chen/Student primary plus auxiliary AutoAttack evidence against immutable
  Hamster files. State import and controller execution remain pending.
- T4/T5, production training, and AutoAttack were not run.
