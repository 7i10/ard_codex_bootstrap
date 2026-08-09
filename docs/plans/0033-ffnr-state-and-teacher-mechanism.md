# 0033 — FF/NR 3-state routing, margin response, and teacher mechanism

## Scope and frozen decisions

This phase is a read-only mechanism analysis. It does not launch Route A/B
treatments, change q or coefficients, add dynamic routing, open official test,
or run AutoAttack. Existing Chen L2/L4 CE-PGD20 replay and online-state
artifacts are reused; missing IRT plateau artifacts remain explicitly blocked.

Frozen endpoint: train-split stable-ID universe, epochs `[189, 194, 199]`,
primary `majority` (at least 2/3 wrong), secondary `all` (3/3 wrong). Feature
anchors are `[39, 59, 79]`, and every feature must be available at or before
its anchor. Student/teacher clean correctness remain separate flags.

Margins use probability space with positive meaning true-label side:

`m_S_clean`, `m_S_adv`, `m_T_clean`, `m_T_adv`,
`Delta_S = m_S_clean - m_S_adv`, and `Delta_T = m_T_clean - m_T_adv`.

The analysis first reports continuous risk surfaces, then preregistered
quantile/change-point candidates without automatically choosing a final
threshold. Cross-seed fit/evaluation is required for predictive comparisons.

## Checklist

- [x] Reconcile `origin/master`, read current FF/NR reports and plans.
- [ ] Add a tracked-clean CPU CLI/config with fail-closed input identity,
      stable-ID/class joins, margin-sign consistency, and non-overwrite output.
- [ ] Compute Student/Teacher one-dimensional risk curves and Student×Teacher
      two-dimensional surfaces for both endpoint definitions.
- [ ] Compare quantile candidate 3-state partitions, 2-state reductions, and
      continuous low-complexity models under L2→L4 and L4→L2 evaluation.
- [ ] Decompose Teacher clean difficulty versus Student→Teacher attack
      response, condition on Student/Teacher correctness strata, and compare
      Student response as a control.
- [ ] Overlay existing Route A/B pilot response tables where the saved schema
      permits; do not infer causal effects from training-state observations.
- [ ] Record IRT as blocked unless an identical `[39,59,79]` feature and
      `[189,194,199]` endpoint artifact is recovered without new training.
- [ ] Run focused unit/lint/type/impact checks and write the machine-readable
      report and Markdown interpretation.
- [ ] Only if the preregistered Teacher-response gate passes, propose (do not
      silently launch) Teacher self-attack or gradient-alignment follow-up.

## Risks and acceptance criteria

- A truncated or mismatched validation/replay lineage fails closed.
- No state threshold is selected using same-seed outcome optimization.
- `m_T_clean`, `m_T_adv`, and `Delta_T` are not entered together in one linear
  model because of exact linear dependence.
- Analysis distinguishes prediction, association, conditional association,
  transfer evidence, geometry evidence, and causal treatment effect.
- Completion means the CPU point report is reproducible from a clean commit,
  focused tests pass, and blocked GPU/IRT stages are explicitly listed. No new
  training or official evaluation is part of this milestone.

## Execution log

| Date | Action | Result |
|---|---|---|
| 2026-08-09 | Repo reconciliation and attachment review | Started from `1db6554`; existing Chen CE-PGD20 and route artifacts available; IRT exact endpoint unavailable. |
