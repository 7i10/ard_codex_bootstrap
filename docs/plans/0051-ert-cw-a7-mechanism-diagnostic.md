# ERT Clean-Wrong A7 mechanism diagnostic

Status: in progress; read-only/no-update analysis.

## Objective

Explain why the frozen A7 Teacher positive-floor probability-margin arm was
more stable than A5/A6/A8 using only existing Chen L2/L4 trajectories,
checkpoint-bound no-update replay, and existing CE-PGD20 endpoint rows.

## Frozen contract

- Arms: A5 fixed target, A6 Teacher target clipped at zero/cap, A7
  Teacher positive-floor target, A8 Teacher-abstain target.
- Parents and fixed Clean-Wrong masks are the hashes recorded in plan 0050 and
  the calibration artifact; no checkpoint substitution is allowed.
- Training attack for replay: pixel-space Teacher-clean KL-PGD10,
  epsilon `8/255`, step `2/255`, 10 steps, random start.
- Frozen margin coefficient, floor, fixed target, and cap come from
  `ert_cw_margin_calibration_v1`; no sweep or outcome-based retuning.
- No optimizer/scheduler/state mutation, new training, seed, official test, or
  AutoAttack. Existing endpoint rows are joined by stable ID only.

## Execution

1. Audit source formulas and all replay/endpoint lineage; fail closed on any
   mismatch.
2. Run one real-checkpoint public-CLI smoke and verify schema, lineage, stable
   IDs, and report creation.
3. Replay epoch 79 parent and A5--A8 checkpoints at 84/89/94 on fixed CW IDs,
   retaining Teacher/Student margins, targets, deficits, active flags, and
   regime transitions.
4. Run deterministic no-update gradient probes on a fixed pre-treatment
   sample panel where feasible; report norms/cosines as mechanism diagnostics,
   not causal proof.
5. Join endpoint rescue/harm, pre-treatment Q1--Q5, and non-CW spillover
   artifacts; write the machine JSON and human report.
6. Stop after the diagnostic report; propose (without launching) the next
   A7-without-CleanCE, lambda, and floor/cap sensitivity priorities.

## Outputs

- `docs/ERT_CW_A7_MECHANISM_DIAGNOSTIC.md`
- `docs/experiments/ert_cw_a7_mechanism_diagnostic_v1.json`

The report must distinguish training-time R0--R3 regimes from pre-treatment
Q1--Q5 subtypes, and label all sample/bootstrap summaries as conditional
descriptive evidence rather than training-seed inference.
