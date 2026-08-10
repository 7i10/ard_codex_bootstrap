# 0035 — ERT state-conditioned mechanism pilot

Status: `M0 source fix complete / clean output regeneration pending`
Date: 2026-08-11

## Objective

Test whether the Chen ERT epoch-79 state decomposition identifies a useful
training component without extending the failed Route A/Route B intervention.
This is the second mechanism-design iteration after the closed Route A/B pilot;
it is not a new model/schema version.
The primary goal is a best-validation robust-accuracy improvement, with rescue,
harm, clean harm, and probability-margin change as mechanism diagnostics.

The proposal is a hypothesis, not an accepted result. No threshold, routing
rule, coefficient, or arm will be selected from an endpoint after it is seen.

## Frozen scientific contract

- Teacher: Chen2021LTD_WRN34_10 (ERT), development seeds L2/seed 1 and L4/seed 2.
- Parent: the exact epoch-79 observed RSLAD checkpoint for each seed. Parent
  model, optimizer, scheduler, scaler, RNG, sampler, and SampleStateStore are
  restored; no epoch-79 retraining and no new selector fit are allowed.
- Training continuation: same controlled CIFAR-10 ResNet-18 protocol, KL-PGD10
  training attack, unchanged optimizer/scheduler/data order, fixed masks, and
  +5 epoch Stage A (terminal checkpoint is epoch 83 under the existing zero-based
  convention). No dynamic state transitions.
- Endpoint: eval-mode, independently generated CE-PGD20 in pixel `[0,1]`,
  Linf epsilon `8/255`, step `2/255`, random start, 20 steps, hard-label CE.
  Control and every treatment generate their own attack; no adversarial tensor
  is shared across arms.
- The anchor-79 state table is train-only and contains no future outcome:
  stable sample ID/class, student/teacher clean and adversarial correctness,
  probability margins, `DeltaS`, `DeltaT`, signed teacher dominance, and
  student-history primitives. Primary state quantiles are fixed at 10%; 20% is
  a preregistered secondary overlay. Thresholds are computed from the anchor
  cohort only and are not tuned after seeing treatment outcomes.
- No official test, AutoAttack, unused seed, q/coefficient tuning, self-target,
  friendly early stopping, or 200-epoch claim is permitted in this pilot.

## Frozen state definitions and arms

`S3` is student clean-correct and strong adversarial-wrong at anchor 79.
Teacher state is computed independently: `T1` safe/correct, `T2` fragile/correct,
and `T3` adversarial-wrong; teacher clean-correct/wrong is retained as a
modifier. Persistence is descriptive only and does not create a dynamic route.

The common no-intervention C79 continuation is run once per seed and is shared
as the control for all component arms. This avoids duplicate baseline training
while preserving the exact same parent and schedule. Unique Stage-A treatments
are:

| family | fixed treatment arms |
|---|---|
| S3×T1 | AdvCE `.25`; AdvCE `.50`; KD target softened with AdvCE `0`; KD target softened with AdvCE `.25` |
| S3×T2 | AdvCE `.25`; AdvCE `.50` |
| S3×T3 | AdvCE `.25`; KD multiplier `.5` + AdvCE `.25`; KD multiplier `0` + AdvCE `.25` |
| clean-wrong/recovery | Clean CE only; Teacher-clean-correct → Clean CE + Clean KD, teacher-clean-wrong → Clean CE only; all → Clean CE + Clean KD |

The treatment formulas, mask hashes, and state-table hash are materialized in
machine-readable configs before training. `C0` labels in the attachment refer
to the shared C79 control, not additional stochastic replicas.

## Execution order

### M0 — identity and CPU overlay (before GPU launch)

- Verify `origin/master`, clean source SHA, both parent checkpoint hashes, parent
  manifests/attestations, train ID/class universe, and all six existing CE20
  endpoint bundles.
- Build one hash-bound anchor-79 state table per seed from registered replay and
  online-state inputs. Emit 10% primary/20% secondary masks and a fixed mask
  manifest; never overwrite the immutable CE20 endpoint reports.
- Join the existing CE20 horizons 84/89/94 without new training. Report the
  complete anchor S1/S2/S3 partition and the registered pilot S3×T1/T2/T3
  endpoint cohorts, plus old Route-B cohorts, clean flags, `DeltaT`, `mT_adv`,
  and treatment-control rescue/harm/net/margin/clean-harm. S1/S2 endpoint
  effects are not part of this old-arm overlay because those masks were not
  registered for the endpoint bundles. The overlay is exploratory; it cannot
  launch an arm.
- One real-data smoke must invoke the public CLI, validate Parquet schema,
  stable-ID/class join, lineage/hash bindings, and non-overwriting output.

### M1 — Stage A component screen

- Generate configs from the frozen state/mask bundle and exact parent. Use the
  shared C79 control plus the 12 unique treatments above for each of L2/L4.
- Run independent jobs on available GPUs, longest jobs first; record GPU/runtime
  identity and W&B/local manifest lineage. Do not use a second writer or repeat
  unchanged review/test commands.
- Primary: validation CE-PGD20 robust accuracy at the fixed +5 endpoint and
  best metric within the registered short continuation. Secondary: selected
  state-cohort robust accuracy, rescue/harm/net rescue, probability-margin delta,
  clean accuracy/clean harm, and non-selected spillover.
- The screen is descriptive and component-level. No candidate is promoted from
  a single seed or from validation inspection alone.

### M2 — pre-registered extension gate

Only arms that improve the fixed primary relative to shared C79 in both seeds,
with no unacceptable clean-harm signal, may receive +10 and +15 continuation.
The gate is written before endpoint results are inspected. If none pass, stop
the intervention branch and report a negative/inconclusive component result.
Matched-random controls are added only after a component passes this gate.

## Lineage and outputs

Every state table, mask, config, checkpoint, endpoint, and report records:
source Git SHA/dirty state, parent checkpoint SHA, parent config and manifest
hashes, state/mask SHA, treatment config hash, seed, world size/per-rank batch,
GPU/runtime, endpoint attack identity/hash, and output SHA. Machine reports are
written under `.cache/analysis/` and the human report is
`docs/ERT_STATE_CONDITIONED_MECHANISM_RESULTS.md`.

## Tests and acceptance criteria

- State/mask unit tests: no future columns; exact stable-ID/class universe;
  10/20% threshold determinism; S/T state partition; mask hash and lineage.
- Overlay tests: CE20 attack identity and endpoint join; selected/control
  rescue/harm and clean-harm arithmetic; non-overwrite behavior.
- Config/trainer tests: every arm changes only the registered objective fields;
  teacher stays frozen; clean/adversarial CE/KD branches match the stated
  formulas; checkpoint resume is epoch-boundary exact.
- One real-data end-to-end CPU smoke before any full overlay.
- `scripts/verify.py --changed` and focused tests once after the implementation
  delta. One scientific review only if metrics, resume, or artifact lineage
  changes; no review is used to wait on GPU jobs.

## Risks and rejected shortcuts

- A high AUROC or endpoint rescue rate is not a treatment effect; it cannot
  select a route or threshold.
- Training-state KL-PGD10 observations are not substituted for the common
  CE-PGD20 endpoint.
- The existing Route A/Route B result is not extended as a new route because
  its selected-minus-random sign is not stable across horizons/seeds.
- If the exact L4 parent or a required state/endpoint input is unavailable, the
  affected arm is blocked rather than silently reconstructed.

## Completion conditions

Plan M0 is complete only after both CPU state overlays and immutable manifests
pass. Stage A is complete only after all registered L2/L4 arms have either a
completed +5 checkpoint or an explicit fail-closed reason. No paper-level claim
is made until a preregistered confirmation and saved-checkpoint official
evaluation are separately approved.

## Progress ledger

- [x] Origin/master and existing CE20/state artifacts reconciled.
- [x] Exact L2/L4 epoch-79 parent files, checkpoint hashes, and fork lineage
  verified.
- [x] Public CPU state-overlay CLI, fixed 10/20% masks, and CE20 joins
  implemented and tested.
- [ ] Real-data overlay output regenerated from the final tracked-clean source
  after endpoint parent-binding fixes; the previous output remains exploratory
  and is not reused as final evidence.
- [ ] Freeze and implement the numeric moderate KD temperature and exact clean
  CE/Clean-KD equations before any Stage-A GPU launch.
- [ ] Canary and 12 unique treatment arms per seed.
