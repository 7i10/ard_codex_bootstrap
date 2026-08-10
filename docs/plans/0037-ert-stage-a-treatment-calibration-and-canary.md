# 0037 — ERT Stage A treatment calibration and canary

Status: planned, implementation not started
Date: 2026-08-11

## Objective

Freeze the ERT Stage A treatment formulas from the implemented RSLAD
objective, calibrate coefficients without optimizer updates or outcome access,
implement the fixed-anchor runtime, run an engineering canary, and only then
run the two-seed five-epoch causal screen. Stage B, dynamic routing, official
test, AutoAttack, and new seeds are explicitly out of scope.

## Reconciliation decisions

- The production RSLAD implementation is `5/6 * adversarial_kd + 1/6 *
  clean_kd`, with `KL(teacher_clean || student)` and the configured
  temperature/`T^2` behavior. The exact resolved config is recorded before
  calibration; no coefficients are inferred from prior documents.
- Existing `ert_state_overlay_v1` and proxy v1/final10 artifacts remain
  historical. Stage A uses explicit predicates and new versioned masks/output.
- The review blocker for equal-rank population is non-blocking only if the
  Stage A selector does not consume that composite. Otherwise execution stops
  for a human decision.
- The prompt's S3-T3 `KD=1.0` arm is interpreted as the baseline adversarial-KD
  coefficient/multiplier of 1.0, not an unweighted raw KD sum. This preserves
  the implemented `5/6` and `1/6` RSLAD coefficients.
- CleanCE calibration references the *effective* baseline training loss after
  `RSLADBaselinePolicy` (`5/6 AdvKD + 1/6 CleanKD`). The latent hard-CE field
  returned by `RSLADObjective` is not part of the baseline update because its
  policy weight is zero.
- Teacher-only softening uses a separate target temperature `tau=2.0`; the
  student objective temperature and `T^2` setting are not changed. The
  calibrated softening multiplier is applied only to the softened adversarial
  KD branch.

## Execution gates

- [ ] Audit RSLAD formula and call chain; write exact equation and resolved
  attack identity.
- [ ] Verify L2/L4 epoch-79 parent, teacher/checkpoint SHA, stable-ID masks,
  and that Stage A does not consume the equal-rank composite.
- [ ] Implement no-update gradient calibration and immutable machine artifact.
- [ ] Run calibration on deterministic class-stratified L2/L4 cohorts; freeze
  `tau`, `alpha_soft`, weak/moderate AdvCE, and weak CleanCE once.
- [ ] Implement treatment runtime with full-batch mean, teacher frozen, and
  Clean-Wrong attack subset skip.
- [ ] Add formula, gradient, mask, attack, no-future-information, and
  calibration mutation tests.
- [ ] Run `scripts/verify.py --changed` and a clean engineering canary.
- [ ] Run exactly L2/L4 +5 epoch Stage A arms from immutable epoch-79 parents.
- [ ] Evaluate every arm with common eval-mode CE-PGD20 and write results.
- [ ] Stop after report; do not promote a winner automatically.

## Scientific stop conditions

Stop before GPU Stage A on any wrong parent/teacher/attack identity, mask or
state mismatch, calibration mutation, future leakage, nonselected baseline
drift, full-batch normalization drift, attacked Clean-Wrong sample,
unfrozen coefficient artifact, dirty code tree, unresolved P0/P1, or unexpected
equal-rank dependency.

## Expected outputs

- `docs/ERT_STAGE_A_CALIBRATION.md`
- `docs/experiments/ert_stage_a_calibration_v1.json`
- `docs/ERT_STAGE_A_TREATMENT_RESULTS.md`
- `docs/experiments/ert_stage_a_results_v1.json`

No production run or official test is authorized by this plan until the
calibration and canary gates are complete.
