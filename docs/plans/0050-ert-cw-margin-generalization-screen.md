# ERT Clean-Wrong Generalizable Robust Action Screen

Status: in preparation; GPU launch blocked pending exact L4 epoch-79 parent recovery.

## Objective

Compare fixed Clean-Wrong interventions that provide clean-label recovery,
plain adversarial hard-label pressure, and adversarial probability-margin
learning.  The screen uses the fixed epoch-79 Clean-Wrong masks for Chen L2
and L4, keeps the RSLAD KL-PGD10 inner attack unchanged, and evaluates each
continuation at epochs 84, 89, and 94 with independent CE-PGD20.

## Frozen scientific contract

- Parents: L2 SHA `ad43d72da2a02f205c65b96485379c9acb5fc2b07d6823d09820439aedc8f78c`;
  L4 SHA `026a36d3fe057386fe19225fed23b56625ab23da80be3dd42cf3e478e5080bf1`.
- Clean-Wrong masks: L2 n=8623, SHA
  `0859507a2d86023f016ac4d7af890b556735ccfcd56faf14110dd161c1989d8b`;
  L4 n=8925, SHA
  `fe818e755e4b2da7a5beb7e1a791a52ab9290295f01064870237972bb58344a6`.
- Training attack: pixel-space KL-PGD10, epsilon 8/255, step 2/255,
  random start, Teacher-clean target.  The generated Student adversarial
  example is reused by all outer treatments.
- Student and Teacher probability margin are
  `p(y)-max_{c!=y} p(c)`, with Teacher evaluated on the frozen Teacher using
  the Student-crafted adversarial input.
- Teacher margin calibration uses only pre-treatment positive margins from
  the hash-bound KL10 feature artifact: Q25/Q50/Q75 define floor/fixed/cap.
  No endpoint result may affect these values.
- One shared no-update gradient-calibrated AdvCE coefficient and one shared
  margin coefficient are frozen across L2/L4; invalid or zero denominators
  fail closed.
- Arms: A0 BASE, A1 CleanCE .15, A2 plain AdvCE, A3 CleanCE+AdvCE,
  A4 fixed margin, A5 CleanCE+fixed margin, A6 Teacher target clipped at
  zero/cap, A7 Teacher positive-floor target, A8 Teacher-abstain target.
  All treatment masks remain the fixed Clean-Wrong cohort.
- No new seed, threshold sweep, dynamic routing, official test, or
  AutoAttack.  W&B records metrics/lineage only; model and run-bundle uploads
  remain disabled under the repository retention policy.

## Required implementation and analysis

1. Add a public plan-bound calibration artifact/CLI for gamma quantiles and
   no-update AdvCE/margin gradient calibration.
2. Extend the shared Trainer with per-sample margin hinge treatments while
   preserving full-batch reduction, frozen Teacher, and baseline empty-mask
   equivalence.  Add formula-level tests for fixed, zero, floor, cap, and
   abstain targets and Teacher no-gradient behavior.
3. Extend the Stage-A runtime/CLI with the nine arm specifications and bind
   every arm to parent, mask, calibration, source, and attack identities.
4. Run a deterministic canary and focused verification before any GPU
   continuation.
5. Run 18 trajectories only after both exact parent checkpoints are present;
   evaluate train and fixed internal validation at 84/89/94 and produce the
   direct, non-CW spillover, held-out, overall, CE20/KL10 quantile, rescue/
   harm, margin, Pareto, and runtime-cost reports.

## Current blocker and accepted non-substitution rule

L2 exact parent is present locally and hashes correctly.  The local L4 file
`.cache/analysis/ffnr-causal-pilot-epoch79/chen-rslad-observed-s2-confirm-v2/last.pt`
hashes to `9b51bca767871ada6c80c75ad92997f9b7f246c0c1e35f3edad35d4e787a4a9c`,
not the frozen `026a...` SHA.  No local checkpoint with the frozen L4 SHA was
found.  The `9b51...` file must not be substituted.  GPU launch therefore
remains blocked until the exact L4 parent is restored and hash-verified.

CPU implementation, contract tests, and artifact preparation may proceed in
the meantime.  Once the parent is restored, rerun the preflight and canary;
do not start production from a dirty or mismatched lineage.

## Output

- `docs/ERT_CW_MARGIN_GENERALIZATION_SCREEN.md`
- `docs/experiments/ert_cw_margin_generalization_screen_v1.json`

The campaign stops after the specified endpoints and does not promote a
winner automatically.
