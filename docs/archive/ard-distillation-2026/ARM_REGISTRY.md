# Arm Registry

This is a reference table, not an essay. It lists every experimental "arm" (a named
training-time treatment condition) found across the project's experiment reports, plus
the arms that only exist in code (`src/ard/config/schema.py`) with no results document yet.
The point is to make it visible when two arms with different names are actually the same
treatment, or when two arms with the same or similar name are actually different
treatments. See **Collisions**, below the main table, for the payoff.

Machine-readable version: [`docs/experiments/ard_arm_registry_v1.json`](experiments/ard_arm_registry_v1.json)
(118 arms; same data as this table plus exact parameter values, full source-record paths,
and per-arm notes).

## Terms used once, here

- **Arm**: one named training configuration — normally "the usual training method, plus one
  extra loss term or one extra rule, applied to some subset of samples."
- **Cohort**: the subset of training samples an arm's extra loss term is applied to. `all`
  means every sample; `none` means the arm adds no loss term at all (it is a control).
- **Cohort rule**: how the cohort is decided.
  - `fixed(e79)` — cohort membership is computed once, from the model as it existed at
    epoch 79, and never rechecked again for the rest of training.
  - `online` — cohort membership is recomputed every training step from whatever the model
    currently looks like, so a sample can enter and leave the cohort as training proceeds.
  - `none` — the arm is not cohort-based (a pure control, or a full-dataset augmentation
    schedule).
- **Anchor epoch**: for a `fixed` cohort, the epoch its membership was computed from.
- **Horizon**: the epoch range the arm was actually trained over, written `start->end`.
- **Held-out result**: accuracy on a validation split the treated cohort's mask was never
  computed from — the honest test of whether the treatment generalizes, as opposed to
  "direct" or "selected-cohort" effects, which are measured on the same samples the
  treatment was applied to and are optimistic by construction.
- **pp**: percentage points.
- **Seeds**: `L2`/`L4` denote two Chen-ERT training seeds (informal names, not epoch
  numbers); `dev-1`/`dev-2` denote two I100-lineage development seeds; these are two
  different seed families from two different dataset/parent lineages and are never pooled
  together. Two seeds give a directional read for those two seeds, never a population
  claim (project convention).
- **Canonical form**: this registry's own naming convention, `LOSS(param) @COHORT
  /RULE(anchor)`, used so that arms doing the same thing sort next to each other regardless
  of what a report happened to call them. It is not a name used in any source document.

## Loss families

`AdvCE` (extra adversarial cross-entropy term), `CleanCE` (extra clean-image
cross-entropy term), `TargetSoften` (softens or mixes the training target), `KDScale`
(rescales the existing adversarial knowledge-distillation term, no new term), `TPFM`
(target-probability-floor-margin: a probability-margin hinge toward a Teacher-derived
floor/cap target), `MF` (margin floor: a probability-margin hinge toward a *fixed* target),
`PM` (pair margin: a logit-margin hinge between Student and Teacher), `BD` (detached
boundary distance: the pair-margin hinge normalized by input-gradient norm), `SBD` (secant
boundary distance: a further, numerically fragile variant of `BD`), `Aug` (a data
augmentation schedule, not a loss term), `Order` (a batch/sample ordering policy, not a
loss term), `AnchorRetain` (mixes the current Teacher target with a frozen earlier
checkpoint's target), `PGDPrefix` (shortens the attack instead of changing the loss).
Two families had to be invented because nothing on this list fit precisely — `AdvCE-BCE`
(a MART-style boundary-CE variant) and `AdvCE-structural` (gates or reweights the existing
loss without adding a new term) — both flagged where they occur.

## Main table

Sorted by loss family, then cohort, then arm name. `—` means none. Params are compressed
into the canonical form; exact values, plus every field the JSON schema asks for, are in
the JSON file.

| Name | Aliases | Canonical form | Cohort rule | Horizon | Seeds | Comparator | Source doc |
|---|---|---|---|---|---|---|---|
| A2 | — | AdvCE(beta=0.0773) @CW /fixed(e79) | fixed(e79) | 79->84/89/94 | L2,L4 | A0 | ERT_CW_MARGIN_GENERALIZATION_SCREEN.md |
| CLEAN_WRONG_PLAIN_ADVCE | PLAIN_ADVCE | AdvCE(?) @CW(I100) /fixed(e99) | fixed(e99) | 100->114 | dev-1,dev-2 | I100_CONTROL | ERT_RSLAD_I100_ACTION_TRANSFER_SCREEN.md |
| RAR | — | AdvCE(ce=0.250,kd=0.500) @RouteA.Rand /fixed(e79) | fixed(e79) | 79->84/89/94 | L2,L4 | RA | FFNR_CAUSAL_PILOT_RESULTS.md |
| RA | — | AdvCE(ce=0.250,kd=0.500) @RouteA.Strong /fixed(e79) | fixed(e79) | 79->84/89/94 | L2,L4 | RAR, C79 | FFNR_CAUSAL_PILOT_RESULTS.md |
| RBR | — | AdvCE(ce=0.250,kd=1.000) @RouteB.Rand /fixed(e79) | fixed(e79) | 79->84/89/94 | L2,L4 | RB | FFNR_CAUSAL_PILOT_RESULTS.md |
| RB | — | AdvCE(ce=0.250,kd=1.000) @RouteB.Strong /fixed(e79) | fixed(e79) | 79->84/89/94 | L2,L4 | RBR, C79 | FFNR_CAUSAL_PILOT_RESULTS.md |
| ST1M | — | AdvCE(beta=0.142) @S3T1 /fixed(e79) | fixed(e79) | 79->84 | L2,L4 | C79 | ERT_STAGE_A_TREATMENT_RESULTS.md |
| ST1W | ST1W-rerun | AdvCE(beta=0.0710) @S3T1 /fixed(e79) | fixed(e79) | 79->84 | L2,L4 | C79 | ERT_STAGE_A_TREATMENT_RESULTS.md |
| T1WCONF | — | AdvCE(beta=0.075) @S3T1 /fixed(e79) | fixed(e79) | 80->94 | L2,L4 | C79CONF | ERT_CONFIRMATORY_T123_RESULTS.md |
| PILOT_S3_T1_WEAK_ADVCE | — | AdvCE(beta=0.118) @S3T1(I100) /fixed(e99) | fixed(e99) | 100->114 | dev-1,dev-2 | I100_CONTROL | ERT_RSLAD_I100_ACTION_TRANSFER_SCREEN.md |
| ST2M | — | AdvCE(beta=0.142) @S3T2 /fixed(e79) | fixed(e79) | 79->84 | L2,L4 | C79 | ERT_STAGE_A_TREATMENT_RESULTS.md |
| ST2W | — | AdvCE(beta=0.0710) @S3T2 /fixed(e79) | fixed(e79) | 79->84 | L2,L4 | C79 | ERT_STAGE_A_TREATMENT_RESULTS.md |
| T2WCONF | — | AdvCE(beta=0.075) @S3T2 /fixed(e79) | fixed(e79) | 80->94 | L2,L4 | C79CONF | ERT_CONFIRMATORY_T123_RESULTS.md |
| INST075 | — | AdvCE(beta=0.075) @S3nonCW.Tcorrect(instant) /online | online(from e80) | 79->94 | L2,L4 | BASE(S3hist) | ERT_S3_HISTORY_PRODUCTION_RESULTS.md |
| M3E2_075 | M3E2 | AdvCE(beta=0.075) @S3nonCW.Tcorrect(majority3+exit2) /online | online | 79->94 | L2,L4 | BASE(S3hist) | ERT_S3_HISTORY_PRODUCTION_RESULTS.md |
| M3_075 | — | AdvCE(beta=0.075) @S3nonCW.Tcorrect(majority3) /online | online | 79->94 | L2,L4 | BASE(S3hist) | ERT_S3_HISTORY_PRODUCTION_RESULTS.md |
| S3DYN075 | — | AdvCE(beta=0.075) @S3nonCW.Tcorrect /online | online | 80->94 | L2,L4 | DYNBASE | ERT_DYNAMIC_S3_RECOVERY_RESULTS.md |
| S3FIX075 | — | AdvCE(beta=0.075) @S3nonCW.Tcorrect /fixed(e80) | fixed(e80) | 80->94 | L2,L4 | DYNBASE | ERT_DYNAMIC_S3_RECOVERY_RESULTS.md |
| C12 | — | AdvCE-BCE(beta_bce=0.0889,bce=True) @CW /fixed(e79) | fixed(e79) | 79->84 | L2,L4 | C0 | ERT_CLEAN_WRONG_BROAD_SCREEN_RESULTS.md |
| C13 | — | AdvCE-structural(adaptive_pressure=True) @CW /fixed(e79) | fixed(e79) | 79->84 | L2,L4 | C0 | ERT_CLEAN_WRONG_BROAD_SCREEN_RESULTS.md |
| C14 | — | AdvCE-structural(teacher_gate=True) @CW /fixed(e79) | fixed(e79) | 79->84 | L2,L4 | C0 | ERT_CLEAN_WRONG_BROAD_SCREEN_RESULTS.md |
| C15 | — | AdvCE-structural(iad_inspired=True) @CW /fixed(e79) | fixed(e79) | 79->84 | L2,L4 | C0 | ERT_CLEAN_WRONG_BROAD_SCREEN_RESULTS.md |
| PF_RET_H | — | AnchorRetain(teacher=0.75,anchor=0.25) @PV3.PF.Hist /fixed(e79) | fixed(e79) | 80->129 | seed1(L1),seed2(L3) | PF_RET_R | src/ard/config/schema.py |
| PF_RET_R | — | AnchorRetain(teacher=0.75,anchor=0.25) @PV3.PF.Rand /fixed(e79) | fixed(e79) | 80->129 | seed1(L1),seed2(L3) | PF_RET_H | src/ard/config/schema.py |
| BASE | BASE(aug) | Aug(canonical) @all /none | none | 0->199 | 1,2 | CROPSHIFT | ERT_RSLAD_STATIC_TRAJECTORY_STABILIZATION.md |
| BASE | BASE(unseen) | Aug(canonical) @all /none | none | 0->199 | confirm-a,b,c | (is the reference) | ERT_RSLAD_UNSEEN_CONFIRMATION_RESULTS.md |
| CROPSHIFT | — | Aug(CropShift) @all /none | none | 0->199 | 1,2 | BASE(aug) | ERT_RSLAD_STATIC_TRAJECTORY_STABILIZATION.md |
| CROP_PREFIX | — | Aug(CropShift@e0-99,prefix-only) @all /none | none | 0->199 | confirm-a,b,c | (diagnostic only) | ERT_RSLAD_UNSEEN_CONFIRMATION_RESULTS.md |
| CROP_RE | — | Aug(CropRE) @all /none | none | 0->199 | 1,2 | CROPSHIFT | ERT_RSLAD_STATIC_AUGMENTATION_FAMILY.md |
| CROP_SUFFIX | — | Aug(CropShift->CROP_RE@e100) @all /none | none | 0->199 | confirm-a,b,c | BASE(unseen) | ERT_RSLAD_UNSEEN_CONFIRMATION_RESULTS.md |
| I0 | — | Aug(IDBH_WEAK@e0) @all /none | none | 0->199 | 1,2 | I100 | ERT_RSLAD_SINGLE_SWITCH_TIMING.md |
| I100 | — | Aug(CropShift->IDBH_WEAK@e100) @all /none | none | 0->199 | 1,2 | CROPSHIFT | ERT_RSLAD_STAGEWISE_AUGMENTATION.md |
| I100_SUFFIX | — | Aug(CropShift->IDBH_WEAK@e100) @all /none | none | 0->199 | confirm-a,b,c | CROP_SUFFIX | ERT_RSLAD_UNSEEN_CONFIRMATION_RESULTS.md |
| I125 | — | Aug(CropShift->IDBH_WEAK@e125) @all /none | none | 0->199 | 1,2 | I100 | ERT_RSLAD_SINGLE_SWITCH_TIMING.md |
| I150 | — | Aug(CropShift->IDBH_WEAK@e150) @all /none | none | 0->199 | 1,2 | CROPSHIFT | ERT_RSLAD_STAGEWISE_AUGMENTATION.md |
| I50 | — | Aug(CropShift->IDBH_WEAK@e50) @all /none | none | 0->199 | 1,2 | I100 | ERT_RSLAD_SINGLE_SWITCH_TIMING.md |
| I75 | — | Aug(CropShift->IDBH_WEAK@e75) @all /none | none | 0->199 | 1,2 | I100 | ERT_RSLAD_SINGLE_SWITCH_TIMING.md |
| IDBH_WEAK | — | Aug(IDBH_WEAK) @all /none | none | 0->199 | 1,2 | CROPSHIFT | ERT_RSLAD_STATIC_AUGMENTATION_FAMILY.md |
| R100 | — | Aug(CropShift->CROP_RE@e100) @all /none | none | 0->199 | 1,2 | CROPSHIFT | ERT_RSLAD_STAGEWISE_AUGMENTATION.md |
| R150 | — | Aug(CropShift->CROP_RE@e150) @all /none | none | 0->199 | 1,2 | CROPSHIFT | ERT_RSLAD_STAGEWISE_AUGMENTATION.md |
| D-BDD | DBDD | BD(coef=31.65) @S2T1 /fixed(e99) | fixed(e99) | 100->114 | dev-1,dev-2 | I100_CONTROL | ERT_RSLAD_I100_S2_DYNAMIC_BDD_RECOVERY_RESULTS.md |
| OS_DBDP | DBDP, OS-DBDP, D-BDP | BD(coef=31.65) @S2T1 /online | online(from e100) | 100->114 | dev-1,dev-2 | I100_CONTROL | ERT_RSLAD_I100_ONLINE_STATE_S2_PRESERVATION.md |
| A1 | — | CleanCE(beta=0.150) @CW /fixed(e79) | fixed(e79) | 79->84/89/94 | L2,L4 | A0 | ERT_CW_MARGIN_GENERALIZATION_SCREEN.md |
| C10 | — | CleanCE(beta=0.15) @CW /fixed(e79) | fixed(e79) | 79->84 | L2,L4 | C0 | ERT_CLEAN_WRONG_BROAD_SCREEN_RESULTS.md |
| C4 | — | CleanCE(beta=0.075) @CW /fixed(e79) | fixed(e79) | 79->84 | L2,L4 | C0 | ERT_CLEAN_WRONG_BROAD_SCREEN_RESULTS.md |
| C5 | — | CleanCE(beta=0.075,eps=4/255) @CW /fixed(e79) | fixed(e79) | 79->84 | L2,L4 | C0 | ERT_CLEAN_WRONG_BROAD_SCREEN_RESULTS.md |
| CW1 | — | CleanCE(beta=0.078) @CW /fixed(e79) | fixed(e79) | 79->84 | L2,L4 | C79 | ERT_STAGE_A_TREATMENT_RESULTS.md |
| CW2 | — | CleanCE(?) @CW /fixed(e79) | fixed(e79) | 79->84 | L2,L4 | C79 | ERT_STAGE_A_TREATMENT_RESULTS.md |
| CW3 | — | CleanCE(?) @CW /fixed(e79) | fixed(e79) | 79->84 | L2,L4 | C79 | ERT_STAGE_A_TREATMENT_RESULTS.md |
| F1 | =A1 (reused) | CleanCE(beta=0.150) @CW /fixed(e79) | fixed(e79) | 79->84/89/94 | L2,L4 | F0 | ERT_CW_A7_CLEANCE_ABLATION.md |
| G1_CW_ALL_CE015 | G1 | CleanCE(beta=0.150) @CW /fixed(e79) | fixed(e79) | 79->84/89/94 | L2,L4 | G0_BASE | ERT_CW_RELIABILITY_GATED_CE015_RESULTS.md |
| G2_CW_R_CE20_CE015 | G2 | CleanCE(beta=0.150) @CW.CE20reliable /fixed(e79) | fixed(e79) | 79->84/89/94 | L2,L4 | G0_BASE | ERT_CW_RELIABILITY_GATED_CE015_RESULTS.md |
| G3_CW_R_KL10_CE015 | G3 | CleanCE(beta=0.150) @CW.KL10reliable /fixed(e79) | fixed(e79) | 79->84/89/94 | L2,L4 | G0_BASE | ERT_CW_RELIABILITY_GATED_CE015_RESULTS.md |
| A3 | — | CleanCE+AdvCE(0.150,beta=0.0773) @CW /fixed(e79) | fixed(e79) | 79->84/89/94 | L2,L4 | A0 | ERT_CW_MARGIN_GENERALIZATION_SCREEN.md |
| C6 | — | CleanCE+KDScale(0.075,mult=0.5) @CW /fixed(e79) | fixed(e79) | 79->84 | L2,L4 | C0 | ERT_CLEAN_WRONG_BROAD_SCREEN_RESULTS.md |
| C7 | — | CleanCE+KDScale(0.075,mult=0.5,eps=4/255) @CW /fixed(e79) | fixed(e79) | 79->84 | L2,L4 | C0 | ERT_CLEAN_WRONG_BROAD_SCREEN_RESULTS.md |
| A5 | — | CleanCE+MF(0.150,lambda=0.239) @CW /fixed(e79) | fixed(e79) | 79->84/89/94 | L2,L4 | A0 | ERT_CW_MARGIN_GENERALIZATION_SCREEN.md |
| F3 | — | CleanCE+TPFM(0.150,lambda=0.239) @CW /fixed(e79) | fixed(e79) | 79->84/89/94 | L2,L4 | F0 | ERT_CW_A7_CLEANCE_ABLATION.md |
| C0 | — | Control @CW /fixed(e79) | fixed(e79) | 79->84 | L2,L4 | (is the comparator) | ERT_CLEAN_WRONG_BROAD_SCREEN_RESULTS.md |
| C1 | — | Control(eps=4/255) @CW /fixed(e79) | fixed(e79) | 79->84 | L2,L4 | C0 | ERT_CLEAN_WRONG_BROAD_SCREEN_RESULTS.md |
| C8 | — | Control(eps=2/255) @CW /fixed(e79) | fixed(e79) | 79->84 | L2,L4 | C0 | ERT_CLEAN_WRONG_BROAD_SCREEN_RESULTS.md |
| DYNBASE | — | Control @S3nonCW.Tcorrect /online | online | 80->94 | L2,L4 | (is the comparator) | ERT_DYNAMIC_S3_RECOVERY_RESULTS.md |
| A0 | BASE(CW-margin) | Control @all /fixed(e79) | none | 79->84/89/94 | L2,L4 | (is the comparator) | ERT_CW_MARGIN_GENERALIZATION_SCREEN.md |
| B0_BASE | — | Control @all /fixed(e79) | none | 79->84/89/94 | L2-R1/R2,L4-R1/R2 | (is the comparator) | ERT_CW_MARGIN_LOCAL_LAMBDA_STABILITY.md |
| BASE | BASE(S3hist) | Control @all /fixed(e79) | none | 79->94 | L2,L4 | (is the comparator) | ERT_S3_HISTORY_PRODUCTION_RESULTS.md |
| C | — | Control @all /fixed(e99) | none | 99->? | n/a | (is the comparator) | src/ard/config/schema.py |
| C79 | C | Control @all /fixed(e79) | none | 79->84 | L2,L4 | (is the comparator) | ERT_STAGE_A_TREATMENT_RESULTS.md |
| C79CONF | — | Control @all /fixed(e79) | none | 80->94 | L2,L4 | (is the comparator) | ERT_CONFIRMATORY_T123_RESULTS.md |
| F0 | =A0 (reused) | Control @all /fixed(e79) | none | 79->84/89/94 | L2,L4 | (is the comparator) | ERT_CW_A7_CLEANCE_ABLATION.md |
| G0_BASE | G0 | Control @all /fixed(e79) | none | 79->84/89/94 | L2,L4 | (is the comparator) | ERT_CW_RELIABILITY_GATED_CE015_RESULTS.md |
| I100_CONTROL | CONTROL | Control @all /fixed(e99) | none | 100->114 | dev-1,dev-2 | (comparator for SBF,TPFM,DPM,D-BDD,S-BDD,OS_PMP,OS_DBDP) | ERT_RSLAD_I100_CANONICAL_S2_ROBUST_BOUNDARY_PRESERVATION.md |
| I100_CONTROL | CONTROL | Control @all /fixed(e99) | none | 100->114 | dev-1,dev-2 | (comparator for PILOT/PLAIN_ADVCE/A7_MARGIN_ONLY) | ERT_RSLAD_I100_ACTION_TRANSFER_SCREEN.md |
| L0_BASE | =A0 (reused) | Control @all /fixed(e79) | none | 79->84/89/94 | L2,L4 | (is the comparator) | ERT_CW_MARGIN_LAMBDA_SENSITIVITY.md |
| C11 | — | KDScale(mult=1.5) @CW /fixed(e79) | fixed(e79) | 79->84 | L2,L4 | C0 | ERT_CLEAN_WRONG_BROAD_SCREEN_RESULTS.md |
| C2 | — | KDScale(mult=0.5) @CW /fixed(e79) | fixed(e79) | 79->84 | L2,L4 | C0 | ERT_CLEAN_WRONG_BROAD_SCREEN_RESULTS.md |
| C3 | — | KDScale(mult=0.5,eps=4/255) @CW /fixed(e79) | fixed(e79) | 79->84 | L2,L4 | C0 | ERT_CLEAN_WRONG_BROAD_SCREEN_RESULTS.md |
| C9 | — | KDScale(mult=0.25) @CW /fixed(e79) | fixed(e79) | 79->84 | L2,L4 | C0 | ERT_CLEAN_WRONG_BROAD_SCREEN_RESULTS.md |
| HD | — | KDScale(mult=0.500) @HistSel99 /fixed(e99) | fixed(e99) | 99->? | n/a | RD | src/ard/config/schema.py |
| RD | — | KDScale(mult=0.500) @RandSel99 /fixed(e99) | fixed(e99) | 99->? | n/a | HD | src/ard/config/schema.py |
| ST3K0 | — | KDScale(mult=0.0) @S3T3 /fixed(e79) | fixed(e79) | 79->84 | L2,L4 | C79 | ERT_STAGE_A_TREATMENT_RESULTS.md |
| ST3K05 | — | KDScale(mult=0.5) @S3T3 /fixed(e79) | fixed(e79) | 79->84 | L2,L4 | C79 | ERT_STAGE_A_TREATMENT_RESULTS.md |
| ST3K1 | — | KDScale(mult=1.0) @S3T3 /fixed(e79) | fixed(e79) | 79->84 | L2,L4 | C79 | ERT_STAGE_A_TREATMENT_RESULTS.md |
| T3LP05CONF | — | KDScale(mult=0.5) @S3T3 /fixed(e79) | fixed(e79) | 80->94 | L2,L4 | C79CONF | ERT_CONFIRMATORY_T123_RESULTS.md |
| A4 | — | MF(mode=fixed,lambda=0.239) @CW /fixed(e79) | fixed(e79) | 79->84/89/94 | L2,L4 | A0 | ERT_CW_MARGIN_GENERALIZATION_SCREEN.md |
| A6 | — | MF(mode=teacher_zero,lambda=0.239) @CW /fixed(e79) | fixed(e79) | 79->84/89/94 | L2,L4 | A0 | ERT_CW_MARGIN_GENERALIZATION_SCREEN.md |
| SBF | — | MF(coef=0.236) @S2T1 /fixed(e99) | fixed(e99) | 100->114 | dev-1,dev-2 | I100_CONTROL | ERT_RSLAD_I100_CANONICAL_S2_ROBUST_BOUNDARY_PRESERVATION.md |
| CONTROL(order-v1,blocked) | CONTROL | Order(epoch_shuffle) @all /none (BLOCKED) | none | never run | dev-1,dev-2 | HISTORY_BALANCED | ERT_RSLAD_HISTORY_BALANCED_ORDERING_DEV.md |
| HISTORY_BALANCED | — | Order(history_balanced,v1) @all /none (BLOCKED) | none | never run | dev-1,dev-2 | CONTROL(order-v1,blocked) | ERT_RSLAD_HISTORY_BALANCED_ORDERING_DEV.md |
| NEW_CONTROL | — | Order(epoch_shuffle) @all /none | none | 100->199 | 1,2 | NEW_HISTORY | ERT_RSLAD_HISTORY_BALANCED_ORDERING_DEV_V2.md |
| NEW_HISTORY | — | Order(history_balanced,v2) @all /none | none | 100->199 | 1,2 | NEW_CONTROL | ERT_RSLAD_HISTORY_BALANCED_ORDERING_DEV_V2.md |
| SHUFFLE_PLUS_0..7 | (8 schedules) | Order(pure-order-probe,8 schedules) @all /none | none | 100->114 | dev-1,dev-2 | cross-schedule | ERT_RSLAD_ORDERING_MECHANISM_DISCOVERY.md |
| NR_PFX_H | — | PGDPrefix(step=5/10) @PV3.NR.Hist /fixed(e79) | fixed(e79) | 80->99 | seed1(L1),seed2(L3) | NR_PFX_R | src/ard/config/schema.py |
| NR_PFX_R | — | PGDPrefix(step=5/10) @PV3.NR.Rand /fixed(e79) | fixed(e79) | 80->99 | seed1(L1),seed2(L3) | NR_PFX_H | src/ard/config/schema.py |
| DPM | — | PM(lambda=0.054) @S2T1 /fixed(e99) | fixed(e99) | 100->114 | dev-1,dev-2 | I100_CONTROL | ERT_RSLAD_I100_S2_DYNAMIC_BDD_RECOVERY_RESULTS.md |
| OS_PMP | PMP, OS-PMP | PM(lambda=0.054) @S2T1 /online | online(from e100) | 100->114 | dev-1,dev-2 | I100_CONTROL | ERT_RSLAD_I100_ONLINE_STATE_S2_PRESERVATION.md |
| S-BDD | — | SBD(coef=1.522) @S2T1 /fixed(e99) | fixed(e99) | 100->114 | dev-1,dev-2 | I100_CONTROL | ERT_RSLAD_I100_S2_DYNAMIC_BDD_RECOVERY_RESULTS.md |
| A100 | — | TPFM(lambda=0.239) @CW /fixed(e79) | fixed(e79) | 79->84/89/94 | L2-R1/R2,L4-R1/R2 | B0_BASE | ERT_CW_MARGIN_LOCAL_LAMBDA_STABILITY.md |
| A7 | CW-TPFM, "A7 margin-only" | TPFM(lambda=0.239) @CW /fixed(e79) | fixed(e79) | 79->84/89/94 | L2,L4 | A0 | ERT_CW_MARGIN_GENERALIZATION_SCREEN.md |
| F2 | =A7 (reused) | TPFM(lambda=0.239) @CW /fixed(e79) | fixed(e79) | 79->84/89/94 | L2,L4 | F0 | ERT_CW_A7_CLEANCE_ABLATION.md |
| L1_010 | — | TPFM(lambda=0.100) @CW /fixed(e79) | fixed(e79) | 79->84/89/94 | L2,L4 | L0_BASE | ERT_CW_MARGIN_LAMBDA_SENSITIVITY.md |
| L2_CAL | =A7 (reused) | TPFM(lambda=0.239) @CW /fixed(e79) | fixed(e79) | 79->84/89/94 | L2,L4 | L0_BASE | ERT_CW_MARGIN_LAMBDA_SENSITIVITY.md |
| L3_025 | — | TPFM(lambda=0.250) @CW /fixed(e79) | fixed(e79) | 79->84/89/94 | L2,L4 | L0_BASE | ERT_CW_MARGIN_LAMBDA_SENSITIVITY.md |
| L4_050 | — | TPFM(lambda=0.500) @CW /fixed(e79) | fixed(e79) | 79->84/89/94 | L2,L4 | L0_BASE | ERT_CW_MARGIN_LAMBDA_SENSITIVITY.md |
| N105 | — | TPFM(lambda=0.251) @CW /fixed(e79) | fixed(e79) | 79->84/89/94 | L2-R1/R2,L4-R1/R2 | B0_BASE | ERT_CW_MARGIN_LOCAL_LAMBDA_STABILITY.md |
| N110 | — | TPFM(lambda=0.263) @CW /fixed(e79) | fixed(e79) | 79->84/89/94 | L2-R1/R2,L4-R1/R2 | B0_BASE | ERT_CW_MARGIN_LOCAL_LAMBDA_STABILITY.md |
| N90 | — | TPFM(lambda=0.215) @CW /fixed(e79) | fixed(e79) | 79->84/89/94 | L2-R1/R2,L4-R1/R2 | B0_BASE | ERT_CW_MARGIN_LOCAL_LAMBDA_STABILITY.md |
| N95 | — | TPFM(lambda=0.227) @CW /fixed(e79) | fixed(e79) | 79->84/89/94 | L2-R1/R2,L4-R1/R2 | B0_BASE | ERT_CW_MARGIN_LOCAL_LAMBDA_STABILITY.md |
| CLEAN_WRONG_A7_MARGIN_ONLY | TPFM(action-transfer) | TPFM(coef=0.316) @CW(I100) /fixed(e99) | fixed(e99) | 100->114 | dev-1,dev-2 | I100_CONTROL | ERT_RSLAD_I100_ACTION_TRANSFER_SCREEN.md |
| TPFM | I100 canonical TPFM | TPFM(coef=0.167) @S2T1 /fixed(e99) | fixed(e99) | 100->114 | dev-1,dev-2 | I100_CONTROL | ERT_RSLAD_I100_CANONICAL_S2_ROBUST_BOUNDARY_PRESERVATION.md |
| A8 | — | TPFM-abstain(lambda=0.239) @CW /fixed(e79) | fixed(e79) | 79->84/89/94 | L2,L4 | A0 | ERT_CW_MARGIN_GENERALIZATION_SCREEN.md |
| HS | — | TargetSoften(rho=0.500) @HistSel99 /fixed(e99) | fixed(e99) | 99->? | n/a | RS | src/ard/config/schema.py |
| NR_TA | — | TargetSoften(mix=0.5/0.5) @OnlineHistSel39.NR /fixed(e39) | fixed(e39) | 40->199 | seed1,seed2 | NR_R, C | HISTORY_ROUTING_V2_RESULTS.md |
| PF_TA | — | TargetSoften(mix=0.5/0.5) @OnlineHistSel39.PF /fixed(e39) | fixed(e39) | 40->199 | seed1,seed2 | PF_R, C | HISTORY_ROUTING_V2_RESULTS.md |
| NR_R | — | TargetSoften(mix=0.5/0.5) @RandSel39.NR /fixed(e39) | fixed(e39) | 40->199 | seed1,seed2 | NR_TA | HISTORY_ROUTING_V2_RESULTS.md |
| PF_R | — | TargetSoften(mix=0.5/0.5) @RandSel39.PF /fixed(e39) | fixed(e39) | 40->199 | seed1,seed2 | PF_TA | HISTORY_ROUTING_V2_RESULTS.md |
| RS | — | TargetSoften(rho=0.500) @RandSel99 /fixed(e99) | fixed(e99) | 99->? | n/a | HS | src/ard/config/schema.py |
| ST1S | — | TargetSoften(tau=2.0,alpha=1.252) @S3T1 /fixed(e79) | fixed(e79) | 79->84 | L2,L4 | C79 | ERT_STAGE_A_TREATMENT_RESULTS.md |
| ST2S | ST2S-rerun | TargetSoften(tau=2.0,alpha=1.252) @S3T2 /fixed(e79) | fixed(e79) | 79->84 | L2,L4 | C79 | ERT_STAGE_A_TREATMENT_RESULTS.md |

Notes on reading this table:
- `?` in a canonical form means the source document does not state the coefficient (CW2,
  CW3, `CLEAN_WRONG_PLAIN_ADVCE`).
- `n/a` for seeds means the arm exists only in code, with no results document in the
  sources swept for this registry.
- Rows with the same `Name` and different `Aliases` (three `BASE` rows, two `I100_CONTROL`
  rows) are genuinely different documents using the same literal name; the aliases column
  and each JSON entry's `notes` field say whether they are the same underlying run or not.

## Collisions

This is the point of the exercise. Two arms "collide" if they share the same loss family
and the same cohort — meaning, on paper, they are the same kind of treatment applied to
the same kind of samples. When a collision group's members also share the same coefficient
and a similar horizon, they are very likely the same treatment tested more than once under
different names, and any two verdicts about them should be read as one piece of evidence,
not two independent ones.

### Confirmed near-duplicates (the headline finding)

**ST1W vs. S3FIX075.** These are the closest look-alike pair that is *not* an exact
duplicate, and the one this task was seeded with.

| | ST1W | S3FIX075 |
|---|---|---|
| Loss | AdvCE, beta=0.0710 | AdvCE, beta=0.075 |
| Cohort | S3 ∩ Teacher-T1 | S3 ∩ Teacher-adversarial-correct |
| Anchor | epoch 79 | epoch 80 |
| Horizon | 79->84 | 80->94 |
| Held-out L2 | +1.500 pp, CI [+0.680, +2.320] | +1.82 pp (e94) |
| Held-out L4 | +0.040 pp, CI [-0.760, +0.860] | +0.62 pp (e94) |
| Comparator | C79 | DYNBASE |

Both are positive on both seeds at their final horizon. **The cohorts are close but not
identical.** S3 is defined the same way in both ("student clean-correct and adversarial-
wrong"). The difference is the Teacher-side condition: ST1W's Teacher-T1 additionally
excludes the bottom decile by Teacher margin ("Teacher-T2" in the Stage-A partition) —
those weakest-margin teacher-correct samples are routed to a separate arm (ST2W) instead.
S3FIX075's "Teacher adversarial correct" keeps that bottom decile in. So S3FIX075's cohort
is the strict superset T1∪T2, anchored one epoch later (80, not 79). They are two
overlapping-but-different screens of essentially the same idea — add a small adversarial
CE term on top of RSLAD, restricted to samples the Student gets wrong on the attack but
gets right when clean and the Teacher still gets right — run under two names in two plans
(Stage A, and the later Dynamic-S3 screen) that never cross-reference each other's numbers.

**Its near-twin: ST1W and T1WCONF.** The Confirmatory-T123 campaign re-ran the same
treatment on the same cohort (S3×T1, n=9,889/9,368) from the same epoch-79 parent, with
byte-identical selected-ID masks, extended to epoch 94 and relabeled.
`docs/archive/ard-distillation-2026/ERT_RSLAD_HISTORICAL_TREATMENT_RESPONSE_ANALYSIS.md` calls T1WCONF a re-run of ST1W.

The coefficient is *not* identical, and the difference matters. Stage A used the calibrated
`beta_advce_weak = 0.07095924764871597`
(`docs/ERT_STAGE_A_TREATMENT_RESULTS.md:35`); Confirmatory used "the frozen **rounded**
`beta_advce=0.075`" (`docs/ERT_CONFIRMATORY_T123_RESULTS.md:127`), which is +5.7%. So the
pair is the project's only executed test of what happens when a calibrated coefficient is
rounded for reporting. The held-out robust effect on L2 went from `+1.500 pp`
(sample-bootstrap CI `[+0.680,+2.320]`) to `-0.14 pp`; L4 was near zero under both. The sign
changed — but the change is inside the measured floor for that regime (about 1.14 pp, see
`docs/archive/ard-distillation-2026/MEASUREMENT_DESIGN.md`), so it cannot be attributed to the rounding rather than to
run-to-run variation. The correct reading is that this pair establishes neither that the
treatment works nor that rounding is safe.

The same relabeling repeats twice more in the same campaign: **ST2W / T2WCONF** (S3×T2) and
**ST3K05 = T3LP05CONF** (S3×T3, AdvKD multiplier 0.5, "LP05" = low-pressure 0.5; this one
*is* an exact coefficient match). Each pair should be scored as one line of evidence, not two.

**S3DYN075 ≈ INST075.** `S3DYN075`'s online cohort rule ("recompute the predicate every
visit") is described in `docs/archive/ard-distillation-2026/ERT_S3_HISTORY_PRODUCTION_RESULTS.md` as the "Instant" rule
— the same online recomputation, same predicate, same beta=0.075 — just given a new name
(`INST075`) in the follow-on smoothing screen that also introduced `M3_075` and `M3E2_075`
(majority-of-3-visits smoothing, with and without exit hysteresis). All five arms in this
row of the main table (S3FIX075, S3DYN075, INST075, M3_075, M3E2_075) share loss family
AdvCE and cohort `S3nonCW.Tcorrect`; they differ only in how the online/fixed decision is
smoothed, and none beats its own control at the final horizon in both seeds.

**A1 = C10 = G1_CW_ALL_CE015 = F1.** The single largest duplicate cluster: "add a clean-
image cross-entropy term with coefficient 0.15 to every sample in the fixed Clean-Wrong
cohort, anchored at epoch 79" was independently run and separately reported under four
names, from four different plans:

| Name | Plan / doc | Horizon | Held-out robust Δ vs. its own control (e94) |
|---|---|---|---|
| A1 | 0050, CW-margin screen | 79->84/89/94 | L2 -1.64 pp, L4 -0.64 pp |
| C10 | 0044, CW broad screen | 79->84 only | (e84 only: L2 +0.04 pp, L4 -0.56 pp) |
| G1_CW_ALL_CE015 | 0047, reliability-gated screen | 79->84/89/94 | L2 -1.64 pp, L4 -0.64 pp |
| F1 | 0052, CleanCE ablation | 79->84/89/94 | (reused from A1, not retrained) |

A1 and G1's e94 numbers match exactly (both -1.64/-0.64 pp), which is the strongest
possible evidence they are the same trained run reused, not merely the same recipe run
twice. F1 is explicitly documented as "= A1 (reused)". C10 only has an e84 endpoint, so
its number cannot be checked against the others at the same horizon, but the recipe,
coefficient, and cohort all match.

A weaker, non-identical near-duplicate sits one level down in coefficient: **CW1**
(Stage A, beta=0.07825280725955963) and **C4**/**C5** (broad screen, beta=0.075) apply the
same CleanCE-only mechanism to the same Clean-Wrong cohort at nearly the same coefficient —
close enough that they were very likely calibrated to hit the same target and should not
be treated as three independent confirmations of "CleanCE ~0.075-0.078 doesn't help."

**DPM = OS_PMP, and D-BDD = OS_DBDP.** This is the pair the task's own worked examples
already point at, and it recurs at the coefficient level: DPM's pair-margin coefficient
(0.05380932585058825) and D-BDD's boundary-distance coefficient (31.649566509850324) are
*exactly* the coefficients reused, unchanged, by OS_PMP and OS_DBDP respectively. Same
mechanism (`PM`, `BD`), same target cohort (`S2T1`), same anchor lineage (both fork from
byte-identical epoch-99 parents) — the only real difference is `cohort_rule`: DPM/D-BDD
freeze `S2T1` membership at epoch 99 and never touch it again; OS_PMP/OS_DBDP recompute
`S2T1` membership every step from a threshold frozen at a one-epoch-later e100 prefix. See
**Hazard 2** below — this pair is a direct instance of the "dynamic" ambiguity.

### Same acronym, different calibration, different cohort (not a strict collision, but a hazard)

`TPFM` (target-probability-floor-margin) is not merely reused as a mechanism — the exact
acronym is recalibrated from scratch three separate times, for three different cohorts, in
three different plans, and a reader skimming for "TPFM" without checking the cohort will
conflate them:

| Instance | Cohort | Anchor | lambda / coefficient | floor | cap |
|---|---|---|---|---|---|
| A7 (= CW-TPFM) | CW (Clean-Wrong), epoch 79 | e79 | 0.2388051152229309 | 0.03221710026264191 | 0.13952550292015926 |
| `CLEAN_WRONG_A7_MARGIN_ONLY` | CW, I100/epoch-99 lineage | e99 | 0.316427398202933 | 0.17963354289531708 | 0.5595575273036957 |
| `TPFM` (canonical S2 boundary screen) | S2T1, epoch 99 | e99 | 0.16676844691071563 | 0.16590790450572968 | 0.32364362478256226 |

These are not the same arm (different cohort, different lineage, independently
recalibrated numbers) but they are extremely easy to mis-cite as "TPFM's effect" without
naming which one. Every occurrence in this registry's canonical form includes the cohort
for exactly this reason — never cite a bare "TPFM result" without its `@COHORT`.

### Other collision groups found (same family, same cohort, distinct treatments)

These share a family+cohort bucket but differ enough in coefficient or exact mechanism
that they are legitimately separate arms, not duplicates — listed so the grouping is
visible, not to claim they are the same:

- **KDScale @ S3T3**: ST3K1 (mult=1.0), ST3K05 (mult=0.5, = T3LP05CONF), ST3K0 (mult=0.0)
  — a genuine dose sweep, not a duplicate.
- **KDScale @ CW**: C2 (0.5), C3 (0.5, reduced attack budget), C9 (0.25), C11 (1.5) — a
  dose/budget factorial from the same broad screen, all distinct cells.
- **TPFM @ CW** (11 members: A7/F2/L2_CAL/A100 all at lambda≈0.239; L1_010, L3_025,
  L4_050, N90, N95, N105, N110 at other lambdas): this is one deliberate lambda sweep
  spread across three plans (0050, 0053, 0054/local-stability); A7=F2=L2_CAL=A100 all
  hit the *same* lambda (0.2388051152229309) but A100 is stated explicitly to be a fresh
  replicate run on different hardware, not a byte-reuse of A7 — a useful example of
  "same treatment, same name pattern, genuinely re-run rather than reused."
- **MF @ CW**: A4 (fixed pooled-Q50 target) vs. A6 (teacher-conditioned, clipped to
  `[0,cap]`) — same family label, different target definition; flagged here because both
  use the word "fixed" for their target-computation mode in the source docs, which is a
  third, narrower use of "fixed" than the cohort-rule sense used everywhere else in this
  registry (see Hazard 2 for the other two senses of "dynamic"/"fixed").
- **Aug @ all** (16 members): one continuous design space (which post-switch augmentation
  policy, and at what epoch to switch to it), not 16 independent ideas. `I100_SUFFIX` is a
  confirmation-seed re-run of `I100` itself; `CROP_SUFFIX` is a confirmation-seed re-run of
  `R100`'s mechanism.
- **Order @ none**: `CONTROL(order-v1,blocked)`/`HISTORY_BALANCED` were never actually
  trained (blocked by an RNG confound) and were re-run, after a fix, as `NEW_CONTROL`/
  `NEW_HISTORY` — same intended arms, same names in spirit, different literal names before
  and after the fix.
- **Control @ none** (12 members): every family's zero-treatment baseline collapses into
  this bucket by construction; not a scientific collision, just confirmation that "no
  treatment" is reinvented under a new name in nearly every plan (`C`, `C79`, `C79CONF`,
  `A0`, `B0_BASE`, `F0`, `G0_BASE`, `L0_BASE`, three flavors of `BASE`, two identical-numeric
  copies of `I100_CONTROL`). One genuine finding inside this bucket: **`I100_CONTROL` in
  the canonical-boundary, dynamic-BDD, and online-state reports is not just the same
  recipe — it is the exact same run, byte-for-byte identical clean/robust numbers at every
  reported epoch and seed**, reused across three separately-written documents. That is
  correct practice, not a hazard, but it means a reader must not count it as three
  independent Control observations.

### Cohort predicates that look alike but are not

- **ST1W's `S3T1` vs. S3FIX075/S3DYN075's `S3nonCW.Tcorrect`** — covered above; the
  difference is whether the bottom Teacher-margin decile is excluded (T1 only) or included
  (all Teacher-correct).
- **`MF`'s two senses of "fixed" target**: SBF's target is the *Student's own* frozen
  epoch-99 margin quantile; A4's target is a *pooled constant* (Q50 across the cohort).
  Both call themselves a "fixed" margin target in their source docs; they are not
  computing the same number.
- **A8 is the one CW-margin arm that gates *within* its own cohort.** A1 through A7 apply
  their loss to every sample in the fixed Clean-Wrong mask, every step. A8 additionally
  requires `teacher_margin > 0` at that step — so on any given step A8's *effective* cohort
  is a moving subset of the nominally fixed CW mask, even though its `cohort_rule` is
  still recorded as `fixed` (the mask is fixed; the per-step activation is not).
- **`BASE` means three unrelated things.** The S3-history-smoothing screen's `BASE`
  (Chen-ERT epoch-79 lineage, AdvCE-based comparator family) and the augmentation family's
  `BASE` (I100 lineage, canonical-augmentation full run) and the unseen-confirmation
  bundle's `BASE` (also augmentation lineage, independent run) share a name and nothing
  else — different dataset lineages, different mechanisms being screened, different
  epochs. Do not merge robust-accuracy numbers across these three `BASE` rows.

## Two definitional hazards

### Hazard 1 — the legacy `ert_state_overlay_v1` states are not the canonical states

Two different three-way partitions of "how did this sample do" exist in this project and
use the *same* labels S1/S2/S3 for *different* things:

- **Legacy `ert_state_overlay_v1`** (used by the Stage-A pilot and older plans 0035/0036):
  S1 = adversarially correct; **S2 = adversarially wrong AND clean wrong**; **S3 = clean
  correct AND adversarially wrong**.
- **Canonical routing-proxy contract** (`ert_online_routing_proxy.py` v2, and everything
  downstream of it, including the I100 canonical-boundary/dynamic-BDD/online-state
  families): S1 = safe-correct; **S2 = fragile-correct** (correct on clean, a positive but
  small margin under attack — the "positive-margin q10" quantile); **S3 = wrong**.

These are genuinely different partitions, not a relabeling of the same thing — a sample
that is "S3" under the legacy pilot scheme (clean-correct, adversarially wrong) is exactly
this project's "Clean-Wrong-adjacent" S3 cohort used throughout Stage A, while "S2" or "S3"
under the canonical contract is defined from a margin quantile, not from a raw correct/
wrong split. `docs/archive/ard-distillation-2026/reviews/0001-ert-routing-contract-review.md` records this as an
*intentional*, documented divergence: legacy v1 masks and hashes are preserved as-is and
are never silently migrated to canonical labels; a new canonical overlay would need to be
a new, separately versioned artifact. **Practical rule for this registry:** every arm in
the "Stage A" and "FFNR causal pilot" sections above (S3, S3T1, S3T2, S3T3, CW) uses the
*legacy* partition; every arm in the "I100 canonical S2 boundary/dynamic-BDD/online-state"
sections (`S2T1`) uses the *canonical* partition. Do not compare an "S3" cohort from one
side against an "S2"/"S3" cohort from the other as if they were the same population.

### Hazard 2 — "dynamic" does not mean the same thing in plan 0079 and plan 0087

- **Plan 0079** (`DPM`/`D-BDD`/`S-BDD`): "dynamic" describes the *loss target*, not the
  cohort. The `S2T1` cohort is frozen at epoch 99 and never reselected ("no dynamic
  selector," stated explicitly in the plan). What is dynamic is that the margin/boundary-
  distance quantities feeding the loss are recomputed every step from the *current* (
  moving) Student and Teacher state, even though *which samples* receive the loss never
  changes after epoch 99. This registry's `cohort_rule` for these three arms is `fixed`.
- **Plan 0087** (`OS_PMP`/`OS_DBDP`): "dynamic"/"online" describes the *cohort itself*.
  `Online-S2×T1` membership is recomputed every training step from live margins against a
  per-seed threshold frozen once from a one-epoch treatment-free e100 prefix — a sample can
  enter and leave the treated set as training proceeds (the report records ~38,000 action
  switches over 14 epochs per seed for `OS_PMP`). This registry's `cohort_rule` for these
  two arms is `online`.
- **A third, consistent-with-0087 usage**: `S3DYN075` uses "DYN" to mean the same thing as
  plan 0087's "dynamic" — a recomputed cohort, not a recomputed target — so plan 0079 is
  the outlier, not the majority usage. A reader who assumes "dynamic X" always means
  "dynamic cohort" (0087's and `S3DYN075`'s sense) will misread plan 0079's arms, whose
  cohort was frozen the entire time.

**Practical rule for this registry:** never trust the word "dynamic" or "online" alone;
always check the `cohort_rule` field. `/fixed(eNN)` arms have a frozen cohort no matter
what their loss computes; `/online` arms have a moving cohort no matter how their name
reads.

## Coverage

Everything above was verified against a source document, a source JSON record, or
`src/ard/config/schema.py` — nothing in the main table or the collisions section was
invented. The following gaps are known and explicit rather than silently papered over:

- **CW2 and CW3** (Stage A): the exact coefficient distinguishing them from CW1 (and from
  each other) is never stated in the Stage-A documents read for this registry. Listed with
  `CleanCE(?)`.
- **`CLEAN_WRONG_PLAIN_ADVCE`** (I100 action-transfer screen): its AdvCE coefficient was
  not resolved from the source excerpt available; listed with `AdvCE(?)`.
- **Legacy H3 arms (`C`, `HS`, `RS`, `HD`, `RD`)**, **history-routing v2 exact per-seed
  counts for `PF_R`/`NR_R`**, and **`PrescriptiveV3Config` (`PF_RET_H/R`, `NR_PFX_H/R`)**:
  fully specified in `src/ard/config/schema.py`, but no results document with numeric
  outcomes was found among the sources swept. Their rows carry mechanism and parameters
  only, no headline result.
- **`SHUFFLE_PLUS_0` through `SHUFFLE_PLUS_7`** (pure-order mechanism probe) and the
  **shuffle-vs-augmentation RNG decomposition arms** (`REF1/REF2/SHUF1/SHUF2/AUG1/AUG2/
  BOTH1/BOTH2` in `docs/archive/ard-distillation-2026/ERT_RSLAD_SHUFFLE_AUGMENTATION_RESULTS.md`) are grouped/summarized
  rather than given one row each — they are RNG-mechanism diagnostics, not named
  treatments competing for promotion, and itemizing all sixteen individually would not
  change the collision analysis.
- **Not read at all in this pass** (named in the task's source list or discovered
  adjacent to it, but out of scope for time): `docs/archive/ard-distillation-2026/ERT_RSLAD_I100_CW_GAP_COMPLETION_AND_S2_BRIDGE.md`,
  `docs/archive/ard-distillation-2026/ERT_RSLAD_I100_CW_HELDOUT_GENERALIZATION_GAP.md`, `docs/archive/ard-distillation-2026/ERT_RSLAD_I100_CLEAN_WRONG_LONG_HORIZON.md`
  (a light skim by one research pass surfaced that these discuss `PLAIN_ADVCE` and `TPFM`
  continuing to epoch 199, already captured above by name, but their held-out e199 numbers
  are not independently re-verified here), `docs/archive/ard-distillation-2026/ERT_CLEAN_WRONG_RELIABILITY_PROXY_SAFETY.md`,
  `docs/archive/ard-distillation-2026/ERT_CLEAN_WRONG_RELIABILITY_STRATIFIED.md`, `docs/archive/ard-distillation-2026/ERT_CLEAN_WRONG_RESCUE_SUBTYPES.md`
  (these three are confirmed to be read-only re-analyses of already-listed arms — C0/C10/
  C12/C13 and the CW-R/CW-U reliability split — not new arms, but their internal quintile/
  subtype tables were not transcribed), and the runtime/RNG-audit-only documents under
  `docs/ERT_RSLAD_RUNTIME_*`, `docs/ERT_RSLAD_*_RNG_*`, and `docs/ERT_RSLAD_ORDER_RNG_AUDIT.md`
  (confirmed by name to contain no new named arms).
- **`docs/archive/ard-distillation-2026/essays/SIGNAL_AUDIT.md`**: skimmed; confirmed to report model/teacher configuration
  comparisons (`stored_risk_kind: student` vs. `joint`), not training arms, so nothing was
  added to this registry from it.
- Every arm's held-out result above is quoted as printed in its source document. Where a
  document reports only a training-time/selected-cohort effect and no held-out number
  (several `src/ard/config/schema.py`-only rows), the table says so rather than
  substituting a different horizon's number.

## Verification performed

- `python3 -m json.tool docs/experiments/ard_arm_registry_v1.json` — passes (valid JSON).
- Every `source_doc` path in the JSON was checked with `ls`/`os.path.exists`; all 118
  arms resolve to an existing file (a `docs/...md` report or `src/ard/config/schema.py`).
- Every arm has a non-empty `canonical` field.
- Arm count: **118** (more than the roughly-60-80 originally expected, because this sweep
  also surfaced confirmed duplicate-under-a-new-name arms — `T1WCONF`/`T2WCONF`/
  `T3LP05CONF`/`C79CONF`, the I100 action-transfer arms, and the ordering family's
  before/after-RNG-fix pairs — that were not in the original arm-name list but are
  necessary to complete the collisions section).
