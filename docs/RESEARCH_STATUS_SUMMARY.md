# Mobile-robustness / self-distillation research status summary

Status: evidence ledger for the post-pivot research direction (mobile-scale
robustness via teacher-free self-distillation, `docs/decisions/0009-mobile-robustness-direction-and-first-step.md`).
Started 2026-09-10, after the project moved away from the ARD/RSLAD-family
distillation direction. That closed direction's own status summary is frozen
at `docs/archive/ard-distillation-2026/ERT_RESEARCH_STATUS_SUMMARY.md` and is
not extended by new rows here or there -- this document is the one live
skills (`/experiment-postrun`, `/experiment-decide`, `/biweekly-review`)
append to going forward.

## Evidence ledger

| question | result | decision |
|---|---|---|
| Does teacher-free EMA self-distillation (ADR) help MobileNetV2 more than ResNet-18 on CIFAR-10 (decision packet 0009 option B)? | SIGN_CONFIRMED: MobileNetV2 mean AutoAttack gain +2.79pp (n=3) exceeds ResNet-18 Nesterov-matched mean gain +1.22pp (n=3), best checkpoint, model weights. EMA weights show the same ordering (+3.18pp vs +2.08pp). TRADES+ADR reaches 49.50% (EMA), matching/exceeding literature comparators (AdaAD 49.21, RSLAD 49.23, RSLAD-300 49.50) that plain TRADES (47.55%) could not. `trades_49k_validation` pilot: narrowing `validation_fraction` did not close TRADES' literature gap (47.64% vs 47.87% standard-split). | See `docs/decisions/0011-*.md` (packet to follow) for the ImageNet-track decision this licenses or withholds. |
| Plan 0103 Phase 0 2x2: for MobileNetV4-Conv-Small plain PGD-AT on ImageNet-1k, does a clean-pretrained init or a 100-epoch (vs 50) budget raise robustness? | Seed 0, internal validation (2% held-out train slice), PGD-10 l_inf 4/255, last checkpoint: pretrained/100 55.21% clean / 30.70% PGD-10; random-init/100 54.56 / 30.58; pretrained/50 (Arm A) 54.57 / 30.58 (seed 1: 54.33 / 30.59). Every difference is <=0.65pt clean and <=0.35pt PGD-10 (best or last), inside any credible noise floor. The pretrained head start is erased during the lr-0.05 phase, which is a plateau. n=1 per 100-epoch cell, no AutoAttack. Details: plan 0103 Progress log, 2026-09-25. | `docs/decisions/0019-*.md` (pending). |
