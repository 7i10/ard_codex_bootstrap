# ERT Clean-Wrong Broad Treatment Screen

Status: completed only after all 32 arms and independent epoch-84 endpoints exist; no automatic promotion.

Direct is the fixed epoch-79 Clean-Wrong train cohort; spillover is the remaining train IDs;
held-out is the fixed internal validation split. Bootstrap intervals are sample uncertainty,
not training-seed uncertainty.

| seed | arm | direct robust Δ | spillover robust Δ | held-out robust Δ | held-out clean Δ |
|---|---|---:|---:|---:|---:|
| L2 | C0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| L2 | C1 | -0.000696 | -0.009264 | 0.000800 | -0.004600 |
| L2 | C2 | -0.007654 | -0.022542 | -0.015800 | 0.000800 |
| L2 | C3 | -0.002667 | -0.008989 | -0.005600 | 0.003200 |
| L2 | C4 | 0.004059 | -0.002227 | -0.003200 | 0.009400 |
| L2 | C5 | 0.005451 | -0.010611 | -0.005200 | 0.012200 |
| L2 | C6 | 0.001044 | -0.012013 | -0.011400 | -0.002600 |
| L2 | C7 | -0.008002 | -0.031641 | -0.020200 | 0.009200 |
| L2 | C8 | -0.001856 | -0.022074 | -0.012400 | 0.008600 |
| L2 | C9 | -0.007886 | -0.024741 | -0.017800 | -0.008200 |
| L2 | C10 | 0.013104 | -0.012178 | 0.000400 | 0.012400 |
| L2 | C11 | 0.006958 | -0.012041 | -0.006000 | 0.010800 |
| L2 | C12 | 0.010321 | -0.014322 | -0.001800 | 0.004000 |
| L2 | C13 | 0.005798 | -0.000605 | 0.003400 | 0.001000 |
| L2 | C14 | 0.000116 | -0.037634 | -0.027000 | -0.001600 |
| L2 | C15 | -0.006726 | -0.029277 | -0.015000 | -0.006600 |
| L4 | C0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| L4 | C1 | -0.008627 | -0.002661 | -0.008000 | 0.010400 |
| L4 | C2 | -0.009076 | -0.002717 | -0.003600 | 0.002200 |
| L4 | C3 | -0.014454 | -0.006071 | -0.008600 | 0.007000 |
| L4 | C4 | -0.000112 | -0.005655 | -0.006200 | -0.005200 |
| L4 | C5 | -0.003810 | -0.030049 | -0.021000 | 0.011800 |
| L4 | C6 | -0.009748 | -0.008205 | -0.010800 | 0.010000 |
| L4 | C7 | -0.009188 | -0.006376 | -0.002800 | 0.012600 |
| L4 | C8 | -0.005714 | -0.005267 | -0.006600 | 0.024800 |
| L4 | C9 | -0.013557 | -0.010478 | -0.008400 | -0.006000 |
| L4 | C10 | 0.005938 | -0.004213 | -0.005600 | 0.014200 |
| L4 | C11 | 0.006275 | -0.002744 | -0.005600 | -0.014400 |
| L4 | C12 | 0.009748 | -0.012585 | -0.008400 | -0.003400 |
| L4 | C13 | -0.001569 | -0.006764 | -0.005000 | 0.001400 |
| L4 | C14 | -0.005490 | -0.002134 | 0.000000 | -0.013400 |
| L4 | C15 | -0.008067 | -0.012086 | -0.010800 | -0.002200 |

No winner, coefficient, threshold, +15 continuation, official test, or AutoAttack was selected automatically.

## Frozen lineage and calibration

- Source SHA used by all valid trajectories: `cbe03a7b3be0b11fa1555b573c6f453a3d10f27b`.
- Endpoint: epoch 84, independent eval-mode CE-PGD20, pixel `[0,1]`,
  $\epsilon=8/255$, step $2/255$, 20 steps, random start, hard-label CE.
- Endpoint attack identity SHA:
  `7081101693340e70d24d522563f3c26bb935198a72865a5a8a26a5f305dcc4f2`.
- Fixed epoch-79 Clean-Wrong mask sizes: L2 8,623; L4 8,925.
- C12 BCE coefficient was calibrated without optimizer steps or endpoint
  metrics: $\beta_{BCE}=0.08891977369785309$, pooled median target gradient
  ratio 0.25. It was not retuned after results.
- C0 endpoint baselines: L2 train clean/robust `0.78998/0.50762`, validation
  `0.77820/0.47140`; L4 train `0.79222/0.51067`, validation
  `0.79020/0.46420`.
- Machine artifact includes all 16 arm definitions, mask hashes, endpoint
  checkpoint/row/attack hashes, parent lineage, and W&B run URLs.

## Interpretation (screen, not promotion)

No non-baseline arm improves held-out robust accuracy in both seeds. The
strongest apparent case is C13 (adaptive high-pressure): L2 `+0.34 pp` but L4
`-0.50 pp`. C1 (epsilon reduction) is L2 `+0.08 pp` and L4 `-0.80 pp`;
C14 (teacher-clean reliability gate) is L2 `-2.70 pp` and L4 `0.00 pp`.
Thus none is a two-seed robust-compatible recovery.

C10 (extra CleanCE `0.15`) gives direct clean changes of `+5.45 pp` (L2) and
`+7.71 pp` (L4), while held-out clean changes are `+1.24 pp` and `+1.42 pp`;
held-out robust changes are `+0.04 pp` and `-0.56 pp`. This is a clean/robust
trade-off, not evidence that CleanCE solves Clean-Wrong samples. C12 BCE
similarly improves direct clean (`+2.88/+3.76 pp`) but held-out robust is
`-0.18/-0.84 pp`. C9 (KD×0.25) and C15 (IAD-inspired detached self-target)
are harmful on held-out robust for both seeds (`-1.78/-0.84 pp` and
`-1.50/-1.08 pp`, respectively).

For C0–C7, average held-out robust contrasts (L2/L4 mean) are:

| contrast | mean change |
|---|---:|
| epsilon 8→4, KD=1, CleanCE=0 | `-0.36 pp` |
| epsilon 8→4, KD=1, CleanCE=.075 | `-0.84 pp` |
| epsilon 8→4, KD=.5, CleanCE=0 | `+0.26 pp` |
| epsilon 8→4, KD=.5, CleanCE=.075 | `-0.04 pp` |
| KD 1→.5, epsilon=8, CleanCE=0 | `-0.97 pp` |
| KD 1→.5, epsilon=4, CleanCE=0 | `-0.35 pp` |
| CleanCE 0→.075, epsilon=8, KD=1 | `-0.47 pp` |
| CleanCE 0→.075, epsilon=4, KD=1 | `-0.95 pp` |

These are descriptive two-seed contrasts, not population-level significance
tests. They do not support lowering AdvKD or reducing the selected attack
budget as a robust solution. C13 merits analysis as a hypothesis only; its
sign reversal blocks promotion.

## Hypothesis screen

| hypothesis | assessment | evidence |
|---|---|---|
| H1 attack is too difficult | unsupported as transferable | C1/C8 do not improve held-out robust in both seeds; C8 is `-1.24/-0.66 pp`. |
| H2 adversarial supervision is too strong | unsupported | KD×0.5 and KD×0.25 are negative or mixed on held-out robust. |
| H3 clean recovery signal is missing | mixed, clean/robust trade-off | C10 recovers clean accuracy but not robust accuracy in both seeds. |
| H4 hard samples need stronger supervision | unsupported | C11 and C12 do not produce held-out robust gains in both seeds. |
| H5 teacher reliability should gate treatment | unsupported in this fixed gate | C14 is `-2.70/0.00 pp` held-out robust and harms L2 clean. |

All valid trajectories were tracked as W&B online production runs. One
duplicate L4 C0 launch and one endpoint-directory precreation error were
stopped before valid evaluation; the L4 campaign was rerun in a unique
namespace and only the complete lineage is included. These are operational
events, not scientific results. No official test, AutoAttack, new seed, +15
continuation, dynamic routing, or automatic winner promotion was performed.
