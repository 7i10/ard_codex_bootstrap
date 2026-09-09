# I100 Secant Boundary-Distance Forensic Audit

## Classification

**SBPF2 — FORMULA_SINGULARITY_SUPPORTED.**  The corrected v2 Student parameter graph agrees in sign and scale with the audited scalar and directional finite differences, so this audit does not retain evidence for a v2 implementation/normalization bug.  At the same time, real selected samples show a strongly tailed gradient-ratio distribution tied descriptively to the reciprocal secant geometry.  The historical v2 training became non-finite on both development seeds.  The frozen median calibration did not control that tail.  This classification describes the audited formula/parameterization; it does not select an epsilon, a cap, or a replacement method.

S-BDD: **NUMERICALLY_UNSUPPORTED** — corrected secant formulation became non-finite reproducibly in both dev seeds; excluded from causal utility comparison.

## Source-to-equation audit

| version | Student secant denominator | Student graph | Teacher terms | result |
| --- | --- | --- | --- | --- |
| v1 historical | $q_S=\lvert m_S^{adv}-m_S^{clean}\rvert/(\rho+\epsilon)$ | $q_S$ detached | detached | superseded; not the registered recovery intervention |
| v2 corrected | same $q_S$ | preserved; no detach | $m_T^{adv},m_T^{clean},q_T,d_T$ detached | audited here and used in both failed S-BDD recovery runs |

For v2, $d_S=m_S^{adv}/(q_S+\epsilon)$, $d_T=m_T^{adv}/(q_T+\epsilon)$, and $L=\tfrac12[\max(0,d_T-d_S)]^2$.  The selected mask, positive-Teacher gate, and $\rho$ are detached; the caller retains the full-batch mean and does not selected-count-normalize.
 The audited source and frozen v2 contract agree on the pair, clean/adversarial views, epsilon placement, detach ownership, gates, full-batch reduction, and a single coefficient application.

## No-update derivative and restoration checks

| seed | Teacher-pair-gated fixed-mask samples across 4 natural batches | scalar FD max abs/relative error | parameter FD best abs/relative error | best step | Student forward mode | state restored bitwise |
| --- | ---: | --- | --- | ---: | --- | --- |
| dev-1 | 19 | 0.000168 / 9.23e-06 | 4.85e-09 / 0.0029 (`model.linear.weight`) | 3e-04 | `train_with_full_state_restore` | true |
| dev-2 | 17 | 1.81e-06 / 5.67e-07 | 6.22e-06 / 0.0382 (`model.linear.weight`) | 1e-03 | `train_with_full_state_restore` | true |

Scalar partials are reported only after their own $m_S^{adv}$ or $m_S^{clean}$ abs/ReLU regions remain unchanged at both perturbations.  Parameter checks likewise require preserved Student-margin abs sign, hinge, Teacher-pair gate, and $\rho$ gate, and restore the full Student parameter/buffer state before every train-mode forward.  The finite-difference audit is an implementation check, not an optimizer update or a training replay.

## Real-checkpoint tail diagnostics at coefficient 1

| seed | $q_S$ min / median / max | $1/(q_S+\epsilon)$ p95 / max | ratio p50 / p95 / p99 / max | raw loss p50 / p99 / max | Spearman ratio vs $1/q_S$ |
| --- | --- | --- | --- | --- | ---: |
| dev-1 | 0.445 / 16.1 / 46.8 | 1.14 / 2.25 | 0.0118 / 12.1 / 98 / 119 | 0.00205 / 1.62 / 1.97 | 0.221 |
| dev-2 | 1.69 / 15.5 / 33.3 | 0.463 / 0.591 | 0.0423 / 16.3 / 21.4 / 22.7 | 0.00341 / 0.808 / 0.819 | 0.814 |

The observed $q_S$ values do not approach $\epsilon$ in these four-batch probes.  The supported instability claim is therefore narrower: the reciprocal secant parameterization is highly sensitive to small clean–adversarial Student-margin differences, producing a heavy intervention-gradient tail even away from the literal epsilon limit.

## Epsilon diagnostic only

This scalar sensitivity holds the recorded real-batch values fixed.  It is not a coefficient/epsilon selection and is not evidence for a stabilized training variant.

| seed | epsilon | raw loss median | raw loss max |
| --- | ---: | ---: | ---: |
| dev-1 | 0.0001 | 0.00206068 | 1.98026 |
| dev-1 | 1e-06 | 0.00204778 | 1.97044 |
| dev-1 | 1e-09 | 0.00204765 | 1.97034 |
| dev-1 | 1e-12 | 0.00204765 | 1.97034 |
| dev-2 | 0.0001 | 0.00343062 | 0.823513 |
| dev-2 | 1e-06 | 0.00340913 | 0.818604 |
| dev-2 | 1e-09 | 0.00340892 | 0.818554 |
| dev-2 | 1e-12 | 0.00340892 | 0.818554 |

## Largest audited gradient-ratio samples

| seed | stable ID | ratio at coefficient 1 | $q_S$ | $|m_S^{adv}-m_S^{clean}|$ | $d_S$ | $d_T$ | gap | raw loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev-1 | 30822 | 119 | 0.98 | 0.0307 | 0.287 | 2.27 | 1.99 | 1.97 |
| dev-1 | 9941 | 0.171 | 11.6 | 0.365 | 0.0441 | 0.174 | 0.13 | 0.00849 |
| dev-1 | 6788 | 0.139 | 13.3 | 0.417 | 0.117 | 0.261 | 0.144 | 0.0103 |
| dev-1 | 40404 | 0.117 | 4.49 | 0.141 | 0.0558 | 0.11 | 0.0543 | 0.00147 |
| dev-1 | 39885 | 0.099 | 14 | 0.438 | 0.042 | 0.131 | 0.0886 | 0.00392 |
| dev-2 | 4502 | 22.7 | 2.32 | 0.0728 | 0.465 | 1.06 | 0.595 | 0.177 |
| dev-2 | 34313 | 14.7 | 1.69 | 0.0531 | 0.661 | 1.02 | 0.363 | 0.066 |
| dev-2 | 17124 | 7.32 | 3.94 | 0.124 | 0.163 | 1.39 | 1.23 | 0.753 |
| dev-2 | 334 | 2.05 | 5.57 | 0.175 | 0.0017 | 1.28 | 1.28 | 0.819 |
| dev-2 | 3861 | 0.461 | 9.8 | 0.307 | 0.157 | 0.202 | 0.0445 | 0.00099 |

## Historical v2 calibration and failure evidence

The pooled v2 calibration used the frozen coefficient `1.52196388329` for a median target of `0.25`.  Its achieved ratios were min `0.051948`, median `0.25`, max `20.86`, IQR `1.5622`.  Thus median matching did not bound the observed tail; this is a calibration limitation that coexists with, rather than replaces, the formula-sensitivity evidence above.

| seed | source / host | v2 formula and coefficient | last retained finite evidence | first non-finite / terminal evidence |
| --- | --- | --- | --- | --- |
| dev-1 | Hamster GPU1 | `student_parameter_graph_v2`, coefficient 1.52196388329, $\epsilon=1e-12$ | e105; loss 8.58034e+10; $|w|_{max}$ 3.20128e+27 | epoch 106: trainer raised FloatingPointError(non-finite training loss) |
| dev-2 | Ferret GPU0 | `student_parameter_graph_v2`, coefficient 1.52196388329, $\epsilon=1e-12$ | e101; loss 5.66103e+09 | not retained locally; terminal run had no valid checkpoint or endpoint after e101 |

Both failures used `student_parameter_graph_v2`, the same frozen v2 calibration artifact and epsilon, but occurred on Hamster GPU1 and Ferret GPU0.  This rules out a host/GPU-specific explanation at the available resolution.  Control, DPM, and D-BDD completed e114 with finite telemetry and registered endpoints, so the main causal comparison remains evaluable.

## Causal utility remains evaluable without S-BDD

| seed | e114 DPM − Control held-out robust | e114 D-BDD − Control | e114 D-BDD − DPM |
| --- | ---: | ---: | ---: |
| dev-1 | +0.08 pp | +0.04 pp | -0.04 pp |
| dev-2 | +0.12 pp | +0.20 pp | +0.08 pp |

D-BDD versus DPM is mixed across the two development seeds, so this audit does not support a D-BDD promotion or e199 extension.  No floor/cap/smoothed reciprocal is tried here.  The current S-BDD contract is closed as numerically unsupported; any stabilized secant variant remains a discussion-only redesign candidate and would require a separate scientific contract, calibration, and experiment.

Machine artifact: `docs/experiments/ert_rslad_i100_s2_secant_boundary_distance_forensic_v1.json` (SHA-256 `8a1267f41ad475644e186698347eeb0a3df3bf3c0cf8e36d096940b6b25a5739`).
