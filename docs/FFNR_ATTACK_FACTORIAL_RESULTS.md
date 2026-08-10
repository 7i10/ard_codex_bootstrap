# FFNR CE/KL × PGD10/20 factorial

This is a point-estimate diagnostic for the Chen ERT runs L2 and L4. It is
not a new training result and it is not an official-test or AutoAttack result.
The fixed endpoint is epochs `189, 194, 199`; rows are robust failures under
the named replay attack on the same 45,000-sample train/validation panel.
No bootstrap was preregistered for this diagnostic.

Reproducibility note: the summary below is an archived historical report.
The current checkout does not contain the hash-bound raw CE-PGD20 Parquet
artifacts at the configured L2/L4 factorial paths, so this summary must not be
treated as a reusable input for a new analysis. The current online-routing
proxy reports those cells as unavailable and does not silently recreate them.

## Frozen attack matrix

All cells use pixel-space `[0, 1]`, `L_inf`, `epsilon=8/255`,
`step_size=2/255`, random start, and the same stable-ID/class universe.
`CE` uses hard-label cross entropy. `KL` uses teacher-clean logits as the
target. The number after `PGD` is the step count. Existing CE-PGD20 cells
were reused when their identity and epoch coverage matched; the other three
cells were replayed through the public CLI.

Hamster ran the L2 cells and Ferret ran the L4 cells. The L4 source
checkpoints were recovered from the pinned W&B artifact versions and copied
with SHA-256 verification. Smoke outputs were kept separate from full replay
outputs so an existing smoke could not be mistaken for a completed cell.

| condition | epoch | L2 failure | L4 failure | Jaccard | chance-adjusted | κ |
|---|---:|---:|---:|---:|---:|---:|
| ce_pgd10 | 189 | 0.2687 | 0.2661 | 0.8387 | 0.8093 | 0.8803 |
| ce_pgd10 | 194 | 0.2658 | 0.2672 | 0.8395 | 0.8103 | 0.8810 |
| ce_pgd10 | 199 | 0.2648 | 0.2661 | 0.8394 | 0.8104 | 0.8811 |
| ce_pgd20 | 189 | 0.2766 | 0.2741 | 0.8385 | 0.8078 | 0.8788 |
| ce_pgd20 | 194 | 0.2738 | 0.2761 | 0.8393 | 0.8088 | 0.8795 |
| ce_pgd20 | 199 | 0.2740 | 0.2740 | 0.8415 | 0.8116 | 0.8815 |
| kl_pgd10 | 189 | 0.2010 | 0.1840 | 0.5780 | 0.5278 | 0.6690 |
| kl_pgd10 | 194 | 0.1905 | 0.1859 | 0.5764 | 0.5273 | 0.6690 |
| kl_pgd10 | 199 | 0.1841 | 0.1825 | 0.5844 | 0.5377 | 0.6788 |
| kl_pgd20 | 189 | 0.2114 | 0.1941 | 0.5803 | 0.5270 | 0.6670 |
| kl_pgd20 | 194 | 0.2004 | 0.1957 | 0.5778 | 0.5256 | 0.6663 |
| kl_pgd20 | 199 | 0.1941 | 0.1911 | 0.5865 | 0.5372 | 0.6772 |

| condition | frequency Spearman | frequency Pearson |
|---|---:|---:|
| ce_pgd10 | 0.9177 | 0.9246 |
| ce_pgd20 | 0.9172 | 0.9248 |
| kl_pgd10 | 0.7724 | 0.7852 |
| kl_pgd20 | 0.7762 | 0.7852 |

## Interpretation

- CE attacks have high cross-seed agreement: Jaccard is approximately
  `0.839–0.842`, chance-adjusted Jaccard `0.808–0.812`, and Cohen's κ
  `0.879–0.882`.
- KL attacks have materially lower agreement: Jaccard is approximately
  `0.576–0.587`, chance-adjusted Jaccard `0.526–0.538`, and κ
  `0.666–0.679`.
- Changing PGD10 to PGD20 barely changes agreement within either objective.
  In this panel, the objective (CE versus KL), rather than step count, is the
  dominant explanation for the CE/KL difference. This does not establish
  that CE is a better training attack; it only describes failure-mask
  agreement under replay.
- CE failure prevalence is about `26.5–27.7%`, while KL is about
  `18.3–21.1%`. These are replay-panel failure rates, not CIFAR-10 test
  accuracy.

The report is therefore evidence that the attack objective changes which
samples are called failures. It does not by itself select an intervention,
prove causality, or justify changing the training protocol.

## Provenance and limits

The three new L4 cells use the recovered W&B run-bundle manifest and
hash-bound checkpoint inventory; their local source files and transferred
outputs were verified. The pre-existing L4 CE-PGD20 parquet was reused from
the earlier Ferret artifact, so its original local lineage is not recreated
in this report. This limitation is recorded rather than silently treating
the reused cell as a newly replayed one.

The Bartoldson dense-checkpoint audit remains separate: seed-1 periodic
checkpoints contain complete state, but no L3/seed-2 dense continuation was
launched. No new training was started, and no official PGD or AutoAttack was
run in this phase.
