# Seed-0 signal audit

This is a read-only diagnostic over the four Student/Joint seed-0 runs.
All four artifact inventories, deterministic historical replays, and formal
prospective reports completed on 2026-07-30. The historical checkpoint is
epoch 99 (the checkpoint after epoch 100); each
config fixes 45,000 train samples, 10 classes, split/bootstrap seeds, and
1,000 bootstrap replicates.  Student configs use `stored_risk_kind: student`;
Joint configs use `joint`.

The 2026-07-30 read-only W&B preflight found `last` versions v0--v39 for all
eight canonical runs.  Best versions are deduplicated when bytes do not
change, so periodic `last` is the primary time series.  Association remains
exploratory.  Artifact-only diagnostics do not establish the formal Signal
decision: a prospective teacher-risk comparison requires deterministic
historical teacher-risk replay from saved checkpoints.  The official test
split is excluded.

The selected Student/Joint checkpoints were verified against W&B `last:v19`
(epoch 99) and `last:v39` (epoch 199), including local-byte MD5 and size.
Epoch-99 teacher risk was reconstructed from saved checkpoints using the exact
raw-train split and KL PGD-10 training attack. All four reports satisfy the
preregistered **Signal Go** allocation rule:

| Teacher / method | Forgetting prevalence | Teacher AUROC | Augmented AUROC | Delta | Bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| Chen / Student | 0.539 | 0.853 | 0.898 | +0.045 | [0.040, 0.051] |
| Chen / Joint | 0.547 | 0.859 | 0.896 | +0.037 | [0.031, 0.043] |
| Bartoldson / Student | 0.684 | 0.586 | 0.923 | +0.337 | [0.325, 0.349] |
| Bartoldson / Joint | 0.694 | 0.602 | 0.937 | +0.335 | [0.323, 0.347] |

| Teacher / method | Teacher / augmented AUPRC | Teacher / augmented log-loss |
|---|---:|---:|
| Chen / Student | 0.830 / 0.841 | 0.471 / 0.382 |
| Chen / Joint | 0.849 / 0.839 | 0.462 / 0.385 |
| Bartoldson / Student | 0.772 / 0.946 | 0.607 / 0.309 |
| Bartoldson / Joint | 0.795 / 0.955 | 0.594 / 0.280 |

Held-out log-loss improves in every row. Chen Joint AUPRC decreases slightly
despite higher AUROC and lower log-loss, so the Signal Go result should not be
read as uniform improvement under every ranking metric. Overall this supports retaining the
student signal, especially for Bartoldson; it does **not** establish that the
current target-softening intervention improves accuracy. Confidence intervals
are conditional on seed-0 training runs, not uncertainty across training seeds.

### Complete held-out model comparison

The tables below extract all four preregistered logistic models from the same
formal JSON reports and the same held-out prospective
`subsequent_forgetting_increment` outcome. Here, `c_i` is historical teacher
low-entropy risk, `u_i` is historical student robust-margin risk, and
`u_i c_i` is their product. Higher AUROC/AUPRC and lower log-loss are better.

| Teacher / run | Teacher-only `c_i` | Student-only `u_i` | Main effects `u_i,c_i` | Main + product `u_i,c_i,u_i c_i` |
|---|---:|---:|---:|---:|
| **AUROC** |||||
| Chen / Student | 0.8529 | 0.8989 | 0.8968 | 0.8982 |
| Chen / Joint | 0.8593 | 0.8952 | 0.8955 | 0.8964 |
| Bartoldson / Student | 0.5857 | 0.9184 | 0.9194 | 0.9230 |
| Bartoldson / Joint | 0.6023 | 0.9338 | 0.9340 | 0.9373 |
| **AUPRC** |||||
| Chen / Student | 0.8296 | 0.8461 | 0.8384 | 0.8414 |
| Chen / Joint | 0.8490 | 0.8401 | 0.8398 | 0.8395 |
| Bartoldson / Student | 0.7719 | 0.9363 | 0.9387 | 0.9460 |
| Bartoldson / Joint | 0.7950 | 0.9469 | 0.9475 | 0.9552 |
| **Log-loss** |||||
| Chen / Student | 0.4710 | 0.3895 | 0.3853 | 0.3822 |
| Chen / Joint | 0.4623 | 0.3981 | 0.3882 | 0.3853 |
| Bartoldson / Student | 0.6071 | 0.3137 | 0.3134 | 0.3086 |
| Bartoldson / Joint | 0.5936 | 0.2857 | 0.2860 | 0.2796 |

The three requested incremental comparisons are:

1. **Teacher versus student (`c_i` vs `u_i`).** Student-only AUROC is higher
   in all four reports: `+0.0461`, `+0.0359`, `+0.3326`, and `+0.3315`.
   Student-only log-loss also improves in every report. Chen Joint is the
   exception for AUPRC (`0.8401` vs teacher `0.8490`), so the student signal
   does not dominate the teacher under every ranking metric.
2. **Adding teacher to student (`u_i` vs `u_i,c_i`).** AUROC changes are
   `-0.0021`, `+0.0004`, `+0.0010`, and `+0.0002`. Teacher entropy therefore
   adds little ranking information after student history in these seed-0
   reports. It improves Chen log-loss, is nearly neutral for Bartoldson
   Student, and slightly worsens Bartoldson Joint (`+0.0003`).
3. **Adding the product (`u_i,c_i` vs `u_i,c_i,u_i c_i`).** AUROC changes are
   `+0.0014`, `+0.0009`, `+0.0035`, and `+0.0033`; log-loss improves in all
   four reports by `0.0030`, `0.0030`, `0.0048`, and `0.0064`. The product
   provides a small incremental calibration/ranking gain, strongest for
   Bartoldson, but is not the main source of predictive power.

These are held-out point estimates. The existing reports contain paired
bootstrap inference for teacher-only versus main-plus-product, not for every
adjacent comparison above. Therefore the small main-effect/product increments
are descriptive and should not yet be called statistically established.
Predictive interaction also does not prove that multiplying the risks is the
correct causal intervention or target-softening rule.

The final-state association remains exploratory. Final student risk is strongly
associated with same-run final robust error (AUROC 0.961/0.957 for
Chen/Bartoldson Student), while final low-entropy teacher risk is inversely
associated (0.156/0.192). These post-treatment associations are not used for
the Signal Go decision.

## Canonical RSLAD common-trajectory result

The intervention-specific Student/Joint reports above cannot cleanly compare
signals because their losses already depend on those signals.  We therefore
replayed the 40 periodic checkpoints from each canonical RSLAD seed-0 run with
the same raw 45,000-sample train partition and independent fixed KL PGD-10
feature/outcome seed domains.  Epoch 99 is the anchor; only checkpoint-panel
transitions after that anchor define the primary forgetting outcome.

The table gives held-out metrics for the frozen models.  Higher AUROC/AUPRC
and lower log-loss are better.

| Teacher | Model | AUROC | AUPRC | Log-loss |
|---|---|---:|---:|---:|
| Chen | Teacher entropy | 0.8003 | 0.5753 | 0.4928 |
| Chen | Current correctness | 0.7191 | 0.4752 | 0.5399 |
| Chen | Instantaneous margin | 0.7598 | 0.5217 | 0.5331 |
| Chen | Current main effects | 0.7598 | 0.5217 | 0.5273 |
| Chen | Student history | 0.8220 | 0.5533 | 0.4531 |
| Chen | Current + history | 0.8216 | 0.5509 | 0.4534 |
| Chen | History + teacher | 0.8295 | 0.5619 | 0.4450 |
| Chen | History + teacher interactions | 0.8298 | 0.5626 | 0.4480 |
| Bartoldson | Teacher entropy | 0.5961 | 0.5356 | 0.6641 |
| Bartoldson | Current correctness | 0.6889 | 0.5584 | 0.6104 |
| Bartoldson | Instantaneous margin | 0.7391 | 0.5893 | 0.5821 |
| Bartoldson | Current main effects | 0.7391 | 0.5893 | 0.5824 |
| Bartoldson | Student history | 0.7916 | 0.6368 | 0.5237 |
| Bartoldson | Current + history | 0.7917 | 0.6338 | 0.5234 |
| Bartoldson | History + teacher | 0.7990 | 0.6354 | 0.5171 |
| Bartoldson | History + teacher interactions | 0.7992 | 0.6373 | 0.5156 |

Both teachers pass the preregistered History gate.  Against the best
current-state baseline, history-only improves Chen AUROC by `0.06220`
(paired 95% CI `[0.05470, 0.07001]`) and Bartoldson AUROC by `0.05250`
(`[0.04549, 0.06027]`); held-out log-loss improves by `0.07418` and
`0.05837`, respectively.  Adding teacher entropy to history provides only
`+0.00752` and `+0.00739` AUROC.  The multiplicative teacher interactions add
`+0.00029` for Chen and `+0.00019` for Bartoldson; they worsen Chen log-loss
but slightly improve Bartoldson log-loss.  Thus the prospective evidence
supports student history as the main signal and teacher information as a
secondary gate, not as the dominant multiplicative risk term.

These are sample-conditional seed-0 checkpoint-panel results, not uncertainty
across training seeds and not evidence that any intervention improves
robustness.  Result identities are:

- Chen report:
  `cb5305182bb942b9f9d44036c67700c9d8fff54116ad3f7111be0a73f65016fa`;
  lineage:
  `f485f72341b276351098e97514e4becc46b278a005694be2672811c5aaf5a808`.
- Bartoldson report:
  `d44ee166f8866b77067ebd07757d394a060242c9cf1cdc5d4513f127897981f8`;
  lineage:
  `9b6ea091dc9ed4ff81bb579bf05d6650ac8e6d4ab6104981c446f29069e4a64e`.

Historical replay is a separate, read-only CUDA command. It is fixed to the
epoch-99 checkpoint, batch size 128, the exact training KL PGD-10 attack,
raw unaugmented train samples, and a clean reviewed Git/source identity:

```bash
PYTHONPATH=src python -m ard.cli.replay_teacher_risk \
  --config configs/analysis/seed0_chen_student.yaml \
  --output outputs/analysis/seed0/chen-student-e99-teacher-risk.json \
  --device cuda:0 --batch-size 128

PYTHONPATH=src python -m ard.cli.signal_audit \
  --config configs/analysis/seed0_chen_student.yaml \
  --teacher-risk-replay outputs/analysis/seed0/chen-student-e99-teacher-risk.json \
  --output outputs/analysis/seed0/chen-student-formal.json
```

The replay command rejects a dirty worktree, non-CUDA backend, DDP execution,
checkpoint/config mismatch, or existing output path. The formal audit validates
the replay source hashes, Git SHA, attack/domain, dataset/split identity,
checkpoint world size, sample labels, and perturbation bound.

Configs are host-specific because the CLI resolves paths literally.  Run on
the owning host with, for example:

```bash
PYTHONPATH=src python -m ard.cli.signal_audit \
  --config configs/analysis/seed0_chen_student.yaml \
  --output /tmp/seed0-chen-student-audit.json
```

Use the analogous `seed0_{bartoldson,chen}_{student,joint}.yaml` file.  A
successful report contains `checkpoint_inventory`,
`final_state_association`, `artifact_only_temporal_diagnostics`, and
`prospective_prediction`, plus `config_hash` and input hashes.  The
prospective section is eligible for a decision only when the validated
historical replay envelope is supplied. Without it, the CLI deliberately
returns `insufficient_data`.

## Result identities

The ignored local formal reports are bound by SHA-256:

- Chen Student: `8c657354d94a8353953499ed89449257ca4c4fb2a051a1d3b654c4775914d61a`
- Chen Joint: `121112b38f1be8564efa807df2648516caadef975dea6f5d749c1032afef7b20`
- Bartoldson Student: `cb7d31e924b9ea20b102971b47f1064fd1a298c80f7e9a31ee338d50a2f142ab`
- Bartoldson Joint: `d6b6e4334bd97e9d5c78360702d87d168ba0939534808937ff27b47157866bb7`
