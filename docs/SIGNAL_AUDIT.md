# Seed-0 signal audit

This is a read-only diagnostic over the four Student/Joint seed-0 runs.
All four artifact-only reports have been executed. It is exploratory and is
not yet a formal Signal decision. The
historical checkpoint is epoch 99 (the checkpoint after epoch 100); each
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
All four reports correctly return `insufficient_data` for the formal prospective
decision because no historical teacher-risk replay has yet been supplied.
The current exploratory result is nevertheless diagnostic: final student risk
is strongly associated with same-run final robust error (AUROC 0.961 for Chen,
0.957 for Bartoldson), while low-entropy teacher risk is inversely associated
(0.156 and 0.192).  These are post-training sample-level associations from
seed 0, not evidence of future-failure prediction or method improvement.

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
prospective section must retain an `insufficient_data` decision until
historical teacher risk is replayed.
