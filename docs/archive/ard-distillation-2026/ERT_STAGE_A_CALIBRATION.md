# ERT Stage A calibration

Status: complete, no-update calibration only

The exact baseline formula is documented in
[ERT_STAGE_A_FORMULA_AUDIT.md](ERT_STAGE_A_FORMULA_AUDIT.md). Calibration was
run on the fixed epoch-79 L2/L4 parents using the registered S3×T1, S3×T2,
S3×T3, and Clean-Wrong masks. It used 8 deterministic class-stratified
mini-batches of 64 per cohort per seed, with no optimizer, scheduler, sample
state, checkpoint, validation, or future-outcome mutation.

Frozen values:

```text
tau                  = 2.0
alpha_soft           = 1.2522921562194824
beta_advce_weak      = 0.07095924764871597
beta_advce_moderate  = 0.14191849529743195
beta_cleance_weak    = 0.07825280725955963
```

Achieved median gradient ratios were 0.9955 for softened AdvKD, 0.2494 for
weak AdvCE, 0.4988 for moderate AdvCE, and 0.2490 for weak CleanCE. The
AdvKD–AdvCE gradient cosine median was 0.7658. These are mechanism-calibration
diagnostics, not downstream performance choices.

The full measurement artifact is
`.cache/analysis/ert-stage-a-calibration-v1.json` with sidecar hash
`0e3b98b4e1cfcc7727786fd23da57a82903fa2c8b95a16b6e12ca1425d34da16`. The
tracked compact manifest is
`docs/experiments/ert_stage_a_calibration_v1.json`.

No Stage A training, Stage B extension, official test, or AutoAttack was
started by calibration.
