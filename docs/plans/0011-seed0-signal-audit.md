# Seed-zero signal audit

Status: in progress

## Goal

Determine whether student robust learnability adds prospective information
beyond teacher entropy, and whether the current Joint intervention targets the
right samples, before any Student/Joint seed expansion or v2 training.

This milestone is analysis-only. It does not modify completed training,
evaluation, checkpoint, or W&B artifacts and does not use the official test
split to design a new method.

## Frozen analysis distinction

- **Final-state association:** epoch-199 features and epoch-199 train robust
  error. This is exploratory contemporaneous association only.
- **Prospective prediction:** features from a fixed historical checkpoint and
  subsequent forgetting/last error. Only this analysis is eligible for the
  Signal Go/No-Go decision.

Sample-level bootstrap confidence intervals are conditional on the single
seed-zero training run. They do not represent training-seed uncertainty.
Bootstrap is class-stratified and, for repeated epochs, clustered by stable
sample ID. AUPRC is always reported with positive prevalence.

## Progress

- [x] Confirm the local final train Parquet schema and best/last checkpoint
  SampleStateStore coverage.
- [x] Enumerate W&B model artifact version history read-only for all eight
  canonical train runs.
- [x] Record run ID, artifact name/version, checkpoint epoch, checkpoint
  SHA-256, SampleStateStore presence/count, config hash, and scientific Git
  SHA.
- [x] Prefer periodic checkpoint versions when their identity and state
  validate; fall back to best/last only when unavailable or incomplete.
- [x] Implement fail-closed artifact inventory and dataset-namespace checks.
- [x] Implement final-state association and prospective prediction as separate
  analysis outputs.
- [ ] Run the frozen seed-zero audit without new training. (All four
  artifact-only reports complete; historical teacher-risk replay pending.)
- [ ] Perform one consolidated scientific review and update the experiment
  dashboard with results and limitations.

W&B API evidence collected on 2026-07-30 shows 40 `last` versions for every
canonical run.  The corresponding `best` version counts are Chen
RSLAD/Entropy/Student/Joint = 21/24/19/23 and Bartoldson
RSLAD/Entropy/Student/Joint = 11/15/12/9.  Every run is `finished`, and `v39`
has the terminal `last` alias.  Because unchanged best bytes do not create a
new version, periodic `last` is the only complete five-epoch time series.

The owning-host run bundles each contain 40 content-addressed `last` and 40
`best` publications and agree on scientific Git SHA
`2d54b8230b8d14d13c1ea7472ccba53491b4d38d`.  Direct loads of all eight
terminal checkpoints found epoch 199 and matching run/config identities.
Student and Joint each contain 45,000 state records; RSLAD and Entropy contain
no `SampleStateStore`, as expected.  Historical teacher risk is not stored in
the checkpoint state.  Therefore artifact-only temporal analysis can measure
student-state dynamics, but the preregistered teacher-only versus augmented
prospective comparison remains `insufficient_data` until a deterministic
saved-checkpoint replay produces historical teacher risk.

Selected epoch-99/199 Student/Joint checkpoint bytes match W&B
`last:v19/v39`. All four reports ran successfully on 2026-07-30 and remain
`insufficient_data` by construction. Student final-state associations show
student-risk AUROC 0.961/0.957 and teacher-risk AUROC 0.156/0.192 for
Chen/Bartoldson same-run final robust error; Joint shows 0.956/0.949 and
0.150/0.194 respectively. These numbers are exploratory and do not satisfy
the prospective Signal Go/No-Go contract.

## Data contracts

- Training sample IDs and official-test sample IDs are distinct namespaces;
  numeric equality never authorizes a join.
- Train Parquet must contain exactly 45,000 unique stable IDs and official test
  Parquet exactly 10,000 unique IDs.
- Stored Student/Joint risk is recomputed from the documented formulas and must
  match within a justified FP32 tolerance. Implied `rho` must be in `[0, 0.5]`.
- W&B artifact metadata is read-only. Existing files are not relabeled,
  aliased, deleted, or rewritten.
- Every selected checkpoint is bound to run ID, artifact version, content
  SHA-256, checkpoint epoch, config hash, scientific Git SHA, and sample-state
  count.
- Cross-method effects conditioned on a post-intervention Joint risk are
  exploratory only, not causal intervention evidence.

## Prospective primary comparison

On a fixed hash-based held-out split, compare:

1. teacher risk only;
2. student risk only;
3. teacher and student main effects;
4. main effects plus their product.

Report AUROC, AUPRC, positive prevalence, and log-loss with fixed-seed
class-stratified sample-cluster bootstrap intervals.

## Resource-allocation thresholds

- Signal Go: teacher-only versus student-augmented held-out
  `delta AUROC >= 0.02`, bootstrap lower bound above zero, and improved
  held-out log-loss.
- Signal No-Go: `delta AUROC` upper bound below `0.01`, or no held-out
  log-loss improvement.
- Otherwise: inconclusive. Do not expand the full Student/Joint matrix; add at
  most one no-intervention instrumentation run if temporal evidence remains
  insufficient.

These are preregistered compute-allocation rules, not claims about
training-seed uncertainty.

## Test selection

- Unit: artifact identity/schema rejection, epoch extraction, state presence
  and count, dataset namespace, formula recomputation, split determinism,
  bootstrap clustering, metric/prevalence reporting.
- Integration: synthetic versioned checkpoint inventory and deterministic
  analysis output under an offline/mock tracker.
- Operational: read-only W&B artifact enumeration followed by local
  content-addressed checkpoint inspection when hashes match.
- No GPU training, production run, or AutoAttack in this milestone.

## Completion conditions

- Periodic artifact availability is established from actual W&B API evidence.
- Selected periodic or fallback checkpoints have exact lineage and state
  inventories.
- Association and prospective results cannot be confused in files, tables, or
  Go/No-Go logic.
- Dataset namespace mistakes and post-treatment causal claims fail closed.
- Analysis results are reproducible from a resolved analysis config and input
  hashes.
