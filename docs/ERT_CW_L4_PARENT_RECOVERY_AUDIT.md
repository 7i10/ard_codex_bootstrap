# ERT Clean-Wrong L4 parent recovery audit

Date: 2026-08-21  
Purpose: verify the historical L4 epoch-79 causal parent before any new GPU
production run.

## Decision

The requested L4 screen parent is recovered exactly:

```text
SHA256 = 026a36d3fe057386fe19225fed23b56625ab23da80be3dd42cf3e478e5080bf1
epoch  = 79 (epoch boundary: end)
seed   = 2
```

The byte-exact checkpoint is available at:

```text
.cache/analysis/ffnr-causal-pilot-screens-e79-94/L4/C79/last.pt
```

It was materialized for subsequent use at:

```text
.cache/analysis/ert-cw-l4-parent-recovery-audit/parent-epoch79.pt
```

The materialized copy is independently SHA-256 verified to be identical to
the source.  This cache is intentionally untracked; the checkpoint is not
committed to Git.

No GPU production training was started during this audit.  The L4 parent
blocker is therefore resolved without changing the frozen SHA contract.  The
0050 screen may proceed only through its already specified calibration,
canary, and lineage gates.

## Historical provenance

The audit gives priority to `fork_lineage.parent_checkpoint_sha256` for a
continuation fork, as required by the experiment contract.  The generic
`parent_lineage` fields are retained as provenance for the original source
run, but are not substituted for the continuation parent.

### Clean-Wrong Broad Screen

Source namespace:

```text
source commit: cbe03a7b3be0b11fa1555b573c6f453a3d10f27b
namespace:     ert-clean-wrong-broad-v1-l4r2
```

The L4 C0--C15 run bundles under
`.cache/analysis/ert-clean-wrong-broad-v1/L4/` consistently record:

```text
fork_lineage.experiment_parent_checkpoint_sha256 = 026a...
fork_lineage.parent_checkpoint_sha256            = 026a...
fork_lineage.parent_epoch                        = 79
fork_lineage.source_git_sha                       = cbe03a7...
```

The corresponding generic lineage in these manifests points to the original
observed source checkpoint `9b51...`.  That is not a contradiction: the C0--C15
children were forked from the already materialized C79 checkpoint `026a...`.

### Reliability replay

The L4 feature artifacts used by the Clean-Wrong reliability analyses are
bound to checkpoint `026a...`, epoch 79, and the fixed L4 Clean-Wrong mask
(`fe818e...`).  This holds for both the CE-PGD20 replay (source commit
`4a81f40f2c1265d966baac26f08b167949d8a5db`) and the KL-PGD10 practical proxy
replay.  The L4 validation CE20/KL10 replay metadata also binds the same
checkpoint SHA.

### Gated CleanCE experiment

The L4 G0--G3 run bundles under
`.cache/analysis/ert-cw-reliability-gated-ce015-v1/L4/` likewise record:

```text
fork_lineage.experiment_parent_checkpoint_sha256 = 026a...
fork_lineage.parent_checkpoint_sha256            = 026a...
fork_lineage.parent_epoch                        = 79
```

Their generic `parent_lineage.checkpoint_sha256` remains `9b51...` because it
describes the original observed source run.  The fork field is the causal
parent for these treatment continuations and is the field used here.

Representative run-bundle identities are:

| historical source | L4 run-bundle example | fork parent | generic source parent |
|---|---|---|---|
| Broad Screen | `ert-ert-clean-wrong-broad-v1-l4r2-2-C0-cbe03a7` | `026a...` | `9b51...` |
| gated CleanCE | `ert-ert-cw-reliability-gated-ce015-v3-2-G1_CW_ALL_CE015-8544fed` | `026a...` | `9b51...` |

The original source manifest is
`.cache/analysis/ffnr-causal-pilot-epoch79/run-bundles/chen-rslad-observed-s2-confirm-v2/manifest.json`.
Its local lineage attestation is
`.cache/analysis/ffnr-causal-pilot-lineage-e79-94/L4/artifact-attestation.json`.
The corresponding W&B-local checkpoint identity is
`model-chen-rslad-observed-s2-confirm-v2-last:v15` with digest
`local-9b51bca767871ada6c80c75ad92997f9b7f246c0c1e35f3edad35d4e787a4a9c`.

## Exact checkpoint recovery

| item | value |
|---|---|
| source path | `.cache/analysis/ffnr-causal-pilot-screens-e79-94/L4/C79/last.pt` |
| source size | 104,256,283 bytes |
| source SHA-256 | `026a36d3fe057386fe19225fed23b56625ab23da80be3dd42cf3e478e5080bf1` |
| recovered path | `.cache/analysis/ert-cw-l4-parent-recovery-audit/parent-epoch79.pt` |
| recovered size | 104,256,283 bytes |
| recovered SHA-256 | same as source |
| checkpoint epoch | 79 |
| global step | 28,160 |
| checkpoint config hash | `f5c64f153784564a44a7e483e2677ad28cdab2c74e8e4b807cb47ac9f3584d5e` |
| tracker run | `ffnr-causal-l4-c79-e79-94-v1` |
| fork source commit | `b296b8bd1f677cbebb163849cc4d90b4c9fb6a1f` |

The recovered checkpoint's own `fork_lineage` identifies it as the C79
continuation child whose parent was the original observed `9b51...` state.  It
is nevertheless the exact checkpoint SHA recorded as the parent by all
downstream C0--C15 and gated continuations, and by the replay artifacts.

## What `9b51...` is

The other local checkpoint is:

```text
path: .cache/analysis/ffnr-causal-pilot-epoch79/chen-rslad-observed-s2-confirm-v2/last.pt
SHA:  9b51bca767871ada6c80c75ad92997f9b7f246c0c1e35f3edad35d4e787a4a9c
size: 104,257,179 bytes
epoch: 79 (end), global step: 28,160
run: chen-rslad-observed-s2-confirm-v2
config hash: 8986d2711a55ca635c27c8699da34087c2e54c00249584cf3e3f86ec2b85a1d7
best metric: 0.476
```

The `026a...` payload explicitly records this checkpoint as its
`fork_lineage.parent_checkpoint_sha256`, with the original source run, source
Git SHA `8254a8899ae7373c2f541d108593e5c8185b26f5`, and the original raw
config hash.  Thus `9b51...` is the upstream observed source for the C79
fork, not an unexplained replacement discovered after the fact.

## Component-level comparison

Both checkpoints were loaded without modifying them.  A canonical recursive
hash was computed for each saved state component: mappings are key-sorted,
tensors are moved to CPU and hashed by dtype/shape/contiguous bytes, and
sequences/scalars are serialized deterministically.

| component | equal? | canonical SHA-256 |
|---|---:|---|
| Student model state (124 tensors) | yes; 0 differing tensors | `7b677d8b5ba74618471e0bfb584c17ab36264f600558637a3e0e9381edcd8ecd` |
| optimizer state | yes | `e2faec8a98eab1e25aec48a03ea7f7068f0354e9b2d9c15c4505031b5bfac9e4` |
| scheduler state | yes | `c313f3eec4f43fc3b7e8a4f60f28b52f0264c18bf13dc8fa9a663bb61cc9a12e` |
| AMP scaler state | yes | `45a544e3656c045a985458ca664531e60771d74fdfa465fceab8bcd8e391c768` |
| RNG state | yes | `fa7522fe598eca90050e9f5c42afe7839ee4b30bcf6687efd78e2d0f9bded6ef` |
| sampler epoch/state | yes | `d2532961b0d13d497e988944b25db4d2e1687227f609141664dc6a8daa5db6fb` / `6f63e70fccada38c8d953e49fa492f82a29f92f0e3c4c8c490089678374731ef` |
| sample state | yes | `46476ed8b65d684310d130121d0b1e5486a9bd8fa6323b6cdfc3e1c94a0830c4` |
| global step / epoch | yes | `28,160` / `79` |

The byte-level checkpoint SHA differs because the containers have different
metadata and fork provenance, not because the saved trainable or resumable
state components differ.  In particular:

- `026a...` has config hash `f5c64f...`, tracker
  `ffnr-causal-l4-c79-e79-94-v1`, `best_metric=-inf`, and a `fork_lineage`
  block;
- `9b51...` has config hash `8986...`, tracker
  `chen-rslad-observed-s2-confirm-v2`, `best_metric=0.476`, and no embedded
  `fork_lineage` block.

Therefore the audit establishes exact byte recovery of `026a...` and strong
component-level state equivalence with `9b51...`; it does **not** relabel the
two byte hashes as identical, nor does it replace the frozen byte-SHA
contract with a semantic-only contract.

## Launch decision

This is Case A from the recovery plan:

```text
exact 026a... recovered: YES
9b51... substitution needed: NO
GPU production during audit: NO
```

The L4 parent blocker is closed.  Before the 0050 production continuations,
the next required steps remain the plan's calibration, coefficient freeze,
deterministic canary, and clean-lineage launch checks.  No checkpoint SHA,
mask, threshold, or treatment rule was changed by this audit.
