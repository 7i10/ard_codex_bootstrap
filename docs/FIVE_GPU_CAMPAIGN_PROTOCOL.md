# Five-GPU campaign protocol

This campaign runs one independent process per physical GPU. It is not five-GPU DDP. Every result is identified as
`ws1_prb128_gb128_localbn_v1` (world size 1, per-rank/global batch 128, ordinary local BatchNorm) and must not be
pooled with the existing `ws2_prb64_gb128_localbn` run.

## Safety boundary

- A campaign template is inert until a full Git SHA is bound and its host state is armed.
- Each host uses a detached fixed-SHA worktree, atomic job state, an append-only event log, and per-GPU `flock`.
- A phase is adopted only when PID, process start time, cwd, argv digest, run ID, and SHA agree.
- External GPU processes are permitted only with `ARD_CAMPAIGN_ALLOW_EXTERNAL_GPU_PROCESSES=1` and at least
  `ceil(1.25 * pilot peak reserved MiB)` free memory. The runner never changes batch size or scientific settings.
- Stopping the controller does not stop a detached train/evaluation phase. Restarting the controller reconciles it.
- Exit code zero is insufficient: train requires resolved config, best/last, and a completed run bundle; evaluation
  additionally requires `evaluation-results.json`.
- Each host stops at `awaiting_scientific_review` after all of its preregistered core jobs reach a terminal state.
  Failed/blocked jobs do not unlock seed extension.

The protected Ferret run `chen-rslad-production-s0-0ca90ad` retains GPUs 0/1 until it completes. It remains a
two-GPU reference result and is never mutated or used to fill the single-GPU matrix. Its reservation is released
only by an exact marker containing the protected run ID, Git SHA, execution profile, completed W&B sync, and
completed saved-checkpoint PGD evaluation.

## Bounded pilots

The final campaign SHA first runs:

| Host/GPU | Cell | Epochs | Purpose |
|---|---|---:|---|
| Hamster 0 | Chen / RSLAD / seed 0 | 1 | local launch, lineage, best/last and PGD transition |
| Hamster 1 | Chen / Joint / seed 0 | 3 | epoch-0 warmup and post-warmup signal activation |
| Ferret 2 | Bartoldson / RSLAD / seed 0 | 1 | large-teacher batch-128 memory and remote detachment |

These lengths are engineering gates, not efficacy estimates. All three still evaluate official CIFAR-10 best and last
with the fixed CE PGD-20 attack. Production is not armed if any pilot is non-finite, OOM, lacks terminal lineage, or
has an unresolved P0/P1 finding. A source/config correction creates a new SHA and requires fresh pilots.

## Production queue

`configs/campaigns/five_gpu_single_process_v1.yaml` preregisters the eight seed-0 cells:
Chen/Bartoldson × RSLAD/entropy/student/joint. Training and mandatory PGD evaluation outrank optional AutoAttack.
AutoAttack is selected for RSLAD, entropy, and joint; student-only stops after PGD. No seed 1/2 run is automatically
created.

The controllers use `scripts/campaign/manage.py`; the shell entry points only supply host-local paths. Typical local
preparation is:

```bash
SHA=<full-pushed-sha>
RUN_ID=c10-r18-ws1-pilots-v1-${SHA:0:7}
ROOT=/home/shunsukenaito/workspace-local/ard-campaign-runs/ard_codex_bootstrap

scripts/campaign/hamster-campaign prepare \
  --source-repo "$PWD" --run-root "$ROOT" --run-id "$RUN_ID" --sha "$SHA"
scripts/campaign/hamster-campaign start \
  --run-dir "$ROOT/$RUN_ID" \
  --campaign configs/campaigns/five_gpu_single_process_pilots_v1.yaml \
  --host hamster
```

Ferret is prepared through the existing fixed-SHA `ferret-prepare`, then its committed wrapper is invoked inside that
detached worktree with `start --sha "$SHA" --host ferret`. This does not edit or switch the source clone.

Status and controller-only stop:

```bash
scripts/campaign/campaign-status --run-dir <run-dir>
scripts/campaign/campaign-stop --run-dir <run-dir>
```

Outputs are below `<run-dir>/outputs`, state below `<run-dir>/state`, and controller/phase logs below `control`.
Production online W&B is mandatory. The fixed SHA suffix is appended to every configured base run ID.

## Acceptance and handoff

Before production launch, inspect finite train/evaluation metrics, best/last results, W&B terminal manifests, measured
peak reserved VRAM, the Joint signal distribution after warmup, and restart/adoption evidence. Do not author the
acceptance JSON by hand. `scripts/campaign/accept_pilots.py` derives it from the exact three terminal job states,
online W&B manifests, resolved configs, metrics, full best/last PGD-20 results, and Joint Parquet statistics; source
artifacts are bound by SHA-256. `scripts/campaign/manage.py start --pilot-evidence ...` accepts only that exact check
and identity schema for the same pushed Git SHA.

Before unattended handoff, both host controllers must be detached and at least one production job per available host
must show the correct SHA/GPU/profile/run identity and finite progress. Full campaign completion is not awaited in a
foreground Codex session.
