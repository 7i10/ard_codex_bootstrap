# Five-GPU campaign handoff

This record is populated from executed evidence. Empty fields are not claims.

## Immutable identity

- Scientific implementation/campaign Git SHA: `2d54b8230b8d14d13c1ea7472ccba53491b4d38d`
- Single-GPU execution profile: `ws1_prb128_gb128_localbn_v1`
- Protected two-GPU run: `chen-rslad-production-s0-0ca90ad` at
  `0ca90ad3d48fe019151363b00c6da2160d64eb99`

## Pilot evidence

| Pilot | State | W&B ID/URL | Train clean/robust | Best/last PGD-20 | Peak reserved VRAM |
|---|---|---|---|---|---|
| Hamster Chen RSLAD | completed | `pilot-h-chen-rslad-s0-2d54b82` | 0.2336 / 0.1766 | 0.2428 / 0.2428 | 2058 MiB |
| Hamster Chen Joint | completed | `pilot-h-chen-joint-s0-2d54b82` | 0.4413 / 0.2811 | 0.2949 / 0.2949 | 2070 MiB |
| Ferret Bartoldson RSLAD | completed | `pilot-f-bart-rslad-s0-2d54b82` | 0.2297 / 0.1690 | 0.2342 / 0.2342 | 3688 MiB |

Every best/last PGD value above is from exactly 10,000 CIFAR-10 test examples. Joint epoch-2 risk min/mean/max was
`0.002404 / 0.134750 / 0.609255`; canonical uniform KD weight remained `1.0` while target mixing was active.
Ferret controller restart/adoption completed without restarting the scientific child. Exact source hashes and checks
are in the pilot run's `control/pilot-acceptance.json`.

## Production handoff

- Production was armed from hash-bound pilot evidence with SHA-256
  `0501366f632d15d9557cf8deb9e2c61023f33b29e78b424f09fbb2c69e7`.
- Hamster run:
  `/home/shunsukenaito/workspace-local/ard-campaign-runs/ard_codex_bootstrap/c10-r18-ws1-b128-core-s0-v1-2d54b82`
  - Chen RSLAD: 200 epochs and official PGD complete. Best/last clean accuracy is `0.8318 / 0.8322`; best/last
    PGD-20 accuracy is `0.5565 / 0.5544` over 10,000 test examples.
  - Bartoldson entropy: 200 epochs and official PGD complete. Best/last clean is `0.8524 / 0.8511`; best/last PGD-20
    is `0.5009 / 0.4872`.
  - Bartoldson student: 200 epochs and official PGD complete. Best/last clean is `0.8471 / 0.8500`; best/last PGD-20
    is `0.5053 / 0.4598`.
  - Chen entropy began after recovery. At the 2026-07-29 observation it had reached epoch 8 with finite validation
    clean/PGD `0.5240 / 0.2926`; its measured epoch time was about 71 seconds.
- Ferret run:
  `/home/shunsukenaito/workspace-local/ard-runs/ard_codex_bootstrap/c10-r18-ws1-b128-core-s0-v1-2d54b82`
  - Chen Joint completed 200 epochs, official best/last PGD, and full best/last AutoAttack without rerun.
  - Bartoldson RSLAD, Bartoldson Joint, and Chen Student never launched before the driver outage. Their exact
    inventory-failure records are retained and are the only jobs eligible for transient recovery.
  - The protected `chen-rslad-production-s0-0ca90ad` W&B sync and strict best/last PGD evaluation completed on
    2026-07-29. Best/last clean is `0.8351 / 0.8342`; best/last PGD-20 is `0.5588 / 0.5561` over 10,000 examples.
    Its exact release marker is now present for GPUs 0/1.
- Online W&B runs observed:
  - [Chen RSLAD](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/prod-chen-rslad-s0-2d54b82)
  - [Bartoldson entropy](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/prod-bart-entropy-s0-2d54b82)
  - [Chen Joint](https://wandb.ai/shunsuke-n-waseda-university/single-teacher-ard/runs/prod-chen-joint-s0-2d54b82)
- Host/GPU/job assignments and phase order are fixed in
  `configs/campaigns/five_gpu_single_process_v1.yaml`. At measured throughput, the training portions of the two-job
  queues are approximately 31 hours per GPU before saved-checkpoint evaluation and selected AutoAttack.
- Stop command: `scripts/campaign/campaign-stop --run-dir <host-run-dir>`
- Recovery: rerun the matching host `start` command; it adopts a matching live phase and never retries a nonzero
  scientific phase automatically. Run status/recovery only from a host-visible shell: a restricted PID namespace can
  read the shared state files but cannot validate host PIDs and may report a false `controller_live=false`.
- GPU-outage recovery and the successor-phase regression are documented in
  `docs/debugging/0012-gpu-outage-campaign-recovery.md`. The persistent watchdog uses a clean committed control-plane
  revision while every scientific phase continues to execute from the immutable scientific SHA above.

Seed extension, direct-training baselines, full SAAD, and MobileNetV2 remain intentionally deferred.
