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

- Hamster run/state/controller: pending
- Ferret run/state/controller: pending
- Host/GPU/job assignments: see the fixed campaign YAML; live evidence pending
- Stop command: `scripts/campaign/campaign-stop --run-dir <host-run-dir>`
- Recovery: rerun the matching host `start` command; it adopts a matching live phase and never retries a nonzero
  scientific phase automatically.

Seed extension, direct-training baselines, full SAAD, and MobileNetV2 remain intentionally deferred.
