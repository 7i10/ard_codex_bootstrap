# Five-GPU campaign handoff

This record is populated from executed evidence. Empty fields are not claims.

## Immutable identity

- Implementation/campaign Git SHA: pending
- Single-GPU execution profile: `ws1_prb128_gb128_localbn_v1`
- Protected two-GPU run: `chen-rslad-production-s0-0ca90ad` at
  `0ca90ad3d48fe019151363b00c6da2160d64eb99`

## Pilot evidence

| Pilot | State | W&B ID/URL | Train clean/robust | Best/last PGD-20 | Peak reserved VRAM |
|---|---|---|---|---|---|
| Hamster Chen RSLAD | pending | pending | pending | pending | pending |
| Hamster Chen Joint | pending | pending | pending | pending | pending |
| Ferret Bartoldson RSLAD | pending | pending | pending | pending | pending |

Joint post-warmup signal distribution, GPU launch snapshots, and controller restart/adoption evidence: pending.

## Production handoff

- Hamster run/state/controller: pending
- Ferret run/state/controller: pending
- Host/GPU/job assignments: see the fixed campaign YAML; live evidence pending
- Stop command: `scripts/campaign/campaign-stop --run-dir <host-run-dir>`
- Recovery: rerun the matching host `start` command; it adopts a matching live phase and never retries a nonzero
  scientific phase automatically.

Seed extension, direct-training baselines, full SAAD, and MobileNetV2 remain intentionally deferred.
