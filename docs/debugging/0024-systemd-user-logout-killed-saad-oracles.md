# Long SAAD user services stopped on logout

## Failure signature

The Chen WRN34-20 U and P full-SAAD services started from clean SHA `52affda`
on Hamster GPUs 0 and 1 at 2026-08-07 06:43:05 JST. Both were healthy until
2026-08-07 17:23:41, when they stopped in the same second without
`terminal.json`. U was in epoch 137 and P in epoch 134; neither reached the
only upstream model save at the end of training.

## Evidence and root cause

- The user journal records `Activating special unit exit.target`, then stops
  both SAAD services together with D-Bus, PipeWire and the rest of the user
  manager. The host did not reboot.
- `loginctl show-user shunsukenaito` reports `Linger=no`. The jobs were
  transient `systemd --user` services and therefore did not outlive the last
  login session.
- U/P telemetry contains no GPU error. Peak memory was `7,566/7,506 MiB`, peak
  temperature was `63/76 C`, and both last telemetry samples showed active GPU
  computation.
- The pinned upstream `saad.py` saves only a final SWA `state_dict` after all
  epochs and final PGD/CW/FGSM evaluation. It has no optimizer, scheduler, RNG,
  SWA or epoch checkpoint and no resume entrypoint.

The root cause is a missing host-lifetime preflight, not OOM, numerical failure
or a scientific code error. `systemd-run --user` was incorrectly treated as
detached persistence while lingering was disabled.

## Bounded fix

Before any long `systemd --user` job, require:

```bash
test "$(loginctl show-user "$USER" -p Linger --value)" = yes
```

Enable lingering once through the host's normal administrator policy, then
recheck the value before launch. Reuse the existing successful scientific and
GPU smoke evidence. Restart exact U/P oracles from epoch 0 in fresh output
directories; do not retrofit resume into the unmodified-upstream identity.

## Verification boundary

No attack, objective, teacher, optimizer, source or runtime input changed, so
numerical/unit/GPU smoke tests are not rerun. Verification is the static
`Linger=yes` preflight, fresh non-overwriting output paths, clean SHA/source
identity, and process survival after the initiating SSH/VS Code session ends.
The interrupted per-epoch test PGD values are exploratory operational evidence
and must not be used for checkpoint or hyperparameter selection.
