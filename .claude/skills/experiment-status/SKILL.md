---
name: experiment-status
description: Report the current state of the ARD research platform — GPU hosts, running and terminal campaigns, hand-run bundles, pending decision packets, recent headless postruns. Use at session start, when the user asks "what is running" / "where are we", or before deciding whether a launch is safe.
---

# experiment-status

`$ARGUMENTS` is normally empty. Ignore anything you do not recognise.

## Steps

1. Run the full status report (Markdown, no flags):

   ```bash
   python3 scripts/ardx/status.py
   ```

   Run it from the repo root. It finishes in under 15 s even when Ferret is
   unreachable; do not add a timeout wrapper and do not run it twice.
2. If it exits non-zero, show its stderr and stop. Do not hand-reconstruct the
   report by reading `state.json` files yourself.
3. Summarise in **12 lines or fewer**, in this order, dropping empty sections:
   - hosts: one line each for Hamster and Ferret (GPU free/used, or `unreachable`);
   - watcher: active / inactive, plus the newest event time;
   - campaigns: only non-terminal ones by name and job counts; then one line
     for terminal-but-unimported ones ("terminal, postrun not recorded");
   - hand-run bundles that are `running`, `stale` or `failed`;
   - pending decisions: count and their ids;
   - newest postrun: id and outcome.
4. State the derived campaign status exactly as `status.py` derived it (all jobs
   in `{completed,failed,blocked,orphaned}` → terminal; all `completed` → success).
   Never re-derive it from the campaign-level `status` field.
5. If the watcher is inactive, say so and quote the install path:

   ```bash
   scripts/ardx/install_units.sh --dry-run   # then, after the user reads it:
   scripts/ardx/install_units.sh --install
   ```

   Say that the user runs `--install`; you do not.
6. If a campaign is terminal and its record is not yet in `docs/experiments/`,
   end with one line naming `/experiment-postrun <campaign>` as the next step.
   Do not start it unless asked.

## Stop conditions

- `scripts/ardx/status.py` is missing or exits non-zero → report the error and stop.
- The report shows a campaign in a state you cannot classify → say so plainly
  and stop; do not guess success or failure.
- More than one campaign is non-terminal → report both and stop before
  recommending any launch.

## Never

- Never run `install_units.sh --install`, `systemctl --user enable/start`, or any
  other command that changes daemon state.
- Never poll: no `sleep`, no `watch`, no repeated `nvidia-smi`, no loop over
  `status.py`. To wait, arm `Monitor` or end the turn.
- Never launch, retry, aggregate, commit or edit anything from this skill.
- Never read GPU utilisation or PIDs as evidence that a job succeeded; the only
  completion signal is the campaign/hand-run contract in `status.py`.
