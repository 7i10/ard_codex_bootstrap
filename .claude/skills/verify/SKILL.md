---
name: verify
description: Run the repository's impact-selected test gate (scripts/verify.py --changed) with the project Python, showing the selection first and reusing cached passes. Use before any commit, after edits to src/ard or configs, and whenever the user says verify / run the tests / check the gate.
---

# verify

`$ARGUMENTS` is optional. The only supported token is `--force`; ignore anything
else and say you ignored it.

## Steps

1. Preview the selection (never skip this):

   ```bash
   cd /home/islab/workspace-local/shunsuke.naito/ard_codex_bootstrap
   PYTHONPATH=src /home/shunsukenaito/.conda/envs/adv/bin/python \
     scripts/verify.py --changed --dry-run
   ```

   Show the selected commands verbatim and which are cached passes. If the
   selection is empty, say "no tests implicated" and stop.
2. If the selection is unexpectedly large or includes GPU-bound tests, say so
   and ask before running. Otherwise run it for real:

   ```bash
   PYTHONPATH=src /home/shunsukenaito/.conda/envs/adv/bin/python \
     scripts/verify.py --changed
   ```

   Add `--force` **only** when `$ARGUMENTS` contains it or the user asked. Never
   add `--non-scientific` on your own initiative.
3. Report: passed / failed counts, and which commands were reused from cache
   rather than re-executed.
4. On failure, show the exact failing command and the **last 30 lines** of its
   output. Diagnose in one paragraph. Fix only if the fix is obviously
   mechanical (import, typo, fixture path); otherwise report and stop.
5. After a fix, re-run step 2 once. Do not loop.

## Notes

- Bare `python` does not exist here; always use the absolute conda path above.
- `--tier`, `--failed`, `--smoke`, `--base` exist but are not part of this skill;
  use one only when the user names it.
- Do not preview twice, and do not run a second full invocation as a "status
  check" — one dry-run plus at most one real run per fix cycle.

## Stop conditions

- `scripts/verify.py` or the conda Python is missing → report the path and stop.
- A test fails for a scientific reason (accuracy, tolerance, attack strength,
  determinism) → stop and report; that is a finding, not a fix task.
- The same command fails twice → stop and report; escalate to `/ard-bug-hunt`.
- A run needs a GPU that is busy with a live campaign → stop and say so.

## Never

- Never widen a tolerance, lower an epsilon or step count, weaken an attack,
  skip or xfail a test, or set an env var to make the gate pass.
- Never pass `--force` to hide a stale cache problem, and never delete or edit
  the pass-cache to force a re-run.
- Never re-run an unchanged passing command without `--force`.
- Never commit, push or edit `src/ard/**` scientific behaviour from this skill.
- Never `sleep` or poll while a test runs.
