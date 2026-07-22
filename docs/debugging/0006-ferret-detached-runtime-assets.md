# Ferret detached worktree omitted verified teacher runtime assets

## Failure

The first two-GPU Chen pilot at fixed SHA `4b1b30c` stopped before W&B initialization or training with:

```text
RobustBench teacher preflight failed: teacher config does not exactly match registry entry chen2021_ltd_wrn34_10
```

The configured checkpoint was the verified shared dataset source, while `TeacherRegistry.validate_config` intentionally requires the canonical project cache path. A detached Git worktree contains neither ignored `teacher_cache/` nor `.external/robustbench`, so the next strict check would also have lacked the pinned constructor/license checkout.

## Root cause and fix

`ferret-prepare` created only tracked files and did not attach host-local, Git-ignored runtime assets. It now links exactly the source clone's `.external` and `teacher_cache` directories into each detached worktree and records those resolved shared paths in the run manifest. The shared source assets are bootstrapped once from the pinned external lock and hash-verified checkpoint sources; run worktrees do not duplicate the large checkpoint bytes.

The regression test checks the allowlisted shared asset names, link command, and manifest lineage. Attack, normalization, checkpoint SHA, teacher architecture, and objective settings are unchanged. The failed run remains preserved; retry uses a new run ID and fixed SHA.
