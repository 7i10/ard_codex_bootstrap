# ARD workspace contract

`configs/workspace/ard_workspace_v1.json` is the authority for future ARD
runtime paths on Hamster and Ferret.  It deliberately distinguishes a shared,
canonical logical path from the local bind-mount/symlink realization.

```text
repository:    /home/shunsukenaito/workspace-local/ard_codex_bootstrap
datasets:      /home/shunsukenaito/workspace-local/datasets
runtime root:  /home/shunsukenaito/workspace-local/ard-runtime/ard_codex_bootstrap
```

Future runtime writes belong only under the registered runtime root:

```text
runs/  analysis/  staging/  worktrees/  orchestration/
task-context/  locks/  tmp/
```

Use `ard.workspace.load_workspace_contract()` in Python and
`scripts/workspace_doctor.py --json` for the compact operational view.  The
doctor is not a scientific launch gate: parent, Teacher, attack, mask, and
checkpoint lineage are still checked by the production-launch gate.

For long-task recovery, use the runtime-only [task-context protocol](TASK_CONTEXT_PROTOCOL.md).

Historical roots such as `ard-runs`, `ard-analysis`, and
`ard-campaign-runs` remain read-only because frozen reports/configs reference
them.  Do not migrate or delete them merely to satisfy this contract.  The
cleanup inventory records explicit evidence before any candidate deletion.

The tracked registry may list host GPU UUIDs and observed throughput solely as
operational placement metadata.  They are not scientific hyperparameters.
