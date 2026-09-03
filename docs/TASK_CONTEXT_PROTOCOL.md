# Task-context protocol

Use a runtime-only task context for long operational or scientific tasks:

```bash
python scripts/task_context.py init \
  --task-id <stable-task-id> \
  --goal '<concise objective>' \
  --source-sha "$(git rev-parse HEAD)" \
  --authoritative-file docs/ERT_RESEARCH_STATUS_SUMMARY.md \
  --pending-milestone '<next milestone>' \
  --stop-rule '<explicit stop rule>'
```

The default location is the tracked workspace registry's
`task_context_root`.  It records navigation facts only: the task goal, source
and workspace-contract hashes, audited authorities, decisions, milestones,
blockers, active jobs, and stop rules.  Use `append` after a bounded milestone
and `show` after context compression or a session handoff.

Task context never replaces a registered result artifact, exact report,
checkpoint hash, or source commit.  If context disagrees with the actual
workspace, the actual workspace wins and the context must be corrected.
