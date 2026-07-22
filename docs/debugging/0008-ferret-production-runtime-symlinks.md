# Ferret production runtime symlinks rejected as untracked

## Failure

After the Chen two-GPU pilot and saved-checkpoint evaluation completed, the
first canonical production launch failed before creating tracking output. The
production guard reported untracked repository files in the detached Ferret
worktree.

## Root cause and fix

`ferret-prepare` intentionally attaches only `.external` and `teacher_cache` as
symlinks to verified shared runtime assets. The corresponding `.gitignore`
patterns ended in `/`, so Git ignored real directories but not the symlink
entries themselves. Production correctly rejected those entries as untracked.

The ignore patterns now name `.external` and `teacher_cache` without a trailing
slash, covering both directories and symlinks. The production lineage guard is
unchanged: arbitrary untracked files remain forbidden, while external commits,
dirty state, and teacher checkpoint hashes are still verified independently.

A focused regression creates the same two symlinks in a temporary Git
repository and requires an empty porcelain status for those paths. The failed
production run is retained; the retry must use a fresh pushed SHA and run ID.
