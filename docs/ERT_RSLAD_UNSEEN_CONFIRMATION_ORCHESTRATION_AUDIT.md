# ERT/RSLAD Unseen Confirmation — Orchestration Audit

This is a non-invasive, pre-completion audit of the unseen-seed confirmation
campaign. It was assembled from records already visible to Codex; no running
production process, GPU assignment, seed, checkpoint, launcher, W&B setting,
or completion marker was changed.

## What is recorded

- Five technical smoke retries were separated from scientific failures.
- The successful Ferret smoke reached real data/teacher initialization, finite
  epoch-0 metrics, W&B initialization, and checkpoint/output creation.
- The frozen seed registry and source commit are recorded as lineage events.
- Prior measured throughput is retained as a scheduling reference: Hamster was
  about 679 images/s, Ferret GPU0/1 about 600 images/s, and Ferret GPU2 about
  425 images/s in the existing runtime audit.
- Production per-run timing, parent-to-child delay, completion-marker use, and
  final completion are intentionally `unknown` until a later non-invasive
  collection turn.

## Retry classification

The wrong Python path, missing `ARD_SEED`, and Hamster-path leakage were
orchestration/path errors. The production and pilot one-epoch rejections were
contract safeguards, not experiment failures. No retry changed a scientific
seed or treatment.

## Stable launch and monitoring

The bounded smoke was stable once it had finite metrics and a writable output.
The cancellation helper rejected a symlink-resolved Ferret cwd; the named
smoke process was subsequently terminated by its own process group, but the
manifest status update was incomplete because the remote shell did not expose
`python` under that name. This does not change the production campaign.

No completion polling was performed for this audit. Unknown timestamps and
production fields are left unknown rather than inferred.

## Postmortem actions

The reusable improvements are: resolve remote paths during preflight, schedule
by measured longest-processing-time-first, bind dependency launches to a
single hash-bound inventory/marker, and perform one bounded stable check before
handing the campaign back. The cancellation helper's canonical-path comparison
is a repo/skill follow-up, separate from the scientific source.

Machine-readable details are in
[`ert_rslad_unseen_confirmation_orchestration_audit_v1.json`](experiments/ert_rslad_unseen_confirmation_orchestration_audit_v1.json).
