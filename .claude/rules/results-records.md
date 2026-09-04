---
paths: ["src/ard/analysis/**", "scripts/aggregate_*.py", "scripts/analysis/**", "docs/**"]
---
# Results and records
A campaign is finished when its record is committed, not when its GPUs go idle.

## Aggregation output
- An aggregate script builds **one result dict** and emits both the JSON record and the Markdown report from it.
  Never hand-write numbers into a report, and never let the two drift.
- Every record carries `contract` (a versioned string), `source_git_sha`, and hash-bound lineage for each input
  it consumed (run IDs, checkpoint filenames + SHA-256, threshold/prefix manifests + their `.sha256`).
  Refuse to aggregate when a contract string or a lineage hash mismatches; do not "fix" it by relaxing the check.
- Write `<record>.sha256` next to every record: a single bare 64-hex line, no filename, no extra text.
- The Markdown report links its record file by path and states the same contract string.
- **Non-overwrite**: a record/report path is written once. A correction is a new versioned file (or an explicit
  amendment record) plus a line saying what it supersedes — never an in-place edit of a committed result.

## Commit boundary
- The first work after a campaign ends: import record + report into `docs/`, tick the plan's milestones and
  write its completion section, commit all of it **in one commit**, then write the decision packet.
- Do not open a postmortem, retro or narrative document before the record itself is committed.

## Decision packets
`docs/decisions/NNNN-<slug>.md`, zero-padded, next = max existing + 1. YAML frontmatter keys, in order:
`id`, `status` (`pending|decided|superseded`), `created`, `campaign`, `question`, `options` (mapping option id →
one line: what runs, GPU hours, what it decides, preregistered rule), `recommendation`, `chosen` (`null` until a
human fills it). Body = evidence, numbers, links. No new scientific job launches while `chosen` is null.

## What docs may be
Documents are **generated** (aggregate output) or **decisions** (packets, plans). There is no third category:
no process narration, no "what I did today", no rule accretion. Retire a doc with its subsystem.

## Claims discipline
- Two seeds give a directional verdict for those seeds. Never phrase it as a population or "method X is better"
  claim, and never report mean/std over two runs as if it estimated a distribution.
- Internal validation (training-time PGD, pilot epochs, smoke, synthetic or mocked adapters) is never an
  official test result. Keep clean / PGD / AutoAttack and best / last separate everywhere they are reported.
- Report the fixed identity with every number: dataset, student, method, teacher, training + evaluation seed,
  checkpoint (best or last), complete threat identity, world size and effective global batch size. Runs that
  differ in world size or global batch size are not pooled.
