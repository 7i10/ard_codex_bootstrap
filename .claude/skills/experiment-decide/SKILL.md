---
name: experiment-decide
description: Turn a just-imported campaign result into docs/decisions/NNNN-<slug>.md — 2-4 costed options with what each would establish, the preregistered decision rule, the noise floor, and a recommendation — then stop for the human. Use after /experiment-postrun, or when the user asks what next / write a decision packet.
---

# experiment-decide

`$ARGUMENTS` = a campaign id (or the contract name of the record just imported).

## 1. Gather the evidence

Read, in this order: `docs/experiments/<contract>.json` (numbers), the matching
`docs/<REPORT>.md` (interpretation), the plan's Completion report, and the
evidence ledger in `docs/RESEARCH_STATUS_SUMMARY.md` (the live status summary
for the current research direction; the ARD/RSLAD-era one is frozen at
`docs/archive/ard-distillation-2026/ERT_RESEARCH_STATUS_SUMMARY.md` and is
background reading only, not a live constraint).
Options must not contradict standing decisions already recorded there; if one
does, say why explicitly.

## 2. Cost each option

Use `configs/workspace/ard_workspace_v1.json` host throughput and known per-epoch
costs. Anchor: **one e101–114 arm ≈ 28 min on a Hamster 4090**. Scale by arms ×
seeds × epochs, add endpoint evaluation and aggregation, and state the assumption
you scaled from. Give a range, not a false-precision single number.

## 3. State the noise floor

RNG alone moves e114 held-out accuracy by **1–2 pp**. An effect below that is not
decidable with 2 seeds. For every option, say whether its expected effect clears
the floor, and if not, what would (more seeds, a different endpoint, a longer
horizon). Never propose reading a sub-floor difference as a result.

## 4. Write the packet

`docs/decisions/NNNN-<slug>.md`, `NNNN` = zero-padded `max(existing) + 1`
(`0001` if the directory is empty). Frontmatter keys exactly:

```yaml
---
id: NNNN
status: pending
created: YYYY-MM-DD
campaign: <campaign id>
question: <one line, the actual choice>
options:
  A: <one line>
  B: <one line>
recommendation: A
chosen: null
---
```

Body, in Markdown, short: the evidence summary with the actual numbers and links
to record and report; then one subsection per option with **GPU-hours estimate**,
**what it would establish**, **preregistered decision rule**, and **risks**; then
one paragraph justifying the recommendation, including what would change it.

2–4 options. Include the option of doing nothing further when it is real.

## 5. Stop

Print the packet path and the one-line question. Say the human fills `chosen:`
(by editing the file or answering in chat) and that no new scientific job starts
until then. End the turn.

## Stop conditions

- No imported record for `$ARGUMENTS` → run `/experiment-postrun` first; stop.
- A decision packet for this campaign already exists with `status: pending` →
  report its path and stop. Do not open a second one.
- You cannot state a preregistered decision rule for an option → drop that option.
- Fewer than 2 defensible options remain → say so and stop.

## Never

- Never set `chosen`, never write `status: decided`, never edit a packet the
  human already decided (supersede it with a new NNNN instead).
- Never launch, queue, arm or pre-stage any option, and never call
  `/experiment-launch` from here.
- Never propose an option that changes epsilon, steps, step size, random start,
  normalization, temperature, schedule, checkpoint selection or evaluation
  attacks without naming it as a new scientific contract needing a new plan.
- Never present a sub-noise-floor difference as a finding, and never restate a
  post-hoc threshold or subgroup as if it had been preregistered.
