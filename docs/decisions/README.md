# Decision packets
Scientific decisions (new arm, seed, horizon, official test, promotion) are the human's: Claude writes a packet
here and stops, launching no scientific job while `chosen` is `null`. A packet is `NNNN-<slug>.md` (zero-padded,
next = highest + 1) = the frontmatter below + a Markdown body of evidence; a reversal is a new, superseding one.
```yaml
id: 0001
status: pending      # pending | decided | superseded
created: 2026-09-05
campaign: <campaign id>
question: <the one question being decided>
options:             # id -> one line: what runs, GPU hours, what it decides, preregistered rule
  A: ...
recommendation: A
chosen: null         # the human writes an option id here (or says so in chat)
```
