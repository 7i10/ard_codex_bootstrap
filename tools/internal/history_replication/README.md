# History-replication provenance

This directory preserves the frozen preregistration and gate attestation used
to authorize the Bartoldson seed-1 history-replication run at Git SHA
`0fdfaeb5809e3b08a0825e2e5caf0bebfa215047`.

These documents are evidence for that completed launch decision. They are not
runtime configuration, are not imported by `ard.cli.train`, and must not be
reused as a generic launch gate. Exact resume of that run uses the original
Git SHA; future non-intervened observation runs use ordinary `rslad` plus an
explicit `observation.profile`.

`bridge/` is a bounded, one-time clean-SHA migration verifier used to decide
whether the historical observation API and the public observation-profile API
produce exactly equal optimization/sample-state checkpoints. It is internal
research provenance tooling, not a training method or installed runtime API.
