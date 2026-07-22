# M2 entropy-policy DDP minimum correction

## Failure signature

`rslad_entropy` originally subtracted each rank's unmasked local entropy
minimum. Different shards therefore assigned different weights to the same
entropy value, and a low-entropy DDP padding row could change valid-sample loss
and gradients.

## Root cause and correction

The Trainer constructed its padding mask after policy evaluation, while the
policy had no validity or cross-rank reduction context. `PolicyContext` now
provides the boolean valid mask and an injected global-min reducer. Every rank
reduces a scalar candidate after masking invalid rows to infinity. The policy
fails if no finite valid entropy exists globally, computes exactly
`5 * (H_i - global_valid_min)`, detaches the result, and assigns zero weight to
invalid rows. The final padding mask and DDP `world_size/global_count` gradient
scaling remain unchanged.

The coefficient five is part of the method identity and is no longer a config
parameter. Fixed sharded loss/gradient and padded-minimum regressions cover the
correction.
