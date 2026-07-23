# W&B canonical group length gate

## Failure

The corrected output-layout campaign at `712b8789f78f0590ff57274a1566911e7230985b` reached W&B initialization.
Both Chen pilots initialized online and entered training, but the Bartoldson pilot failed before model construction:

```text
wandb.errors.errors.CommError: invalid parameters: 128 limit exceeded for GroupName
```

The configured comparison base was short. `canonical_run_group` then appended the full protocol ID, teacher registry
ID, and execution identity. Bartoldson's longer registry ID pushed the resulting W&B group beyond the service limit.
No Bartoldson training batch or scientific metric was produced.

## Fix

The complete canonical group string remains the source identity. Strings at or below the limit are unchanged.
Longer strings retain a readable prefix and end in `-sha256-<16 hex>`, where the digest is computed from the complete
unshortened identity. This keeps groups deterministic and distinguishes teacher, protocol, and execution profile while
meeting W&B's 128-character gate. Method and seed remain excluded from the comparison-group identity.

Tests cover the long-group bound, determinism, complete-identity sensitivity, and unchanged short-group behavior.
Because tracking identity and Git SHA changed, production acceptance requires all three pilots from the next fixed SHA;
the two still-running Chen jobs at `712b878` are operational evidence only.
