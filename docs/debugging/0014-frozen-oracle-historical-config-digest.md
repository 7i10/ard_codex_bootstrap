# Frozen-oracle historical resolved-config digest

## Symptom

The frozen-oracle builder rejected the Bartoldson/RSLAD epoch-99 checkpoint:
the checkpoint and saved raw `resolved_config.yaml` used
`b105b3dacec6bf68c722e95b01616c5800cc1febb743e5d54719aa095cbc222a`,
while re-validating that YAML through the current schema then hashing
`model_dump` produced a different digest (`d505…`).

## Cause and correction

The current schema introduced default fields after the source run. Pydantic
injected them during validation, so its current resolved mapping was not the
historical checkpoint identity. The builder now hashes the saved YAML mapping
directly, validates that mapping independently, and passes that immutable hash
through checkpoint, W&B inventory, replay, and generated-mask lineage checks.
It never derives historical identity from a current `model_dump`.

## Regression

`test_historical_raw_resolved_mapping_hash_survives_new_schema_defaults`
constructs an old raw mapping without the new fields, proves its hash differs
from the current schema-expanded hash, and requires oracle construction to bind
the old checkpoint hash exactly.
