# M5 test-gate and DDP artifact race corrections

## Scope

M5 gate triage confirmed two runtime bugs: visible-CUDA cache fingerprinting could retain a non-JSON CUDA UUID
wrapper, and every DDP rank could write the same sample-statistics Parquet temporary path. Neither bug changed an
attack, objective, optimizer update, sample policy, checkpoint-selection rule, or other training mathematics.

## Visible-CUDA cache serialization

The visible-GPU path failed while hashing the test-cache payload with:

```text
TypeError: Object of type _CUuuid is not JSON serializable
```

`torch.cuda.get_device_properties(...).uuid` can be a private `_CUuuid` wrapper rather than a Python string.
`environment_identity()` had allowed that implementation object into the payload consumed by `json.dumps`, so the
failure appeared only when CUDA was visible and occurred before the selected test command could be cached.

The runtime identity now explicitly canonicalizes every GPU field to JSON primitives: integer index, string name,
two integer capability values, and UUID as either `str(uuid)` or `None`. PyTorch and CUDA versions are likewise
strings or `None`. The UUID remains part of the fingerprint, so changing the visible physical GPU changes the cache
key rather than erasing device identity.

Regression coverage is
`tests/unit/test_verify_gate.py::test_environment_identity_canonicalizes_cuda_uuid_and_fingerprint`. It injects an
opaque `_CUuuid`, requires a JSON round trip, verifies that two UUID strings produce different fingerprints, and
checks the `None` case.

## DDP Parquet temporary-path race and phase skew

The original training CLI let every rank call `write_sample_parquet` for the same destination. Each call used the
same `sample-stats-train.parquet.tmp` name before replacing the final file. The primary race signature was therefore:

```text
FileNotFoundError: [Errno 2] No such file or directory:
'.../sample-stats-train.parquet.tmp' -> '.../sample-stats-train.parquet'
```

One rank could complete the replacement while another still expected the shared temporary file. Success and failure
ranks then entered different normal-finalization or exception-tracking phases, creating collective phase skew and
incoherent secondary failure or hang behavior.

Sample-statistics publication is now a named rank-zero filesystem phase. Rank zero alone runs the Parquet writer;
`run_rank_zero_phase(..., phase="sample statistics write")` broadcasts a primitive success/error outcome before any
rank proceeds to tracker finalization. A writer exception is reconstructed coherently on every rank. The injected
failure contract is exact on both ranks:

```text
ARD_SAMPLE_STATS_FAILURE_RANK=<rank>: rank-zero sample statistics write failed (OSError): injected sample statistics failure
```

Regression coverage is
`tests/integration/test_synthetic_training.py::test_two_process_gloo_sample_statistics_are_rank_zero_only_and_fail_coherently`
with `tests/integration/torchrun_sample_stats.py`. The success branch rejects any nonzero-rank writer call and checks
the genuine Parquet output. The failure branch checks the same propagated error on ranks 0 and 1, absence of the final
Parquet file, a failed manifest, and absence of the completion marker.

## Stale production assertion

A separate gate failure in
`tests/integration/test_synthetic_training.py::test_cli_rejects_invalid_production_tracking_before_output_write` was
an assertion against an older production-tracking error contract, not a production runtime defect. The runtime still
rejected the invalid production configuration before creating its output; the stale assertion/fixture contract, not
the guard implementation, required correction.

This note records the confirmed fixes and their regressions only. It does not claim that the final M5 gate has passed.
