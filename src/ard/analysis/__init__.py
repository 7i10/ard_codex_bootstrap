"""Post-training analysis helpers with no training-time dependencies."""

from .aggregate import summarize, summarize_checkpoint_groups
from .sample_stats import ParquetDependencyError, fixed_panel_ids, write_sample_parquet
from .signal_audit import (
    CheckpointInventory,
    SampleKey,
    SignalAuditError,
    artifact_only_temporal_diagnostics,
    associate_wandb_versions,
    audit_report,
    binary_metrics,
    deterministic_hash_split,
    inventory_run_bundle,
    load_final_sample_stats,
    periodic_last_checkpoints,
    select_prospective_checkpoints,
    validate_sample_partitions,
)

__all__ = [
    "ParquetDependencyError",
    "CheckpointInventory",
    "SampleKey",
    "SignalAuditError",
    "artifact_only_temporal_diagnostics",
    "associate_wandb_versions",
    "audit_report",
    "binary_metrics",
    "deterministic_hash_split",
    "fixed_panel_ids",
    "inventory_run_bundle",
    "load_final_sample_stats",
    "periodic_last_checkpoints",
    "select_prospective_checkpoints",
    "validate_sample_partitions",
    "write_sample_parquet",
    "summarize",
    "summarize_checkpoint_groups",
]
