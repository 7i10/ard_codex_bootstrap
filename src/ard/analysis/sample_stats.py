"""Small, stable sample-level evaluation artifacts.

Parquet is intentionally not emulated.  Users who request sample statistics
must install the declared optional dependency; otherwise evaluation fails before
reporting a misleading file with a ``.parquet`` suffix.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


class ParquetDependencyError(RuntimeError):
    """Raised when a requested parquet artifact cannot be written faithfully."""


def fixed_panel_ids(sample_ids: Iterable[int], *, seed: int, size: int) -> tuple[int, ...]:
    """Choose a deterministic sparse panel independent of loader order."""
    if size < 0:
        raise ValueError("panel size must be non-negative")
    unique = sorted({int(sample_id) for sample_id in sample_ids})
    ranked = sorted(
        unique,
        key=lambda sample_id: hashlib.sha256(f"{seed}:{sample_id}".encode()).digest(),
    )
    return tuple(ranked[:size])


def write_sample_parquet(rows: Iterable[Mapping[str, Any]], path: Path) -> Path:
    """Write genuine Parquet or fail with an actionable optional-dependency error."""
    materialized = [dict(row) for row in rows]
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised without optional extra
        raise ParquetDependencyError(
            "sample statistics require pyarrow; install `ard[tracking]` before requesting Parquet output"
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(materialized), temporary)
    temporary.replace(path)
    return path
