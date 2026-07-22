"""Bounded, observational training diagnostics keyed by stable source ID."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ard.analysis import fixed_panel_ids
from ard.engine.distributed import gather_objects, get_rank


@dataclass
class TrainingDiagnostics:
    panel_ids: tuple[int, ...]
    mode: Literal["summary", "panel"] = "panel"
    pending: list[dict[str, Any]] = field(default_factory=list)
    all_rows: dict[int, dict[str, Any]] = field(default_factory=dict)
    panel_rows: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def for_ids(
        cls, ids: list[int], *, seed: int, size: int, mode: Literal["summary", "panel"] = "panel"
    ) -> TrainingDiagnostics:
        return cls(fixed_panel_ids(ids, seed=seed, size=size) if mode == "panel" else (), mode=mode)

    def record(self, **values: Any) -> None:
        if not bool(values.pop("valid")):
            return
        values["rank"] = get_rank()
        values["order"] = len(self.pending)
        if self.mode != "panel" or int(values["sample_id"]) not in self.panel_ids:
            for field in ("clean_image", "adversarial_image", "perturbation_visualization"):
                values.pop(field, None)
        self.pending.append(values)

    def flush(self) -> None:
        rows = [row for rank_rows in gather_objects(self.pending) for row in rank_rows]
        self.pending = []
        canonical: dict[int, dict[str, Any]] = {}
        for row in sorted(rows, key=lambda row: (int(row["sample_id"]), int(row["rank"]), int(row["order"]))):
            canonical.setdefault(int(row["sample_id"]), row)
        public = {
            sample_id: {key: value for key, value in row.items() if key not in {"rank", "order"}}
            for sample_id, row in canonical.items()
        }
        self.all_rows.update(public)
        self.panel_rows = [public[sample_id] for sample_id in self.panel_ids if sample_id in public]
