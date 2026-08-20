from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ard.analysis.ert_cw_reliability_gated import prepare_selector_bundle


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    mask = tmp_path / "mask.json"
    mask.write_text(
        json.dumps(
            {
                "contract": "ert_state_overlay_v1",
                "anchor_epoch": 79,
                "masks": {"student_clean_wrong": {"selected_ids": [1, 2, 3], "selected_labels": [0, 1, 2]}},
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {"sample_id": i, "true_label": i - 1, "teacher_adv_margin": margin}
        for i, margin in [(1, 0.1), (2, -0.1), (3, 0.2)]
    ]
    ce_rows, kl_rows = tmp_path / "ce.parquet", tmp_path / "kl.parquet"
    pq.write_table(pa.Table.from_pylist(rows), ce_rows)
    pq.write_table(
        pa.Table.from_pylist([{**row, "teacher_adv_margin": -row["teacher_adv_margin"]} for row in rows]), kl_rows
    )

    def meta(path: Path, contract: str, rows_path: Path) -> Path:
        value = {
            "feature_epoch": 79,
            "full_train_order_replayed": True,
            "checkpoint_sha256": "ad43d72da2a02f205c65b96485379c9acb5fc2b07d6823d09820439aedc8f78c",
            "mask_sha256": __import__("hashlib").sha256(mask.read_bytes()).hexdigest(),
            "contract": contract,
            "rows_sha256": __import__("hashlib").sha256(rows_path.read_bytes()).hexdigest(),
        }
        out = path.with_suffix(".json")
        out.write_text(json.dumps(value), encoding="utf-8")
        return out

    return (
        mask,
        meta(tmp_path / "ce", "ert_clean_wrong_c0_ce_pgd20_features_v1", ce_rows),
        ce_rows,
        meta(tmp_path / "kl", "ert_clean_wrong_c0_kl_pgd10_features_v1", kl_rows),
        kl_rows,
    )


def test_prepare_bundle_uses_fixed_positive_margin_and_confusion_groups(tmp_path: Path) -> None:
    mask, ce_meta, ce_rows, kl_meta, kl_rows = _fixture(tmp_path)
    result = prepare_selector_bundle(
        run="L2",
        mask_path=mask,
        ce_meta=ce_meta,
        ce_rows=ce_rows,
        kl_meta=kl_meta,
        kl_rows=kl_rows,
        output_dir=tmp_path / "out",
    )
    assert result["counts"]["CE20_reliable"] == 2
    assert result["counts"]["KL10_reliable"] == 1
    assert result["counts"]["RR"] == 0
    assert result["counts"]["RU"] == 2
    assert result["counts"]["UR"] == 1
    assert result["counts"]["UU"] == 0
