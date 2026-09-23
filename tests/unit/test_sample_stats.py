from __future__ import annotations


def test_write_sample_parquet_keeps_columns_that_first_appear_in_later_rows(tmp_path) -> None:
    import pyarrow.parquet as pq

    from ard.analysis.sample_stats import write_sample_parquet

    path = write_sample_parquet([{"a": 1}, {"a": 2, "b": 3.0}], tmp_path / "rows.parquet")
    table = pq.read_table(path)
    assert table.column_names == ["a", "b"]
    assert table.column("b").to_pylist() == [None, 3.0]

