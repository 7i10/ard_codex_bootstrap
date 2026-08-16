from __future__ import annotations

from ard.analysis.ert_clean_wrong_reliability_proxy import (
    _pearson_spearman,
    _quantile_bins,
)


def test_proxy_correlations_are_finite_and_rank_tie_aware() -> None:
    result = _pearson_spearman([0.0, 1.0, 1.0, 3.0], [0.0, 2.0, 2.0, 6.0])
    assert result["pearson"] is not None
    assert result["spearman"] is not None
    assert abs(float(result["pearson"]) - 1.0) < 1e-12
    assert abs(float(result["spearman"]) - 1.0) < 1e-12


def test_quantile_bins_are_disjoint_and_cover_ids() -> None:
    ids = list(range(10))
    values = {item: float(item) for item in ids}
    bins = _quantile_bins(ids, values, bins=5)
    assert [len(item) for item in bins] == [2, 2, 2, 2, 2]
    assert sorted(sample_id for item in bins for sample_id in item) == ids
    assert not (set(bins[0]) & set(bins[1]))
