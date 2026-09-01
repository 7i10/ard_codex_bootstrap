from scripts.analysis.aggregate_ert_rslad_pure_order_probes import _auc


def test_auc_accepts_adjacent_probe_points() -> None:
    rows = [
        {"epoch": epoch, "metric": float(epoch - 100)}
        for epoch in range(100, 115)
    ]

    # Trapezoidal integral of 0..14 divided by the 14-epoch span.
    assert _auc(rows, "metric") == 7.0
