from types import SimpleNamespace

import torch

from ard.analysis.ordering_telemetry import OrderingTelemetry, descriptor_summary
from ard.cli.train import _finish_ordering_telemetry_epoch
from ard.data import IndexedBatch


def _store(margins: dict[int, float]) -> SimpleNamespace:
    return SimpleNamespace(
        records={sample_id: SimpleNamespace(margin_ema=margin) for sample_id, margin in margins.items()}
    )


def test_descriptor_summary_has_frozen_d1_to_d6() -> None:
    rows = [
        {"risk_mean": 1.0, "risk_sd": 0.1, "high_risk_fraction": 1.0},
        {"risk_mean": 2.0, "risk_sd": 0.2, "high_risk_fraction": 0.0},
        {"risk_mean": 3.0, "risk_sd": 0.3, "high_risk_fraction": 1.0},
        {"risk_mean": 4.0, "risk_sd": 0.4, "high_risk_fraction": 0.0},
    ]
    result = descriptor_summary(rows, risk_snapshot_epoch=99, risk_snapshot_sha256="abc")
    assert result["risk_definition"] == "-margin_ema"
    assert set(key for key in result if key.startswith("D")) == {
        "D1_batch_mean_risk_sd",
        "D2_within_batch_risk_sd_mean",
        "D3_high_risk_fraction_sd",
        "D4_lag1_batch_mean_risk_acf",
        "D5_hard_batch_longest_run",
        "D6_position_vs_batch_mean_risk_spearman",
    }


def test_ordering_telemetry_uses_valid_stable_ids_and_pre_epoch_snapshot(tmp_path) -> None:
    telemetry = OrderingTelemetry(tmp_path / "telemetry")
    store = _store({index: float(index) for index in range(5)})
    batch = IndexedBatch(
        images=torch.zeros(3, 3, 2, 2),
        labels=torch.tensor([0, 1, 2]),
        sample_ids=torch.tensor([4, 1, 0]),
        state_update_mask=torch.tensor([True, True, False]),
        multiplicity=torch.ones(3, dtype=torch.long),
    )
    telemetry.on_batch(100, 0, batch, store)
    telemetry.finish_epoch(100)
    row = telemetry.batch_path.read_text(encoding="utf-8").strip()
    assert "\"valid_sample_ids\":[4,1]" in row
    assert telemetry.last_descriptor is not None
    assert telemetry.last_descriptor["risk_snapshot_epoch"] == 99
    assert telemetry.last_descriptor["valid_sample_count"] == 2


def test_train_epoch_finalization_flushes_ordering_telemetry() -> None:
    class _Recorder:
        def __init__(self) -> None:
            self.epochs: list[int] = []

        def finish_epoch(self, epoch: int) -> None:
            self.epochs.append(epoch)

    recorder = _Recorder()
    _finish_ordering_telemetry_epoch(recorder, 114)
    _finish_ordering_telemetry_epoch(None, 114)
    assert recorder.epochs == [114]
