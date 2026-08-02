"""Read-only H5-Early common-PGD panel analysis."""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ard.analysis.rslad_signal_replay import FEATURE_EPOCHS, OUTCOME_EPOCHS, PANEL_EMA_BETA, repository_root_from_source
from ard.analysis.signal_audit import SignalAuditError, binary_metrics, bootstrap_metric_delta, sha256_file


class HistoryEarlyError(ValueError):
    pass


ANCHORS = (39, 59, 79, 99)
PEAK_WINDOW = (99, 104, 109)
OBS_COLS = (
    "namespace",
    "sample_id",
    "class_id",
    "epoch",
    "teacher_entropy_normalized",
    "student_probability_margin",
    "student_margin_risk",
    "robust_correct",
)


def _tracked_clean_provenance() -> dict[str, Any]:
    """Bind a canonical H5-Early report to its exact clean analysis sources."""
    root = repository_root_from_source()
    paths = {
        "history_early": Path(__file__).resolve(),
        "history_early_cli": root / "src/ard/cli/history_early.py",
        "rslad_signal_replay": root / "src/ard/analysis/rslad_signal_replay.py",
    }
    try:
        relative = [str(path.relative_to(root)) for path in paths.values()]
        subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", *relative], check=True, capture_output=True
        )
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise HistoryEarlyError("H5-Early requires tracked source files and Git identity") from exc
    if len(sha) != 40 or dirty:
        raise HistoryEarlyError("H5-Early requires a tracked-clean revision")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    return {"git": {"sha": sha, "dirty": False}, "source_files": hashes}


def _json(path: Path) -> dict[str, Any]:
    try:
        v = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise HistoryEarlyError("unreadable lineage") from e
    if not isinstance(v, dict):
        raise HistoryEarlyError("lineage must be mapping")
    return v


def _parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq

        t = pq.read_table(path)
    except Exception as e:
        raise HistoryEarlyError("unreadable observation panel") from e
    if not set(OBS_COLS).issubset(t.column_names):
        raise HistoryEarlyError("observation column contract drifted")
    return [dict(x) for x in t.to_pylist()]


def _i(x: object, n: str) -> int:
    if isinstance(x, bool) or not isinstance(x, int) or x < 0:
        raise HistoryEarlyError(f"{n} must be nonnegative integer")
    return x


def _f(x: object, n: str, lo: float, hi: float) -> float:
    if (
        isinstance(x, bool)
        or not isinstance(x, (int, float))
        or not math.isfinite(float(x))
        or not lo <= float(x) <= hi
    ):
        raise HistoryEarlyError(f"{n} outside contract")
    return float(x)


def _panel(
    rows: Sequence[Mapping[str, Any]], epochs: Sequence[int], count: int, name: str
) -> dict[int, dict[int, dict[str, Any]]]:
    out = {e: {} for e in epochs}
    for r in rows:
        e = _i(r.get("epoch"), name + " epoch")
        sid = _i(r.get("sample_id"), name + " sample_id")
        cls = _i(r.get("class_id"), name + " class_id")
        if e not in out or r.get("namespace") != "train" or sid >= 50000 or cls >= 10 or sid in out[e]:
            raise HistoryEarlyError(name + " epoch/ID contract drifted")
        m = _f(r.get("student_probability_margin"), name + " margin", -1, 1)
        risk = _f(r.get("student_margin_risk"), name + " risk", 0, 1)
        _f(r.get("teacher_entropy_normalized"), name + " entropy", 0, 1)
        if not isinstance(r.get("robust_correct"), bool) or not math.isclose(risk, (1 - m) / 2, abs_tol=1e-7):
            raise HistoryEarlyError(name + " signal contract drifted")
        out[e][sid] = {"c": cls, "m": m, "r": risk, "ok": r["robust_correct"]}
    if any(len(v) != count for v in out.values()):
        raise HistoryEarlyError(name + " lacks exact stable ID coverage")
    ref = next(iter(out.values()))
    if any(set(v) != set(ref) or any(v[k]["c"] != ref[k]["c"] for k in ref) for v in out.values()):
        raise HistoryEarlyError(name + " class identity drifted")
    return out


def _lineage(path: Path, obs: Path, key: str, count: int) -> dict[str, Any]:
    x = _json(path)
    if (
        x.get("schema_version") != 1
        or x.get("observation_schema_version") != 2
        or x.get("train_expected_count") != count
        or x.get(key) != sha256_file(obs)
    ):
        raise HistoryEarlyError("lineage byte/count binding drifted")
    for k in (
        "run_id",
        "config_hash",
        "scientific_git_sha",
        "attack_identity",
        "dataset_identity",
        "teacher",
        "feature_protocol" if key.startswith("feature") else "outcome_protocol",
    ):
        if k not in x:
            raise HistoryEarlyError("lineage identity incomplete")
    return x


def _mid(v: Mapping[int, float]) -> dict[int, float]:
    order = sorted((x, k) for k, x in v.items())
    out = {}
    s = 0
    while s < len(order):
        e = s + 1
        while e < len(order) and order[e][0] == order[s][0]:
            e += 1
        for _, k in order[s:e]:
            out[k] = (s + (e - s) / 2) / len(order)
        s = e
    return out


def _score(panel: Mapping[int, Mapping[int, Mapping[str, Any]]], a: int) -> dict[str, dict[int, float]]:
    es = tuple(e for e in FEATURE_EPOCHS if e <= a)
    hist = es[:-1]
    if not hist or es[-1] != a:
        raise HistoryEarlyError("anchor schedule drifted")
    freq = {}
    margin = {}
    instant = {}
    curr = {}
    for sid in panel[a]:
        ema = None
        hits = 0
        for e in hist:
            ema = (
                panel[e][sid]["m"] if ema is None else PANEL_EMA_BETA * ema + (1 - PANEL_EMA_BETA) * panel[e][sid]["m"]
            )
            hits += int(panel[e][sid]["ok"])
        freq[sid] = 1 - hits / len(hist)
        margin[sid] = (1 - ema) / 2
        instant[sid] = panel[a][sid]["r"]
        curr[sid] = float(not panel[a][sid]["ok"])
    fr, mr, ir, cr = _mid(freq), _mid(margin), _mid(instant), _mid(curr)
    return {
        "adaptive_history": {k: (fr[k] + mr[k]) / 2 for k in freq},
        "frequency_only": freq,
        "margin_only": margin,
        "instantaneous_margin": instant,
        "current_correctness": curr,
        "outcome_free_current_rank": {k: (ir[k] + cr[k]) / 2 for k in freq},
    }


def _metric(scores: Mapping[int, float], y: Mapping[int, int]) -> dict[str, float | None]:
    try:
        r = binary_metrics([y[k] for k in sorted(y)], [scores[k] for k in sorted(y)])
    except SignalAuditError:
        return {"auroc": None, "auprc": None, "prevalence": sum(y.values()) / len(y)}
    return {"auroc": r["auroc"], "auprc": r["auprc"], "prevalence": r["prevalence"]}


def analyze_history_early(
    *,
    feature_observations: Path,
    outcome_observations: Path,
    feature_lineage: Path,
    outcome_lineage: Path,
    expected_count: int,
    analysis_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fmeta = _lineage(feature_lineage, feature_observations, "feature_observations_sha256", expected_count)
    ometa = _lineage(outcome_lineage, outcome_observations, "outcome_observations_sha256", expected_count)
    if any(
        fmeta[k] != ometa[k]
        for k in ("run_id", "config_hash", "scientific_git_sha", "attack_identity", "dataset_identity", "teacher")
    ):
        raise HistoryEarlyError("feature/outcome lineage identity drifted")
    f = _panel(_parquet(feature_observations), FEATURE_EPOCHS, expected_count, "feature")
    o = _panel(_parquet(outcome_observations), OUTCOME_EPOCHS, expected_count, "outcome")
    if set(f[99]) != set(o[99]) or any(f[99][k]["c"] != o[99][k]["c"] for k in f[99]):
        raise HistoryEarlyError("feature/outcome epoch99 join drifted")
    peak = {k: int(sum(not o[e][k]["ok"] for e in PEAK_WINDOW) >= 2) for k in o[99]}
    post = {}
    for k in o[109]:
        if o[109][k]["ok"]:
            prev = True
            forgot = False
            for e in (x for x in OUTCOME_EPOCHS if x > 109):
                now = o[e][k]["ok"]
                forgot |= prev and not now
                prev = now
            post[k] = int(forgot)
    tables = {}
    gate_rows: dict[str, list[dict[str, Any]]] = {}
    for a in ANCHORS:
        s = _score(f, a)
        post_scores = {n: {k: v for k, v in values.items() if k in post} for n, values in s.items()}
        tables[str(a)] = {
            "peak_window_error": {n: _metric(v, peak) for n, v in s.items()},
            "post_peak_forgetting": {n: _metric(values, post) for n, values in post_scores.items()},
        }
        gate_rows[str(a)] = [
            {
                "sample_id": sample_id,
                "class_id": o[109][sample_id]["c"],
                "outcome": post[sample_id],
                "baseline": post_scores["outcome_free_current_rank"][sample_id],
                "candidate": post_scores["adaptive_history"][sample_id],
            }
            for sample_id in sorted(post)
        ]
    return {
        "schema_version": 1,
        "contract": "h5_early_common_pgd_v1",
        "input_identity": {
            "run_id": fmeta["run_id"],
            "config_hash": fmeta["config_hash"],
            "feature_observations_sha256": sha256_file(feature_observations),
            "outcome_observations_sha256": sha256_file(outcome_observations),
            "feature_attack_domain": fmeta["feature_protocol"],
            "outcome_attack_domain": ometa["outcome_protocol"],
        },
        "anchors": list(ANCHORS),
        "outcomes": {
            "peak_window_error": "wrong_at_least_two_of_99_104_109",
            "post_peak_forgetting": "epoch109_correct_then_later_correct_to_wrong",
        },
        "tables": tables,
        "_post_peak_gate_rows": gate_rows,
        "analysis_provenance": dict(
            _tracked_clean_provenance() if analysis_provenance is None else analysis_provenance
        ),
    }


def collection_gate(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows = {}
    for a in ANCHORS:
        d = {}
        for label in ("L1", "L3"):
            t = reports.get(label, {}).get("tables", {}).get(str(a), {}).get("post_peak_forgetting", {})
            baseline = t.get("outcome_free_current_rank", {}).get("auroc")
            ad = t.get("adaptive_history", {}).get("auroc")
            d[label] = ad - baseline if isinstance(ad, float) and isinstance(baseline, float) else None
        point_pass = all(delta is not None and delta >= 0.02 for delta in d.values())
        result: dict[str, Any] = {
            "adaptive_minus_predeclared_current_rank_auroc": d,
            "predeclared_current_rank_diagnostic": "outcome_free_current_rank",
            "point_threshold_met": point_pass,
            "paired_lower_bound": None,
        }
        if not point_pass:
            result["status"] = "no_go_point_gate"
        else:
            bounds: dict[str, Any] = {}
            try:
                for label in ("L1", "L3"):
                    sample_rows = reports[label]["_post_peak_gate_rows"][str(a)]
                    bounds[label] = bootstrap_metric_delta(
                        sample_rows,
                        baseline=[float(row["baseline"]) for row in sample_rows],
                        candidate=[float(row["candidate"]) for row in sample_rows],
                        seed=2026073102,
                        replicates=2000,
                    )
            except (KeyError, SignalAuditError, TypeError, ValueError):
                result["status"] = "pending_paired_ci_inputs"
            else:
                result["paired_lower_bound"] = {label: float(bounds[label]["lower"]) for label in bounds}
                result["paired_bootstrap"] = bounds
                result["status"] = "pending_admissibility_review"
        rows[str(a)] = result
    return {
        "outcome": "post_peak_forgetting",
        "criterion": (
            "point delta>=0.02 on L1/L3, then frozen class-stratified paired bootstrap "
            "(seed=2026073102, n=2000) against outcome_free_current_rank; "
            "no automatic Go before admissibility review"
        ),
        "status": "sequential_no_automatic_go",
        "anchors": rows,
    }
