#!/usr/bin/env python3
# ruff: noqa: E501
"""Prepare the preregistered FF/NR epoch-79 causal-pilot masks.

This is a train-split, selection-time-only operation.  It never reads terminal
outcomes, validation metrics, or test data.  Route A is a rule-based joint-hard
selector; Route B is a strong-current-correct boundary selector at fixed q.
Matched-random controls use the same class/state/margin-stratum budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ANCHOR = 79
Q_VALUES = (0.05, 0.10)
RANDOM_SEED = 20260809
EXPECTED_COUNT = 45_000


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stable_choice(seed: int, key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big")


def _read(path: Path) -> list[dict[str, Any]]:
    rows = pq.read_table(path).to_pylist()
    selected = [row for row in rows if int(row["epoch"]) == ANCHOR]
    if len(selected) != EXPECTED_COUNT or len({int(row["sample_id"]) for row in selected}) != EXPECTED_COUNT:
        raise ValueError(f"{path} lacks the exact epoch-{ANCHOR} 45k train panel")
    return selected


def _teacher_wrong(row: dict[str, Any]) -> bool:
    probs = [float(v) for v in row["teacher_adversarial_probabilities"]]
    label = int(row["class_id"])
    return max(range(len(probs)), key=lambda i: probs[i]) != label


def _stratum(row: dict[str, Any]) -> tuple[int, bool, bool, int]:
    # Four-way state plus class and strong-margin decile.  The decile is
    # computed globally by the caller and is passed back through the row.
    return (
        int(row["class_id"]),
        bool(row["student_clean_correct"]),
        bool(row["student_robust_correct"]),
        int(row["_margin_decile"]),
    )


def _matched_random(*, selected: list[dict[str, Any]], candidates: list[dict[str, Any]], seed: int) -> list[int]:
    by_stratum: dict[tuple[int, bool, bool, int], list[dict[str, Any]]] = defaultdict(list)
    target_counts = Counter(_stratum(row) for row in selected)
    for row in candidates:
        by_stratum[_stratum(row)].append(row)
    result: list[int] = []
    for key, count in sorted(target_counts.items(), key=lambda item: item[0]):
        pool = sorted(by_stratum.get(key, []), key=lambda row: _stable_choice(seed, f"{key}:{row['sample_id']}"))
        if len(pool) < count:
            raise ValueError(f"matched-random stratum is underfull: {key} need={count} have={len(pool)}")
        result.extend(int(row["sample_id"]) for row in pool[:count])
    return sorted(set(result))


def _mask_payload(
    *, label: str, ids: list[int], rows: list[dict[str, Any]], source: str, seed: int | None
) -> dict[str, Any]:
    selected = set(ids)
    classes = Counter(int(row["class_id"]) for row in rows if int(row["sample_id"]) in selected)
    payload = {
        "schema_version": 1,
        "contract": "ffnr_causal_pilot_mask_v1",
        "label": label,
        "source": source,
        "anchor_epoch": ANCHOR,
        "selected_ids": ids,
        "selected_count": len(ids),
        "selected_class_counts": {str(k): int(v) for k, v in sorted(classes.items())},
        "random_seed": seed,
        "selection_inputs": {
            "train_only": True,
            "future_outcome_used": False,
            "official_test_used": False,
            "autoattack_used": False,
            "strong_attack": "CE-PGD20, Linf, epsilon=8/255, step=2/255, random_start=true",
        },
    }
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    payload["sha256"] = _sha256_bytes(encoded)
    return payload


def build(*, l2: Path, l4: Path, output: Path) -> dict[str, Any]:
    runs = {"L2": _read(l2), "L4": _read(l4)}
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema_version": 1,
        "contract": "ffnr_causal_pilot_masks_v1",
        "anchor_epoch": ANCHOR,
        "random_seed": RANDOM_SEED,
        "q_candidates": list(Q_VALUES),
        "future_outcome_used": False,
        "runs": {},
    }
    for run, rows in runs.items():
        ordered = sorted(rows, key=lambda row: (-float(row["student_adversarial_logit_margin"]), int(row["sample_id"])))
        for index, row in enumerate(ordered):
            row["_margin_decile"] = min(9, index * 10 // len(ordered))
        route_a = [
            row
            for row in rows
            if not bool(row["student_robust_correct"])
            and not bool(row["student_clean_correct"])
            and _teacher_wrong(row)
        ]
        # Route A random is matched inside the same observable eligibility side
        # (student clean/robust wrong) and the same class/margin strata.
        route_a_pool = [
            row for row in rows if not bool(row["student_robust_correct"]) and not bool(row["student_clean_correct"])
        ]
        masks: dict[str, dict[str, Any]] = {}
        a_ids = sorted(int(row["sample_id"]) for row in route_a)
        masks["route_a_selected"] = _mask_payload(
            label=f"{run}:route_a:selected", ids=a_ids, rows=rows, source="ffnr_route_a_strong_ce_pgd20", seed=None
        )
        a_random = _matched_random(
            selected=route_a, candidates=route_a_pool, seed=RANDOM_SEED + (0 if run == "L2" else 1)
        )
        masks["route_a_random"] = _mask_payload(
            label=f"{run}:route_a:random",
            ids=a_random,
            rows=rows,
            source="ffnr_route_a_matched_random",
            seed=RANDOM_SEED,
        )
        route_b_pool = [row for row in rows if bool(row["student_robust_correct"]) and not _teacher_wrong(row)]
        route_b_all = sorted(
            route_b_pool, key=lambda row: (float(row["student_adversarial_logit_margin"]), int(row["sample_id"]))
        )
        masks["route_b_pool"] = {
            "count": len(route_b_pool),
            "class_counts": dict(Counter(int(row["class_id"]) for row in route_b_pool)),
        }
        for q in Q_VALUES:
            n = int(round(q * len(route_b_pool)))
            selected = route_b_all[:n]
            key = f"route_b_selected_q{int(q * 100):02d}"
            masks[key] = _mask_payload(
                label=f"{run}:{key}",
                ids=sorted(int(row["sample_id"]) for row in selected),
                rows=rows,
                source="ffnr_route_b_strong_ce_pgd20",
                seed=None,
            )
            random_ids = _matched_random(
                selected=selected,
                candidates=route_b_pool,
                seed=RANDOM_SEED + 100 + int(q * 100) + (0 if run == "L2" else 1),
            )
            masks[f"route_b_random_q{int(q * 100):02d}"] = _mask_payload(
                label=f"{run}:route_b:random:q{int(q * 100):02d}",
                ids=random_ids,
                rows=rows,
                source="ffnr_route_b_matched_random",
                seed=RANDOM_SEED,
            )
        run_out = output / run
        run_out.mkdir()
        for name, payload in masks.items():
            path = run_out / f"{name}.json"
            path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            payload["path"] = str(path)
        report["runs"][run] = masks
    (output / "manifest.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--l2", type=Path, required=True)
    parser.add_argument("--l4", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build(l2=args.l2, l4=args.l4, output=args.output)
    for run, masks in report["runs"].items():
        print(run, {key: value.get("selected_count", value.get("count")) for key, value in masks.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
