"""Freeze the three allocation masks plan 0096 compares.

The plan asks which images should receive the richer augmentation from epoch
100.  The answer has to be decided from each parent's own state at epoch 99 and
then frozen, because an allocation that keeps changing is a different experiment
from an allocation that is fixed, and only the fixed one isolates the direction.

Three masks per parent, all the same size so the comparison is about *which*
images and not *how many*:

* ``s1``       -- adversarially correct with margin above the frozen tenth
                  percentile: the samples the model already handles safely.
* ``fragile``  -- the same number drawn from the rest, nearest the threshold
                  first, so it is the complement's most-nearly-safe end rather
                  than an arbitrary slice of it.
* ``random``   -- a class-matched random draw of the same size, which is the
                  control that turns "we chose well" into a testable claim.

The random draw's seeds are declared in the plan before any of this runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from ard.policies.fixed_mask import selected_ids_sha256

SCHEMA_VERSION = 1
NUM_CLASSES = 10
SOURCES = {
    "s1": "stagewise_allocation_s1_epoch100_v1",
    "fragile": "stagewise_allocation_fragile_epoch100_v1",
    "random": "stagewise_allocation_matched_random_epoch100_v1",
}


def _write_mask(
    path: Path, *, selected: list[int], labels: dict[int, int], provenance: dict[str, Any]
) -> dict[str, Any]:
    selected = sorted(set(selected))
    counts = Counter(labels[sample_id] for sample_id in selected)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "namespace": "train",
        "num_classes": NUM_CLASSES,
        "selected_ids": selected,
        "selected_ids_sha256": selected_ids_sha256(tuple(selected)),
        "selected_count": len(selected),
        "selected_class_counts": {str(c): counts[c] for c in sorted(counts)},
        "provenance": provenance,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "selected_ids_sha256": payload["selected_ids_sha256"],
        "selected_count": payload["selected_count"],
        "selected_class_counts": payload["selected_class_counts"],
        "provenance": provenance,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True, help="epoch-100 online-state parquet for one parent")
    parser.add_argument("--thresholds", type=Path, required=True, help="that parent's frozen threshold artifact")
    parser.add_argument("--seed-label", required=True, help="the parent's label, recorded in provenance")
    parser.add_argument("--random-seed", type=int, required=True, help="declared in the plan before running")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))["thresholds"]
    student_q10 = float(thresholds["student_global_logit_q10"])
    rows = pq.read_table(args.state).to_pylist()
    labels = {int(r["sample_id"]): int(r["class_id"]) for r in rows}

    s1, rest = [], []
    for row in rows:
        sample_id = int(row["sample_id"])
        if bool(row["student_adv_correct"]) and float(row["student_global_margin"]) > student_q10:
            s1.append(sample_id)
        else:
            rest.append((float(row["student_global_margin"]), sample_id))
    if not s1 or len(rest) < len(s1):
        raise SystemExit(f"cannot build equal-sized masks: |S1|={len(s1)}, |rest|={len(rest)}")

    # The fragile arm takes the complement's highest-margin end, so the two arms
    # differ in which side of the threshold they sit on rather than in how far
    # from it they are.
    fragile = [sample_id for _, sample_id in sorted(rest, reverse=True)[: len(s1)]]

    # Class-matched random: same count per class as S1, drawn by a hash of the
    # declared seed so the draw is reproducible from the plan alone.
    by_class: dict[int, list[int]] = defaultdict(list)
    for sample_id in labels:
        by_class[labels[sample_id]].append(sample_id)
    wanted = Counter(labels[sample_id] for sample_id in s1)
    random_ids: list[int] = []
    for class_id, count in sorted(wanted.items()):
        ranked = sorted(
            by_class[class_id],
            key=lambda sid: hashlib.sha256(f"{args.random_seed}:{class_id}:{sid}".encode()).digest(),
        )
        if len(ranked) < count:
            raise SystemExit(f"class {class_id} has {len(ranked)} images, need {count}")
        random_ids.extend(ranked[:count])

    args.output.mkdir(parents=True, exist_ok=True)
    base = {"seed_label": args.seed_label, "switch_epoch": 100, "state": str(args.state.resolve())}
    manifest = {
        "contract": "ard_stagewise_allocation_masks_v1",
        "seed_label": args.seed_label,
        "student_global_logit_q10": student_q10,
        "masks": {
            "s1": _write_mask(
                args.output / "s1.json", selected=s1, labels=labels, provenance={"source": SOURCES["s1"], **base}
            ),
            "fragile": _write_mask(
                args.output / "fragile.json",
                selected=fragile,
                labels=labels,
                provenance={"source": SOURCES["fragile"], **base},
            ),
            "random": _write_mask(
                args.output / "random.json",
                selected=random_ids,
                labels=labels,
                provenance={"source": SOURCES["random"], "random_seed": args.random_seed, **base},
            ),
        },
    }
    sizes = {name: spec["selected_count"] for name, spec in manifest["masks"].items()}
    if len(set(sizes.values())) != 1:
        raise SystemExit(f"masks must be the same size so the contrast is about which, not how many: {sizes}")
    if manifest["masks"]["s1"]["selected_class_counts"] != manifest["masks"]["random"]["selected_class_counts"]:
        raise SystemExit("the random draw is not class-matched to S1")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
