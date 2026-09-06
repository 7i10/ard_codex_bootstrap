"""Freeze the three allocation masks plan 0096 compares.

The plan asks which images should receive the richer augmentation from epoch 100.
Three masks per parent, **identical in size and in class composition**, so that a
difference between arms cannot be a dose effect or a class-balance effect:

* ``safe``     -- adversarially correct with margin above the frozen tenth
                  percentile: the samples the model already handles well.
* ``fragile``  -- per class, the **lowest**-margin images outside that set: the
                  genuine opposite end, which is what AROID's vulnerability
                  direction means.  An earlier version took the highest-margin
                  end of the complement and therefore excluded the most
                  vulnerable images entirely; a null from that arm could not have
                  closed the direction it was standing for.
* ``random``   -- a draw of the same per-class counts, keyed on the parent as
                  well as the declared seed.  Keying on the seed alone made the
                  six parents' controls near-identical (Jaccard 0.99) while their
                  treatments varied (Jaccard 0.45), so the control would have had
                  one effective realisation across all six blocks.

The draw seeds are declared in the plan before any of this runs.
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
    "safe": "stagewise_allocation_s1_epoch100_v1",
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

    thresholds_payload = json.loads(args.thresholds.read_text(encoding="utf-8"))
    student_q10 = float(thresholds_payload["thresholds"]["student_global_logit_q10"])
    state_sha = hashlib.sha256(args.state.read_bytes()).hexdigest()
    rows = pq.read_table(args.state).to_pylist()
    labels = {int(r["sample_id"]): int(r["class_id"]) for r in rows}

    epochs = {int(row["epoch"]) for row in rows}
    if epochs != {100}:
        raise SystemExit(f"state file must be the epoch-100 boundary, found epochs {sorted(epochs)}")
    if thresholds_payload.get("prefix_state_sha256") not in (None, state_sha):
        raise SystemExit(
            "the thresholds artifact was frozen against a different state file; "
            "pairing one parent's state with another's thresholds silently changes every mask"
        )

    # Split the training set at the frozen threshold, then take a symmetric slice
    # from each side.  The safe side is class-imbalanced -- easy classes have more
    # safe images than hard ones -- so the largest arm that can be matched on both
    # size and class composition is min(|safe|, |rest|) per class, taken from the
    # extreme end of each side.  Matching matters more than size here: an
    # unmatched pair would confound the direction of allocation with class
    # balance, and the class ratio between the two sides reaches 2.5x.
    safe_by_class: dict[int, list[tuple[float, int]]] = defaultdict(list)
    rest_by_class: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for row in rows:
        sample_id = int(row["sample_id"])
        margin = float(row["student_global_margin"])
        target = safe_by_class if (bool(row["student_adv_correct"]) and margin > student_q10) else rest_by_class
        target[labels[sample_id]].append((margin, sample_id))

    safe: list[int] = []
    fragile: list[int] = []
    random_ids: list[int] = []
    by_class: dict[int, list[int]] = defaultdict(list)
    for sample_id, class_id in labels.items():
        by_class[class_id].append(sample_id)

    for class_id in sorted(by_class):
        safe_side = sorted(safe_by_class[class_id], reverse=True)   # highest margin first
        rest_side = sorted(rest_by_class[class_id])                 # lowest margin first
        count = min(len(safe_side), len(rest_side))
        if count == 0:
            raise SystemExit(f"class {class_id} has no images on one side of the threshold")
        safe.extend(sample_id for _, sample_id in safe_side[:count])
        fragile.extend(sample_id for _, sample_id in rest_side[:count])
        ranked = sorted(
            by_class[class_id],
            key=lambda sid: hashlib.sha256(
                f"{args.random_seed}:{args.seed_label}:{class_id}:{sid}".encode()
            ).digest(),
        )
        random_ids.extend(ranked[:count])

    args.output.mkdir(parents=True, exist_ok=True)
    # Identity, never a path: a path in a hashed provenance differs between hosts.
    base = {"seed_label": args.seed_label, "switch_epoch": 100, "state_sha256": state_sha}
    manifest = {
        "contract": "ard_stagewise_allocation_masks_v1",
        "seed_label": args.seed_label,
        "student_global_logit_q10": student_q10,
        "masks": {
            "safe": _write_mask(
                args.output / "safe.json", selected=safe, labels=labels, provenance={"source": SOURCES["safe"], **base}
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
    counts = {name: spec["selected_class_counts"] for name, spec in manifest["masks"].items()}
    if len({json.dumps(c, sort_keys=True) for c in counts.values()}) != 1:
        raise SystemExit(f"all three masks must share one class composition, got {counts}")
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "manifest.json.sha256").write_text(
        hashlib.sha256(manifest_path.read_bytes()).hexdigest() + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
