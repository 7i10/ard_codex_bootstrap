#!/usr/bin/env python3
"""Audit whether changing sample order changes existing PGD random starts.

The audit is deliberately read-only.  It records the current Trainer/PGD
contract and a small deterministic reproduction of the batch-position
coupling.  A positive coupling result is a launch blocker for an
ordering-only intervention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiments/ert_rslad_order_rng_audit_v1.json")
    args = parser.parse_args()
    trainer_path = ROOT / "src/ard/engine/trainer.py"
    pgd_path = ROOT / "src/ard/attacks/pgd.py"
    trainer = trainer_path.read_text(encoding="utf-8")
    pgd = pgd_path.read_text(encoding="utf-8")
    # These source anchors are part of the audit evidence, not a substitute
    # for the deterministic reproduction below.
    generator_anchor = re.search(
        r"def _attack_generator\(self\).*?return torch\.Generator\(device=self\.device\)\.manual_seed\(seed\)",
        trainer,
        re.DOTALL,
    )
    random_start_anchor = re.search(
        r"delta\.uniform_\(-1\.0, 1\.0, generator=request\.generator\)", pgd
    )
    if generator_anchor is None or random_start_anchor is None:
        raise RuntimeError("expected Trainer/PGD RNG source anchors were not found")

    batch_size = 4
    shape = (batch_size, 3, 4, 4)
    seed = 17
    control_order = list(range(8))
    reordered = [1, 0, 2, 3, 4, 5, 6, 7]

    def draws(order: list[int]) -> dict[int, torch.Tensor]:
        by_id: dict[int, torch.Tensor] = {}
        for batch_index, start in enumerate(range(0, len(order), batch_size)):
            generator = torch.Generator().manual_seed(seed + 1_000_003 * batch_index)
            delta = torch.empty(shape).uniform_(-1.0, 1.0, generator=generator)
            for position, sample_id in enumerate(order[start : start + batch_size]):
                by_id[sample_id] = delta[position].clone()
        return by_id

    control = draws(control_order)
    treatment = draws(reordered)
    changed_ids = [sample_id for sample_id in control if not torch.equal(control[sample_id], treatment[sample_id])]
    report = {
        "schema_version": 1,
        "kind": "ert_rslad_order_rng_audit",
        "source_git_sha": git_sha(),
        "read_only": True,
        "sources": {
            "trainer": {"path": str(trainer_path), "sha256": sha256(trainer_path), "anchor": generator_anchor.group(0)},
            "pgd": {"path": str(pgd_path), "sha256": sha256(pgd_path), "anchor": random_start_anchor.group(0)},
        },
        "current_contract": {
            "trainer_seed": "seed + 1000003*global_step + 10007*rank",
            "generator_scope": "one freshly seeded torch.Generator per training batch",
            "random_start": "batched delta.uniform_ using request.generator",
            "augmentation_contract": "source/epoch keyed (separate from this blocker)",
        },
        "reproduction": {
            "batch_size": batch_size,
            "control_order": control_order,
            "reordered_order": reordered,
            "changed_sample_ids": changed_ids,
            "changed_count": len(changed_ids),
            "order_changes_random_start_assignment": bool(changed_ids),
        },
        "decision": "BLOCK_PRODUCTION_ORDERING_ONLY"
        if changed_ids
        else "PASS_ORDER_RNG_AUDIT",
        "reason": (
            "The existing batch-position random-start stream assigns different initial perturbations to fixed "
            "sample IDs "
            "when the DataLoader order changes.  Changing order therefore changes attack randomness, so the requested "
            "ordering-only causal interpretation is not valid without a separately reviewed RNG-contract change."
            if changed_ids
            else "No coupling observed in the bounded reproduction."
        ),
        "no_training": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"output": str(args.output), "decision": report["decision"], "changed_count": len(changed_ids)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
