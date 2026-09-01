#!/usr/bin/env python3
"""Freeze the attack-random-start seed registry before any outcome is read."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DOMAIN = "ert_rslad_attack_seed_probe_v1"


def attack_seed(index: int) -> int:
    if not 0 <= index < 8:
        raise ValueError("attack seed index must be in [0, 7]")
    digest = hashlib.sha256(f"{DOMAIN}|{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def build_registry(*, source_sha: str, parent_hashes: dict[str, str]) -> dict[str, object]:
    if len(source_sha) != 40 or any(c not in "0123456789abcdef" for c in source_sha):
        raise ValueError("source SHA must be a lowercase 40-character Git SHA")
    if set(parent_hashes) != {"seed1", "seed2"}:
        raise ValueError("registry requires seed1 and seed2 parent hashes")
    return {
        "schema_version": 1,
        "kind": "ert_rslad_attack_seed_probe_registry_v1",
        "status": "frozen_before_training",
        "domain": DOMAIN,
        "source_git_sha": source_sha,
        "parents": parent_hashes,
        "parent_payload_epoch": 99,
        "probe_epoch_start": 100,
        "probe_epoch_end_exclusive": 115,
        "attack": {
            "loss": "kl",
            "target": "teacher_clean",
            "epsilon": "8/255",
            "step_size": "2/255",
            "steps": 10,
            "random_start": True,
            "random_start_distribution": "Uniform[-epsilon,+epsilon]",
            "random_start_keying": "sample_keyed_v1",
        },
        "invariant_streams": [
            "model_init",
            "data_order",
            "augmentation",
            "evaluation_attack",
            "python_numpy_torch_other_rng",
        ],
        "seeds": [{"label": f"ATTACK_SEED_{i}", "index": i, "value": attack_seed(i)} for i in range(8)],
        "phase_a": {
            "risk": "-margin_ema_at_epoch_99",
            "strata": ["HIGH", "MID", "LOW"],
            "fractions": [0.2, 0.6, 0.2],
            "selection_rule": "descriptive_only_no_outcome_threshold_tuning",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--parent-seed1", required=True)
    parser.add_argument("--parent-seed2", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    registry = build_registry(
        source_sha=args.source_sha,
        parent_hashes={"seed1": args.parent_seed1, "seed2": args.parent_seed2},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(registry, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({"path": str(args.output.resolve()), "sha256": digest, "seed_count": 8}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
