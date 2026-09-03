#!/usr/bin/env python3
"""Aggregate canonical I100 S2 trajectories without launching GPU work."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from ard.analysis.ert_i100_s2_longitudinal import canonical_action_states

SEEDS = ("dev-1", "dev-2")
ARMS = ("control", "dpm", "dbdd")
EPOCHS = (99, 104, 109, 114)
ACTIVITY_BRANCHES = ("Clean-Wrong", "S3-non-CW", "S2xT1", "S2xT2T3", "S1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    result = {int(row["sample_id"]): dict(row) for row in pq.read_table(path).to_pylist()}
    if len(result) != 45_000:
        raise ValueError(f"{path}: expected exactly 45,000 unique train IDs")
    return result


def target(state: Mapping[str, str]) -> bool:
    return state["branch"] == "S2" and state["teacher"] == "T1"


def activity_branch(state: Mapping[str, str]) -> str:
    """Required fixed-mask activity table branches, with Teacher split only for S2."""
    if state["branch"] != "S2":
        return str(state["branch"])
    return "S2xT1" if state["teacher"] == "T1" else "S2xT2T3"


def _patterns(states: list[Mapping[str, str]]) -> tuple[str, str]:
    branch = ">".join(item["branch"] for item in states)
    joint = ">".join(item["joint"] for item in states)
    return branch, joint


def fixed_cohort_trajectory(
    *, selected: set[int], state_by_epoch: Mapping[int, Mapping[int, Mapping[str, str]]]
) -> dict[str, Any]:
    """Observed-endpoint transitions; P flags are explicitly non-exclusive."""
    pattern_counts: Counter[str] = Counter()
    branch_path_counts: Counter[str] = Counter()
    flags: Counter[str] = Counter()
    explicit: Counter[str] = Counter()
    teacher_transitions: Counter[str] = Counter()
    for sample_id in sorted(selected):
        states = [state_by_epoch[epoch][sample_id] for epoch in EPOCHS]
        membership = [target(state) for state in states]
        branch_path, joint_path = _patterns(states)
        pattern_counts["".join("1" if value else "0" for value in membership)] += 1
        branch_path_counts[branch_path] += 1
        if all(membership):
            flags["P1_all_observed_S2xT1"] += 1
        # P2 is a terminal S1 departure: no later observed Student S2 state.
        first_s1 = next((i for i, state in enumerate(states) if state["branch"] == "S1"), None)
        if first_s1 is not None and all(state["branch"] != "S2" for state in states[first_s1 + 1 :]):
            flags["P2_S1_then_outside_S2_later"] += 1
        if any(state["branch"] == "S3-non-CW" for state in states[1:]):
            flags["P3_observed_S3_nonCW"] += 1
        if any(state["branch"] == "Clean-Wrong" for state in states[1:]):
            flags["P4_observed_CleanWrong"] += 1
        reentries = sum(not membership[index - 1] and membership[index] for index in range(1, len(membership)))
        if reentries:
            flags["P5_leave_then_reenter_S2xT1"] += 1
        if reentries >= 2:
            flags["P6_multiple_exit_reentry"] += 1
        # Explicit routes are observed (not continuous) state paths.  A
        # Teacher T2/T3 S2 visit cannot satisfy these target re-entry routes.
        for index in range(2, len(states)):
            if target(states[index]) and target(states[index - 2]) and not target(states[index - 1]):
                middle = states[index - 1]["branch"]
                if middle == "S1":
                    explicit["S2_to_S1_to_S2"] += 1
                elif middle == "S3-non-CW":
                    explicit["S2_to_S3_nonCW_to_S2"] += 1
                elif middle == "Clean-Wrong":
                    explicit["S2_to_CleanWrong_to_S2"] += 1
                else:
                    explicit["S2_to_other_to_S2"] += 1
        for earlier, later in zip(states, states[1:], strict=True):
            if earlier["branch"] == "S2" or later["branch"] == "S2":
                teacher_transitions[f"{earlier['teacher']}_to_{later['teacher']}"] += 1
    if sum(pattern_counts.values()) != len(selected):
        raise AssertionError("fixed-cohort membership patterns are not conservative")
    return {
        "n": len(selected),
        "membership_patterns": dict(sorted(pattern_counts.items())),
        "branch_paths": dict(branch_path_counts.most_common()),
        "overlapping_observed_indicators": dict(sorted(flags.items())),
        "explicit_observed_reentry_routes": dict(sorted(explicit.items())),
        "teacher_transitions_when_student_S2_at_either_endpoint": dict(sorted(teacher_transitions.items())),
        "terminology": (
            "P1-P6 are overlapping observed-endpoint indicators; membership_patterns are the disjoint partition."
        ),
    }


def entrant_summary(
    *, initial: Mapping[int, Mapping[str, str]], state_by_epoch: Mapping[int, Mapping[int, Mapping[str, str]]]
) -> dict[str, Any]:
    entrants: list[int] = []
    origins: Counter[str] = Counter()
    persistence: Counter[str] = Counter()
    for sample_id, state in initial.items():
        if target(state):
            continue
        membership = [target(state_by_epoch[epoch][sample_id]) for epoch in (104, 109, 114)]
        if not any(membership):
            continue
        entrants.append(sample_id)
        if state["branch"] == "S2":
            origin = "e99_S2xT2T3"
        else:
            origin = f"e99_{state['branch']}"
        origins[origin] += 1
        changes = sum((not membership[i - 1]) and membership[i] for i in range(1, len(membership)))
        if changes:
            persistence["re-entry"] += 1
        elif sum(membership) == 1:
            persistence["one-endpoint-only"] += 1
        else:
            persistence["repeated"] += 1
    return {
        "n": len(entrants),
        "e99_origin": dict(sorted(origins.items())),
        "persistence": dict(sorted(persistence.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--e99-dev1", type=Path, required=True)
    parser.add_argument("--e99-dev2", type=Path, required=True)
    parser.add_argument("--mask-dev1", type=Path, required=True)
    parser.add_argument("--mask-dev2", type=Path, required=True)
    parser.add_argument("--runtime-activity-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    e99_paths = {"dev-1": args.e99_dev1, "dev-2": args.e99_dev2}
    masks = {"dev-1": args.mask_dev1, "dev-2": args.mask_dev2}
    result: dict[str, Any] = {
        "schema_version": 1,
        "contract": "ert_rslad_i100_s2_longitudinal_state_audit_v1",
        "observation_contract": {
            "primary": "unaugmented raw train view + sample-keyed CE-PGD20 with fixed epoch-99 key",
            "secondary_excluded": (
                "historical augmented/batch-keyed state replays are not joined into primary trajectories"
            ),
            "continuity": "observed endpoints only; no continuous state claim",
        },
        "seeds": {},
    }
    for seed in SEEDS:
        initial_rows = rows(e99_paths[seed])
        initial = canonical_action_states(initial_rows.values())["state_by_id"]
        mask = set(json.loads(masks[seed].read_text(encoding="utf-8"))["masks"]["s2_t1"]["selected_ids"])
        anchor_target = {sample_id for sample_id, state in initial.items() if target(state)}
        if mask != anchor_target:
            raise ValueError(f"{seed}: fixed S2xT1 mask does not equal canonical e99 action state")
        seed_result: dict[str, Any] = {
            "e99_rows": {"path": str(e99_paths[seed]), "sha256": sha256(e99_paths[seed])},
            "fixed_mask": {
                "n": len(mask),
                "sha256": hashlib.sha256(json.dumps(sorted(mask), separators=(",", ":")).encode()).hexdigest(),
            },
            "arms": {},
        }
        for arm in ARMS:
            by_epoch: dict[int, Mapping[int, Mapping[str, str]]] = {99: initial}
            metadata: dict[str, Any] = {}
            for epoch in (104, 109, 114):
                base = args.canonical_root / seed.replace("-", "") / arm / f"e{epoch}"
                replay = json.loads((base / "state-replay.json").read_text(encoding="utf-8"))
                if replay.get("contract") != "ert_rslad_i100_s2_longitudinal_ce20_state_v1":
                    raise ValueError(f"{seed}/{arm}/e{epoch}: incorrect canonical state contract")
                current_rows = rows(base / "state-rows.parquet")
                if set(current_rows) != set(initial_rows):
                    raise ValueError(f"{seed}/{arm}/e{epoch}: stable-ID universe differs")
                by_epoch[epoch] = canonical_action_states(current_rows.values())["state_by_id"]
                metadata[str(epoch)] = {
                    "checkpoint_sha256": replay["checkpoint_sha256"],
                    "rows_sha256": replay["rows_sha256"],
                    "state_summary": replay["state_summary"],
                }
            seed_result["arms"][arm] = {
                "endpoint_metadata": metadata,
                "fixed_cohort": fixed_cohort_trajectory(selected=mask, state_by_epoch=by_epoch),
                "new_entrants": entrant_summary(initial=initial, state_by_epoch=by_epoch),
                "fixed_mask_current_branch_counts": {
                    str(epoch): dict(sorted(Counter(by_epoch[epoch][sample_id]["joint"] for sample_id in mask).items()))
                    for epoch in EPOCHS
                },
            }
        result["seeds"][seed] = seed_result
    if args.runtime_activity_dir is not None:
        proxy_rows: dict[tuple[str, str, int], dict[str, Any]] = {}
        for path in sorted(args.runtime_activity_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("contract") != "ert_rslad_i100_s2_checkpoint_no_update_runtime_activity_proxy_v1":
                continue
            key = (str(payload["seed"]), str(payload["arm"]), int(payload["checkpoint_epoch"]))
            if key in proxy_rows:
                raise ValueError(f"duplicate runtime activity proxy: {key}")
            proxy_rows[key] = payload
        required = {(seed, arm, epoch) for seed in SEEDS for arm in ("dpm", "dbdd") for epoch in (104, 109, 114)}
        if set(proxy_rows) != required:
            raise ValueError(f"runtime proxy matrix incomplete: missing={sorted(required - set(proxy_rows))}")
        activity_summary: dict[str, Any] = {}
        for seed in SEEDS:
            seed_summary: dict[str, Any] = {}
            for arm in ARMS:
                arm_summary: dict[str, Any] = {}
                for epoch in (104, 109, 114):
                    if arm == "control":
                        row_path = (
                            args.canonical_root / seed.replace("-", "") / arm / f"e{epoch}" / "state-rows.parquet"
                        )
                        current_states = canonical_action_states(rows(row_path).values())["state_by_id"]
                        fixed_ids = set(
                            json.loads(masks[seed].read_text(encoding="utf-8"))["masks"]["s2_t1"]["selected_ids"]
                        )
                        current = Counter(activity_branch(current_states[sample_id]) for sample_id in fixed_ids)
                        arm_summary[str(epoch)] = {
                            "scope": "Control has no extra intervention loss",
                            "by_current_branch": {
                                branch: {
                                    "fixed_mask_n": int(current.get(branch, 0)),
                                    "current_state_n": int(current.get(branch, 0)),
                                    "extra_loss_active_n": 0,
                                    "extra_loss_active_fraction": 0.0,
                                }
                                for branch in ACTIVITY_BRANCHES
                            },
                        }
                        continue
                    payload = proxy_rows[(seed, arm, epoch)]
                    # Rebuild the current action branch from the canonical rows
                    # so the activity proxy remains explicitly cross-view.
                    row_path = args.canonical_root / seed.replace("-", "") / arm / f"e{epoch}" / "state-rows.parquet"
                    current_states = canonical_action_states(rows(row_path).values())["state_by_id"]
                    grouping: dict[str, dict[str, int]] = {}
                    for sample in payload["per_sample"]:
                        branch = activity_branch(current_states[int(sample["sample_id"])])
                        entry = grouping.setdefault(branch, {"fixed_mask_n": 0, "extra_loss_active_n": 0})
                        entry["fixed_mask_n"] += 1
                        entry["extra_loss_active_n"] += int(bool(sample["extra_loss_positive"]))
                    arm_summary[str(epoch)] = {
                        "scope": payload["scope"],
                        "runtime_contract": payload["runtime_contract"],
                        "by_current_branch": {
                            branch: {
                                "fixed_mask_n": grouping.get(branch, {}).get("fixed_mask_n", 0),
                                "current_state_n": grouping.get(branch, {}).get("fixed_mask_n", 0),
                                "extra_loss_active_n": grouping.get(branch, {}).get("extra_loss_active_n", 0),
                                "extra_loss_active_fraction": (
                                    grouping[branch]["extra_loss_active_n"] / grouping[branch]["fixed_mask_n"]
                                    if branch in grouping and grouping[branch]["fixed_mask_n"]
                                    else 0.0
                                ),
                            }
                            for branch in ACTIVITY_BRANCHES
                        },
                    }
                seed_summary[arm] = arm_summary
            activity_summary[seed] = seed_summary
        result["checkpoint_no_update_runtime_activity_proxy"] = activity_summary
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.with_name(args.output.name + ".sha256").write_text(sha256(args.output) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
