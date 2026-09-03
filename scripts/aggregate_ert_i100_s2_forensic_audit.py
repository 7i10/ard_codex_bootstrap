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
CANONICAL_STATE_CONTRACT = "ert_rslad_i100_s2_longitudinal_ce20_state_v1"
RUNTIME_PROXY_CONTRACT = "ert_rslad_i100_s2_checkpoint_no_update_runtime_activity_proxy_v1"
TEACHER_SHA256 = "fc398a4890e6856b5dd80856076000ec9e2debdd12d9f78a66171b9ffc383983"
CANONICAL_CE20_ATTACK_SHA256 = "675a8d4e3cd16d345acd7fe9e5d1e721f834fcf63c7225e9035e1736ed0c07b6"
RUNTIME_KL10_ATTACK_SHA256 = "97a41870008f5946af3b10dd0d7f145324fe5265b12d3c523bf3f8d099623d4d"


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
        for earlier, later in zip(states, states[1:]):
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
        "observability": {
            "P6_multiple_exit_reentry": {
                "observable": False,
                "reason": (
                    "With four observations beginning in the target state, two leave-and-reenter cycles cannot be "
                    "observed; this endpoint cadence cannot distinguish an unobserved second cycle from no second "
                    "cycle."
                ),
            }
        },
        "terminology": (
            "P1-P5 are overlapping observed-endpoint indicators; membership_patterns are the disjoint partition. "
            "P6 is not observable under this cadence."
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
        active_runs = sum(value and (index == 0 or not membership[index - 1]) for index, value in enumerate(membership))
        active_endpoints = sum(membership)
        if active_endpoints == 1:
            persistence["one-endpoint-only"] += 1
        elif active_runs >= 2:
            persistence["re-entry"] += 1
        else:
            persistence["repeated"] += 1
    return {
        "n": len(entrants),
        "e99_origin": dict(sorted(origins.items())),
        "persistence": dict(sorted(persistence.items())),
    }


def runtime_proxy_payloads(activity_dir: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Load the complete nested checkpoint-proxy matrix fail-closed."""
    proxy_rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in sorted(activity_dir.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("contract") != RUNTIME_PROXY_CONTRACT:
            continue
        key = (str(payload["seed"]), str(payload["arm"]), int(payload["checkpoint_epoch"]))
        if key in proxy_rows:
            raise ValueError(f"duplicate runtime activity proxy: {key}")
        proxy_rows[key] = payload
    return proxy_rows


def _registered_checkpoint(registered_results: Mapping[str, Any], *, seed: str, arm: str, epoch: int) -> str:
    try:
        return str(registered_results["held_out"][seed][arm][str(epoch)]["checkpoint_sha256"])
    except (KeyError, TypeError) as error:
        raise ValueError(f"registered Dynamic-BDD result lacks {seed}/{arm}/e{epoch} checkpoint lineage") from error


def validate_canonical_replay(
    *,
    replay: Mapping[str, Any],
    rows_path: Path,
    seed: str,
    arm: str,
    epoch: int,
    registered_results: Mapping[str, Any],
) -> None:
    """Fail closed on replay payload/content/parent lineage disagreement."""
    prefix = f"{seed}/{arm}/e{epoch}"
    if replay.get("contract") != CANONICAL_STATE_CONTRACT:
        raise ValueError(f"{prefix}: incorrect canonical state contract")
    if int(replay.get("checkpoint_epoch", -1)) != epoch or int(replay.get("payload_epoch", -1)) != epoch:
        raise ValueError(f"{prefix}: checkpoint/payload epoch differs from requested endpoint")
    if int(replay.get("row_count", -1)) != 45_000:
        raise ValueError(f"{prefix}: replay row count is not 45,000")
    if str(replay.get("checkpoint_sha256")) != _registered_checkpoint(
        registered_results, seed=seed, arm=arm, epoch=epoch
    ):
        raise ValueError(f"{prefix}: replay checkpoint differs from registered Dynamic-BDD result")
    if str(replay.get("rows_sha256")) != sha256(rows_path):
        raise ValueError(f"{prefix}: state row SHA-256 differs from replay metadata")
    if str(replay.get("teacher_checkpoint_sha256")) != TEACHER_SHA256:
        raise ValueError(f"{prefix}: Teacher SHA differs from frozen lineage")
    observation = replay.get("observation")
    if not isinstance(observation, Mapping):
        raise ValueError(f"{prefix}: canonical observation metadata missing")
    if (
        str(observation.get("attack_identity_sha256")) != CANONICAL_CE20_ATTACK_SHA256
        or int(observation.get("observation_epoch_key", -1)) != 99
        or observation.get("clean_view") != "raw unaugmented train split"
        or observation.get("random_start_keying") != "sample_keyed_v1"
        or observation.get("student_mode") != "eval"
        or observation.get("teacher_mode") != "eval"
    ):
        raise ValueError(f"{prefix}: canonical CE-PGD20 observation identity differs")


def validate_runtime_proxy(
    *,
    payload: Mapping[str, Any],
    seed: str,
    arm: str,
    epoch: int,
    expected_checkpoint_sha256: str,
    expected_mask_sha256: str,
    expected_mask_ids: set[int],
    expected_config_sha256: str,
) -> None:
    """Validate a no-update KL10 proxy before joining it to CE20 state rows."""
    prefix = f"runtime proxy {seed}/{arm}/e{epoch}"
    if payload.get("contract") != RUNTIME_PROXY_CONTRACT:
        raise ValueError(f"{prefix}: contract differs")
    if (payload.get("seed"), payload.get("arm"), int(payload.get("checkpoint_epoch", -1))) != (seed, arm, epoch):
        raise ValueError(f"{prefix}: identity differs")
    if str(payload.get("checkpoint_sha256")) != expected_checkpoint_sha256:
        raise ValueError(f"{prefix}: checkpoint differs from canonical/registered endpoint")
    if str(payload.get("mask_sha256")) != expected_mask_sha256:
        raise ValueError(f"{prefix}: fixed-mask SHA differs")
    if str(payload.get("config_sha256")) != expected_config_sha256:
        raise ValueError(f"{prefix}: host-rebased execution config differs")
    if payload.get("scope") != "checkpoint no-update rank-local training proxy; not historical per-visit activity":
        raise ValueError(f"{prefix}: scope differs")
    contract = payload.get("runtime_contract")
    expected_contract = {
        "student_train_mode": True,
        "teacher_eval_frozen": True,
        "full_rank_batch_mean": True,
        "selected_count_normalization": False,
        "attack_identity_sha256": RUNTIME_KL10_ATTACK_SHA256,
        "train_view_augmentation_epoch": epoch,
    }
    if not isinstance(contract, Mapping) or any(contract.get(key) != value for key, value in expected_contract.items()):
        raise ValueError(f"{prefix}: KL10 no-update runtime contract differs")
    if payload.get("student_state_hash_before_after") != "identical":
        raise ValueError(f"{prefix}: Student state was not restored bitwise")
    counts = payload.get("counts")
    per_sample = payload.get("per_sample")
    if not isinstance(counts, Mapping) or not isinstance(per_sample, list):
        raise ValueError(f"{prefix}: incomplete runtime-proxy payload")
    proxy_ids = {int(row["sample_id"]) for row in per_sample}
    if (
        int(counts.get("seen", -1)) != 45_000
        or int(counts.get("selected", -1)) != len(expected_mask_ids)
        or len(per_sample) != len(expected_mask_ids)
        or proxy_ids != expected_mask_ids
    ):
        raise ValueError(f"{prefix}: fixed mask was not covered exactly once")


def endpoint_checkpoint(longitudinal_result: Mapping[str, Any], *, seed: str, arm: str, epoch: int) -> str:
    """Return the same-seed canonical endpoint checkpoint for a KL10 proxy."""
    try:
        return str(
            longitudinal_result["seeds"][seed]["arms"][arm]["endpoint_metadata"][str(epoch)]["checkpoint_sha256"]
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"longitudinal result lacks {seed}/{arm}/e{epoch} endpoint metadata") from error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--e99-dev1", type=Path, required=True)
    parser.add_argument("--e99-dev2", type=Path, required=True)
    parser.add_argument("--mask-dev1", type=Path, required=True)
    parser.add_argument("--mask-dev2", type=Path, required=True)
    parser.add_argument("--execution-config-dev1", type=Path, required=True)
    parser.add_argument("--execution-config-dev2", type=Path, required=True)
    parser.add_argument("--registered-results", type=Path, required=True)
    parser.add_argument("--analysis-source-sha", required=True)
    parser.add_argument("--runtime-activity-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    e99_paths = {"dev-1": args.e99_dev1, "dev-2": args.e99_dev2}
    masks = {"dev-1": args.mask_dev1, "dev-2": args.mask_dev2}
    execution_configs = {"dev-1": args.execution_config_dev1, "dev-2": args.execution_config_dev2}
    execution_config_hashes = {seed: sha256(path) for seed, path in execution_configs.items()}
    registered_results = json.loads(args.registered_results.read_text(encoding="utf-8"))
    if registered_results.get("contract") != "ert_rslad_i100_s2_dynamic_bdd_recovery_results_v1":
        raise ValueError("registered results are not the accepted Dynamic-BDD recovery artifact")
    result: dict[str, Any] = {
        "schema_version": 1,
        "contract": "ert_rslad_i100_s2_longitudinal_state_audit_v1",
        "analysis_source_git_sha": args.analysis_source_sha,
        "observation_contract": {
            "primary": "unaugmented raw train view + sample-keyed CE-PGD20 with fixed epoch-99 key",
            "secondary_excluded": (
                "historical augmented/batch-keyed state replays are not joined into primary trajectories"
            ),
            "continuity": "observed endpoints only; no continuous state claim",
        },
        "lineage_contract": {
            "teacher_checkpoint_sha256": TEACHER_SHA256,
            "canonical_ce20_attack_identity_sha256": CANONICAL_CE20_ATTACK_SHA256,
            "runtime_kl10_attack_identity_sha256": RUNTIME_KL10_ATTACK_SHA256,
            "registered_results": {"path": str(args.registered_results), "sha256": sha256(args.registered_results)},
            "execution_configs": {
                seed: {"path": str(path), "sha256": execution_config_hashes[seed]}
                for seed, path in execution_configs.items()
            },
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
                "sha256": sha256(masks[seed]),
                "stable_id_set_sha256": hashlib.sha256(
                    json.dumps(sorted(mask), separators=(",", ":")).encode()
                ).hexdigest(),
            },
            "arms": {},
        }
        fixed_ids = mask
        for arm in ARMS:
            by_epoch: dict[int, Mapping[int, Mapping[str, str]]] = {99: initial}
            metadata: dict[str, Any] = {}
            for epoch in (104, 109, 114):
                base = args.canonical_root / seed.replace("-", "") / arm / f"e{epoch}"
                replay = json.loads((base / "state-replay.json").read_text(encoding="utf-8"))
                rows_path = base / "state-rows.parquet"
                validate_canonical_replay(
                    replay=replay,
                    rows_path=rows_path,
                    seed=seed,
                    arm=arm,
                    epoch=epoch,
                    registered_results=registered_results,
                )
                current_rows = rows(rows_path)
                if set(current_rows) != set(initial_rows):
                    raise ValueError(f"{seed}/{arm}/e{epoch}: stable-ID universe differs")
                by_epoch[epoch] = canonical_action_states(current_rows.values())["state_by_id"]
                metadata[str(epoch)] = {
                    "checkpoint_sha256": replay["checkpoint_sha256"],
                    "rows_sha256": replay["rows_sha256"],
                    "replay_sha256": sha256(base / "state-replay.json"),
                    "observation_attack_identity_sha256": replay["observation"]["attack_identity_sha256"],
                    "state_summary": replay["state_summary"],
                }
            seed_result["arms"][arm] = {
                "endpoint_metadata": metadata,
                "fixed_cohort": fixed_cohort_trajectory(selected=mask, state_by_epoch=by_epoch),
                "new_entrants": entrant_summary(initial=initial, state_by_epoch=by_epoch),
                "current_target_n_by_epoch": {
                    str(epoch): sum(target(state) for state in by_epoch[epoch].values()) for epoch in EPOCHS
                },
                "fixed_mask_current_branch_counts": {
                    str(epoch): dict(sorted(Counter(by_epoch[epoch][sample_id]["joint"] for sample_id in mask).items()))
                    for epoch in EPOCHS
                },
            }
        result["seeds"][seed] = seed_result
    if args.runtime_activity_dir is not None:
        proxy_rows = runtime_proxy_payloads(args.runtime_activity_dir)
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
                    expected_checkpoint = endpoint_checkpoint(result, seed=seed, arm=arm, epoch=epoch)
                    validate_runtime_proxy(
                        payload=payload,
                        seed=seed,
                        arm=arm,
                        epoch=epoch,
                        expected_checkpoint_sha256=expected_checkpoint,
                        expected_mask_sha256=sha256(masks[seed]),
                        expected_mask_ids=fixed_ids,
                        expected_config_sha256=execution_config_hashes[seed],
                    )
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
