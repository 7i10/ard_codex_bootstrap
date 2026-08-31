#!/usr/bin/env python3
"""Build a non-overwriting endpoint-only recovery manifest.

The training jobs in the history-ordering campaign completed successfully, but
the endpoint worker and evaluator disagreed about who creates the output
directory.  This manifest retries only the endpoint jobs against the existing
training checkpoints and keeps their original result paths so the registered
aggregator can consume them.
"""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ORIGINAL = REPO / "docs/experiments/ert_rslad_history_balanced_ordering_dev_v2_manifest.json"
RUN_ROOT = Path(
    "/home/islab/workspace-local/shunsuke.naito/ard-runs/ard_codex_bootstrap/"
    "ert-rslad-history-ordering-v2-final"
)
RECOVERY_ROOT = RUN_ROOT.parent / "ert-rslad-history-ordering-v2-endpoint-recovery"
OUTPUT_JSON = REPO / "docs/experiments/ert_rslad_history_balanced_ordering_dev_v2_results.json"
OUTPUT_MD = REPO / "docs/ERT_RSLAD_HISTORY_BALANCED_ORDERING_DEV_V2.md"


def current_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def endpoint_recovery_job(job: dict[str, object]) -> dict[str, object]:
    original_id = str(job["job_id"])
    recovered = copy.deepcopy(job)
    recovered["job_id"] = f"{original_id}-recovery"
    recovered["run_id"] = f"{job['run_id']}-recovery"
    recovered["dependencies"] = []
    recovered["retry_policy"] = {"max_attempts": 1}
    command = [str(item) for item in job["command"]]  # type: ignore[index]
    config = command[command.index("--config") + 1]
    checkpoint = command[command.index("--checkpoint") + 1]
    recovered["required_paths"] = [config, checkpoint]
    recovered["completion_marker"] = "orchestration/recovery-completion.json"
    identity = dict(job["scientific_identity"])  # type: ignore[arg-type]
    identity.update(
        {
            "technical_recovery_of": original_id,
            "technical_fix": "allow_orchestrator_created_endpoint_directory",
        }
    )
    recovered["scientific_identity"] = identity
    return recovered


def main() -> None:
    original = json.loads(ORIGINAL.read_text(encoding="utf-8"))
    endpoint_jobs = [
        endpoint_recovery_job(job)
        for job in original["jobs"]
        if str(job["job_id"]).startswith("endpoint-")
    ]
    endpoint_ids = [str(job["job_id"]) for job in endpoint_jobs]
    aggregate = {
        "job_id": "aggregate-report-recovery",
        "run_id": "ert-rslad-history-ordering-aggregate-report-recovery",
        "host": "hamster",
        "gpu_count": 0,
        "command": [
            "/home/shunsukenaito/.conda/envs/adv/bin/python",
            "scripts/aggregate_history_ordering_campaign.py",
            "--output-json",
            str(OUTPUT_JSON),
            "--output-md",
            str(OUTPUT_MD),
        ],
        "cwd": str(REPO),
        "env": {"PYTHONPATH": str(REPO / "src"), "PYTHONUNBUFFERED": "1"},
        "output_dir": str(RECOVERY_ROOT / "aggregate"),
        "completion_marker": "orchestration/completion.json",
        "dependencies": endpoint_ids,
        "estimated_work": 1,
        "retry_policy": {"max_attempts": 1},
        "scientific_identity": {
            "method_id": "aggregate_report",
            "campaign": "ert-rslad-history-ordering-dev-v2",
            "technical_recovery": True,
        },
    }
    manifest = {
        "schema_version": 1,
        "campaign_id": "ert-rslad-history-ordering-dev-v2-endpoint-recovery",
        "source": {
            "git_sha": current_sha(),
            "scientific_source_note": (
                "Technical recovery only: existing training checkpoints and endpoint attack identity are reused; "
                "the source change only resolves output-directory ownership."
            ),
        },
        "state_path": str(RECOVERY_ROOT / "orchestration.state.json"),
        "reservation_root": "/home/shunsukenaito/.cache/ard-experiment-orchestrator/reservations",
        "hosts": copy.deepcopy(original["hosts"]),
        "jobs": endpoint_jobs + [aggregate],
    }
    path = REPO / "docs/experiments/ert_rslad_history_balanced_ordering_endpoint_recovery_v1_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(path), "jobs": len(manifest["jobs"]), "source_sha": manifest["source"]["git_sha"]}))


if __name__ == "__main__":
    main()
