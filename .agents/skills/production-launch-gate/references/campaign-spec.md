# Campaign spec v1

The gate accepts JSON as the canonical format. YAML is an input convenience;
the resolved output is always JSON and is consumed by the existing
multi-GPU orchestrator.

```json
{
  "schema_version": 1,
  "campaign_id": "example-v1",
  "source": {
    "git_sha": "<40 hex>",
    "repo_path": "/work/repo"
  },
  "source_policy": {
    "allow_dirty": false,
    "registered_files": {"src/train.py": "<64 hex>"}
  },
  "dataset": {
    "identity": "cifar10_controlled_split_v1",
    "split_identity": "<split digest>",
    "host_paths": {"hamster": "/data/cifar10"},
    "required_files": ["train.index", "val.index"]
  },
  "teacher": {"identity": "teacher-v1", "sha256": "<64 hex>"},
  "training": {
    "scientific_start_epoch": 0,
    "scientific_final_epoch": 114,
    "checkpoint_epochs": [79, 114],
    "endpoint_epochs": [84, 114]
  },
  "attacks": {
    "train": {
      "loss": "kl", "epsilon": "8/255", "step_size": "2/255",
      "steps": 10, "random_start": true, "target": "teacher_clean"
    }
  },
  "augmentation_identity": "crop-shift-v1",
  "rng_contract": {"augmentation": "source-keyed-v1", "attack": "sample-keyed-v1"},
  "hosts": {
    "hamster": {
      "backend": "local", "repo_path": "/work/repo",
      "python": "/opt/env/bin/python",
      "dataset_paths": {"cifar10_controlled_split_v1": "/data/cifar10"},
      "gpus": [{"index": 0, "uuid": "GPU-...", "throughput": 679.0}]
    }
  },
  "canary": {
    "jobs": [{"job_id": "train", "command": ["/bin/true"], "timeout_seconds": 20}]
  },
  "jobs": [{
    "job_id": "train", "arm": "BASE", "seed": "dev-1",
    "host": "hamster", "command": ["${PYTHON}", "train.py", "--epochs", "115"],
    "output_dir": "outputs/train", "attack": "train",
    "parent": {"path": "parents/e79.json", "sha256": "<64 hex>", "epoch": 79},
    "expected_outputs": [{"path": "last.json", "epoch": 114}],
    "retry_policy": {"max_attempts": 2}
  }]
}
```

`scientific_final_epoch` is inclusive and the runtime `--epochs` value is
exclusive. The gate resolves the latter from the former and records both.
`dataset.host_paths` and a host profile's `dataset_paths` should agree; a job's
optional `dataset_path` is checked against the selected host mapping. Use
`kind: dependency_output` with `producer_job_id` for an input created by an
earlier job; it is not required to exist during preflight, but its producer and
path must be declared in the same manifest.

## External-host and collection additions

An external host adds a bounded, machine-readable remote preflight. The command
is owned by the existing remote executor (for example `run-on-ferret`), not by
the gate. It receives expected values in `ARD_LAUNCH_GATE_REMOTE_EXPECTED` and
returns schema-v1 JSON with matching source, Python, GPU, input-artifact,
output-writability, disk, launcher, and completion-probe evidence.

```json
{
  "backend": "external",
  "repo_path": "/remote/repo",
  "python": "/remote/adv/bin/python",
  "dataset_paths": {"cifar10_controlled_split_v1": "/remote/data/cifar10"},
  "teacher_paths": {"teacher-v1": "/remote/cache/teacher.pt"},
  "gpus": [{"index": 0, "uuid": "GPU-...", "throughput": 600.0}],
  "remote_preflight": {
    "command": ["bash", "scripts/remote-preflight.sh"],
    "timeout_seconds": 60,
    "minimum_disk_free_bytes": 10000000000,
    "output_root": "/remote/results",
    "launcher": {"argv": ["bash", "scripts/remote-launch.sh"], "executable": false},
    "completion_probe": {"argv": ["bash", "scripts/remote-status.sh"], "executable": false}
  }
}
```

Each `external_probe` job requires a `remote_command`, `host_confirm_probe`,
and bounded confirmation timing. The confirmation must prove a live PID and
the exact campaign/job/identity/source/GPU/argv binding. A direct
non-executable shell wrapper is invalid in either launcher role; use its
interpreter explicitly.

External campaigns also declare an explicit collection/inventory boundary:

```json
{
  "artifact_collection": {
    "manifest_path": "artifacts/inventory.json",
    "collection_job_id": "collect-endpoints",
    "inventory_job_id": "validate-endpoint-inventory"
  }
}
```

The referenced schema-v1 inventory contains `origin.host`, `origin.path`, a
local `staging_path`, canonical local `collected_path`, expected `sha256`, and
identity fields `campaign_id`, `source_sha`, `job_id`, `seed`, `arm`, `epoch`,
`split`, and `attack_identity`. The collection job copies only bytes; the
inventory job validates the required matrix before any aggregate job runs.
