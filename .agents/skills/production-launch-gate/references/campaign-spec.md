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
