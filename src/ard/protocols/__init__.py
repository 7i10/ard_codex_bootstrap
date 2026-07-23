"""Versioned, immutable experiment protocol identities.

The registry records historical/public settings without allowing audit records
to accidentally become runnable local experiments.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class ProtocolSpec:
    id: str
    runnable_locally: bool
    local_train_reason: str | None
    metadata: Mapping[str, object]


_CONTROLLED_METADATA = MappingProxyType(
    {
        "dataset": MappingProxyType(
            {
                "name": "cifar10",
                "split": "train",
                "download": False,
                "num_classes": 10,
                "image_size": 32,
            }
        ),
        "student": MappingProxyType(
            {
                "architecture": "saad_resnet18_cifar_v1",
                "num_classes": 10,
                "normalization_profile": "cifar10_raw_identity",
                "preprocessing_owner": "student_adapter",
            }
        ),
        "training": MappingProxyType(
            {
                "epochs": 200,
                "global_batch_size": 128,
                "validation_fraction": 0.1,
                "deterministic": True,
                "batchnorm_mode": "local_per_rank",
            }
        ),
        "seeds": MappingProxyType({"split": 20260722, "evaluation_attack": 0}),
        "evaluation": MappingProxyType({"seed": 0}),
        "optimizer": MappingProxyType(
            {"id": "sgd", "learning_rate": 0.1, "momentum": 0.9, "weight_decay": 5e-4, "nesterov": False}
        ),
        "scheduler": MappingProxyType(
            {"id": "multistep", "milestones": (100, 150), "gamma": 0.1, "step_at": "epoch_end"}
        ),
        "train_augmentation": "RandomCrop(32,padding=4)+RandomHorizontalFlip",
        "train_attack": MappingProxyType(
            {
                "loss": "kl",
                "kl_target": "teacher_clean",
                "temperature": 1.0,
                "temperature_squared": True,
                "steps": 10,
                "epsilon": "8/255",
                "step_size": "2/255",
                "random_start": True,
                "norm": "linf",
                "input_domain": "pixel_0_1",
                "student_mode": "eval",
                "teacher_mode": "eval",
            }
        ),
        "selection_attack": MappingProxyType(
            {
                "loss": "ce",
                "kl_target": None,
                "temperature": 1.0,
                "temperature_squared": True,
                "steps": 20,
                "epsilon": "8/255",
                "step_size": "2/255",
                "random_start": True,
                "norm": "linf",
                "input_domain": "pixel_0_1",
                "student_mode": "eval",
                "teacher_mode": "eval",
            }
        ),
    }
)

_PILOT_METADATA = MappingProxyType(
    {
        **_CONTROLLED_METADATA,
        "training": MappingProxyType(
            {
                "epochs": 5,
                "global_batch_size": 128,
                "validation_fraction": 0.1,
                "deterministic": True,
                "batchnorm_mode": "local_per_rank",
            }
        ),
    }
)


def _pilot_metadata(epochs: int) -> Mapping[str, object]:
    return MappingProxyType(
        {
            **_CONTROLLED_METADATA,
            "training": MappingProxyType(
                {
                    "epochs": epochs,
                    "global_batch_size": 128,
                    "validation_fraction": 0.1,
                    "deterministic": True,
                    "batchnorm_mode": "local_per_rank",
                }
            ),
        }
    )


_PAPER_METADATA = MappingProxyType(
    {
        "dataset": "cifar10",
        "train_set": "official_full_50k",
        "validation": "none",
        "checkpoint_lifecycle": "last_or_published",
        "optimizer": MappingProxyType({"id": "sgd", "weight_decay": 5e-4}),
        "attack": MappingProxyType({"algorithm": "pgd", "steps": 10, "epsilon": "8/255", "step_size": "2/255"}),
    }
)

_CODE_AUDIT_METADATA = MappingProxyType(
    {
        "optimizer": MappingProxyType({"id": "sgd", "weight_decay": 2e-4}),
        "source_attack_call": MappingProxyType({"step_size": "8/255", "epsilon": "2/255"}),
        "test_each_epoch": True,
        "swa": MappingProxyType({"enabled": True, "start_epoch": 95}),
        "parallelism": "DataParallel",
    }
)

PROTOCOLS: Mapping[str, ProtocolSpec] = MappingProxyType(
    {
        "synthetic_smoke_v2": ProtocolSpec(
            id="synthetic_smoke_v2",
            runnable_locally=True,
            local_train_reason=None,
            metadata=MappingProxyType({"fixture": True}),
        ),
        "controlled_cifar10_r18_v1": ProtocolSpec(
            id="controlled_cifar10_r18_v1",
            runnable_locally=True,
            local_train_reason=None,
            metadata=_CONTROLLED_METADATA,
        ),
        "controlled_cifar10_r18_pilot_v1": ProtocolSpec(
            id="controlled_cifar10_r18_pilot_v1",
            runnable_locally=True,
            local_train_reason=None,
            metadata=_PILOT_METADATA,
        ),
        "controlled_cifar10_r18_pilot_1ep_v1": ProtocolSpec(
            id="controlled_cifar10_r18_pilot_1ep_v1",
            runnable_locally=True,
            local_train_reason=None,
            metadata=_pilot_metadata(1),
        ),
        "controlled_cifar10_r18_pilot_3ep_v1": ProtocolSpec(
            id="controlled_cifar10_r18_pilot_3ep_v1",
            runnable_locally=True,
            local_train_reason=None,
            metadata=_pilot_metadata(3),
        ),
        "saad_paper_reproduction_v1": ProtocolSpec(
            id="saad_paper_reproduction_v1",
            runnable_locally=False,
            local_train_reason=(
                "saad_paper_reproduction_v1 is a public/full-50k protocol record, not a controlled local train protocol"
            ),
            metadata=_PAPER_METADATA,
        ),
        "saad_code_295121c_audit_v1": ProtocolSpec(
            id="saad_code_295121c_audit_v1",
            runnable_locally=False,
            local_train_reason=(
                "saad_code_295121c_audit_v1 is an audit-only record; it must not run as controlled training"
            ),
            metadata=_CODE_AUDIT_METADATA,
        ),
    }
)


def get_protocol(protocol_id: str) -> ProtocolSpec:
    try:
        return PROTOCOLS[protocol_id]
    except KeyError as exc:
        raise ValueError(f"unknown protocol ID: {protocol_id}") from exc


def ensure_local_trainable(protocol_id: str) -> ProtocolSpec:
    spec = get_protocol(protocol_id)
    if not spec.runnable_locally:
        assert spec.local_train_reason is not None
        raise ValueError(spec.local_train_reason)
    return spec


__all__ = ["PROTOCOLS", "ProtocolSpec", "ensure_local_trainable", "get_protocol"]
