"""Frozen teacher adapter with checkpoint provenance and explicit preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, field_validator
from torch import nn

from ard.config.schema import NormalizationConfig, TeacherConfig

from .registry import PixelNormalization, build_architecture
from .teacher_registry import TeacherRegistry, normalize_state_dict, sha256_file


class TeacherMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    architecture: str
    num_classes: int
    normalization: NormalizationConfig
    checkpoint_sha256: str
    registry_id: str | None = None
    upstream_model_id: str | None = None
    external_commit: str | None = None
    preprocessing_owner: str = "teacher_adapter"
    preprocessing_profile: str | None = None
    threat_model: dict[str, str] | None = None

    @field_validator("checkpoint_sha256")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest")
        return value


class TeacherAdapter(nn.Module):
    def __init__(self, model: nn.Module, metadata: TeacherMetadata) -> None:
        super().__init__()
        self.model = model
        self.metadata = metadata
        self.normalization: nn.Module
        if metadata.preprocessing_owner == "teacher_adapter":
            self.normalization = PixelNormalization(metadata.normalization)
        elif metadata.preprocessing_owner == "model_embedded":
            if (
                metadata.registry_id != "bartoldson2024_adversarial_wrn94_16"
                or metadata.upstream_model_id != "Bartoldson2024Adversarial_WRN-94-16"
            ):
                raise ValueError("model_embedded preprocessing is restricted to the Bartoldson RobustBench teacher")
            self.normalization = nn.Identity()
        else:
            raise ValueError(f"unsupported teacher preprocessing owner: {metadata.preprocessing_owner}")
        self.freeze()

    def freeze(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        super().train(False)
        self.model.eval()

    def train(self, mode: bool = True) -> TeacherAdapter:
        # A frozen single teacher is always an evaluation model.  In particular,
        # PGD must never update nested BatchNorm running statistics even if a
        # caller asks for train mode.  Input gradients remain available.
        super().train(False)
        self.model.eval()
        return self

    def logits(self, pixels: torch.Tensor, *, require_input_grad: bool) -> torch.Tensor:
        """Return logits with an explicit input-gradient contract and frozen parameters."""
        if require_input_grad and not pixels.requires_grad:
            raise ValueError("teacher input gradients require a requires_grad pixel tensor")
        if require_input_grad:
            # This remains usable even when an enclosing forward-only caller
            # entered no_grad. Teacher parameters stay frozen independently.
            with torch.enable_grad():
                return self.model(self.normalization(pixels))
        with torch.no_grad():
            return self.model(self.normalization(pixels))

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.logits(pixels, require_input_grad=pixels.requires_grad)

    @classmethod
    def from_checkpoint(cls, path: Path, metadata: TeacherMetadata) -> TeacherAdapter:
        observed = sha256_file(path)
        if observed != metadata.checkpoint_sha256:
            raise ValueError(f"teacher checkpoint hash mismatch: expected {metadata.checkpoint_sha256}, got {observed}")
        model = build_architecture(metadata.architecture, metadata.num_classes)
        payload: Any = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(normalize_state_dict(payload), strict=True)
        return cls(model, metadata)


def build_teacher(config: TeacherConfig, *, tier: str) -> TeacherAdapter:
    if config.source == "fixture":
        if tier not in {"dev", "smoke"}:
            raise ValueError("fixture teachers are restricted to dev/smoke tiers")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.fixture_seed)
            model = build_architecture(config.architecture, config.num_classes)
        metadata = TeacherMetadata(
            architecture=config.architecture,
            num_classes=config.num_classes,
            normalization=config.normalization,
            checkpoint_sha256="0" * 64,
        )
        return TeacherAdapter(model, metadata)
    if config.source == "robustbench":
        registry = TeacherRegistry.load()
        spec = registry.validate_config(config)
        assert config.checkpoint is not None
        # This local-file check is deliberately before the external import or
        # constructor: RobustBench's downloader is never reachable from ARD.
        registry.validate_local_checkpoint(spec, config.checkpoint)
        model = registry.constructor(spec)
        metadata = TeacherMetadata(
            architecture=spec.architecture,
            num_classes=10,
            normalization=spec.preprocessing.normalization(),
            checkpoint_sha256=spec.checkpoint_sha256 or "",
            registry_id=spec.registry_id,
            upstream_model_id=spec.upstream_model_id,
            external_commit=registry.repository_commit,
            preprocessing_owner=spec.preprocessing.owner,
            preprocessing_profile=spec.preprocessing.profile,
            threat_model={
                "norm": spec.threat.norm,
                "epsilon": spec.threat.epsilon,
                "input_domain": spec.threat.input_domain,
            },
        )
        payload: Any = torch.load(config.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(normalize_state_dict(payload), strict=True)
        return TeacherAdapter(model, metadata)
    assert config.checkpoint is not None and config.checkpoint_sha256 is not None
    metadata = TeacherMetadata(
        architecture=config.architecture,
        num_classes=config.num_classes,
        normalization=config.normalization,
        checkpoint_sha256=config.checkpoint_sha256,
    )
    return TeacherAdapter.from_checkpoint(config.checkpoint, metadata)
