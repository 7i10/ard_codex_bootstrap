"""Frozen teacher adapter with checkpoint provenance and explicit preprocessing."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, field_validator
from torch import nn

from ard.config.schema import NormalizationConfig, TeacherConfig

from .registry import PixelNormalization, build_architecture


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TeacherMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    architecture: str
    num_classes: int
    normalization: NormalizationConfig
    checkpoint_sha256: str

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
        self.normalization = PixelNormalization(metadata.normalization)
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

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.model(self.normalization(pixels))

    @classmethod
    def from_checkpoint(cls, path: Path, metadata: TeacherMetadata) -> TeacherAdapter:
        observed = sha256_file(path)
        if observed != metadata.checkpoint_sha256:
            raise ValueError(f"teacher checkpoint hash mismatch: expected {metadata.checkpoint_sha256}, got {observed}")
        model = build_architecture(metadata.architecture, metadata.num_classes)
        payload: Any = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = payload.get("model") if isinstance(payload, dict) and "model" in payload else payload
        if not isinstance(state_dict, dict):
            raise TypeError("teacher checkpoint must contain a state dictionary")
        model.load_state_dict(state_dict, strict=True)
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
    assert config.checkpoint is not None and config.checkpoint_sha256 is not None
    metadata = TeacherMetadata(
        architecture=config.architecture,
        num_classes=config.num_classes,
        normalization=config.normalization,
        checkpoint_sha256=config.checkpoint_sha256,
    )
    return TeacherAdapter.from_checkpoint(config.checkpoint, metadata)
