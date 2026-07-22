"""Explicit CIFAR model and teacher adapters."""

from .registry import (
    FixtureCNN,
    PixelModel,
    PixelNormalization,
    SAADBasicBlock,
    SAADResNet18CIFAR,
    build_architecture,
    build_student,
)
from .teacher import TeacherAdapter, TeacherMetadata, build_teacher, sha256_file
from .teacher_registry import FactorySpec, TeacherRegistry, TeacherRegistryError, TeacherSpec, normalize_state_dict

__all__ = [
    "FixtureCNN",
    "SAADBasicBlock",
    "SAADResNet18CIFAR",
    "PixelModel",
    "PixelNormalization",
    "TeacherAdapter",
    "TeacherMetadata",
    "TeacherRegistry",
    "TeacherRegistryError",
    "TeacherSpec",
    "FactorySpec",
    "build_architecture",
    "build_student",
    "build_teacher",
    "sha256_file",
    "normalize_state_dict",
]
