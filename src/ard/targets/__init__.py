"""Detached teacher-target transformations for outer-objective branches."""

from .teacher import (
    AnchoredTeacherTargetPolicy,
    IdentityTeacherTargetPolicy,
    TeacherOnlyTemperatureTargetPolicy,
    TeacherTargetOutput,
    TeacherTargetPolicy,
    TrueLabelMixTeacherTargetPolicy,
    UniformSofteningTeacherTargetPolicy,
)

__all__ = [
    "AnchoredTeacherTargetPolicy",
    "IdentityTeacherTargetPolicy",
    "TeacherTargetOutput",
    "TeacherOnlyTemperatureTargetPolicy",
    "TeacherTargetPolicy",
    "TrueLabelMixTeacherTargetPolicy",
    "UniformSofteningTeacherTargetPolicy",
]
