"""Detached teacher-target transformations for outer-objective branches."""

from .teacher import (
    IdentityTeacherTargetPolicy,
    TeacherTargetOutput,
    TeacherTargetPolicy,
    TrueLabelMixTeacherTargetPolicy,
    UniformSofteningTeacherTargetPolicy,
)

__all__ = [
    "IdentityTeacherTargetPolicy",
    "TeacherTargetOutput",
    "TeacherTargetPolicy",
    "TrueLabelMixTeacherTargetPolicy",
    "UniformSofteningTeacherTargetPolicy",
]
