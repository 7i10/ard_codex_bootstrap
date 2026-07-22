"""Detached teacher-target transformations for outer-objective branches."""

from .teacher import (
    IdentityTeacherTargetPolicy,
    TeacherTargetOutput,
    TeacherTargetPolicy,
    UniformSofteningTeacherTargetPolicy,
)

__all__ = [
    "IdentityTeacherTargetPolicy",
    "TeacherTargetOutput",
    "TeacherTargetPolicy",
    "UniformSofteningTeacherTargetPolicy",
]
