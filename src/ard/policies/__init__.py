"""Signal-to-weight mappings."""

from .base import PolicyContext, PolicyOutput, PolicyWeights, WeightPolicy
from .entropy import EntropyOnlyPolicy
from .rslad import RSLADBaselinePolicy
from .student_aware import JointRiskPolicy, StudentRiskPolicy, student_risk_from_margin, teacher_risk_from_entropy
from .uniform import UniformPolicy

__all__ = [
    "EntropyOnlyPolicy",
    "JointRiskPolicy",
    "PolicyContext",
    "PolicyOutput",
    "PolicyWeights",
    "RSLADBaselinePolicy",
    "StudentRiskPolicy",
    "UniformPolicy",
    "WeightPolicy",
    "student_risk_from_margin",
    "teacher_risk_from_entropy",
]
