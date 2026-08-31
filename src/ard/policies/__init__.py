"""Signal-to-weight mappings."""

from .base import PolicyContext, PolicyOutput, PolicyWeights, WeightPolicy
from .entropy import EntropyOnlyPolicy
from .fixed_mask import FixedInterventionMask, FixedMaskError, load_fixed_intervention_mask, selected_ids_sha256
from .history_order import stable_risk_order, verify_h2_rank_equivalence
from .rslad import RSLADBaselinePolicy
from .student_aware import (
    HardFallbackPolicy,
    JointDownweightPolicy,
    JointRiskPolicy,
    StudentRiskPolicy,
    student_risk_from_margin,
    teacher_risk_from_entropy,
)
from .uniform import UniformPolicy

__all__ = [
    "EntropyOnlyPolicy",
    "FixedInterventionMask",
    "FixedMaskError",
    "HardFallbackPolicy",
    "JointDownweightPolicy",
    "JointRiskPolicy",
    "PolicyContext",
    "PolicyOutput",
    "PolicyWeights",
    "RSLADBaselinePolicy",
    "StudentRiskPolicy",
    "UniformPolicy",
    "WeightPolicy",
    "load_fixed_intervention_mask",
    "selected_ids_sha256",
    "stable_risk_order",
    "verify_h2_rank_equivalence",
    "student_risk_from_margin",
    "teacher_risk_from_entropy",
]
