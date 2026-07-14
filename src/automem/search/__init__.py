"""automem.search — architecture search utilities for AutoMem."""

from .pareto_front import ParetoEntry, ParetoFront
from .attribution import (
    AttributionResult, AttributionType, run_posthoc_audit,
    aggregate_layer_metrics, build_layer_diagnosis,
)

# H-plan modular components (2026-05-13)
from .validator import ArchitectureValidator, ValidationReport
from .scheduler import CandidateScheduler, Role, Phase, ScheduledCandidate
from .acquisition import MutationAcquisition, AcquisitionRecommendation
from .incumbent_gate import IncumbentGate, IncumbentDecision
from .quarantine import QuarantineZone, PromotionDecision

__all__ = [
    # Existing
    "ParetoEntry",
    "ParetoFront",
    "AttributionResult",
    "AttributionType",
    "run_posthoc_audit",
    "aggregate_layer_metrics",
    "build_layer_diagnosis",
    # H-plan modular components
    "ArchitectureValidator",
    "ValidationReport",
    "CandidateScheduler",
    "Role",
    "Phase",
    "ScheduledCandidate",
    "MutationAcquisition",
    "AcquisitionRecommendation",
    "IncumbentGate",
    "IncumbentDecision",
    "QuarantineZone",
    "PromotionDecision",
]
