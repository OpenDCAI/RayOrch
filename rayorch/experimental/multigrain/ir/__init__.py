"""Passive multigrain IR, validation passes, and derived capabilities."""

from .capabilities import (
    PrimitiveCapabilities,
    RelationEvidenceFamily,
    capabilities_for,
)
from .model import *  # noqa: F403
from .model import __all__ as _model_all
from .passes import (
    Diagnostic,
    InsertRebatchAfterExpandPass,
    MarkMapFilterFusionCandidatesPass,
    PassKind,
    PassManager,
    PassResult,
    PlanReduceGroupsPass,
    RebatchCandidatePass,
    RelationSummaryPass,
    VerifyPass,
)

__all__ = [
    *_model_all,
    "Diagnostic",
    "InsertRebatchAfterExpandPass",
    "MarkMapFilterFusionCandidatesPass",
    "PassKind",
    "PassManager",
    "PassResult",
    "PlanReduceGroupsPass",
    "PrimitiveCapabilities",
    "RebatchCandidatePass",
    "RelationEvidenceFamily",
    "RelationSummaryPass",
    "VerifyPass",
    "capabilities_for",
]
