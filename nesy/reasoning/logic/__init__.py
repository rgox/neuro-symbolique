"""
Symbolic Reasoning Logic Module.

Provides Scallop-based logic reasoning over scene graphs.
"""

from nesy.reasoning.logic.engine import ReasoningEngine
from nesy.reasoning.logic.scallop_context import ScallopContext, SCALLOP_AVAILABLE
from nesy.reasoning.logic.advanced import (
    AggregationRules,
    RecursiveRules, 
    ProbabilisticRules,
    RuleValidator,
)

__all__ = [
    "ReasoningEngine",
    "ScallopContext",
    "SCALLOP_AVAILABLE",
    "AggregationRules",
    "RecursiveRules",
    "ProbabilisticRules",
    "RuleValidator",
]
