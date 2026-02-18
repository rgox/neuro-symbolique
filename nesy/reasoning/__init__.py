"""
Reasoning Module - Symbolic and Neural-Symbolic Reasoning.

This module contains:
- VSA (Vector-Symbolic Architectures): Hyperdimensional computing
- Logic: Differentiable logic programming with Scallop/Minalog
- Planning: Goal-directed reasoning (PDDL)
- Base: Abstract interfaces for backend-agnostic reasoning

All engines are registered as plugins via the Registry.

Example:
    >>> from nesy.core.registry import Registry
    >>> from nesy.reasoning.base import LogicEngineBase
    >>> engine = Registry.create(LogicEngineBase, "minalog")
"""

from nesy.reasoning.vsa import (
    HyperVector,
    VSACodebook,
    NeuralGrounding,
    GroundingCache
)

from nesy.reasoning.logic import (
    ScallopContext,
    ReasoningEngine,
)

from nesy.reasoning.base import (
    LogicEngineBase,
    PlannerBase,
)

# Import plugin modules to trigger @register_plugin decorators
import nesy.reasoning.engines  # noqa: F401

__all__ = [
    # ABCs (preferred interfaces)
    'LogicEngineBase',
    'PlannerBase',
    # Legacy classes (backward compatible)
    'HyperVector',
    'VSACodebook',
    'NeuralGrounding',
    'GroundingCache',
    'ScallopContext',
    'ReasoningEngine',
]
