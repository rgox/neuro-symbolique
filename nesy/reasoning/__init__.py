"""
Reasoning Module - Symbolic and Neural-Symbolic Reasoning.

This module contains:
- VSA (Vector-Symbolic Architectures): Hyperdimensional computing
- Logic: Differentiable logic programming with Scallop
- (Future) Planning: Goal-directed reasoning

Components:
- vsa: Vector-Symbolic Architectures for neural-symbolic grounding
- logic: Differentiable Datalog for symbolic inference
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

__all__ = [
    'HyperVector',
    'VSACodebook',
    'NeuralGrounding',
    'GroundingCache',
    'ScallopContext',
    'ReasoningEngine',
]
