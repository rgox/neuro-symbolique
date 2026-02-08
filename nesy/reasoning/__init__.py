"""
Reasoning Module - Symbolic and Neural-Symbolic Reasoning.

This module will contain:
- VSA (Vector-Symbolic Architectures): Hyperdimensional computing
- Logic engines: Scallop, differentiable logic
- Inference: Forward/backward reasoning
- Planning: Goal-directed reasoning

Currently implemented:
- vsa: Vector-Symbolic Architectures for neural-symbolic grounding
"""

from nesy.reasoning.vsa import (
    HyperVector,
    VSACodebook,
    NeuralGrounding,
    GroundingCache
)

__all__ = [
    'HyperVector',
    'VSACodebook',
    'NeuralGrounding',
    'GroundingCache',
]
