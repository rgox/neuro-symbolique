"""
LLM Reasoning Module - Hybrid Neural-Symbolic Reasoning.

Integrates Large Language Models with symbolic logic.
All LLM providers are registered as plugins via the Registry.

Example:
    >>> from nesy.core.registry import Registry
    >>> from nesy.reasoning.llm.base import LLMProviderBase
    >>> llm = Registry.create(LLMProviderBase, "mock")
    >>> response = llm.complete("Extract rules from: cups are on tables")
"""

from nesy.reasoning.llm.hybrid import (
    LLMRuleBridge,
    HybridQueryEngine,
    SelfValidator,
    ExplanationGenerator,
    ExtractedRule,
)
from nesy.reasoning.llm.base import LLMProviderBase

# Trigger plugin registration
import nesy.reasoning.llm.providers  # noqa: F401

__all__ = [
    "LLMProviderBase",
    "LLMRuleBridge",
    "HybridQueryEngine",
    "SelfValidator",
    "ExplanationGenerator",
    "ExtractedRule",
]
