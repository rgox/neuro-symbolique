"""Mock LLM provider for testing without API keys."""

from typing import List, Dict, Any
import re

from nesy.core.registry import register_plugin
from nesy.reasoning.llm.base import LLMProviderBase


@register_plugin(LLMProviderBase, "mock")
class MockLLMProvider(LLMProviderBase):
    """
    Mock LLM that returns deterministic responses for testing.

    No API keys or external services required.
    """

    def __init__(self, **kwargs):
        self.call_count = 0

    def complete(self, prompt: str, **kwargs) -> str:
        self.call_count += 1
        prompt_lower = prompt.lower()

        if "rule" in prompt_lower or "datalog" in prompt_lower:
            return (
                "Based on the description, here are the logical rules:\n"
                "1. on(X, Y) :- placed_on(X, Y)\n"
                "2. in(X, Z) :- on(X, Y), in(Y, Z)\n"
            )
        if "explain" in prompt_lower:
            return "The reasoning follows from spatial transitivity: if A is on B and B is in C, then A is in C."
        if "validate" in prompt_lower:
            return "The rules are logically consistent and well-formed."
        return f"Mock response to: {prompt[:50]}..."

    def extract_rules(self, text: str) -> List[Dict[str, Any]]:
        self.call_count += 1
        rules = []
        # Simple pattern matching for Datalog-like rules in text
        pattern = r'(\w+\([^)]+\)\s*:-\s*[^.]+)'
        matches = re.findall(pattern, text)
        for match in matches:
            rules.append({"rule": match.strip(), "confidence": 0.8})
        if not rules:
            rules.append({
                "rule": "related(X, Y) :- near(X, Y)",
                "confidence": 0.7,
            })
        return rules

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "mock",
            "model": "mock-v1",
            "calls": self.call_count,
        }
