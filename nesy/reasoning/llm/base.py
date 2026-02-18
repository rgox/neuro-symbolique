"""
LLM Provider Base Class - Abstract Interface for Language Model Backends.

Defines the ABC contract for LLM providers (mock, Anthropic, OpenAI, etc.)

Example:
    >>> from nesy.core.registry import Registry
    >>> from nesy.reasoning.llm.base import LLMProviderBase
    >>> llm = Registry.create(LLMProviderBase, "mock")
    >>> response = llm.complete("What color is the sky?")
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class LLMProviderBase(ABC):
    """
    Abstract base class for LLM providers.

    All LLM backends (mock, Anthropic Claude, OpenAI, local, etc.)
    must implement this interface.
    """

    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        """
        Generate a text completion.

        Args:
            prompt: Input prompt
            **kwargs: Provider-specific options (temperature, max_tokens, etc.)

        Returns:
            Generated text
        """
        ...

    @abstractmethod
    def extract_rules(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract logical rules from natural language text.

        Args:
            text: Natural language description

        Returns:
            List of rule dicts with 'rule' key in Datalog syntax
        """
        ...

    def get_model_info(self) -> Dict[str, Any]:
        """Return info about the underlying model."""
        return {"provider": self.__class__.__name__}
