"""OpenAI LLM provider (requires openai package)."""

from typing import List, Dict, Any
import re

from nesy.core.registry import register_plugin
from nesy.reasoning.llm.base import LLMProviderBase


@register_plugin(LLMProviderBase, "openai")
class OpenAIProvider(LLMProviderBase):
    """
    OpenAI API provider.

    Requires:
        pip install openai
        OPENAI_API_KEY environment variable

    Args:
        model: Model name (default: gpt-4o)
        api_key: Optional API key (falls back to OPENAI_API_KEY env var)
        max_tokens: Max response tokens
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str = None,
        max_tokens: int = 1024,
        **kwargs,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self._client = None
        self._api_key = api_key

    def _get_client(self):
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "openai package required. Install: pip install openai"
                )
            kwargs = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def complete(self, prompt: str, **kwargs) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    def extract_rules(self, text: str) -> List[Dict[str, Any]]:
        prompt = (
            "Extract Datalog rules from the following description. "
            "Output each rule on a separate line in the format: "
            "head(X, Y) :- body(X, Z), body2(Z, Y)\n\n"
            f"Description: {text}\n\nRules:"
        )
        response = self.complete(prompt)
        rules = []
        pattern = r'(\w+\([^)]+\)\s*:-\s*[^.\n]+)'
        for match in re.findall(pattern, response):
            rules.append({"rule": match.strip(), "confidence": 0.9})
        return rules

    def get_model_info(self) -> Dict[str, Any]:
        return {"provider": "openai", "model": self.model}
