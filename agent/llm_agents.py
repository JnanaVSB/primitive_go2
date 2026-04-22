"""LLM client abstraction with multi-provider support.

Unified `LLMClient` interface with concrete implementations for Anthropic,
OpenAI, Google Gemini, and Ollama. Provider SDKs are imported lazily so
only the SDK for the provider actually in use needs to be installed.

Usage:
    client = make_client(provider='anthropic', model='claude-sonnet-4-5-20250929')
    response = client.generate(prompt)
"""

import os
import time
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Base exception for LLM client errors."""


class LLMClient(ABC):
    """Abstract base for LLM providers.

    Concrete subclasses implement `_raw_generate(prompt)`. Retry with
    exponential backoff on transient errors is handled by `generate()`.
    """

    MAX_RETRIES = 5
    INITIAL_BACKOFF = 1.0  # seconds

    def __init__(self, model: str, temperature: float = 0.7, max_tokens: int = 2000):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def _raw_generate(self, prompt: str) -> str:
        """Provider-specific API call. Returns the LLM's text response."""

    @abstractmethod
    def _is_retriable(self, exc: Exception) -> bool:
        """Return True if this exception should trigger a retry."""

    def generate(self, prompt: str) -> str:
        """Generate with retry on transient errors."""
        backoff = self.INITIAL_BACKOFF
        last_exc = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return self._raw_generate(prompt)
            except Exception as e:
                if not self._is_retriable(e) or attempt == self.MAX_RETRIES:
                    raise LLMError(
                        f"{type(self).__name__} failed after {attempt} attempt(s): {e}"
                    ) from e
                logger.warning(
                    f"{type(self).__name__} attempt {attempt} failed ({e}); "
                    f"retrying in {backoff:.1f}s"
                )
                last_exc = e
                time.sleep(backoff)
                backoff *= 2
        raise LLMError(f"Unreachable")  # loop always returns or raises


class AnthropicClient(LLMClient):
    """Anthropic Claude via the `anthropic` SDK. Reads ANTHROPIC_API_KEY."""

    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "Anthropic SDK not installed. Run: pip install anthropic"
            ) from e
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY environment variable not set")
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key)

    def _raw_generate(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _is_retriable(self, exc: Exception) -> bool:
        e = self._anthropic
        return isinstance(exc, (
            e.RateLimitError,
            e.APIConnectionError,
            e.APITimeoutError,
            e.InternalServerError,
        ))


class OpenAIClient(LLMClient):
    """OpenAI GPT via the `openai` SDK. Reads OPENAI_API_KEY."""

    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "OpenAI SDK not installed. Run: pip install openai"
            ) from e
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise LLMError("OPENAI_API_KEY environment variable not set")
        self._openai = openai
        self._client = openai.OpenAI(api_key=api_key)

    def _raw_generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    def _is_retriable(self, exc: Exception) -> bool:
        e = self._openai
        return isinstance(exc, (
            e.RateLimitError,
            e.APIConnectionError,
            e.APITimeoutError,
            e.InternalServerError,
        ))


class GeminiClient(LLMClient):
    """Google Gemini via `google-generativeai`. Reads GOOGLE_API_KEY."""

    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        try:
            import google.generativeai as genai
            import google.api_core.exceptions as google_exc
        except ImportError as e:
            raise ImportError(
                "Google Gemini SDK not installed. "
                "Run: pip install google-generativeai"
            ) from e
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise LLMError("GOOGLE_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        self._genai = genai
        self._google_exc = google_exc
        self._model_obj = genai.GenerativeModel(model)

    def _raw_generate(self, prompt: str) -> str:
        response = self._model_obj.generate_content(
            prompt,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
            },
        )
        return response.text

    def _is_retriable(self, exc: Exception) -> bool:
        ge = self._google_exc
        return isinstance(exc, (
            ge.ResourceExhausted,    # rate limit
            ge.ServiceUnavailable,
            ge.DeadlineExceeded,
            ge.InternalServerError,
        ))


class OllamaClient(LLMClient):
    """Ollama via the `ollama` SDK. No API key; requires base_url.

    Use this for self-hosted models on a cluster, e.g.:
        OllamaClient(model='llama3.1:70b', base_url='http://cluster:11434')
    """

    def __init__(self, model: str, base_url: str = "http://localhost:11434", **kwargs):
        super().__init__(model, **kwargs)
        try:
            import ollama
            import httpx
        except ImportError as e:
            raise ImportError(
                "Ollama SDK not installed. Run: pip install ollama"
            ) from e
        self._ollama = ollama
        self._httpx = httpx
        self._client = ollama.Client(host=base_url)

    def _raw_generate(self, prompt: str) -> str:
        response = self._client.generate(
            model=self.model,
            prompt=prompt,
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        )
        return response['response']

    def _is_retriable(self, exc: Exception) -> bool:
        return isinstance(exc, (
            self._httpx.ConnectError,
            self._httpx.TimeoutException,
            self._httpx.RemoteProtocolError,
        ))


PROVIDERS = {
    'anthropic': AnthropicClient,
    'openai': OpenAIClient,
    'gemini': GeminiClient,
    'ollama': OllamaClient,
}


def make_client(provider: str, model: str, **kwargs) -> LLMClient:
    """Factory: instantiate the right LLMClient for a provider name.

    Args:
        provider: one of 'anthropic', 'openai', 'gemini', 'ollama'.
        model:    provider-specific model string.
        **kwargs: forwarded to the client constructor. Common: temperature,
                  max_tokens. Ollama-specific: base_url.

    Returns:
        An LLMClient instance.

    Raises:
        ValueError: unknown provider.
    """
    key = provider.lower()
    if key not in PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider}'. Valid: {sorted(PROVIDERS)}"
        )
    return PROVIDERS[key](model=model, **kwargs)