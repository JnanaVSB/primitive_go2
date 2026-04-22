"""Tests for agent/client.py.

These tests monkeypatch `_raw_generate` and the retriable check, so they
don't require any SDK to be installed and don't hit real APIs.
"""

import pytest
from unittest.mock import MagicMock

from agent.llm_agents import (
    LLMClient, LLMError, make_client, PROVIDERS,
    AnthropicClient, OpenAIClient, GeminiClient, OllamaClient,
)


class FakeClient(LLMClient):
    """Test double: no SDK, controllable behavior."""
    def __init__(self, responses=None, retriable_errors=()):
        super().__init__(model='fake', temperature=0.7, max_tokens=1000)
        self.responses = list(responses or [])
        self.retriable_errors = retriable_errors
        self.call_count = 0

    def _raw_generate(self, prompt):
        self.call_count += 1
        if not self.responses:
            raise RuntimeError("no responses queued")
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def _is_retriable(self, exc):
        return isinstance(exc, self.retriable_errors)


class RetriableError(Exception):
    pass


class NonRetriableError(Exception):
    pass


class TestRetry:
    def test_success_first_try(self):
        c = FakeClient(responses=["hello"])
        assert c.generate("prompt") == "hello"
        assert c.call_count == 1

    def test_retries_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(FakeClient, "INITIAL_BACKOFF", 0.001)
        c = FakeClient(
            responses=[RetriableError("rate limit"), RetriableError("rate limit"), "ok"],
            retriable_errors=(RetriableError,),
        )
        assert c.generate("prompt") == "ok"
        assert c.call_count == 3

    def test_non_retriable_raises_immediately(self):
        c = FakeClient(
            responses=[NonRetriableError("bad")],
            retriable_errors=(RetriableError,),
        )
        with pytest.raises(LLMError):
            c.generate("prompt")
        assert c.call_count == 1

    def test_exhausts_retries(self, monkeypatch):
        monkeypatch.setattr(FakeClient, "INITIAL_BACKOFF", 0.001)
        errs = [RetriableError("rate")] * 10  # more than MAX_RETRIES
        c = FakeClient(responses=errs, retriable_errors=(RetriableError,))
        with pytest.raises(LLMError, match="failed after"):
            c.generate("prompt")
        assert c.call_count == FakeClient.MAX_RETRIES


class TestFactory:
    def test_all_providers_registered(self):
        assert set(PROVIDERS) == {'anthropic', 'openai', 'gemini', 'ollama'}
        assert PROVIDERS['anthropic'] is AnthropicClient
        assert PROVIDERS['openai'] is OpenAIClient
        assert PROVIDERS['gemini'] is GeminiClient
        assert PROVIDERS['ollama'] is OllamaClient

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            make_client(provider='nosuchprovider', model='x')

    def test_provider_name_case_insensitive(self):
        # Should accept 'Anthropic' as well as 'anthropic' — but instantiation
        # will fail without an API key, so we just check that it gets past
        # the name check.
        with pytest.raises((ImportError, LLMError)):
            make_client(provider='ANTHROPIC', model='x')


class TestMissingApiKey:
    def test_anthropic_missing_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        # Skip test if the SDK isn't installed
        try:
            import anthropic  # noqa
        except ImportError:
            pytest.skip("anthropic SDK not installed")
        with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
            AnthropicClient(model='claude-sonnet-4-5-20250929')

    def test_openai_missing_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        try:
            import openai  # noqa
        except ImportError:
            pytest.skip("openai SDK not installed")
        with pytest.raises(LLMError, match="OPENAI_API_KEY"):
            OpenAIClient(model='gpt-4o')