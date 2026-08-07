"""Pluggable LLM backends.

The agent only ever calls ``client.complete(messages) -> str`` where
``messages`` is a list of ``{"role": ..., "content": ...}`` dicts, OpenAI
chat-format. That's the entire interface (see LLMClient below) -- it's
deliberately the smallest possible surface so new backends are cheap to
add and MockLLM can stand in for any of them in tests.
"""

from __future__ import annotations

import json
from typing import Protocol


class LLMClient(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str:
        """Return the model's raw text response to the given conversation."""
        ...


class LLMError(Exception):
    """Raised for backend configuration/dependency problems."""


class MockLLM:
    """Deterministic, offline backend: replays a pre-scripted list of
    actions in order, one per call, regardless of what's in ``messages``.

    Used by the test suite (so the agent loop is testable without any
    network access or API key) and by ``codeagent run --provider mock
    --script actions.json`` for reproducible demos.
    """

    def __init__(self, script: list[dict]):
        self._script = list(script)
        self._index = 0
        self.received_messages: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.received_messages.append(messages)
        if self._index >= len(self._script):
            raise IndexError(
                "MockLLM script exhausted -- the agent asked for more steps "
                f"than the {len(self._script)} scripted"
            )
        entry = self._script[self._index]
        self._index += 1
        # A raw string entry is returned verbatim (handy for scripting a
        # deliberately-malformed response, to test the agent's parse-retry
        # path); a dict entry is JSON-encoded as a normal action would be.
        if isinstance(entry, str):
            return entry
        return json.dumps(entry)

    @property
    def calls_made(self) -> int:
        return self._index


class OpenAIClient:
    """Thin wrapper over the OpenAI chat completions API.

    Requires the optional ``openai`` dependency (``pip install
    codeagent[openai]``) and an API key, either passed directly or via the
    OPENAI_API_KEY environment variable (the underlying SDK reads it).
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        try:
            import openai
        except ImportError as e:
            raise LLMError(
                "the 'openai' package is required for OpenAIClient "
                "(pip install codeagent[openai])"
            ) from e
        self._client = openai.OpenAI(api_key=api_key)
        self.model = model

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Forward ``messages`` as-is (already OpenAI chat format) and
        return the first choice's text, or "" if the model returned no
        content at all (e.g. a tool-call-only response)."""
        response = self._client.chat.completions.create(
            model=self.model, messages=messages
        )
        return response.choices[0].message.content or ""


class AnthropicClient:
    """Thin wrapper over the Anthropic Messages API.

    Requires the optional ``anthropic`` dependency (``pip install
    codeagent[anthropic]``) and an API key, either passed directly or via
    the ANTHROPIC_API_KEY environment variable.
    """

    def __init__(self, model: str = "claude-3-5-sonnet-latest", api_key: str | None = None):
        try:
            import anthropic
        except ImportError as e:
            raise LLMError(
                "the 'anthropic' package is required for AnthropicClient "
                "(pip install codeagent[anthropic])"
            ) from e
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Translate the flat OpenAI-style ``messages`` list into
        Anthropic's shape (system prompt pulled out separately, remaining
        turns passed as-is) and concatenate every text block of the
        response into a single string."""
        # The Anthropic API takes the system prompt separately from the
        # user/assistant turn list, unlike OpenAI's flat messages array.
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        turns = [m for m in messages if m["role"] != "system"]

        response = self._client.messages.create(
            model=self.model,
            system="\n".join(system_parts),
            max_tokens=2048,
            messages=turns,
        )
        return "".join(block.text for block in response.content if block.type == "text")
