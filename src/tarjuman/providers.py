"""Ready-made connections. A provider is a protocol plus a URL, a key and a few quirks."""

from __future__ import annotations

import os

from . import errors
from .openai_chat import OpenAIChat


def openrouter(api_key: str | None = None, **kw) -> OpenAIChat:
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise errors.TarjumanError(errors.INVALID_CREDENTIAL,
                                   "set OPENROUTER_API_KEY (https://openrouter.ai/keys)")
    return OpenAIChat("https://openrouter.ai/api/v1", key, provider="openrouter",
                      headers={"HTTP-Referer": "https://github.com/Zine-Elabidine/diwan",
                               "X-Title": "Diwan"}, **kw)


def openai(api_key: str | None = None, **kw) -> OpenAIChat:
    return OpenAIChat("https://api.openai.com/v1", api_key or os.environ.get("OPENAI_API_KEY"),
                      provider="openai", max_tokens_field="max_completion_tokens", **kw)


def local(base_url: str = "http://localhost:8000/v1", **kw) -> OpenAIChat:
    """vLLM, llama.cpp server, LM Studio, Ollama's OpenAI endpoint..."""
    return OpenAIChat(base_url, os.environ.get("LOCAL_API_KEY"), provider="local", **kw)
