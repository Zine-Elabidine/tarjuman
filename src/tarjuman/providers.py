"""Ready-made connections. A provider is a protocol plus a URL, a key and a few quirks, all
read from the compat table (data/providers.json)."""

from __future__ import annotations

import os
from typing import Any

from . import errors
from .anthropic import Anthropic
from .catalog import providers_table
from .openai_chat import OpenAIChat
from .provider import App

PROTOCOLS: dict[str, type] = {"openai-chat": OpenAIChat, "anthropic-messages": Anthropic}
# compat-table keys that are passed to the protocol's constructor as they are
_QUIRKS = ("headers", "max_tokens_field", "reasoning_style", "reasoning_field", "thinking",
           "vision", "default_max_tokens", "cache_control", "stall")


def names() -> list[str]:
    return list(providers_table())


def connect(name: str, *, api_key: str | None = None, base_url: str | None = None,
            app: App | None = None, **kw: Any) -> OpenAIChat | Anthropic:
    """A client for a provider in the compat table. `base_url` points it elsewhere (a gateway
    speaking the same protocol); `app` names the application to providers that show it;
    keyword arguments override the table's quirks."""
    row = providers_table().get(name)
    if row is None:
        raise errors.TarjumanError(errors.INVALID_REQUEST,
                                   f"unknown provider {name!r}; known: {', '.join(names())}")
    key = api_key or os.environ.get(row["key_env"])
    if not key and not row.get("key_optional") and base_url is None:
        where = f" ({row['key_url']})" if row.get("key_url") else ""
        raise errors.TarjumanError(errors.INVALID_CREDENTIAL, f"set {row['key_env']}{where}")
    quirks = {k: row[k] for k in _QUIRKS if k in row}
    if app:
        named = {h: getattr(app, field) for h, field in (row.get("app_headers") or {}).items()}
        quirks["headers"] = {**{h: v for h, v in named.items() if v}, **quirks.get("headers", {})}
    cls = PROTOCOLS[row["protocol"]]
    url = base_url or row["base_url"]
    args = {"provider": name, "catalog": row.get("catalog"), **quirks, **kw}
    if cls is Anthropic:
        return Anthropic(key, base_url=url, **args)
    return OpenAIChat(url, key, **args)


def _as[P: (OpenAIChat, Anthropic)](kind: type[P], p: OpenAIChat | Anthropic) -> P:
    assert isinstance(p, kind)
    return p


def openrouter(api_key: str | None = None, **kw: Any) -> OpenAIChat:
    return _as(OpenAIChat, connect("openrouter", api_key=api_key, **kw))


def openai(api_key: str | None = None, **kw: Any) -> OpenAIChat:
    return _as(OpenAIChat, connect("openai", api_key=api_key, **kw))


def anthropic(api_key: str | None = None, **kw: Any) -> Anthropic:
    return _as(Anthropic, connect("anthropic", api_key=api_key, **kw))


def deepseek(api_key: str | None = None, **kw: Any) -> OpenAIChat:
    return _as(OpenAIChat, connect("deepseek", api_key=api_key, **kw))


def local(base_url: str = "http://localhost:8000/v1", **kw: Any) -> OpenAIChat:
    """vLLM, llama.cpp server, LM Studio, Ollama's OpenAI endpoint, a gateway..."""
    return _as(OpenAIChat, connect("local", base_url=base_url, **kw))
