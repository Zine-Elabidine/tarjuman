"""How big a model's context window is, and roughly how many tokens a conversation takes.

The window comes from the catalog first; for models it doesn't know (a local vLLM, a gateway,
a model newer than the catalog) from the server's own /models listing. Token counts are exact
only in a provider's usage report, and only for the model that wrote it: every model has its
own tokenizer. Between reports, and when the model changes, the count is an estimate from the
characters the model will read, at a chars-per-token ratio the caller can calibrate per model
from a real report (see `ratio`)."""

from __future__ import annotations

import json

import httpx

from .types import Image, Message, Reasoning, Text, Tool, ToolCall, ToolResult

CHARS_PER_TOKEN = 3.0     # code, JSON and tool output (measured 2.5-3.1); prose is nearer 4
IMAGE_TOKENS = 1_000      # a mid-size image on most providers
MESSAGE_OVERHEAD = 4      # tokens of role markers and separators per message

# what /models listings call the window: OpenRouter / Together, vLLM, Groq
WINDOW_FIELDS = ("context_length", "max_model_len", "context_window")


def served_windows(client: httpx.Client, base_url: str, headers: dict[str, str]) -> dict[str, int]:
    """Context windows from an OpenAI-style GET {base_url}/models. Empty on any failure: this is
    optional knowledge, never a reason to fail a request."""
    try:
        r = client.get(f"{base_url}/models", headers=headers, timeout=15)
        r.raise_for_status()
        rows = r.json().get("data") or []
    except (httpx.HTTPError, ValueError, AttributeError):
        return {}
    out: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get("id"):
            continue
        top = row.get("top_provider") or {}
        for v in (*(row.get(f) for f in WINDOW_FIELDS), top.get("context_length")):
            if isinstance(v, int) and v > 0:
                out[row["id"]] = v
                break
    return out


def chars(messages: list[Message], tools: list[Tool] | None = None) -> int:
    """Characters the model reads for these messages and tool definitions (images excluded)."""
    n = 0
    for m in messages:
        for b in m.content:
            if isinstance(b, Text | Reasoning):
                n += len(b.text)
            elif isinstance(b, ToolCall):
                n += len(b.name) + len(b.arguments)
            elif isinstance(b, ToolResult):
                n += sum(len(c.text) for c in b.content if isinstance(c, Text))
    for t in tools or []:
        n += len(t.name) + len(t.description) + len(json.dumps(t.parameters))
    return n


def images(messages: list[Message]) -> int:
    count = 0
    for m in messages:
        for b in m.content:
            if isinstance(b, Image):
                count += 1
            elif isinstance(b, ToolResult):
                count += sum(isinstance(c, Image) for c in b.content)
    return count


def estimate(messages: list[Message], tools: list[Tool] | None = None,
             chars_per_token: float = CHARS_PER_TOKEN) -> int:
    """Approximate tokens for a request, at `chars_per_token` (calibrate it with `ratio`)."""
    return (round(chars(messages, tools) / chars_per_token) + images(messages) * IMAGE_TOKENS
            + len(messages) * MESSAGE_OVERHEAD)


def ratio(messages: list[Message], tools: list[Tool] | None, prompt_tokens: int) -> float | None:
    """Characters per token for one model, from a request and the prompt tokens its provider
    reported for it. None when the report is too small to say anything."""
    text_tokens = prompt_tokens - images(messages) * IMAGE_TOKENS - len(messages) * MESSAGE_OVERHEAD
    if text_tokens < 200:
        return None
    return max(1.0, min(chars(messages, tools) / text_tokens, 8.0))

