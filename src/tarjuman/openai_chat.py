"""The OpenAI Chat Completions protocol. Serves OpenAI, OpenRouter, DeepSeek, Groq, vLLM,
llama.cpp and anything else that speaks it."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx

from . import errors
from .events import BlockEnd, BlockStart, Event, Finish, ReasoningDelta, TextDelta, ToolCallDelta
from .transform import close_tool_calls
from .types import Message, Reasoning, Text, Tool, ToolCall, ToolResult, Usage

_STOPS = {"stop": "end", "tool_calls": "tool_use", "function_call": "tool_use",
          "length": "max_tokens", "content_filter": "content_filter"}


class OpenAIChat:
    def __init__(self, base_url: str, api_key: str | None, *, provider: str = "openai",
                 headers: dict[str, str] | None = None, max_tokens_field: str = "max_tokens",
                 timeout: float = 600, client: httpx.Client | None = None):
        self.base_url = base_url.rstrip("/")
        self.provider = provider
        self.max_tokens_field = max_tokens_field
        h = {"Content-Type": "application/json", **(headers or {})}
        if api_key:
            h["Authorization"] = f"Bearer {api_key}"
        self._headers = h
        self._client = client or httpx.Client(timeout=httpx.Timeout(timeout, connect=30))

    # --- request ---------------------------------------------------------------------------

    def body(self, model: str, messages: list[Message], tools: list[Tool] | None,
             max_tokens: int | None, extra: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": [m for msg in close_tool_calls(messages) for m in _to_wire(msg)],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            body["tools"] = [{"type": "function", "function": {
                "name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in tools]
        if max_tokens:
            body[self.max_tokens_field] = max_tokens
        body.update(extra)
        return body

    def stream(self, model: str, messages: list[Message], *, tools: list[Tool] | None = None,
               max_tokens: int | None = None, **extra: Any) -> Iterator[Event]:
        body = self.body(model, messages, tools, max_tokens, extra)
        try:
            with self._client.stream("POST", f"{self.base_url}/chat/completions",
                                     headers=self._headers, json=body) as r:
                if r.status_code >= 400:
                    r.read()
                    raise errors.from_http(r.status_code, r.text, r.headers.get("retry-after"))
                yield from _Parser(self.provider, model).parse(_sse(r.iter_lines()))
        except httpx.TransportError as e:
            raise errors.TarjumanError(errors.NETWORK, str(e) or type(e).__name__) from e

    def complete(self, model: str, messages: list[Message], **kw: Any) -> Message:
        """Non-streaming convenience: consume the stream, return the final message."""
        for ev in self.stream(model, messages, **kw):
            if isinstance(ev, Finish):
                return ev.message
        raise errors.TarjumanError(errors.SERVER_ERROR, "stream ended without a finish")


def _to_wire(m: Message) -> list[dict[str, Any]]:
    if m.role == "tool":
        return [{"role": "tool", "tool_call_id": b.call_id, "content": b.content}
                for b in m.content if isinstance(b, ToolResult)]
    if m.role == "assistant":
        # Reasoning from any model is not sent back through this protocol.
        out: dict[str, Any] = {"role": "assistant", "content": m.text or None}
        calls = m.tool_calls
        if calls:
            out["tool_calls"] = [{"id": c.id, "type": "function",
                                  "function": {"name": c.name, "arguments": c.arguments or "{}"}}
                                 for c in calls]
        return [out]
    return [{"role": m.role, "content": m.text}]


def _sse(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    for line in lines:
        if not line.startswith("data:"):
            continue  # comments (": OPENROUTER PROCESSING"), event names, blank lines
        data = line[5:].strip()
        if data == "[DONE]":
            return
        if data:
            yield json.loads(data)


class _Parser:
    """Turns Chat Completions chunks into neutral events and assembles the final message."""

    def __init__(self, provider: str, model: str):
        self.provider, self.model = provider, model
        self.blocks: list[Text | Reasoning | ToolCall] = []
        self.open: int | None = None           # index of the open text/reasoning block
        self.calls: dict[int, int] = {}        # wire tool index -> block index
        self.finish: str | None = None
        self.usage = Usage()

    def _start(self, block: Text | Reasoning | ToolCall) -> Iterator[Event]:
        if self.open is not None:
            yield BlockEnd(self.open)
            self.open = None
        self.blocks.append(block)
        i = len(self.blocks) - 1
        if isinstance(block, ToolCall):
            yield BlockStart(i, "tool_call", block.id, block.name)
        else:
            self.open = i
            yield BlockStart(i, block.type)

    def parse(self, chunks: Iterator[dict[str, Any]]) -> Iterator[Event]:
        for chunk in chunks:
            if "error" in chunk:  # errors that arrive mid-stream with HTTP 200
                err = chunk["error"]
                code = err.get("code") if isinstance(err, dict) else None
                msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                raise errors.from_http(code if isinstance(code, int) else 500, msg)
            if chunk.get("model"):
                self.model = chunk["model"]
            if chunk.get("usage"):
                self.usage = _usage(chunk["usage"])
            for choice in chunk.get("choices") or []:
                yield from self._delta(choice.get("delta") or {})
                if choice.get("finish_reason"):
                    self.finish = choice["finish_reason"]
        if self.open is not None:
            yield BlockEnd(self.open)
        for i in self.calls.values():
            yield BlockEnd(i)
        stop = _STOPS.get(self.finish or "stop", "end")
        if stop == "end" and any(isinstance(b, ToolCall) for b in self.blocks):
            stop = "tool_use"
        yield Finish(Message("assistant", list(self.blocks), self.provider, self.model,
                             self.usage, stop))

    def _delta(self, d: dict[str, Any]) -> Iterator[Event]:
        reasoning = d.get("reasoning") or d.get("reasoning_content")
        if reasoning:
            if self.open is None or not isinstance(self.blocks[self.open], Reasoning):
                yield from self._start(Reasoning(""))
            self.blocks[self.open].text += reasoning
            yield ReasoningDelta(self.open, reasoning)
        if d.get("content"):
            if self.open is None or not isinstance(self.blocks[self.open], Text):
                yield from self._start(Text(""))
            self.blocks[self.open].text += d["content"]
            yield TextDelta(self.open, d["content"])
        for tc in d.get("tool_calls") or []:
            wire = tc.get("index", 0)
            fn = tc.get("function") or {}
            if wire not in self.calls:
                call = ToolCall(tc.get("id") or f"call_{wire}", fn.get("name") or "", "")
                yield from self._start(call)
                self.calls[wire] = len(self.blocks) - 1
            call = self.blocks[self.calls[wire]]
            if fn.get("name") and not call.name:
                call.name = fn["name"]
            if fn.get("arguments"):
                call.arguments += fn["arguments"]
                yield ToolCallDelta(self.calls[wire], fn["arguments"])


def _usage(u: dict[str, Any]) -> Usage:
    prompt = u.get("prompt_tokens") or 0
    details = u.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens") or 0
    written = details.get("cache_write_tokens") or 0
    out_details = u.get("completion_tokens_details") or {}
    return Usage(input=max(prompt - cached - written, 0), cache_read=cached, cache_write=written,
                 output=u.get("completion_tokens") or 0,
                 reasoning=out_details.get("reasoning_tokens") or 0,
                 cost=u.get("cost"))
