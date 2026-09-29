"""The OpenAI Chat Completions protocol. Serves OpenAI, OpenRouter, DeepSeek, Groq, vLLM,
llama.cpp and anything else that speaks it."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

import httpx

from . import errors
from .events import BlockEnd, BlockStart, Event, Finish, ReasoningDelta, TextDelta, ToolCallDelta
from .transform import Target, prepare
from .types import (Image, Message, Reasoning, Request, Text, Tool, ToolCall, ToolResult,
                    Usage)

PROTOCOL = "openai-chat"

_STOPS = {"stop": "end", "tool_calls": "tool_use", "function_call": "tool_use",
          "length": "max_tokens", "content_filter": "refusal"}

# How each dialect turns reasoning on. Until the catalog knows each model's levels, a level
# the dialect lacks goes to the nearest one.
_EFFORT = {"off": "minimal", "minimal": "minimal", "low": "low", "medium": "medium",
           "high": "high", "max": "high"}
REASONING_STYLES: dict[str, Callable[[str], dict[str, Any]]] = {
    "openai": lambda level: {"reasoning_effort": _EFFORT[level]},
    "openrouter": lambda level: ({"reasoning": {"enabled": False}} if level == "off"
                                 else {"reasoning": {"effort": _EFFORT[level]}}),
}


class OpenAIChat:
    protocol = PROTOCOL

    def __init__(self, base_url: str, api_key: str | None, *, provider: str = "openai",
                 headers: dict[str, str] | None = None, max_tokens_field: str = "max_tokens",
                 reasoning_style: str | None = None, reasoning_field: str | None = None,
                 vision: bool = True, load_image: Callable[[str], str] | None = None,
                 timeout: float = 600, client: httpx.Client | None = None):
        """Quirks are arguments (later: rows of the compat table).
        reasoning_style: how to request a reasoning level ("openai", "openrouter", or None).
        reasoning_field: send the model's own reasoning back under this field
            (DeepSeek thinking mode needs "reasoning_content"); None = never send it.
        load_image: turns an Image `ref` into base64 data."""
        self.base_url = base_url.rstrip("/")
        self.provider = provider
        self.max_tokens_field = max_tokens_field
        self.reasoning_style = reasoning_style
        self.reasoning_field = reasoning_field
        self.vision = vision
        self.load_image = load_image
        h = {"Content-Type": "application/json", **(headers or {})}
        if api_key:
            h["Authorization"] = f"Bearer {api_key}"
        self._headers = h
        self._client = client or httpx.Client(timeout=httpx.Timeout(timeout, connect=30))

    def target(self, model: str) -> Target:
        return Target(self.provider, PROTOCOL, model, vision=self.vision)

    # --- request ---------------------------------------------------------------------------

    def body(self, req: Request) -> dict[str, Any]:
        messages = prepare(req.messages, self.target(req.model))
        body: dict[str, Any] = {
            "model": req.model,
            "messages": [w for m in messages for w in self._to_wire(m)],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if req.tools:
            body["tools"] = [_tool(t) for t in req.tools]
            if req.tool_choice != "auto":
                body["tool_choice"] = (req.tool_choice if isinstance(req.tool_choice, str) else
                                       {"type": "function",
                                        "function": {"name": req.tool_choice["name"]}})
        if req.max_tokens:
            body[self.max_tokens_field] = req.max_tokens
        if req.reasoning and self.reasoning_style:
            body.update(REASONING_STYLES[self.reasoning_style](req.reasoning))
        for k in ("temperature", "top_p", "stop"):
            if getattr(req, k) is not None:
                body[k] = getattr(req, k)
        if req.response_format:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "output", "schema": req.response_format, "strict": True}}
        body.update(req.extra or {})
        return body

    def stream(self, request: Request | str, messages: list[Message] | None = None, *,
               tools: list[Tool] | None = None, max_tokens: int | None = None,
               **extra: Any) -> Iterator[Event]:
        """`stream(Request(...))`, or the shortcut `stream(model, messages, tools=...)`."""
        req = request if isinstance(request, Request) else Request(
            request, messages or [], tools, max_tokens=max_tokens, extra=extra or None)
        body = self.body(req)
        try:
            with self._client.stream("POST", f"{self.base_url}/chat/completions",
                                     headers=self._headers, json=body) as r:
                if r.status_code >= 400:
                    r.read()
                    raise errors.from_http(r.status_code, r.text, r.headers.get("retry-after"))
                yield from _Parser(self.provider, req.model).parse(_sse(r.iter_lines()))
        except httpx.TransportError as e:
            raise errors.TarjumanError(errors.NETWORK, str(e) or type(e).__name__) from e

    def complete(self, request: Request | str, messages: list[Message] | None = None,
                 **kw: Any) -> Message:
        """Non-streaming convenience: consume the stream, return the final message."""
        for ev in self.stream(request, messages, **kw):
            if isinstance(ev, Finish):
                return ev.message
        raise errors.TarjumanError(errors.SERVER_ERROR, "stream ended without a finish")

    # --- neutral -> wire ---------------------------------------------------------------------

    def _to_wire(self, m: Message) -> list[dict[str, Any]]:
        if m.role == "tool":
            out, images = [], []
            for b in m.content:
                out.append({"role": "tool", "tool_call_id": b.call_id,
                            "content": b.text or ("(image)" if b.content else "")})
                images += [c for c in b.content if isinstance(c, Image)]
            if images:  # tool messages can't carry images here: they follow as a user message
                out.append({"role": "user", "content": [
                    {"type": "text", "text": "Images returned by the tool calls above:"},
                    *[self._image(i) for i in images]]})
            return out
        if m.role == "assistant":
            # separate blocks (another model's thinking turned into text, then the answer)
            text = "\n\n".join(b.text for b in m.content if isinstance(b, Text))
            msg: dict[str, Any] = {"role": "assistant", "content": text or None}
            if self.reasoning_field:  # after the transform, reasoning left here is the model's own
                thought = "".join(b.text for b in m.content if isinstance(b, Reasoning))
                if thought:
                    msg[self.reasoning_field] = thought
            if m.tool_calls:
                msg["tool_calls"] = [{"id": c.id, "type": "function",
                                      "function": {"name": c.name, "arguments": c.arguments or "{}"}}
                                     for c in m.tool_calls]
            return [msg]
        if any(isinstance(b, Image) for b in m.content):
            return [{"role": m.role, "content": [
                {"type": "text", "text": b.text} if isinstance(b, Text) else self._image(b)
                for b in m.content]}]
        return [{"role": m.role, "content": m.text}]

    def _image(self, img: Image) -> dict[str, Any]:
        if img.url:
            url = img.url
        else:
            data = img.data
            if data is None:
                if self.load_image is None:
                    raise errors.TarjumanError(errors.INVALID_REQUEST,
                                               f"image {img.ref} has no loader")
                data = self.load_image(img.ref)
            url = f"data:{img.mime};base64,{data}"
        return {"type": "image_url", "image_url": {"url": url}}


def _tool(t: Tool) -> dict[str, Any]:
    fn: dict[str, Any] = {"name": t.name, "description": t.description, "parameters": t.parameters}
    if t.strict:
        fn["strict"] = True
    return {"type": "function", "function": fn}


def _sse(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    for line in lines:
        if not line.startswith("data:"):
            continue  # comments (": OPENROUTER PROCESSING"), event names, blank lines
        data = line[5:].strip()
        if data == "[DONE]":
            return
        if data:
            yield json.loads(data)


# --- wire -> neutral -------------------------------------------------------------------------

class _Parser:
    """Turns Chat Completions chunks into neutral events and assembles the final message."""

    def __init__(self, provider: str, model: str):
        self.provider, self.model = provider, model
        self.response_model: str | None = None
        self.blocks: list[Text | Reasoning | ToolCall] = []
        self.open: int | None = None           # index of the open text/reasoning block
        self.calls: dict[int, int] = {}        # wire tool index -> block index
        self.finish: str | None = None
        self.refused = False
        self.usage = Usage()

    def _end(self, i: int) -> BlockEnd:
        return BlockEnd(i, self.blocks[i])

    def _start(self, block: Text | Reasoning | ToolCall) -> Iterator[Event]:
        if self.open is not None:
            yield self._end(self.open)
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
            if chunk.get("model") and chunk["model"] != self.model:
                self.response_model = chunk["model"]
            if chunk.get("usage"):
                self.usage = _usage(chunk["usage"])
            for choice in chunk.get("choices") or []:
                yield from self._delta(choice.get("delta") or {})
                if choice.get("finish_reason"):
                    self.finish = choice["finish_reason"]
        if self.open is not None:
            yield self._end(self.open)
        for i in self.calls.values():
            yield self._end(i)
        stop = "refusal" if self.refused else _STOPS.get(self.finish or "stop", "end")
        if stop == "end" and any(isinstance(b, ToolCall) for b in self.blocks):
            stop = "tool_use"
        yield Finish(Message("assistant", list(self.blocks), self.provider, self.model,
                             self.usage, stop, PROTOCOL, self.response_model))

    def _text(self, text: str) -> Iterator[Event]:
        if self.open is None or not isinstance(self.blocks[self.open], Text):
            yield from self._start(Text(""))
        self.blocks[self.open].text += text
        yield TextDelta(self.open, text)

    def _delta(self, d: dict[str, Any]) -> Iterator[Event]:
        reasoning = d.get("reasoning") or d.get("reasoning_content")
        if reasoning:
            if self.open is None or not isinstance(self.blocks[self.open], Reasoning):
                yield from self._start(Reasoning(""))
            self.blocks[self.open].text += reasoning
            yield ReasoningDelta(self.open, reasoning)
        if d.get("content"):
            yield from self._text(d["content"])
        if d.get("refusal"):
            self.refused = True
            yield from self._text(d["refusal"])
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
