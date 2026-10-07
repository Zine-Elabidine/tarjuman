"""The Anthropic Messages protocol (api.anthropic.com and compatible endpoints).

What differs from OpenAI Chat, all handled here:
- one top-level `system` field; later system messages arrive as reminders (the transform)
- tool results live in user messages; consecutive same-role messages are merged
- thinking blocks carry a signature that must come back unchanged (kept in the replay)
- tool arguments are JSON objects, not strings
- prompt caching is explicit: `cache_control` breakpoints on the last tool, the system
  prompt and the newest message
- `max_tokens` is required"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

import httpx

from . import catalog, errors
from .events import BlockEnd, BlockStart, Event, Finish, ReasoningDelta, TextDelta, ToolCallDelta
from .provider import HTTPProvider
from .transform import prepare
from .types import (Image, Message, Reasoning, Replay, Request, Stop, Text, Tool, ToolCall,
                    ToolResult, Unknown, Usage)

PROTOCOL = "anthropic-messages"
API_VERSION = "2023-06-01"
ID_PATTERN = r"[a-zA-Z0-9_-]{1,64}"

_STOPS: dict[str, Stop] = {"end_turn": "end", "stop_sequence": "end", "tool_use": "tool_use",
          "max_tokens": "max_tokens", "model_context_window_exceeded": "max_tokens",
          "pause_turn": "pause", "refusal": "refusal"}

# Adaptive thinking (current models): the model decides how much, guided by an effort. The
# catalog gives each model's accepted efforts; _EFFORT is for models it doesn't know.
_EFFORT = {"minimal": "low", "low": "low", "medium": "medium", "high": "high", "max": "max"}
# Budget thinking (older models): a token budget, which must stay below max_tokens.
_BUDGET = {"minimal": 1024, "low": 2048, "medium": 8192, "high": 16384, "max": 32000}

_ERROR_TYPES = {"overloaded_error": errors.OVERLOADED, "rate_limit_error": errors.RATE_LIMIT,
                "api_error": errors.SERVER_ERROR, "authentication_error": errors.INVALID_CREDENTIAL,
                "permission_error": errors.INVALID_CREDENTIAL,
                "billing_error": errors.INSUFFICIENT_CREDITS,
                "invalid_request_error": errors.INVALID_REQUEST,
                "request_too_large": errors.INVALID_REQUEST, "not_found_error": errors.INVALID_REQUEST}


class Anthropic(HTTPProvider):
    protocol = PROTOCOL
    path = "/v1/messages"
    models_path = "/v1"
    id_pattern = ID_PATTERN

    def __init__(self, api_key: str | None, *, base_url: str = "https://api.anthropic.com",
                 provider: str = "anthropic", headers: dict[str, str] | None = None,
                 thinking: str = "adaptive", default_max_tokens: int = 8192,
                 vision: bool = True, load_image: Callable[[str], str] | None = None,
                 catalog: str | None = None, timeout: float = 600,
                 client: httpx.Client | None = None, stall: float | None = None):
        """thinking: "adaptive" (effort levels) or "budget", for models the catalog doesn't
        know; the catalog decides per model otherwise.
        default_max_tokens: Anthropic requires max_tokens; used when the request has none
            (capped by the model's own output limit when the catalog knows it).
        catalog: the models.dev provider id to look models up in."""
        h = {"anthropic-version": API_VERSION, **(headers or {})}
        if api_key:
            h["x-api-key"] = api_key
        super().__init__(base_url, provider, h, vision=vision, load_image=load_image,
                         catalog=catalog, timeout=timeout, client=client, stall=stall)
        self.thinking = thinking
        self.default_max_tokens = default_max_tokens

    # --- request ---------------------------------------------------------------------------

    def body(self, req: Request) -> dict[str, Any]:
        cache = None if req.cache == "none" else (
            {"type": "ephemeral", "ttl": "1h"} if req.cache == "long" else {"type": "ephemeral"})
        messages = prepare(req.messages, self.target(req.model))
        system = [m.text for m in messages if m.role == "system"]
        turns = _merge([self._to_wire(m) for m in messages if m.role != "system"])

        info = self.info(req.model)
        default = (min(self.default_max_tokens, info.max_output) if info and info.max_output
                   else self.default_max_tokens)
        body: dict[str, Any] = {"model": req.model, "messages": turns, "stream": True,
                                "max_tokens": req.max_tokens or default}
        if system:
            body["system"] = [{"type": "text", "text": t} for t in system]
        if req.tools:
            body["tools"] = [_tool(t) for t in req.tools]
            if req.tool_choice != "auto":
                body["tool_choice"] = (
                    {"type": "tool", "name": req.tool_choice["name"]}
                    if isinstance(req.tool_choice, dict)
                    else {"type": {"required": "any"}.get(req.tool_choice, req.tool_choice)})
        thinking_on = self._reasoning(body, req, info)
        # rejected together with thinking, and by models that take none at all (catalog)
        if req.temperature is not None and not thinking_on and not (info and not info.temperature):
            body["temperature"] = req.temperature
        if req.top_p is not None:
            body["top_p"] = req.top_p
        if req.stop:
            body["stop_sequences"] = req.stop
        if req.response_format:
            raise errors.TarjumanError(errors.INVALID_REQUEST,
                                       "structured output is not supported for Anthropic yet")
        if cache:
            _mark_cache(body, cache)
        body.update(req.extra or {})
        return body

    def _reasoning(self, body: dict[str, Any], req: Request,
                   info: catalog.ModelInfo | None) -> bool:
        level = req.reasoning
        if level is None:
            return False
        if level == "off":
            body["thinking"] = {"type": "disabled"}
            return False
        mode = {"effort": "adaptive", "budget": "budget"}.get((info.thinking if info else None) or "",
                                                              self.thinking)
        if mode == "adaptive":
            effort = (catalog.nearest(level, tuple(lv for lv in info.levels if lv != "off"))
                      if info and info.levels else _EFFORT[level])
            body["thinking"] = {"type": "adaptive", "display": "summarized"}
            body["output_config"] = {"effort": effort}
        else:
            low = info.budget[0] if info and info.budget else 1024
            budget = min(_BUDGET[level], body["max_tokens"] - 1024)
            if budget < low:  # the minimum budget; make room for it and some output
                body["max_tokens"] = low + 2048
                budget = low
            body["thinking"] = {"type": "enabled", "budget_tokens": budget, "display": "summarized"}
        return True

    def _parse(self, model: str, lines: Iterator[str]) -> Iterator[Event]:
        return _Parser(self.provider, model).parse(_sse(lines))

    def _error(self, status: int, body: str, retry_after: str | None) -> errors.TarjumanError:
        return _http_error(status, body, retry_after)

    # --- neutral -> wire ---------------------------------------------------------------------

    def _to_wire(self, m: Message) -> dict[str, Any]:
        if m.role == "tool":
            return {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": b.call_id, "is_error": b.is_error,
                 "content": [self._part(c) for c in b.content] or ""}
                for b in m.content if isinstance(b, ToolResult)]}
        if m.role == "assistant":
            entries = m.replay.blocks if m.replay and m.replay.blocks else [None] * len(m.content)
            return {"role": "assistant",
                    "content": [p for b, e in zip(m.content, entries, strict=True)
                                for p in _assistant_part(b, e)]}
        return {"role": "user", "content": [self._part(b) for b in m.content
                                            if isinstance(b, Text | Image)]}

    def _part(self, b: Text | Image) -> dict[str, Any]:
        if isinstance(b, Text):
            return {"type": "text", "text": b.text}
        if b.url:
            return {"type": "image", "source": {"type": "url", "url": b.url}}
        data = b.data
        if data is None:
            if self.load_image is None:
                raise errors.TarjumanError(errors.INVALID_REQUEST, f"image {b.ref} has no loader")
            data = self.load_image(b.ref or "")
        return {"type": "image", "source": {"type": "base64", "media_type": b.mime, "data": data}}


def _assistant_part(b: Any, entry: Any) -> list[dict[str, Any]]:
    """One neutral block -> wire blocks. `entry` is its replay entry (only present when the
    message came from this same model: the transform removes it otherwise)."""
    if isinstance(b, Text):
        part: dict[str, Any] = {"type": "text", "text": b.text}
        if b.citations:
            part["citations"] = b.citations
        return [part]
    if isinstance(b, Reasoning):
        if b.redacted:
            return [{"type": "redacted_thinking", "data": entry["data"]}] if entry else []
        if entry and entry.get("signature"):
            return [{"type": "thinking", "thinking": b.text, "signature": entry["signature"]}]
        return [{"type": "text", "text": b.text}] if b.text else []  # unsigned: plain text
    if isinstance(b, ToolCall):
        try:
            args = b.args()
        except ValueError:  # the model wrote invalid JSON; the tool result already said so
            args = {"_invalid_arguments": b.arguments}
        return [{"type": "tool_use", "id": b.id, "name": b.name, "input": args}]
    return []


def _merge(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Consecutive same-role turns become one (tool results + a reminder + the user's text)."""
    out: list[dict[str, Any]] = []
    for t in turns:
        if not t["content"]:
            continue
        if out and out[-1]["role"] == t["role"]:
            out[-1]["content"] = out[-1]["content"] + t["content"]
        else:
            out.append({"role": t["role"], "content": list(t["content"])})
    return out


def _mark_cache(body: dict[str, Any], cache: dict[str, str]) -> None:
    """Breakpoints on the last tool, the last system block and the newest message: everything
    before each one is cached, and the next request finds it by walking back."""
    if body.get("tools"):
        body["tools"][-1]["cache_control"] = cache
    if body.get("system"):
        body["system"][-1]["cache_control"] = cache
    if body["messages"]:
        last = body["messages"][-1]["content"][-1]
        if last["type"] in ("text", "image", "tool_result", "tool_use"):
            last["cache_control"] = cache


def _tool(t: Tool) -> dict[str, Any]:
    d: dict[str, Any] = {"name": t.name, "description": t.description,
                         "input_schema": t.parameters or {"type": "object", "properties": {}}}
    if t.strict:
        d["strict"] = True
    return d


def _sse(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    for line in lines:
        if line.startswith("data:"):
            data = line[5:].strip()
            if data:
                yield json.loads(data)  # every payload names its own "type"


def _http_error(status: int, body: str, retry_after: str | None) -> errors.TarjumanError:
    try:
        err = json.loads(body).get("error") or {}
    except (ValueError, AttributeError):
        err = {}
    e = errors.from_http(status, err.get("message") or body, retry_after)
    if "prompt is too long" in e.message.lower():
        e.code = errors.CONTEXT_WINDOW_EXCEEDED
    return e


# --- wire -> neutral -------------------------------------------------------------------------

class _Parser:
    def __init__(self, provider: str, model: str):
        self.provider, self.model = provider, model
        self.response_model: str | None = None
        self.blocks: list[Any] = []
        self.entries: list[Any] = []            # replay entry per block
        self.wire: dict[int, int] = {}          # Anthropic block index -> our index
        self.response: dict[str, Any] = {}
        self.stop: str | None = None
        self.usage = Usage()
        self.stopped = False

    def parse(self, events: Iterator[dict[str, Any]]) -> Iterator[Event]:
        for ev in events:
            kind = ev.get("type")
            if kind == "message_start":
                msg = ev.get("message") or {}
                self.response["id"] = msg.get("id")
                if msg.get("model") and msg["model"] != self.model:
                    self.response_model = msg["model"]
                self._usage(msg.get("usage") or {})
            elif kind == "content_block_start":
                yield from self._start(ev["index"], ev.get("content_block") or {})
            elif kind == "content_block_delta":
                yield from self._delta(ev["index"], ev.get("delta") or {})
            elif kind == "content_block_stop":
                yield from self._stop(ev["index"])
            elif kind == "message_delta":
                d = ev.get("delta") or {}
                if d.get("stop_reason"):
                    self.stop = d["stop_reason"]
                    self.response["stop_reason"] = d["stop_reason"]
                    if d.get("stop_sequence"):
                        self.response["stop_sequence"] = d["stop_sequence"]
                self._usage(ev.get("usage") or {})
            elif kind == "message_stop":
                self.stopped = True
            elif kind == "error":
                err = ev.get("error") or {}
                raise errors.TarjumanError(_ERROR_TYPES.get(str(err.get("type")), errors.SERVER_ERROR),
                                           err.get("message") or "stream error")
            # "ping" and anything newer: ignored
        if not self.stopped:
            raise errors.TarjumanError(errors.NETWORK, "stream ended before message_stop")
        stop = _STOPS.get(self.stop or "end_turn", "end")
        replay = Replay(self.response, self.entries) if any(
            e is not None for e in self.entries) or self.response else None
        yield Finish(Message("assistant", list(self.blocks), self.provider, self.model,
                             self.usage, stop, PROTOCOL, self.response_model, replay))

    def _add(self, wire: int, block: Any, entry: Any = None) -> int:
        self.blocks.append(block)
        self.entries.append(entry)
        self.wire[wire] = len(self.blocks) - 1
        return self.wire[wire]

    def _start(self, wire: int, cb: dict[str, Any]) -> Iterator[Event]:
        t = cb.get("type")
        if t == "text":
            i = self._add(wire, Text(cb.get("text") or ""))
            yield BlockStart(i, "text")
            if cb.get("text"):
                yield TextDelta(i, cb["text"])
        elif t == "thinking":
            i = self._add(wire, Reasoning(cb.get("thinking") or ""), {"signature": cb.get("signature") or ""})
            yield BlockStart(i, "reasoning")
            if cb.get("thinking"):
                yield ReasoningDelta(i, cb["thinking"])
        elif t == "redacted_thinking":
            i = self._add(wire, Reasoning("", redacted=True), {"data": cb.get("data")})
            yield BlockStart(i, "reasoning")
        elif t == "tool_use":
            given = cb.get("input")
            i = self._add(wire, ToolCall(cb.get("id") or "", cb.get("name") or "",
                                         json.dumps(given) if given else ""))
            yield BlockStart(i, "tool_call", cb.get("id"), cb.get("name"))
        else:  # server tools, web search results...: kept in the message, never sent
            i = self._add(wire, Unknown(t or "unknown", {k: v for k, v in cb.items() if k != "type"}))
            yield BlockStart(i, "unknown")

    def _delta(self, wire: int, d: dict[str, Any]) -> Iterator[Event]:
        i = self.wire.get(wire)
        if i is None:
            return
        b, t = self.blocks[i], d.get("type")
        if t == "text_delta" and isinstance(b, Text):
            b.text += d.get("text", "")
            yield TextDelta(i, d.get("text", ""))
        elif t == "thinking_delta" and isinstance(b, Reasoning):
            b.text += d.get("thinking", "")
            yield ReasoningDelta(i, d.get("thinking", ""))
        elif t == "signature_delta" and isinstance(self.entries[i], dict):
            self.entries[i]["signature"] = self.entries[i].get("signature", "") + d.get("signature", "")
        elif t == "input_json_delta" and isinstance(b, ToolCall):
            b.arguments += d.get("partial_json", "")
            yield ToolCallDelta(i, d.get("partial_json", ""))
        elif t == "citations_delta" and isinstance(b, Text):
            b.citations = (b.citations or []) + [d.get("citation")]

    def _stop(self, wire: int) -> Iterator[Event]:
        i = self.wire.get(wire)
        if i is not None:
            yield BlockEnd(i, self.blocks[i], self.entries[i])

    def _usage(self, u: dict[str, Any]) -> None:
        # message_start carries the input side, message_delta the running output count
        if u.get("input_tokens") is not None:
            self.usage.input = u["input_tokens"]
        if u.get("cache_read_input_tokens") is not None:
            self.usage.cache_read = u["cache_read_input_tokens"]
        if u.get("cache_creation_input_tokens") is not None:
            self.usage.cache_write = u["cache_creation_input_tokens"]
        long = (u.get("cache_creation") or {}).get("ephemeral_1h_input_tokens")
        if long is not None:
            self.usage.cache_write_long = long
        if u.get("output_tokens") is not None:
            self.usage.output = u["output_tokens"]
        thinking = (u.get("output_tokens_details") or {}).get("thinking_tokens")
        if thinking is not None:
            self.usage.reasoning = thinking
