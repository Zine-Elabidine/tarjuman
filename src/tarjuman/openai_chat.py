"""The OpenAI Chat Completions protocol. Serves OpenAI, OpenRouter, DeepSeek, Groq, vLLM,
llama.cpp and anything else that speaks it."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

import httpx

from . import catalog, errors, limits
from .cancel import Cancel, cancellable
from .events import BlockEnd, BlockStart, Event, Finish, ReasoningDelta, TextDelta, ToolCallDelta
from .transform import Target, prepare
from .types import Image, Message, Reasoning, Replay, Request, Text, Tool, ToolCall, Usage

PROTOCOL = "openai-chat"

_STOPS = {"stop": "end", "tool_calls": "tool_use", "function_call": "tool_use",
          "length": "max_tokens", "content_filter": "refusal"}

# How each dialect turns reasoning on, given a level the model accepts. The level comes from
# the catalog (the nearest one the model lists); for models it doesn't know, from FALLBACK.
REASONING_STYLES: dict[str, Callable[[str], dict[str, Any]]] = {
    "openai": lambda lv: {"reasoning_effort": "none" if lv == "off" else lv},
    "openrouter": lambda lv: {"reasoning": {"enabled": False} if lv == "off" else {"effort": lv}},
    "deepseek": lambda lv: ({"thinking": {"type": "disabled"}} if lv == "off" else
                            {"thinking": {"type": "enabled"}, "reasoning_effort": lv}),
}
FALLBACK = {
    "openai": {"off": "minimal", "minimal": "minimal", "low": "low", "medium": "medium",
               "high": "high", "max": "high"},
    "openrouter": {"off": "off", "minimal": "minimal", "low": "low", "medium": "medium",
                   "high": "high", "max": "high"},
    "deepseek": {"off": "off", "minimal": "low", "low": "low", "medium": "high", "high": "high",
                 "max": "max"},
}


class OpenAIChat:
    protocol = PROTOCOL

    def __init__(self, base_url: str, api_key: str | None, *, provider: str = "openai",
                 headers: dict[str, str] | None = None, max_tokens_field: str = "max_tokens",
                 reasoning_style: str | None = None, reasoning_field: str | None = None,
                 vision: bool = True, load_image: Callable[[str], str] | None = None,
                 catalog: str | None = None, cache_control: bool | list[str] = False,
                 timeout: float = 600, client: httpx.Client | None = None):
        """Provider quirks are arguments, filled from data/providers.json by providers.connect.
        reasoning_style: how to request a reasoning level ("openai", "openrouter", "deepseek").
        reasoning_field: send the model's own reasoning back under this field; None = the
            catalog decides per model (DeepSeek: "reasoning_content"), else never.
        vision: assumed for models the catalog doesn't know.
        catalog: the models.dev provider id to look models up in (limits, levels, prices).
        load_image: turns an Image `ref` into base64 data.
        cache_control: models that cache only at explicit breakpoints (Claude, Qwen
            behind OpenRouter): True for every model, or a list of model-id prefixes."""
        self.base_url = base_url.rstrip("/")
        self.provider = provider
        self.max_tokens_field = max_tokens_field
        self.reasoning_style = reasoning_style
        self.reasoning_field = reasoning_field
        self.vision = vision
        self.load_image = load_image
        self.catalog = catalog
        self.cache_control = cache_control
        h = {"Content-Type": "application/json", **(headers or {})}
        if api_key:
            h["Authorization"] = f"Bearer {api_key}"
        self._headers = h
        self._client = client or httpx.Client(timeout=httpx.Timeout(timeout, connect=30))
        self._windows: dict[str, int] | None = None  # from the server's /models, when asked

    def info(self, model: str) -> catalog.ModelInfo | None:
        return catalog.lookup(self.catalog, model) if self.catalog else None

    def context_window(self, model: str) -> int | None:
        """The model's context window in tokens: the catalog's, else what the server's /models
        listing says (asked once), else None."""
        info = self.info(model)
        if info and info.context:
            return info.context
        if self._windows is None:
            self._windows = limits.served_windows(self._client, self.base_url, self._headers)
        return self._windows.get(model)

    def target(self, model: str) -> Target:
        info = self.info(model)
        vision = info.vision if info and info.vision is not None else self.vision
        return Target(self.provider, PROTOCOL, model, vision=vision)

    # --- request ---------------------------------------------------------------------------

    def body(self, req: Request) -> dict[str, Any]:
        info = self.info(req.model)
        field = self.reasoning_field or (info.reasoning_field if info else None)
        if field == "reasoning_details":  # carried by the replay, see _to_wire
            field = None
        messages = prepare(req.messages, self.target(req.model))
        body: dict[str, Any] = {
            "model": req.model,
            "messages": [w for m in messages for w in self._to_wire(m, field)],
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
            level = (catalog.nearest(req.reasoning, info.levels) if info and info.levels
                     else FALLBACK[self.reasoning_style][req.reasoning])
            body.update(REASONING_STYLES[self.reasoning_style](level))
        for k in ("temperature", "top_p", "stop"):
            if getattr(req, k) is not None:
                body[k] = getattr(req, k)
        if info and not info.temperature:  # the model rejects it (catalog)
            body.pop("temperature", None)
        if req.response_format:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "output", "schema": req.response_format, "strict": True}}
        if req.cache != "none" and self._explicit_cache(req.model):
            _mark_cache(body["messages"], {"type": "ephemeral", "ttl": "1h"}
                        if req.cache == "long" else {"type": "ephemeral"})
        body.update(req.extra or {})
        return body

    def _explicit_cache(self, model: str) -> bool:
        if isinstance(self.cache_control, bool):
            return self.cache_control
        return model.startswith(tuple(self.cache_control))

    def stream(self, request: Request | str, messages: list[Message] | None = None, *,
               tools: list[Tool] | None = None, max_tokens: int | None = None,
               cancel: Cancel | None = None, **extra: Any) -> Iterator[Event]:
        """`stream(Request(...))`, or the shortcut `stream(model, messages, tools=...)`.
        With `cancel`, firing it stops the stream at once (see cancel.py)."""
        req = request if isinstance(request, Request) else Request(
            request, messages or [], tools, max_tokens=max_tokens, extra=extra or None)
        body = self.body(req)
        if cancel is None:
            yield from self._events(req, body, None)
        else:
            yield from cancellable(lambda token: self._events(req, body, token), cancel)

    def _events(self, req: Request, body: dict[str, Any], token: Cancel | None) -> Iterator[Event]:
        if token and token.cancelled:
            return
        try:
            with self._client.stream("POST", f"{self.base_url}/chat/completions",
                                     headers=self._headers, json=body) as r:
                if token:  # closing the response is what stops the provider generating
                    token.on_cancel(r.close)
                if r.status_code >= 400:
                    r.read()
                    raise errors.from_http(r.status_code, r.text, r.headers.get("retry-after"))
                for ev in _Parser(self.provider, req.model).parse(_sse(r.iter_lines())):
                    if isinstance(ev, Finish) and self.catalog:
                        catalog.fill_cost(ev.message, self.catalog)
                    yield ev
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

    def _to_wire(self, m: Message, reasoning_field: str | None = None) -> list[dict[str, Any]]:
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
            # the transform keeps a replay only for the same model: send its reasoning back
            # exactly as it came (OpenRouter / Bifrost carry signatures this way)
            details = (m.replay.response or {}).get("reasoning_details") if m.replay else None
            if details:
                msg["reasoning_details"] = details
            elif reasoning_field:  # after the transform, reasoning left here is the model's own
                thought = "".join(b.text for b in m.content if isinstance(b, Reasoning))
                if thought:
                    msg[reasoning_field] = thought
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


def _mark_cache(messages: list[dict[str, Any]], cache: dict[str, str]) -> None:
    """Breakpoints on the system prompt and the newest message: everything before each one is
    cached, and the next request finds it by walking back. Two of Anthropic's four
    (a tool message works too: measured). Marking needs the content as a list of parts."""
    if not messages:
        return
    ends = {0, len(messages) - 1} if messages[0]["role"] == "system" else {len(messages) - 1}
    for i in ends:
        m = messages[i]
        content = m.get("content")
        if isinstance(content, str) and content:
            m["content"] = content = [{"type": "text", "text": content}]
        texts = [part for part in content or [] if part.get("type") == "text"]
        if texts:
            texts[-1]["cache_control"] = cache


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
        self.details: list[dict[str, Any]] = []  # reasoning_details, merged as they stream

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
        replay = Replay({"reasoning_details": self.details}) if self.details else None
        yield Finish(Message("assistant", list(self.blocks), self.provider, self.model,
                             self.usage, stop, PROTOCOL, self.response_model, replay))

    def _text(self, text: str) -> Iterator[Event]:
        if self.open is None or not isinstance(self.blocks[self.open], Text):
            yield from self._start(Text(""))
        self.blocks[self.open].text += text
        yield TextDelta(self.open, text)

    def _delta(self, d: dict[str, Any]) -> Iterator[Event]:
        details = [x for x in d.get("reasoning_details") or [] if _valid_detail(x)]
        for x in details:
            _append_detail(self.details, x)
        reasoning = d.get("reasoning") or d.get("reasoning_content")
        if not reasoning:  # some servers send only the structured form
            reasoning = "".join(x.get("text") or x.get("summary") or "" for x in details
                                if x["type"] != "reasoning.encrypted")
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


_DETAIL_FIELD = {"reasoning.text": "text", "reasoning.summary": "summary",
                 "reasoning.encrypted": "data"}


def _valid_detail(x: Any) -> bool:
    field = _DETAIL_FIELD.get(x.get("type")) if isinstance(x, dict) else None
    return field is not None and isinstance(x.get(field), str)


def _append_detail(details: list[dict[str, Any]], x: dict[str, Any]) -> None:
    """OpenRouter streams reasoning_details in pieces: consecutive text or summary pieces are
    one entry; encrypted entries stay whole and opaque."""
    last = details[-1] if details else None
    field = _DETAIL_FIELD[x["type"]]
    if last and last["type"] == x["type"] and field != "data":
        last[field] += x[field]
        for k, v in x.items():  # a signature, id or format often arrives on a later piece
            if k != field and v is not None and last.get(k) is None:
                last[k] = v
        return
    details.append(dict(x))


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
