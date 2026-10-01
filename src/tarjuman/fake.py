"""A scripted provider for tests: replays prepared assistant messages as streams. It offers the
whole Provider interface, so code written against a real provider runs on it unchanged."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from . import catalog
from .cancel import Cancel, cancelled_error
from .errors import SERVER_ERROR, TarjumanError
from .events import BlockEnd, BlockStart, Event, Finish, ReasoningDelta, TextDelta, ToolCallDelta
from .transform import Target, prepare
from .types import Message, Reasoning, Request, Text, ToolCall, Usage

PROTOCOL = "fake"


class Fake:
    provider = "fake"
    protocol = PROTOCOL

    def __init__(self, script: list[Message | TarjumanError], window: int | None = None):
        self.script = list(script)
        self.window = window
        self.requests: list[list[Message]] = []   # what each call was sent, after the transform

    def info(self, model: str) -> catalog.ModelInfo | None:
        return None

    def context_window(self, model: str) -> int | None:
        return self.window

    def target(self, model: str) -> Target:
        return Target(self.provider, PROTOCOL, model)

    def complete(self, request: Request | str, messages: list[Message] | None = None,
                 **kw: Any) -> Message:
        for ev in self.stream(request, messages, **kw):
            if isinstance(ev, Finish):
                return ev.message
        raise TarjumanError(SERVER_ERROR, "stream ended without a finish")

    def stream(self, request: Request | str, messages: list[Message] | None = None, *,
               cancel: Cancel | None = None, **kw: Any) -> Iterator[Event]:
        for ev in self._events(request, messages):
            if cancel is not None and cancel.cancelled:
                raise cancelled_error()
            yield ev

    def _events(self, request: Request | str, messages: list[Message] | None) -> Iterator[Event]:
        req = request if isinstance(request, Request) else Request(request, messages or [])
        self.requests.append(prepare(req.messages, self.target(req.model)))
        if not self.script:
            raise AssertionError("Fake provider: script exhausted")
        item = self.script.pop(0)
        if isinstance(item, TarjumanError):
            raise item
        entries = item.replay.blocks if item.replay and item.replay.blocks else None
        for i, b in enumerate(item.content):
            if isinstance(b, ToolCall):
                yield BlockStart(i, "tool_call", b.id, b.name)
                yield ToolCallDelta(i, b.arguments)
            elif isinstance(b, Reasoning):
                yield BlockStart(i, "reasoning")
                yield ReasoningDelta(i, b.text)
            elif isinstance(b, Text):
                yield BlockStart(i, "text")
                yield TextDelta(i, b.text)
            yield BlockEnd(i, b, entries[i] if entries else None)
        stop = item.stop or ("tool_use" if item.tool_calls else "end")
        yield Finish(Message("assistant", item.content, self.provider, req.model,
                             item.usage or Usage(1, 0, 0, 1), stop, PROTOCOL,
                             replay=item.replay))
