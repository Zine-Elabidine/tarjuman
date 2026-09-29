"""A scripted provider for tests: replays prepared assistant messages as streams."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .errors import TarjumanError
from .events import BlockEnd, BlockStart, Event, Finish, ReasoningDelta, TextDelta, ToolCallDelta
from .transform import Target, prepare
from .types import Message, Reasoning, Request, Text, ToolCall, Usage

PROTOCOL = "fake"


class Fake:
    provider = "fake"
    protocol = PROTOCOL

    def __init__(self, script: list[Message | TarjumanError]):
        self.script = list(script)
        self.requests: list[list[Message]] = []   # what each call was sent, after the transform

    def stream(self, request: Request | str, messages: list[Message] | None = None,
               **kw: Any) -> Iterator[Event]:
        req = request if isinstance(request, Request) else Request(request, messages or [])
        self.requests.append(prepare(req.messages, Target(self.provider, PROTOCOL, req.model)))
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
