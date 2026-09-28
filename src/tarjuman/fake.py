"""A scripted provider for tests: replays prepared assistant messages as streams."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .errors import TarjumanError
from .events import BlockEnd, BlockStart, Event, Finish, ReasoningDelta, TextDelta, ToolCallDelta
from .types import Message, Reasoning, Text, ToolCall, Usage


class Fake:
    provider = "fake"

    def __init__(self, script: list[Message | TarjumanError]):
        self.script = list(script)
        self.requests: list[list[Message]] = []   # what each call was sent

    def stream(self, model: str, messages: list[Message], **_: Any) -> Iterator[Event]:
        self.requests.append(list(messages))
        if not self.script:
            raise AssertionError("Fake provider: script exhausted")
        item = self.script.pop(0)
        if isinstance(item, TarjumanError):
            raise item
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
            yield BlockEnd(i)
        stop = item.stop or ("tool_use" if item.tool_calls else "end")
        yield Finish(Message("assistant", item.content, "fake", model, item.usage or Usage(1, 0, 0, 1), stop))
