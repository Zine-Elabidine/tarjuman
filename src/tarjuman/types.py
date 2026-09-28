"""The neutral format: messages are lists of typed blocks, the same for every provider."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]
Stop = Literal["end", "tool_use", "max_tokens", "content_filter", "interrupted"]


@dataclass
class Text:
    text: str
    type: Literal["text"] = "text"


@dataclass
class Reasoning:
    text: str
    type: Literal["reasoning"] = "reasoning"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON as the model wrote it; may be invalid
    type: Literal["tool_call"] = "tool_call"

    def args(self) -> dict[str, Any]:
        """Parsed arguments. Raises ValueError if the model wrote invalid JSON."""
        if not self.arguments.strip():
            return {}
        value = json.loads(self.arguments)
        if not isinstance(value, dict):
            raise ValueError("tool arguments must be a JSON object")
        return value


@dataclass
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"


Block = Text | Reasoning | ToolCall | ToolResult
_BLOCKS = {"text": Text, "reasoning": Reasoning, "tool_call": ToolCall, "tool_result": ToolResult}


@dataclass
class Usage:
    input: int = 0          # uncached input tokens
    cache_read: int = 0
    cache_write: int = 0
    output: int = 0         # includes reasoning tokens
    reasoning: int = 0
    cost: float | None = None  # in USD, when the provider reports it

    def __add__(self, other: Usage) -> Usage:
        cost = None if self.cost is None and other.cost is None else (self.cost or 0) + (other.cost or 0)
        return Usage(self.input + other.input, self.cache_read + other.cache_read,
                     self.cache_write + other.cache_write, self.output + other.output,
                     self.reasoning + other.reasoning, cost)


@dataclass
class Message:
    role: Role
    content: list[Block] = field(default_factory=list)
    # set on assistant messages
    provider: str | None = None
    model: str | None = None
    usage: Usage | None = None
    stop: Stop | None = None

    @classmethod
    def user(cls, text: str) -> Message:
        return cls("user", [Text(text)])

    @classmethod
    def system(cls, text: str) -> Message:
        return cls("system", [Text(text)])

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, Text))

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [b for b in self.content if isinstance(b, ToolCall)]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Message:
        blocks = [_BLOCKS[b["type"]](**b) for b in d.get("content", [])]
        usage = Usage(**d["usage"]) if d.get("usage") else None
        return cls(d["role"], blocks, d.get("provider"), d.get("model"), usage, d.get("stop"))


@dataclass
class Tool:
    """A tool the model may call. `parameters` is a JSON Schema object."""
    name: str
    description: str
    parameters: dict[str, Any]
