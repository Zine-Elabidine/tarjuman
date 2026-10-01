"""The neutral format: messages are lists of typed blocks, the same for every provider.

Everything here is plain data that round-trips through JSON (`to_dict` / `from_dict`), because
it is written to logs and will cross a network. The spec is docs/format.md."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal

VERSION = 1

Role = Literal["system", "user", "assistant", "tool"]
Stop = Literal["end", "tool_use", "max_tokens", "refusal", "pause", "interrupted", "error"]
ReasoningLevel = Literal["off", "minimal", "low", "medium", "high", "max"]


@dataclass
class Text:
    text: str
    citations: Any = None  # provider-specific, dropped when the model changes
    type: Literal["text"] = "text"


@dataclass
class Reasoning:
    text: str
    redacted: bool = False  # the provider hid it: text is empty, the payload is in the replay
    summary: bool = False   # the provider gave a summary, not the reasoning itself
    type: Literal["reasoning"] = "reasoning"


@dataclass
class Image:
    """Exactly one of: base64 `data`, a `url`, or a caller `ref` (e.g. "sha256:...") that a
    loader passed to the provider turns into data at request time."""
    mime: str
    data: str | None = None
    url: str | None = None
    ref: str | None = None
    type: Literal["image"] = "image"

    def __post_init__(self) -> None:
        if sum(x is not None for x in (self.data, self.url, self.ref)) != 1:
            raise ValueError("Image needs exactly one of data, url or ref")


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


@dataclass(init=False)
class ToolResult:
    call_id: str
    content: list[Text | Image]
    is_error: bool = False
    name: str | None = None      # the tool's name (Gemini needs it; filled in by the transform)
    type: Literal["tool_result"] = "tool_result"

    def __init__(self, call_id: str, content: str | list[Text | Image], is_error: bool = False,
                 name: str | None = None, type: Literal["tool_result"] = "tool_result"):
        """`content` may be a plain string: it becomes one Text block."""
        self.call_id, self.is_error, self.name, self.type = call_id, is_error, name, type
        self.content = [Text(content)] if isinstance(content, str) else list(content)

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, Text))


@dataclass
class Unknown:
    """A block this version doesn't know: from a newer Tarjuman, or a provider block we don't
    model. Kept and saved back unchanged; never sent to a model."""
    kind: str
    data: dict[str, Any]
    type: Literal["unknown"] = "unknown"


Block = Text | Reasoning | Image | ToolCall | ToolResult | Unknown
_BLOCKS: dict[str, type] = {"text": Text, "reasoning": Reasoning, "image": Image,
                            "tool_call": ToolCall, "tool_result": ToolResult}

# which blocks each role may carry
ALLOWED: dict[str, tuple[type, ...]] = {
    "system": (Text,),
    "user": (Text, Image),
    "assistant": (Text, Reasoning, ToolCall),
    "tool": (ToolResult,),
}


def block_to_dict(b: Block) -> dict[str, Any]:
    if isinstance(b, Unknown):
        return {"type": b.kind, **b.data}
    if isinstance(b, ToolResult):
        d = {"type": b.type, "call_id": b.call_id,
             "content": [block_to_dict(c) for c in b.content], "is_error": b.is_error}
        if b.name is not None:
            d["name"] = b.name
        return d
    return {k: v for k, v in asdict(b).items() if v is not None and v is not False}


def block_from_dict(d: dict[str, Any]) -> Block:
    kind = str(d.get("type") or "")
    data = {k: v for k, v in d.items() if k != "type"}
    cls = _BLOCKS.get(kind)
    # an unknown type, or a known one with fields from a newer version: keep it untouched
    if cls is None or set(data) - {f.name for f in fields(cls)}:
        return Unknown(kind, data)
    if cls is ToolResult and not isinstance(data.get("content", ""), str):
        data["content"] = [block_from_dict(c) for c in data["content"]]
    return cls(**data)


@dataclass
class Usage:
    input: int = 0          # uncached input tokens
    cache_read: int = 0
    cache_write: int = 0
    output: int = 0         # includes reasoning tokens
    reasoning: int = 0
    cost: float | None = None  # USD, from the provider or the catalog; never a guess
    cache_write_long: int = 0  # the part of cache_write written with the long (1 h) lifetime

    def __add__(self, other: Usage) -> Usage:
        cost = None if self.cost is None and other.cost is None else (self.cost or 0) + (other.cost or 0)
        return Usage(self.input + other.input, self.cache_read + other.cache_read,
                     self.cache_write + other.cache_write, self.output + other.output,
                     self.reasoning + other.reasoning, cost,
                     self.cache_write_long + other.cache_write_long)


@dataclass
class Replay:
    """Provider-private data needed to send a response back to the model that wrote it.
    Opaque to everyone else. `blocks` has one entry per content block, in order."""
    response: Any = None
    blocks: list[Any] | None = None


@dataclass
class Message:
    role: Role
    content: list[Block] = field(default_factory=list)
    # assistant messages only
    provider: str | None = None
    model: str | None = None           # the model we asked for
    usage: Usage | None = None
    stop: Stop | None = None
    protocol: str | None = None        # "openai-chat", "anthropic-messages", ...
    response_model: str | None = None  # the model the provider says it used, if different
    replay: Replay | None = None
    error: str | None = None           # when stop == "error"
    partial: list[Block] | None = None  # interrupted: blocks cut off mid-stream; kept, never sent

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
        d: dict[str, Any] = {"v": VERSION, "role": self.role,
                             "content": [block_to_dict(b) for b in self.content]}
        for k in ("provider", "protocol", "model", "response_model", "stop", "error"):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        if self.usage is not None:
            d["usage"] = asdict(self.usage)
        if self.replay is not None:
            d["replay"] = asdict(self.replay)
        if self.partial:
            d["partial"] = [block_to_dict(b) for b in self.partial]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Message:
        """Reads every version so far (v0 dicts have no "v")."""
        return cls(
            d["role"], [block_from_dict(b) for b in d.get("content", [])],
            d.get("provider"), d.get("model"),
            Usage(**d["usage"]) if d.get("usage") else None, d.get("stop"),
            d.get("protocol"), d.get("response_model"),
            Replay(**d["replay"]) if d.get("replay") else None, d.get("error"),
            [block_from_dict(b) for b in d["partial"]] if d.get("partial") else None)


@dataclass
class Tool:
    """A tool the model may call. `parameters` is a JSON Schema object."""
    name: str
    description: str
    parameters: dict[str, Any]
    strict: bool = False


@dataclass
class Request:
    """One model request, as data (so it can be logged or sent to a gateway)."""
    model: str
    messages: list[Message]
    tools: list[Tool] | None = None
    tool_choice: str | dict[str, str] = "auto"  # "auto" | "none" | "required" | {"name": ...}
    max_tokens: int | None = None
    reasoning: ReasoningLevel | None = None     # None = the provider's default
    temperature: float | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    response_format: dict[str, Any] | None = None  # a JSON Schema for structured output
    cache: Literal["none", "short", "long"] = "short"
    session_id: str | None = None
    extra: dict[str, Any] | None = None            # raw fields merged into the wire body last

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"v": VERSION}
        for k in ("model", "tool_choice", "max_tokens", "reasoning", "temperature", "top_p",
                  "stop", "response_format", "cache", "session_id", "extra"):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        d["messages"] = [m.to_dict() for m in self.messages]
        if self.tools:
            d["tools"] = [asdict(t) for t in self.tools]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Request:
        kw = {k: v for k, v in d.items() if k not in ("v", "messages", "tools")}
        return cls(messages=[Message.from_dict(m) for m in d["messages"]],
                   tools=[Tool(**t) for t in d["tools"]] if d.get("tools") else None, **kw)
