"""One stream vocabulary for every provider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .types import Block, Message


@dataclass
class BlockStart:
    index: int
    # every block of the final message is announced, so indices are positions in its content;
    # "unknown" = a provider block we don't model (kept in the message, has no deltas)
    kind: Literal["text", "reasoning", "tool_call", "unknown"]
    id: str | None = None      # tool calls only
    name: str | None = None    # tool calls only


@dataclass
class TextDelta:
    index: int
    text: str


@dataclass
class ReasoningDelta:
    index: int
    text: str


@dataclass
class ToolCallDelta:
    index: int
    arguments: str


@dataclass
class BlockEnd:
    """The block is complete. Carries it assembled, with its replay entry (e.g. a signature),
    so a caller interrupted later can still keep every finished block exactly."""
    index: int
    block: Block | None = None
    replay: Any = None


@dataclass
class Finish:
    """Always the last event: the assembled assistant message, with usage and stop reason."""
    message: Message


Event = BlockStart | TextDelta | ReasoningDelta | ToolCallDelta | BlockEnd | Finish
