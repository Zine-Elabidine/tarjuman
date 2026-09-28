"""One stream vocabulary for every provider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .types import Message


@dataclass
class BlockStart:
    index: int
    kind: Literal["text", "reasoning", "tool_call"]
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
    index: int


@dataclass
class Finish:
    """Always the last event: the assembled assistant message, with usage and stop reason."""
    message: Message


Event = BlockStart | TextDelta | ReasoningDelta | ToolCallDelta | BlockEnd | Finish
