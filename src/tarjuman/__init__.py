"""Tarjuman: one neutral language for every LLM provider."""

from importlib.metadata import version

from . import limits
from .anthropic import Anthropic
from .cancel import Cancel
from .errors import TarjumanError
from .events import BlockEnd, BlockStart, Event, Finish, ReasoningDelta, TextDelta, ToolCallDelta
from .openai_chat import OpenAIChat
from .transform import Target, prepare
from .types import (Block, Image, Message, Reasoning, Replay, Request, Text, Tool, ToolCall,
                    ToolResult, Unknown, Usage)

__version__ = version("tarjuman")   # one source: pyproject.toml

__all__ = ["Anthropic", "Block", "BlockEnd", "BlockStart", "Cancel", "Event", "Finish", "Image",
           "Message", "OpenAIChat", "Reasoning", "ReasoningDelta", "Replay", "Request", "Target",
           "TarjumanError", "Text", "TextDelta", "Tool", "ToolCall", "ToolCallDelta",
           "ToolResult", "Unknown", "Usage", "limits", "prepare"]
