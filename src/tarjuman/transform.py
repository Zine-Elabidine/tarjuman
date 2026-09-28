"""Make any history safe to send, whoever produced it."""

from __future__ import annotations

from .types import Message, ToolResult

NO_RESULT = "No result: the call was interrupted before it ran."


def close_tool_calls(messages: list[Message]) -> list[Message]:
    """Every tool call must be answered before the next assistant or user turn.
    Calls left open (an interrupt, a crash) get a synthetic error result."""
    out: list[Message] = []
    pending: list[str] = []

    def flush() -> None:
        if pending:
            out.append(Message("tool", [ToolResult(cid, NO_RESULT, True) for cid in pending]))
            pending.clear()

    for m in messages:
        if m.role == "tool":
            answered = {b.call_id for b in m.content if isinstance(b, ToolResult)}
            pending[:] = [c for c in pending if c not in answered]
            out.append(m)
            continue
        flush()
        out.append(m)
        if m.role == "assistant":
            pending.extend(c.id for c in m.tool_calls)
    flush()
    return out
