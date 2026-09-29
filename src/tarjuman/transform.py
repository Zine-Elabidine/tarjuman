"""Make any history safe to send to one target model, whoever produced it.

A pure function: the same history and target always give the same request (warm prompt
caches, testable replay). The rules are in docs/format.md §5."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace

from .types import (ALLOWED, Block, Image, Message, Reasoning, Replay, Text, ToolCall,
                    ToolResult)

NO_RESULT = "No result: the call was interrupted before it ran."
NO_VISION = "(image omitted: this model does not accept images)"
REMINDER = "<system-reminder>\n{}\n</system-reminder>"


@dataclass(frozen=True)
class Target:
    """The model a request is for: what the transform needs to know about it."""
    provider: str
    protocol: str
    model: str
    vision: bool = True           # unknown models are assumed to accept images
    id_pattern: str | None = None  # tool-call ids must match this (e.g. Anthropic's)


def same_model(m: Message, t: Target) -> bool:
    return (m.provider, m.protocol, m.model) == (t.provider, t.protocol, t.model)


def prepare(messages: list[Message], target: Target) -> list[Message]:
    ids: dict[str, str] = {}
    out: list[Message] = []
    started = False                    # a non-system message has been seen
    open_calls: dict[str, str] = {}    # unanswered call id -> tool name
    names: dict[str, str] = {}         # every sent call id -> tool name
    held: list[Message] = []           # reminders waiting for open calls to be answered

    def close() -> None:
        if open_calls:
            out.append(Message("tool", [ToolResult(cid, NO_RESULT, True, name)
                                        for cid, name in open_calls.items()]))
            open_calls.clear()
        out.extend(held)
        held.clear()

    for m in messages:
        if m.role == "system":
            text = _text_only(m)
            if not text:
                continue
            if not started:
                out.append(Message.system(text))
            elif open_calls:
                held.append(Message.user(REMINDER.format(text)))
            else:
                out.append(Message.user(REMINDER.format(text)))
            continue
        started = True

        if m.role == "assistant":
            close()
            if m.stop == "error":
                continue
            a = _assistant(m, target, ids)
            if a is None:
                continue
            out.append(a)
            for c in a.tool_calls:
                open_calls[c.id] = c.name
                names[c.id] = c.name

        elif m.role == "tool":
            results = []
            for b in m.content:
                if not isinstance(b, ToolResult):
                    continue
                cid = ids.get(b.call_id, b.call_id)
                if cid not in open_calls:
                    continue  # its call was dropped (errored turn) or already answered
                content = b.content if target.vision else _no_images(b.content)
                results.append(ToolResult(cid, content, b.is_error, names[cid]))
                del open_calls[cid]
            if results:
                out.append(Message("tool", results))
            if not open_calls:
                out.extend(held)
                held.clear()

        else:  # user
            close()
            blocks = _clean(m)
            if not target.vision:
                blocks = _no_images(blocks)
            if blocks:
                out.append(Message("user", blocks))

    close()
    return out


def _assistant(m: Message, t: Target, ids: dict[str, str]) -> Message | None:
    same = same_model(m, t)
    replay = m.replay if same else None
    if replay and replay.blocks is not None and len(replay.blocks) != len(m.content):
        replay = None  # misaligned: never guess which entry belongs to which block
    entries = replay.blocks if replay else None
    blocks: list[Block] = []
    kept: list = []
    for i, b in enumerate(m.content):
        if not isinstance(b, ALLOWED["assistant"]):
            continue
        if isinstance(b, Reasoning) and not same:
            if b.redacted or not b.text.strip():
                continue
            b = Text(b.text)                       # another model's thinking becomes text
        elif isinstance(b, Text):
            if not b.text:
                continue
            if not same and b.citations is not None:
                b = Text(b.text)
        elif isinstance(b, Reasoning) and not b.text and not b.redacted and not (entries and entries[i]):
            continue                               # empty and nothing to replay
        elif isinstance(b, ToolCall) and t.id_pattern and not re.fullmatch(t.id_pattern, b.id):
            ids[b.id] = _new_id(b.id)
            b = replace(b, id=ids[b.id])
        blocks.append(b)
        if entries is not None:
            kept.append(entries[i])
    if not blocks:
        return None
    if replay:
        replay = Replay(replay.response, kept if entries is not None else None)
    return Message("assistant", blocks, m.provider, m.model, m.usage, m.stop, m.protocol,
                   m.response_model, replay)


def _clean(m: Message) -> list[Block]:
    return [b for b in m.content if isinstance(b, ALLOWED[m.role])
            and not (isinstance(b, Text) and not b.text)]


def _text_only(m: Message) -> str:
    return "\n\n".join(b.text for b in m.content if isinstance(b, Text) and b.text)


def _no_images(blocks: list) -> list:
    out: list = []
    for b in blocks:
        if isinstance(b, Image):
            if not (out and isinstance(out[-1], Text) and out[-1].text == NO_VISION):
                out.append(Text(NO_VISION))
        else:
            out.append(b)
    return out


def _new_id(old: str) -> str:
    """A deterministic id that satisfies the strictest rule ([a-zA-Z0-9_-], max 64)."""
    return "call_" + hashlib.sha256(old.encode()).hexdigest()[:24]
