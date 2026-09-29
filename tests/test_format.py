import json

import pytest

from tarjuman import (Image, Message, Reasoning, Replay, Request, Text, Tool, ToolCall,
                      ToolResult, Unknown, Usage)


def roundtrip(m: Message) -> Message:
    return Message.from_dict(json.loads(json.dumps(m.to_dict())))


def test_every_block_and_field_survives_json():
    m = Message("assistant",
                [Reasoning("hm"), Reasoning("", redacted=True), Text("hi", citations=[{"u": 1}]),
                 ToolCall("c1", "read", '{"path": "a"}')],
                "anthropic", "claude-x", Usage(10, 5, 2, 7, 3, 0.01, 1), "tool_use",
                "anthropic-messages", "claude-x-2026", Replay({"id": "msg_1"}, ["sig", None, None, None]),
                partial=[Text("cut off")])
    assert roundtrip(m) == m
    assert m.to_dict()["v"] == 1


def test_user_images_and_rich_tool_results():
    u = Message("user", [Text("look"), Image("image/png", data="AAAA")])
    t = Message("tool", [ToolResult("c1", [Text("shot"), Image("image/png", ref="sha256:ab")],
                                    name="screenshot")])
    assert roundtrip(u) == u and roundtrip(t) == t


def test_tool_result_accepts_a_plain_string():
    r = ToolResult("c1", "done")
    assert r.content == [Text("done")] and r.text == "done"


def test_image_needs_exactly_one_source():
    with pytest.raises(ValueError):
        Image("image/png")
    with pytest.raises(ValueError):
        Image("image/png", data="A", url="https://x")


def test_unknown_blocks_are_kept_unchanged():
    d = {"v": 2, "role": "user", "content": [
        {"type": "text", "text": "hi"},
        {"type": "document", "name": "a.pdf", "pages": 3},              # a newer block type
        {"type": "text", "text": "x", "language": "fr"},                 # a newer field
    ]}
    m = Message.from_dict(d)
    assert isinstance(m.content[1], Unknown) and m.content[1].kind == "document"
    assert isinstance(m.content[2], Unknown)
    assert m.to_dict()["content"] == d["content"]                        # saved back as it was


def test_v0_messages_still_load():
    v0 = {"role": "tool", "content": [{"type": "tool_result", "call_id": "c1",
                                        "content": "out", "is_error": False}]}
    m = Message.from_dict(v0)
    assert m.content[0].text == "out"
    old = {"role": "assistant", "content": [{"type": "text", "text": "t"}],
           "provider": "openrouter", "model": "m", "stop": "end",
           "usage": {"input": 1, "cache_read": 0, "cache_write": 0, "output": 2,
                     "reasoning": 0, "cost": None}}
    assert Message.from_dict(old).usage.output == 2


def test_request_roundtrip():
    r = Request("m", [Message.system("s"), Message.user("u")],
                [Tool("read", "Read", {"type": "object"}, strict=True)],
                tool_choice={"name": "read"}, max_tokens=100, reasoning="high",
                stop=["END"], extra={"top_k": 5})
    assert Request.from_dict(json.loads(json.dumps(r.to_dict()))) == r
