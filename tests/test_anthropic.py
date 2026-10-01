import json

import httpx
import pytest

from tarjuman import (Anthropic, BlockEnd, BlockStart, Message, OpenAIChat, Reasoning, Request,
                      TarjumanError, Text, TextDelta, Tool, ToolCall, ToolResult, Unknown, errors)
from tarjuman.transform import REMINDER


def sse(*events):
    out = []
    for e in events:
        out += [f"event: {e['type']}", f"data: {json.dumps(e)}", ""]
    return "\n".join(out).encode()


def start(model="claude-x", **usage):
    return {"type": "message_start", "message": {
        "id": "msg_01", "type": "message", "role": "assistant", "model": model, "content": [],
        "usage": {"input_tokens": 50, "output_tokens": 1, **usage}}}


def block(i, **cb):
    return {"type": "content_block_start", "index": i, "content_block": cb}


def delta(i, **d):
    return {"type": "content_block_delta", "index": i, "delta": d}


def stop(i):
    return {"type": "content_block_stop", "index": i}


def end(reason="end_turn", **usage):
    return [{"type": "message_delta", "delta": {"stop_reason": reason, "stop_sequence": None},
             "usage": {"output_tokens": 30, **usage}},
            {"type": "message_stop"}]


THINK_THEN_TOOL = [
    start(cache_read_input_tokens=400, cache_creation_input_tokens=100,
          cache_creation={"ephemeral_5m_input_tokens": 100, "ephemeral_1h_input_tokens": 0}),
    block(0, type="thinking", thinking="", signature=""),
    delta(0, type="thinking_delta", thinking="I should "),
    delta(0, type="thinking_delta", thinking="read it."),
    delta(0, type="signature_delta", signature="EqQBsig"),
    stop(0),
    {"type": "ping"},
    block(1, type="text", text=""),
    delta(1, type="text_delta", text="Reading."),
    stop(1),
    block(2, type="tool_use", id="toolu_01", name="read", input={}),
    delta(2, type="input_json_delta", partial_json='{"path":'),
    delta(2, type="input_json_delta", partial_json=' "a.py"}'),
    stop(2),
    *end("tool_use"),
]


def client(events, seen=None, **kw):
    def handler(req):
        if seen is not None:
            seen["body"] = json.loads(req.content)
            seen["headers"] = req.headers
        return httpx.Response(200, content=sse(*events))
    return Anthropic("k", client=httpx.Client(transport=httpx.MockTransport(handler)), **kw)


def test_stream_thinking_signature_text_and_tool_call():
    seen = {}
    events = list(client(THINK_THEN_TOOL, seen).stream("claude-x", [Message.user("hi")]))
    assert seen["headers"]["x-api-key"] == "k" and seen["headers"]["anthropic-version"]
    kinds = [e.kind for e in events if isinstance(e, BlockStart)]
    assert kinds == ["reasoning", "text", "tool_call"]
    ends = [e for e in events if isinstance(e, BlockEnd)]
    assert ends[0].replay == {"signature": "EqQBsig"}          # signature arrives before the end

    msg = events[-1].message
    assert msg.content == [Reasoning("I should read it."), Text("Reading."),
                           ToolCall("toolu_01", "read", '{"path": "a.py"}')]
    assert msg.replay.blocks[0] == {"signature": "EqQBsig"}
    assert msg.replay.response["id"] == "msg_01"
    assert (msg.stop, msg.protocol, msg.model) == ("tool_use", "anthropic-messages", "claude-x")
    u = msg.usage
    assert (u.input, u.cache_read, u.cache_write, u.output) == (50, 400, 100, 30)


def test_history_goes_out_in_anthropic_shape():
    first = client(THINK_THEN_TOOL).complete("claude-x", [Message.user("hi")])
    history = [Message.system("You are Diwan."), Message.user("hi"), first,
               Message("tool", [ToolResult("toolu_01", "print(1)")]),
               Message.system("80% of the budget used"),
               Message.user("keep going")]
    seen = {}
    client(THINK_THEN_TOOL, seen).complete(Request("claude-x", history, [
        Tool("read", "Read a file", {"type": "object", "properties": {"path": {"type": "string"}}})]))
    body = seen["body"]

    assert body["system"] == [{"type": "text", "text": "You are Diwan.",
                               "cache_control": {"type": "ephemeral"}}]
    assert body["max_tokens"] == 8192                             # required: the default fills in
    assert body["tools"][0]["input_schema"]["properties"]["path"] == {"type": "string"}
    assert body["tools"][-1]["cache_control"] == {"type": "ephemeral"}

    user, assistant, results = body["messages"]
    assert user == {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    assert assistant["content"][0] == {"type": "thinking", "thinking": "I should read it.",
                                       "signature": "EqQBsig"}      # same model: replayed exactly
    assert assistant["content"][2] == {"type": "tool_use", "id": "toolu_01", "name": "read",
                                       "input": {"path": "a.py"}}
    # tool result, reminder and the user's text merged into one user turn, result first
    assert [p["type"] for p in results["content"]] == ["tool_result", "text", "text"]
    assert results["content"][1]["text"] == REMINDER.format("80% of the budget used")
    assert results["content"][2] == {"type": "text", "text": "keep going",
                                     "cache_control": {"type": "ephemeral"}}


def test_another_models_history_is_adapted():
    theirs = Message("assistant", [Reasoning("hmm"), ToolCall("call|a.b", "read", "{}")],
                     "openrouter", "deepseek-v4", None, "tool_use", "openai-chat")
    h = [Message.user("go"), theirs, Message("tool", [ToolResult("call|a.b", "x")])]
    seen = {}
    client(THINK_THEN_TOOL, seen).complete("claude-x", h)
    assistant, results = seen["body"]["messages"][1:]
    assert assistant["content"][0] == {"type": "text", "text": "hmm"}   # foreign thinking: text
    new_id = assistant["content"][1]["id"]
    assert new_id != "call|a.b" and results["content"][0]["tool_use_id"] == new_id


def test_redacted_thinking_roundtrips_for_the_same_model():
    events = [start(), block(0, type="redacted_thinking", data="ENCRYPTED"), stop(0),
              block(1, type="text", text=""), delta(1, type="text_delta", text="ok"), stop(1),
              *end()]
    msg = client(events).complete("claude-x", [Message.user("hi")])
    assert msg.content[0] == Reasoning("", redacted=True)
    seen = {}
    client(events, seen).complete("claude-x", [Message.user("hi"), msg, Message.user("more")])
    assert seen["body"]["messages"][1]["content"][0] == {"type": "redacted_thinking",
                                                         "data": "ENCRYPTED"}


def test_reasoning_levels_adaptive_and_budget():
    seen = {}
    client(THINK_THEN_TOOL, seen).complete(
        Request("claude-x", [Message.user("hi")], reasoning="minimal", temperature=0.5))
    assert seen["body"]["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert seen["body"]["output_config"] == {"effort": "low"}
    assert "temperature" not in seen["body"]                    # rejected together with thinking
    client(THINK_THEN_TOOL, seen, thinking="budget").complete(
        Request("claude-x", [Message.user("hi")], reasoning="high", max_tokens=20000))
    assert seen["body"]["thinking"]["budget_tokens"] == 16384
    client(THINK_THEN_TOOL, seen).complete(
        Request("claude-x", [Message.user("hi")], reasoning="off", temperature=0.5))
    assert seen["body"]["thinking"] == {"type": "disabled"} and seen["body"]["temperature"] == 0.5


def test_tool_choice_and_cache_options():
    seen = {}
    tools = [Tool("read", "R", {"type": "object"})]
    client(THINK_THEN_TOOL, seen).complete(
        Request("claude-x", [Message.user("hi")], tools, tool_choice="required", cache="long"))
    assert seen["body"]["tool_choice"] == {"type": "any"}
    assert seen["body"]["tools"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    client(THINK_THEN_TOOL, seen).complete(
        Request("claude-x", [Message.user("hi")], tools, tool_choice={"name": "read"}, cache="none"))
    assert seen["body"]["tool_choice"] == {"type": "tool", "name": "read"}
    assert "cache_control" not in json.dumps(seen["body"])


@pytest.mark.parametrize("reason,expected", [("pause_turn", "pause"), ("refusal", "refusal"),
                                             ("max_tokens", "max_tokens"), ("stop_sequence", "end")])
def test_stop_reasons(reason, expected):
    events = [start(), block(0, type="text", text="x"), stop(0), *end(reason)]
    msg = client(events).complete("claude-x", [Message.user("hi")])
    assert msg.stop == expected and msg.replay.response["stop_reason"] == reason


def test_server_blocks_are_kept_as_unknown():
    events = [start(), block(0, type="server_tool_use", id="srv_1", name="web_search", input={}),
              stop(0), block(1, type="text", text=""), delta(1, type="text_delta", text="found"),
              stop(1), *end()]
    evs = list(client(events).stream("claude-x", [Message.user("hi")]))
    assert [e.kind for e in evs if isinstance(e, BlockStart)] == ["unknown", "text"]
    msg = evs[-1].message
    assert isinstance(msg.content[0], Unknown) and msg.content[0].kind == "server_tool_use"
    assert [e.index for e in evs if isinstance(e, TextDelta)] == [1]    # indices = positions


def test_errors():
    def fails(status, body):
        return Anthropic("k", client=httpx.Client(transport=httpx.MockTransport(
            lambda req: httpx.Response(status, text=json.dumps(body)))))
    with pytest.raises(TarjumanError) as e:
        fails(529, {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}
              ).complete("claude-x", [Message.user("hi")])
    assert e.value.code == errors.OVERLOADED and e.value.retryable
    with pytest.raises(TarjumanError) as e:
        too_long = "prompt is too long: 210000 tokens > 200000 maximum"
        fails(400, {"type": "error", "error": {"type": "invalid_request_error", "message": too_long}}
              ).complete("claude-x", [Message.user("hi")])
    assert e.value.code == errors.CONTEXT_WINDOW_EXCEEDED
    with pytest.raises(TarjumanError) as e:   # mid-stream error event
        client([start(), {"type": "error", "error": {"type": "overloaded_error", "message": "x"}}]
               ).complete("claude-x", [Message.user("hi")])
    assert e.value.code == errors.OVERLOADED
    with pytest.raises(TarjumanError) as e:   # connection dropped before message_stop
        client([start(), block(0, type="text", text="x")]).complete("claude-x", [Message.user("hi")])
    assert e.value.code == errors.NETWORK and e.value.retryable


def test_switch_claude_to_openai_chat_and_back_loses_nothing():
    """The point of Tarjuman: one conversation, two protocols, nothing lost that could be kept."""
    claude = client(THINK_THEN_TOOL)
    turn1 = claude.complete("claude-x", [Message.user("hi")])
    history = [Message.user("hi"), turn1, Message("tool", [ToolResult("toolu_01", "print(1)")])]

    # DeepSeek (OpenAI Chat) continues: sees Claude's thinking as text, never the signature
    seen_ds = {}

    def ds(req):
        seen_ds["body"] = json.loads(req.content)
        chunk = {"choices": [{"index": 0, "delta": {"reasoning_content": "fine", "content": "done"},
                              "finish_reason": "stop"}]}
        return httpx.Response(200, content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode())
    deepseek = OpenAIChat("https://ds.test/v1", "k", provider="deepseek",
                          client=httpx.Client(transport=httpx.MockTransport(ds)))
    turn2 = deepseek.complete("deepseek-v4", history)
    assert "EqQBsig" not in json.dumps(seen_ds["body"])
    assert seen_ds["body"]["messages"][1]["content"] == "I should read it.\n\nReading."

    # back to Claude: its own thinking returns signed; DeepSeek's reasoning comes as text
    seen = {}
    client(THINK_THEN_TOOL, seen).complete("claude-x", [*history, turn2, Message.user("thanks")])
    msgs = seen["body"]["messages"]
    assert msgs[1]["content"][0]["signature"] == "EqQBsig"
    assert msgs[3]["content"][:2] == [{"type": "text", "text": "fine"},
                                      {"type": "text", "text": "done"}]
