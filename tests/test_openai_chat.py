import json

import httpx
import pytest

from tarjuman import Finish, Message, OpenAIChat, TarjumanError, Text, TextDelta, Tool, ToolCall, ToolResult
from tarjuman import errors
from tarjuman.transform import close_tool_calls


def sse(*chunks):
    lines = [": OPENROUTER PROCESSING", ""]
    for c in chunks:
        lines += [f"data: {json.dumps(c)}", ""]
    lines += ["data: [DONE]", ""]
    return "\n".join(lines).encode()


def provider(handler):
    return OpenAIChat("https://x.test/v1", "k", provider="test",
                      client=httpx.Client(transport=httpx.MockTransport(handler)))


def delta(**d):
    return {"choices": [{"index": 0, "delta": d, "finish_reason": None}]}


def test_text_reasoning_and_tool_call_stream():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        seen["auth"] = req.headers["authorization"]
        return httpx.Response(200, content=sse(
            delta(reasoning="Let me "), delta(reasoning="look."),
            delta(content="Reading "), delta(content="it."),
            delta(tool_calls=[{"index": 0, "id": "c1", "type": "function",
                               "function": {"name": "read", "arguments": ""}}]),
            delta(tool_calls=[{"index": 0, "function": {"arguments": '{"path":'}}]),
            delta(tool_calls=[{"index": 0, "function": {"arguments": ' "a.py"}'}}]),
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            {"choices": [], "model": "vendor/m-1",
             "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                       "prompt_tokens_details": {"cached_tokens": 60},
                       "completion_tokens_details": {"reasoning_tokens": 5}, "cost": 0.0012}},
        ))

    p = provider(handler)
    tool = Tool("read", "Read a file", {"type": "object", "properties": {"path": {"type": "string"}}})
    events = list(p.stream("vendor/m", [Message.system("sys"), Message.user("hi")],
                           tools=[tool], max_tokens=100))

    assert seen["auth"] == "Bearer k"
    assert seen["body"]["messages"] == [{"role": "system", "content": "sys"},
                                        {"role": "user", "content": "hi"}]
    assert seen["body"]["tools"][0]["function"]["name"] == "read"
    assert seen["body"]["max_tokens"] == 100
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "Reading it."

    msg = events[-1].message
    assert isinstance(events[-1], Finish)
    assert [b.type for b in msg.content] == ["reasoning", "text", "tool_call"]
    assert msg.content[0].text == "Let me look."
    assert msg.tool_calls[0].args() == {"path": "a.py"}
    assert msg.stop == "tool_use"
    assert msg.model == "vendor/m-1" and msg.provider == "test"
    u = msg.usage
    assert (u.input, u.cache_read, u.output, u.reasoning, u.cost) == (40, 60, 20, 5, 0.0012)


def test_history_is_converted_and_open_calls_are_closed():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, content=sse(delta(content="ok"),
                                               {"choices": [{"delta": {}, "finish_reason": "stop"}]}))

    history = [
        Message.user("go"),
        Message("assistant", [Text("a"), ToolCall("c1", "bash", '{"command":"ls"}'),
                              ToolCall("c2", "bash", '{"command":"pwd"}')]),
        Message("tool", [ToolResult("c1", "file.txt")]),   # c2 never answered (interrupt)
        Message.user("next"),
    ]
    msg = provider(handler).complete("m", history)
    wire = seen["body"]["messages"]
    assert wire[1]["tool_calls"][1]["id"] == "c2"
    assert wire[2] == {"role": "tool", "tool_call_id": "c1", "content": "file.txt"}
    assert wire[3]["role"] == "tool" and wire[3]["tool_call_id"] == "c2"
    assert wire[4] == {"role": "user", "content": "next"}
    assert msg.text == "ok" and msg.stop == "end"


@pytest.mark.parametrize("status,body,code", [
    (401, "bad key", errors.INVALID_CREDENTIAL),
    (402, "no credits", errors.INSUFFICIENT_CREDITS),
    (429, "slow down", errors.RATE_LIMIT),
    (503, "busy", errors.OVERLOADED),
    (400, "This model's maximum context length is 8192 tokens", errors.CONTEXT_WINDOW_EXCEEDED),
    (400, "bad field", errors.INVALID_REQUEST),
])
def test_http_errors_map_to_stable_codes(status, body, code):
    p = provider(lambda req: httpx.Response(status, text=body, headers={"retry-after": "3"}))
    with pytest.raises(TarjumanError) as e:
        list(p.stream("m", [Message.user("hi")]))
    assert e.value.code == code
    assert e.value.retryable == (code in errors.RETRYABLE)


def test_error_inside_stream():
    p = provider(lambda req: httpx.Response(200, content=sse(
        delta(content="par"), {"error": {"code": 502, "message": "upstream died"}})))
    with pytest.raises(TarjumanError) as e:
        list(p.stream("m", [Message.user("hi")]))
    assert e.value.code == errors.SERVER_ERROR


def test_network_error():
    def handler(req):
        raise httpx.ConnectError("refused")
    with pytest.raises(TarjumanError) as e:
        list(provider(handler).stream("m", [Message.user("hi")]))
    assert e.value.code == errors.NETWORK and e.value.retryable


def test_message_roundtrip():
    m = Message("assistant", [Text("t"), ToolCall("c", "n", "{}")], "p", "m", None, "tool_use")
    assert Message.from_dict(json.loads(json.dumps(m.to_dict()))) == m


def test_close_tool_calls_leaves_answered_history_alone():
    h = [Message("assistant", [ToolCall("c1", "x", "{}")]), Message("tool", [ToolResult("c1", "r")])]
    assert close_tool_calls(h) == h
