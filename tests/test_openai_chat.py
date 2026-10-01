import json

import httpx
import pytest

from tarjuman import (BlockEnd, Finish, Image, Message, OpenAIChat, Reasoning, Request,
                      TarjumanError, Text, TextDelta, Tool, ToolCall, ToolResult, errors)


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
    assert msg.model == "vendor/m" and msg.response_model == "vendor/m-1"   # asked vs served
    assert msg.provider == "test" and msg.protocol == "openai-chat"
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


def wire_of(history, request=None, **quirks):
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, content=sse(delta(content="ok")))

    p = OpenAIChat("https://x.test/v1", "k", provider="test", **quirks,
                   client=httpx.Client(transport=httpx.MockTransport(handler)))
    p.complete(request or Request("m", history))
    return seen["body"]


def test_images_go_as_parts_and_tool_images_follow_as_user():
    h = [Message("user", [Text("see"), Image("image/png", data="AAA")]),
         Message("assistant", [ToolCall("c1", "shot", "{}")]),
         Message("tool", [ToolResult("c1", [Text("taken"), Image("image/png", ref="r1")])])]
    body = wire_of(h, load_image=lambda ref: "BBB")
    assert body["messages"][0]["content"][1] == {
        "type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}
    assert body["messages"][2] == {"role": "tool", "tool_call_id": "c1", "content": "taken"}
    assert body["messages"][3]["role"] == "user"
    assert body["messages"][3]["content"][1]["image_url"]["url"] == "data:image/png;base64,BBB"


def test_own_reasoning_is_sent_back_only_with_the_quirk():
    mine = Message("assistant", [Reasoning("plan"), ToolCall("c1", "a", "{}")], "test", "m",
                   None, "tool_use", "openai-chat")
    h = [Message.user("go"), mine, Message("tool", [ToolResult("c1", "ok")])]
    assert "reasoning_content" not in wire_of(h)["messages"][1]
    assert wire_of(h, reasoning_field="reasoning_content")["messages"][1]["reasoning_content"] == "plan"
    other = Message("assistant", [Reasoning("plan"), ToolCall("c1", "a", "{}")], "x", "y",
                    None, "tool_use", "openai-chat")
    body = wire_of([h[0], other, h[2]], reasoning_field="reasoning_content")
    assert "reasoning_content" not in body["messages"][1]       # another model's: sent as text
    assert body["messages"][1]["content"] == "plan"


def test_request_options_map_to_the_wire():
    r = Request("m", [Message.user("hi")], [Tool("read", "R", {"type": "object"}, strict=True)],
                tool_choice={"name": "read"}, max_tokens=50, reasoning="max", temperature=0.2,
                stop=["END"], response_format={"type": "object"}, extra={"top_k": 3})
    body = wire_of(None, r, reasoning_style="openrouter")
    assert body["tool_choice"] == {"type": "function", "function": {"name": "read"}}
    assert body["tools"][0]["function"]["strict"] is True
    assert body["reasoning"] == {"effort": "high"}                # max -> nearest the dialect has
    assert (body["max_tokens"], body["temperature"], body["stop"], body["top_k"]) == (50, 0.2, ["END"], 3)
    assert body["response_format"]["json_schema"]["schema"] == {"type": "object"}
    off = wire_of(None, Request("m", [Message.user("hi")], reasoning="off"), reasoning_style="openrouter")
    assert off["reasoning"] == {"enabled": False}


def test_refusal_and_block_end_carries_the_block():
    p = provider(lambda req: httpx.Response(200, content=sse(
        delta(refusal="I can't help with that."),
        {"choices": [{"delta": {}, "finish_reason": "stop"}]})))
    events = list(p.stream("m", [Message.user("bad")]))
    ends = [e for e in events if isinstance(e, BlockEnd)]
    assert ends[0].block == Text("I can't help with that.")
    assert events[-1].message.stop == "refusal"


def test_reasoning_details_are_merged_kept_and_sent_back_to_the_same_model():
    # OpenRouter / Bifrost stream Claude's thinking with its signature in reasoning_details
    def rd(**d):
        return delta(reasoning_details=[d])
    p = provider(lambda req: httpx.Response(200, content=sse(
        {**delta(reasoning="I will "), "choices": [{"index": 0, "delta": {
            "reasoning": "I will ", "reasoning_details": [
                {"type": "reasoning.text", "text": "I will ", "format": "anthropic-claude-v1", "index": 0}]},
            "finish_reason": None}]},
        rd(type="reasoning.text", text="look.", index=0),
        rd(type="reasoning.text", text="", signature="SIG123", index=0),
        rd(type="reasoning.encrypted", data="ENC==", index=1),
        delta(content="Done."),
        {"choices": [{"delta": {}, "finish_reason": "stop"}]})))
    msg = p.complete("anthropic/claude-x", [Message.user("hi")])
    assert msg.content[0] == Reasoning("I will look.")
    assert msg.replay.response["reasoning_details"] == [
        {"type": "reasoning.text", "text": "I will look.", "format": "anthropic-claude-v1",
         "index": 0, "signature": "SIG123"},
        {"type": "reasoning.encrypted", "data": "ENC==", "index": 1}]

    same = wire_of(None, Request("anthropic/claude-x", [Message.user("hi"), msg, Message.user("more")]))
    assert same["messages"][1]["reasoning_details"] == msg.replay.response["reasoning_details"]

    other = wire_of([Message.user("hi"), msg, Message.user("more")])  # model "m": another model
    assert "reasoning_details" not in other["messages"][1]
    assert other["messages"][1]["content"] == "I will look.\n\nDone."


def test_reasoning_details_alone_still_give_reasoning_text():
    p = provider(lambda req: httpx.Response(200, content=sse(
        delta(reasoning_details=[{"type": "reasoning.summary", "summary": "Plan: "}]),
        delta(reasoning_details=[{"type": "reasoning.summary", "summary": "read."}]),
        delta(content="ok"), {"choices": [{"delta": {}, "finish_reason": "stop"}]})))
    msg = p.complete("m", [Message.user("hi")])
    assert msg.content[0] == Reasoning("Plan: read.")
    assert msg.replay.response["reasoning_details"] == [{"type": "reasoning.summary",
                                                         "summary": "Plan: read."}]


def test_cache_breakpoints_on_system_and_newest_message_for_listed_models():
    h = [Message.system("sys"), Message.user("go"),
         Message("assistant", [ToolCall("c1", "read", "{}")]),
         Message("tool", [ToolResult("c1", "file body")])]
    body = wire_of(h, Request("anthropic/claude-x", h), cache_control=["anthropic/"])
    msgs = body["messages"]
    assert msgs[0]["content"] == [{"type": "text", "text": "sys",
                                   "cache_control": {"type": "ephemeral"}}]
    assert msgs[-1]["role"] == "tool"
    assert msgs[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert msgs[1]["content"] == "go"                               # only two breakpoints

    long = wire_of(h, Request("anthropic/claude-x", h, cache="long"), cache_control=True)
    assert long["messages"][0]["content"][0]["cache_control"]["ttl"] == "1h"
    for req in (Request("deepseek/v4", h), Request("anthropic/claude-x", h, cache="none")):
        assert wire_of(h, req, cache_control=["anthropic/"])["messages"][0]["content"] == "sys"


def test_openrouter_marks_claude_but_not_deepseek():
    from tarjuman import providers
    p = providers.openrouter(api_key="k")
    h = [Message.system("s"), Message.user("u")]
    assert "cache_control" in json.dumps(p.body(Request("anthropic/claude-sonnet-4.5", h)))
    assert "cache_control" not in json.dumps(p.body(Request("deepseek/deepseek-v4-flash", h)))
