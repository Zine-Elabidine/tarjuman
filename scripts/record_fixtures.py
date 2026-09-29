"""Record the golden files in tests/fixtures/ from the current code.

    uv run python scripts/record_fixtures.py

Each file is language-neutral: an input (a Request, or a raw SSE stream) and the exact
output (the wire body, or the assembled message). tests/test_fixtures.py replays them, and a
port to another language passes the same files. Re-record only on purpose, and read the diff:
a changed expected output is a changed wire format."""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from tarjuman import Message, Reasoning, Replay, Request, Text, Tool, ToolCall, ToolResult
from tarjuman.providers import PROTOCOLS

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
READ = Tool("read", "Read a file", {"type": "object", "properties": {"path": {"type": "string"}},
                                    "required": ["path"]})


def sse_anthropic(*events: dict) -> str:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


def sse_openai(*chunks: dict) -> str:
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


def claude_turn() -> Message:
    return Message("assistant", [Reasoning("I should read it."), Text("Reading."),
                                 ToolCall("toolu_01", "read", '{"path": "a.py"}')],
                   "anthropic", "claude-x", None, "tool_use", "anthropic-messages",
                   replay=Replay({"id": "msg_01"}, [{"signature": "EqQBsig"}, None, None]))


WIRE = {
    "anthropic_history_same_model": ("anthropic-messages", {"provider": "anthropic"}, Request(
        "claude-x", [Message.system("You are Diwan."), Message.user("hi"), claude_turn(),
                     Message("tool", [ToolResult("toolu_01", "print(1)")]),
                     Message.system("80% of the budget used"), Message.user("keep going")],
        [READ], reasoning="high")),
    "anthropic_interrupted_turn": ("anthropic-messages", {"provider": "anthropic"}, Request(
        "claude-x", [Message.user("fix it"),
                     Message("assistant", [Reasoning("plan")], "anthropic", "claude-x", None,
                             "interrupted", "anthropic-messages",
                             replay=Replay(None, [{"signature": "sig-1"}]),
                             partial=[Text("Let me look at")]),
                     Message.system("The user interrupted the previous turn on purpose."),
                     Message.user("no, the README")])),
    "openai_chat_from_claude_history": ("openai-chat", {"provider": "deepseek",
                                                         "reasoning_field": "reasoning_content"},
        Request("deepseek-v4", [Message.user("hi"), claude_turn(),
                                Message("tool", [ToolResult("toolu_01", "print(1)")])],
                [READ], tool_choice="required", max_tokens=1000)),
}

STREAM = {
    "anthropic_thinking_then_tool": ("anthropic-messages", {"provider": "anthropic"}, "claude-x",
        sse_anthropic(
            {"type": "message_start", "message": {"id": "msg_01", "model": "claude-x", "usage": {
                "input_tokens": 50, "cache_read_input_tokens": 400, "cache_creation_input_tokens": 100}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "I should read it."}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "EqQBsig"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "ping"},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Reading."}},
            {"type": "content_block_stop", "index": 1},
            {"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "toolu_01", "name": "read", "input": {}}},
            {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": "{\"path\": \"a.py\"}"}},
            {"type": "content_block_stop", "index": 2},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 30}},
            {"type": "message_stop"})),
    "openai_chat_reasoning_details": ("openai-chat", {"provider": "openrouter"}, "anthropic/claude-x",
        sse_openai(
            {"model": "anthropic/claude-x-2026", "choices": [{"index": 0, "delta": {"reasoning": "I will ", "reasoning_details": [
                {"type": "reasoning.text", "text": "I will ", "format": "anthropic-claude-v1", "index": 0}]}}]},
            {"choices": [{"index": 0, "delta": {"reasoning": "look.", "reasoning_details": [
                {"type": "reasoning.text", "text": "look.", "index": 0}]}}]},
            {"choices": [{"index": 0, "delta": {"reasoning_details": [
                {"type": "reasoning.text", "text": "", "signature": "SIG123", "index": 0}]}}]},
            {"choices": [{"index": 0, "delta": {"content": "Done."}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                                      "prompt_tokens_details": {"cached_tokens": 60}, "cost": 0.001}})),
}


def client_for(protocol: str, options: dict, sse: str = ""):
    http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=sse.encode())))
    cls = PROTOCOLS[protocol]
    if protocol == "anthropic-messages":
        return cls("k", client=http, **options)
    return cls("https://x.test/v1", "k", client=http, **options)


def write(name: str, data: dict) -> None:
    (OUT / f"{name}.json").write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n",
                                      encoding="utf-8", newline="\n")
    print("recorded", name)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (protocol, options, req) in WIRE.items():
        body = client_for(protocol, options).body(req)
        write(name, {"kind": "wire", "protocol": protocol, "options": options,
                     "request": req.to_dict(), "expected_body": body})
    for name, (protocol, options, model, sse) in STREAM.items():
        msg = client_for(protocol, options, sse).complete(model, [Message.user("hi")])
        write(name, {"kind": "stream", "protocol": protocol, "options": options, "model": model,
                     "sse": sse, "expected_message": msg.to_dict()})


if __name__ == "__main__":
    main()
