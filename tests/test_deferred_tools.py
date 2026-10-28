"""Deferred tools: sent from the start, shown to the model from a ToolsAdded on. On Anthropic
models that take tool changes, the tools sent never change (tool_addition); elsewhere the added
ones are sent from then on."""

from tarjuman import (Anthropic, Message, OpenAIChat, Request, Target, Text, Tool, ToolCall,
                      ToolResult, ToolsAdded, prepare, tokens, visible_tools)
from tarjuman.types import block_from_dict, block_to_dict

READ = Tool("read", "Read a file.", {"type": "object", "properties": {}})
ISSUES = Tool("issues", "List issues.", {"type": "object", "properties": {}}, deferred=True)
CLOSE = Tool("close", "Close an issue.", {"type": "object", "properties": {}}, deferred=True)
TOOLS = [READ, ISSUES, CLOSE]


def history(loaded=("issues",), after=()):
    """search -> (the load, logged while the search runs) -> its result -> more."""
    return [Message.system("prompt"), Message.user("list the issues"),
            Message("assistant", [ToolCall("c1", "tool_search", "{}")]),
            Message("system", [ToolsAdded(list(loaded))]),     # before the result, as logged
            Message("tool", [ToolResult("c1", "loaded")]), *after]


def native(model="claude-sonnet-5-5"):
    return Anthropic("k", catalog=None).body(Request(model, history(), TOOLS))


def test_the_block_round_trips():
    b = ToolsAdded(["a", "b"])
    assert block_from_dict(block_to_dict(b)) == b


def test_visible_tools_are_the_shown_ones_then_the_added_ones_in_order():
    msgs = history(("close", "issues", "nope"))
    assert [t.name for t in visible_tools(TOOLS, msgs)] == ["read", "close", "issues"]
    assert [t.name for t in visible_tools(TOOLS, history(()))] == ["read"]


def test_the_transform_puts_the_addition_after_the_users_side():
    t = Target("anthropic", "anthropic-messages", "m", tool_changes=True)
    out = prepare(history(after=[Message("system", [Text("a reminder")]),
                                 Message("assistant", [Text("ok")])]), t)
    assert [m.role for m in out] == ["system", "user", "assistant", "tool", "user", "system",
                                     "assistant"]
    assert out[5].content == [ToolsAdded(["issues"])]
    # last when nothing follows yet, and dropped for a target that can't take it
    assert prepare(history(), t)[-1].content == [ToolsAdded(["issues"])]
    plain = Target("x", "openai-chat", "m")
    assert not any(isinstance(b, ToolsAdded) for m in prepare(history(), plain) for b in m.content)


def test_anthropic_sends_every_tool_and_adds_in_place():
    body = native()
    assert [(t["name"], t.get("defer_loading", False)) for t in body["tools"]] == [
        ("read", False), ("issues", True), ("close", True)]
    assert body["messages"][-1] == {"role": "system", "content": [
        {"type": "tool_addition", "tool": {"type": "tool_reference", "name": "issues"}}]}
    assert "cache_control" in body["tools"][0]                    # the last shown tool
    assert "cache_control" in body["messages"][-2]["content"][-1]  # the result, not the addition


def test_the_tools_and_everything_before_stay_the_same_after_a_load():
    a = Anthropic("k", catalog=None)
    before = a.body(Request("claude-opus-5-5", history(()), TOOLS))
    after = a.body(Request("claude-opus-5-5", history(), TOOLS))
    strip = lambda x: json_of(x, drop_cache=True)   # noqa: E731
    assert strip(before["tools"]) == strip(after["tools"])
    assert strip(before["messages"]) == strip(after["messages"][:-1])


def json_of(x, drop_cache=False):
    import json
    s = json.dumps(x, sort_keys=True)
    return s.replace(', "cache_control": {"type": "ephemeral"}', "") if drop_cache else s


def test_the_beta_header_goes_only_with_deferred_tools():
    a = Anthropic("k", catalog=None, headers={"anthropic-beta": "other"})
    assert a.headers(native())["anthropic-beta"] == "other,mid-conversation-tool-changes-2026-07-01"
    plain = a.body(Request("claude-sonnet-5-5", history(), [READ]))
    assert a.headers(plain)["anthropic-beta"] == "other"


def test_unknown_and_unsent_names_are_never_referenced():
    body = Anthropic("k", catalog=None).body(
        Request("claude-sonnet-5-5", history(("issues", "gone")), [READ, ISSUES]))
    assert [b["tool"]["name"] for b in body["messages"][-1]["content"]] == ["issues"]


def test_other_anthropic_models_and_gateways_fall_back_to_sending_the_added_tools():
    for a, model in [(Anthropic("k", catalog=None), "claude-sonnet-5"),
                     (Anthropic("k", catalog=None), "claude-haiku-5-5"),
                     (Anthropic("k", catalog=None, base_url="https://gw.example"), "claude-sonnet-5-5")]:
        body = a.body(Request(model, history(), TOOLS))
        assert [t["name"] for t in body["tools"]] == ["read", "issues"]
        assert all(m["role"] != "system" for m in body["messages"])
        assert "anthropic-beta" not in a.headers(body)


def test_openai_chat_sends_the_added_tools():
    body = OpenAIChat("https://x.example/v1", "k").body(Request("m", history(), TOOLS))
    assert [t["function"]["name"] for t in body["tools"]] == ["read", "issues"]


def test_token_estimates_count_a_deferred_tool_once_added():
    assert tokens.chars(history(()), TOOLS) < tokens.chars(history(), TOOLS)
    assert tokens.chars(history(()), TOOLS) == tokens.chars(history(()), [READ])
