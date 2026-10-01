"""One test per rule of docs/format.md §5."""

from tarjuman import Image, Message, Reasoning, Replay, Target, Text, ToolCall, ToolResult, Unknown, prepare
from tarjuman.transform import NO_RESULT, NO_VISION, REMINDER

CLAUDE = Target("anthropic", "anthropic-messages", "claude-x", id_pattern=r"[a-zA-Z0-9_-]{1,64}")
DEEPSEEK = Target("deepseek", "openai-chat", "deepseek-v4", vision=False)


def by(t: Target, *blocks, replay=None, stop="tool_use", **kw) -> Message:
    return Message("assistant", list(blocks), t.provider, t.model, None, stop, t.protocol,
                   replay=replay, **kw)


def test_same_model_keeps_reasoning_and_replay():
    m = by(CLAUDE, Reasoning("think"), Text("hi"), replay=Replay({"id": 1}, ["sig", None]), stop="end")
    [out] = prepare([m], CLAUDE)
    assert out.content == m.content and out.replay == m.replay


def test_other_model_gets_reasoning_as_text_and_no_replay():
    m = by(CLAUDE, Reasoning("think"), Reasoning("", redacted=True), Text("hi", citations=[1]),
           replay=Replay({"id": 1}, ["s1", "s2", None]), stop="end")
    [out] = prepare([m], DEEPSEEK)
    assert out.content == [Text("think"), Text("hi")]
    assert out.replay is None


def test_misaligned_replay_is_discarded_whole():
    m = by(CLAUDE, Reasoning("t"), Text("hi"), replay=Replay({"id": 1}, ["only one"]), stop="end")
    [out] = prepare([m], CLAUDE)
    assert out.replay is None


def test_dropped_blocks_drop_their_replay_entry():
    m = by(CLAUDE, Text(""), Reasoning("t"), Text("hi"), replay=Replay(None, ["a", "b", "c"]),
           stop="end")
    [out] = prepare([m], CLAUDE)
    assert out.content == [Reasoning("t"), Text("hi")] and out.replay.blocks == ["b", "c"]


def test_errored_turns_are_skipped_and_their_results_too():
    h = [Message.user("go"), by(DEEPSEEK, ToolCall("c1", "bash", "{}"), stop="error"),
         Message("tool", [ToolResult("c1", "out")]), Message.user("again")]
    assert [m.role for m in prepare(h, DEEPSEEK)] == ["user", "user"]


def test_interrupted_turn_sends_finished_blocks_only():
    # finished blocks in content, the cut-off one in partial: partial is never sent
    m = by(CLAUDE, Reasoning("done thinking"), Text("Let me"), replay=Replay(None, ["sig", None]),
           stop="interrupted", partial=[ToolCall("c9", "bash", '{"comm')])
    [out] = prepare([m], CLAUDE)
    assert out.content == [Reasoning("done thinking"), Text("Let me")]
    assert out.replay.blocks == ["sig", None] and out.partial is None


def test_tool_ids_rewritten_for_strict_targets_and_results_follow():
    h = [Message.user("go"), by(DEEPSEEK, ToolCall("fc|abc.123", "read", "{}")),
         Message("tool", [ToolResult("fc|abc.123", "text")])]
    out = prepare(h, CLAUDE)
    new = out[1].tool_calls[0].id
    assert new != "fc|abc.123" and new == prepare(h, CLAUDE)[1].tool_calls[0].id  # deterministic
    assert out[2].content[0].call_id == new
    assert prepare(h, DEEPSEEK)[1].tool_calls[0].id == "fc|abc.123"             # no rule: untouched


def test_tool_results_get_the_tool_name():
    h = [by(DEEPSEEK, ToolCall("c1", "read", "{}")), Message("tool", [ToolResult("c1", "x")])]
    assert prepare(h, DEEPSEEK)[1].content[0].name == "read"


def test_images_become_placeholders_without_vision():
    h = [Message("user", [Text("see"), Image("image/png", data="A"), Image("image/png", data="B")]),
         by(DEEPSEEK, ToolCall("c1", "shot", "{}")),
         Message("tool", [ToolResult("c1", [Image("image/png", data="C")])])]
    out = prepare(h, DEEPSEEK)
    assert out[0].content == [Text("see"), Text(NO_VISION)]
    assert out[2].content[0].content == [Text(NO_VISION)]
    assert prepare(h, CLAUDE)[0].content[1] == Image("image/png", data="A")


def test_orphaned_calls_get_a_synthetic_result():
    h = [by(DEEPSEEK, ToolCall("c1", "a", "{}"), ToolCall("c2", "b", "{}")),
         Message("tool", [ToolResult("c1", "ok")]), Message.user("next")]
    out = prepare(h, DEEPSEEK)
    assert [m.role for m in out] == ["assistant", "tool", "tool", "user"]
    assert out[2].content == [ToolResult("c2", NO_RESULT, True, "b")]


def test_leading_system_stays_later_ones_become_reminders():
    h = [Message.system("prompt"), Message.user("hi"), by(DEEPSEEK, Text("yo"), stop="end"),
         Message.system("80% of the budget used"), Message.user("go on")]
    out = prepare(h, DEEPSEEK)
    assert out[0] == Message.system("prompt")
    assert out[3] == Message.user(REMINDER.format("80% of the budget used"))
    assert out[4] == Message.user("go on")


def test_reminder_waits_until_open_calls_are_answered():
    h = [Message.system("p"), Message.user("hi"), by(DEEPSEEK, ToolCall("c1", "a", "{}")),
         Message.system("tools changed"), Message("tool", [ToolResult("c1", "ok")])]
    out = prepare(h, DEEPSEEK)
    assert [m.role for m in out] == ["system", "user", "assistant", "tool", "user"]
    assert "tools changed" in out[4].text


def test_prefix_is_unchanged_when_a_reminder_is_added():
    # the cache argument: appending a reminder never changes what came before it
    h = [Message.system("p"), Message.user("hi"), by(DEEPSEEK, Text("yo"), stop="end")]
    before = prepare(h, DEEPSEEK)
    after = prepare([*h, Message.system("note"), Message.user("next")], DEEPSEEK)
    assert after[:len(before)] == before


def test_empty_and_unknown_blocks_are_dropped():
    h = [Message("user", [Text(""), Unknown("document", {"name": "a.pdf"}), Text("hi")]),
         by(DEEPSEEK, Text(""), stop="end"), Message("user", [Text("")])]
    assert prepare(h, DEEPSEEK) == [Message.user("hi")]
