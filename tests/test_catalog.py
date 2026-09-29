import json

import httpx
import pytest

from tarjuman import Anthropic, Message, OpenAIChat, Reasoning, Request, TarjumanError, ToolCall, ToolResult, Usage
from tarjuman import catalog, providers

TABLE = {
    "anthropic": {
        "claude-new": {"vision": True, "temperature": False, "reasoning": True, "thinking": "effort",
                       "levels": ["low", "medium", "high", "xhigh", "max"], "max_output": 128000,
                       "price": {"input": 2, "output": 10, "cache_read": 0.2, "cache_write": 2.5}},
        "claude-old": {"vision": True, "reasoning": True, "thinking": "budget", "budget": [1024, None],
                       "max_output": 4000},
    },
    "deepseek": {
        "ds": {"vision": False, "reasoning": True, "thinking": "effort",
               "levels": ["off", "low", "high", "max"], "reasoning_field": "reasoning_content",
               "price": {"input": 0.4, "output": 0.8, "cache_read": 0.004, "cache_write": 0}},
    },
    "openai": {"gpt": {"temperature": False, "reasoning": True, "thinking": "effort",
                       "levels": ["off", "minimal", "low", "medium", "high"]}},
}


@pytest.fixture(autouse=True)
def fixed_table(monkeypatch):
    monkeypatch.setattr(catalog, "_table", lambda: TABLE)


def capture(cls, **kw):
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        if cls is Anthropic:
            ev = [{"type": "message_start", "message": {"id": "m", "model": "x", "usage": {
                "input_tokens": 1000, "cache_read_input_tokens": 2000,
                "cache_creation_input_tokens": 500, "cache_creation": {"ephemeral_1h_input_tokens": 100}}}},
                {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 300}},
                {"type": "message_stop"}]
            return httpx.Response(200, content="".join(f"data: {json.dumps(e)}\n\n" for e in ev).encode())
        chunk = {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 1000, "completion_tokens": 200,
                           "prompt_tokens_details": {"cached_tokens": 400}}}
        return httpx.Response(200, content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode())
    client = httpx.Client(transport=httpx.MockTransport(handler))
    p = cls("k", client=client, **kw) if cls is Anthropic else cls("https://x/v1", "k", client=client, **kw)
    return p, seen


def test_nearest_level():
    assert catalog.nearest("medium", ("off", "low", "high", "max")) == "low"   # tie: the weaker
    assert catalog.nearest("minimal", ("low", "high")) == "low"
    assert catalog.nearest("max", ("minimal", "low", "medium", "high")) == "high"
    assert catalog.nearest("high", None) == "high"                              # unknown: as asked


def test_cost_from_catalog_and_long_cache_writes():
    info = catalog.lookup("anthropic", "claude-new")
    u = Usage(input=1000, cache_read=2000, cache_write=500, output=300, cache_write_long=100)
    # 1000*2 + 300*10 + 2000*0.2 + 400*2.5 + 100*(2*2)  per million
    assert catalog.cost(u, info) == pytest.approx((2000 + 3000 + 400 + 1000 + 400) / 1e6)
    assert catalog.cost(u, None) is None
    assert catalog.cost(u, catalog.lookup("openai", "gpt")) is None              # no price: None


def test_anthropic_reads_effort_levels_temperature_and_cost():
    p, seen = capture(Anthropic, catalog="anthropic")
    msg = p.complete(Request("claude-new", [Message.user("hi")], reasoning="minimal", temperature=0.3))
    assert seen["body"]["output_config"] == {"effort": "low"}                   # nearest it accepts
    assert "temperature" not in seen["body"]
    assert msg.usage.cost == pytest.approx((2000 + 3000 + 400 + 1000 + 400) / 1e6)
    p.complete(Request("claude-new", [Message.user("hi")], temperature=0.3))
    assert "temperature" not in seen["body"]                                    # never, for this model


def test_anthropic_older_model_gets_a_budget_and_its_own_output_cap():
    p, seen = capture(Anthropic, catalog="anthropic")
    p.complete(Request("claude-old", [Message.user("hi")], reasoning="high", temperature=0.3))
    assert seen["body"]["thinking"]["type"] == "enabled"
    assert seen["body"]["max_tokens"] >= seen["body"]["thinking"]["budget_tokens"] + 1024
    p.complete(Request("claude-old", [Message.user("hi")]))
    assert seen["body"]["max_tokens"] == 4000                                   # min(8192, its limit)
    p.complete(Request("claude-unknown", [Message.user("hi")], reasoning="high"))
    assert seen["body"]["thinking"]["type"] == "adaptive"                       # unknown: the default


def test_deepseek_style_levels_reasoning_field_and_no_vision():
    p, seen = capture(OpenAIChat, provider="deepseek", catalog="deepseek", reasoning_style="deepseek")
    mine = Message("assistant", [Reasoning("plan"), ToolCall("c1", "a", "{}")], "deepseek", "ds",
                   None, "tool_use", "openai-chat")
    msg = p.complete(Request("ds", [Message.user("go"), mine, Message("tool", [ToolResult("c1", "ok")])],
                             reasoning="medium"))
    body = seen["body"]
    assert body["thinking"] == {"type": "enabled"} and body["reasoning_effort"] == "low"
    assert body["messages"][1]["reasoning_content"] == "plan"                   # from the catalog
    assert msg.usage.cost == pytest.approx((600 * 0.4 + 400 * 0.004 + 200 * 0.8) / 1e6)
    p.complete(Request("ds", [Message.user("go")], reasoning="off"))
    assert seen["body"]["thinking"] == {"type": "disabled"}
    assert p.target("ds").vision is False and p.target("unknown").vision is True


def test_openai_style_off_and_temperature():
    p, seen = capture(OpenAIChat, catalog="openai", reasoning_style="openai")
    p.complete(Request("gpt", [Message.user("hi")], reasoning="off", temperature=0.5))
    assert seen["body"]["reasoning_effort"] == "none" and "temperature" not in seen["body"]
    p.complete(Request("gpt", [Message.user("hi")], reasoning="max"))
    assert seen["body"]["reasoning_effort"] == "high"


def test_connect_reads_the_compat_table(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dk")
    ds = providers.connect("deepseek")
    assert (ds.provider, ds.base_url, ds.reasoning_style, ds.catalog) == (
        "deepseek", "https://api.deepseek.com", "deepseek", "deepseek")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(TarjumanError) as e:
        providers.connect("anthropic")
    assert "ANTHROPIC_API_KEY" in str(e.value)
    gw = providers.connect("anthropic", base_url="http://gw:8080")             # a gateway: key optional
    assert isinstance(gw, Anthropic) and gw.base_url == "http://gw:8080"
    with pytest.raises(TarjumanError):
        providers.connect("nope")
    assert set(providers.names()) >= {"openrouter", "openai", "anthropic", "deepseek", "local"}


def test_shipped_catalog_loads(monkeypatch):
    monkeypatch.undo()                                                         # the real data files
    catalog._table.cache_clear()
    for p in ("anthropic", "openai", "openrouter", "deepseek"):
        assert catalog._table().get(p), f"no {p} models in data/models.json"
