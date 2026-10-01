import httpx

from tarjuman import Image, Message, OpenAIChat, Tool, ToolCall, ToolResult, tokens


def served(rows, status=200):
    calls = []

    def handler(req):
        calls.append(str(req.url))
        return httpx.Response(status, json={"data": rows})
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return client, calls


def test_window_from_the_catalog_without_asking_the_server():
    client, calls = served([])
    p = OpenAIChat("https://x.test/v1", "k", catalog="openrouter", client=client)
    assert p.context_window("anthropic/claude-haiku-4.5") == 200_000
    assert calls == []


def test_unknown_models_ask_the_server_once():
    client, calls = served([{"id": "qwen-local", "max_model_len": 32768},          # vLLM
                            {"id": "gw/model", "context_length": 128000},         # OpenRouter-style
                            {"id": "groq-x", "context_window": 131072},           # Groq
                            {"id": "nested", "top_provider": {"context_length": 64000}},
                            {"id": "silent"}])
    p = OpenAIChat("http://localhost:8000/v1", None, client=client)
    assert p.context_window("qwen-local") == 32768
    assert p.context_window("gw/model") == 128000
    assert p.context_window("groq-x") == 131072
    assert p.context_window("nested") == 64000
    assert p.context_window("silent") is None and p.context_window("nope") is None
    assert calls == ["http://localhost:8000/v1/models"]


def test_a_failing_models_endpoint_means_unknown_not_an_error():
    client, _ = served([], status=404)
    assert OpenAIChat("http://x.test/v1", None, client=client).context_window("m") is None


def test_estimate_counts_text_calls_results_tools_and_images():
    msgs = [Message.user("a" * 400),
            Message("assistant", [ToolCall("c1", "read", '{"path": "' + "b" * 90 + '"}')]),
            Message("tool", [ToolResult("c1", [Image("image/png", data="x")])])]
    tool = Tool("read", "Read", {"type": "object"})
    n_chars = tokens.chars(msgs, [tool])
    call = len("read") + len('{"path": "') + 90 + len('"}')
    tool_def = len("read") + len("Read") + len('{"type": "object"}')
    assert n_chars == 400 + call + tool_def                    # the image is counted apart
    est = tokens.estimate(msgs, [tool], chars_per_token=4)
    assert est == round(n_chars / 4) + tokens.IMAGE_TOKENS + 3 * tokens.MESSAGE_OVERHEAD


def test_ratio_is_calibrated_from_a_real_report():
    msgs = [Message.user("word " * 2000)]                     # 10,000 characters
    assert tokens.ratio(msgs, None, 2504) == 10_000 / 2500     # 4 overhead tokens removed
    assert tokens.ratio(msgs, None, 100) is None               # too small to trust
