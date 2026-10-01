import httpx

from tarjuman import Anthropic, App, Message, OpenAIChat, Provider, Text, providers
from tarjuman.fake import Fake


def test_every_provider_offers_the_same_interface():
    # checked by pyright too: each assignment must satisfy the Provider protocol
    impls: list[Provider] = [OpenAIChat("https://x.test/v1", "k"), Anthropic("k"),
                             Fake([Message("assistant", [Text("hi")])], window=1000)]
    for p in impls:
        assert p.target("m").protocol == p.protocol
    fake = impls[2]
    assert fake.complete("m", [Message("user", [Text("hello")])]).text == "hi"
    assert fake.context_window("m") == 1000


def test_app_headers_come_from_the_caller():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(req.headers)
        return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"ok"},'
                                        '"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')

    client = httpx.Client(transport=httpx.MockTransport(handler))
    p = providers.connect("openrouter", api_key="k", client=client,
                          app=App("Atelier", "https://example.test/atelier"))
    p.complete("m", [Message("user", [Text("hi")])])
    assert seen["x-title"] == "Atelier"
    assert seen["http-referer"] == "https://example.test/atelier"

    seen.clear()
    providers.connect("openrouter", api_key="k", client=client).complete(
        "m", [Message("user", [Text("hi")])])
    assert "x-title" not in seen and "http-referer" not in seen   # no app, no attribution

    seen.clear()  # a provider without app headers ignores the app
    providers.connect("openai", api_key="k", client=client, app=App("Atelier")).complete(
        "m", [Message("user", [Text("hi")])])
    assert "x-title" not in seen
