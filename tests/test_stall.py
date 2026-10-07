"""A stream that goes silent fails (NETWORK, retried by the caller) instead of hanging, for
providers known to keep a stream alive; the others wait the whole timeout."""

from tarjuman import providers


def test_only_providers_that_send_keep_alives_give_up_on_silence():
    for name in ("openrouter", "anthropic", "deepseek"):
        assert providers.connect(name, api_key="k")._client.timeout.read == 120
    for name in ("openai", "local"):
        assert providers.connect(name, api_key="k")._client.timeout.read == 600
