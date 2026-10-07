"""A stream that goes silent fails (NETWORK, retried by the caller) instead of hanging."""

from tarjuman import providers
from tarjuman.provider import STALL


def test_hosted_providers_give_up_on_a_silent_stream_and_local_ones_wait():
    hosted = providers.connect("openrouter", api_key="k")
    assert hosted._client.timeout.read == STALL < hosted._client.timeout.pool
    local = providers.connect("local")
    assert local._client.timeout.read == 600
