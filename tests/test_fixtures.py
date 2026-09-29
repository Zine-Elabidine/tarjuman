"""Replays the language-neutral golden files in tests/fixtures/ (see scripts/record_fixtures.py).
Any implementation of Tarjuman, in any language, must produce exactly these outputs."""

import json
from pathlib import Path

import httpx
import pytest

from tarjuman import Message, Request
from tarjuman.providers import PROTOCOLS

FIXTURES = sorted((Path(__file__).parent / "fixtures").glob("*.json"))


def client_for(protocol, options, sse=""):
    http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=sse.encode())))
    if protocol == "anthropic-messages":
        return PROTOCOLS[protocol]("k", client=http, **options)
    return PROTOCOLS[protocol]("https://x.test/v1", "k", client=http, **options)


def test_there_are_fixtures():
    assert len(FIXTURES) >= 5


@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_fixture(path):
    f = json.loads(path.read_text(encoding="utf-8"))
    if f["kind"] == "wire":
        body = client_for(f["protocol"], f["options"]).body(Request.from_dict(f["request"]))
        assert body == f["expected_body"]
    else:
        msg = client_for(f["protocol"], f["options"], f["sse"]).complete(f["model"], [Message.user("hi")])
        assert msg.to_dict() == f["expected_message"]
