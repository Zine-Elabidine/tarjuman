import threading
import time

import httpx
import pytest

from tarjuman import Cancel, Message, OpenAIChat, TarjumanError, TextDelta
from tarjuman import errors
from tarjuman.cancel import cancellable


def provider(handler):
    return OpenAIChat("https://x.test/v1", "k", provider="test",
                      client=httpx.Client(transport=httpx.MockTransport(handler)))


def cancel_after(c: Cancel, seconds: float) -> None:
    threading.Timer(seconds, c.cancel).start()


def test_cancel_while_waiting_for_the_first_byte_returns_at_once():
    release = threading.Event()

    def handler(req):
        release.wait(5)  # a slow model: no headers yet
        return httpx.Response(200, content=b"data: [DONE]\n\n")

    c = Cancel()
    cancel_after(c, 0.1)
    start = time.monotonic()
    with pytest.raises(TarjumanError) as e:
        list(provider(handler).stream("m", [Message.user("hi")], cancel=c))
    release.set()
    assert e.value.code == errors.CANCELLED and not e.value.retryable
    assert time.monotonic() - start < 1


def test_cancel_mid_stream_stops_between_chunks_and_closes_the_response():
    release, closed = threading.Event(), threading.Event()

    class Body(httpx.SyncByteStream):
        def __iter__(self):
            yield b'data: {"choices": [{"delta": {"content": "par"}}]}\n\n'
            release.wait(5)  # the model is thinking
            yield b"data: [DONE]\n\n"

        def close(self):
            closed.set()

    c = Cancel()
    seen = []
    with pytest.raises(TarjumanError) as e:
        for ev in provider(lambda req: httpx.Response(200, stream=Body())).stream(
                "m", [Message.user("hi")], cancel=c):
            seen.append(ev)
            if isinstance(ev, TextDelta):
                cancel_after(c, 0.05)
    release.set()
    assert e.value.code == errors.CANCELLED
    assert [ev.text for ev in seen if isinstance(ev, TextDelta)] == ["par"]
    assert closed.wait(1)


def test_a_signal_already_fired_sends_nothing():
    sent = []
    c = Cancel()
    c.cancel()
    with pytest.raises(TarjumanError):
        list(provider(lambda req: sent.append(1) or httpx.Response(200)).stream(
            "m", [Message.user("hi")], cancel=c))
    time.sleep(0.05)
    assert sent == []


def test_stopping_iteration_early_cancels_the_producer():
    stopped = threading.Event()

    def events(token):
        token.on_cancel(stopped.set)
        yield 1
        yield 2

    it = cancellable(events, Cancel())
    assert next(it) == 1
    it.close()
    assert stopped.is_set()


def test_cancel_callbacks_and_wait():
    c, calls = Cancel(), []
    remove = c.on_cancel(lambda: calls.append("a"))
    c.on_cancel(lambda: calls.append("b"))
    remove()
    assert c.wait(0.01) is False
    c.cancel()
    c.cancel()
    assert calls == ["b"] and c.cancelled and c.wait(0)
    c.on_cancel(lambda: calls.append("late"))   # already fired: runs now
    assert calls == ["b", "late"]
