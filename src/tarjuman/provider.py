"""What every provider offers, and the HTTP plumbing the protocols share.

A protocol (OpenAIChat, Anthropic) subclasses HTTPProvider and supplies only what differs on
the wire: the request body, the endpoint path, how the stream is parsed and how errors read.
Connection, model lookups, cancelling and the non-streaming shortcut live here, once."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

import httpx

from . import catalog, errors, tokens
from .cancel import Cancel, cancellable
from .events import Event, Finish
from .transform import Target
from .types import Message, Request, Tool


class Provider(Protocol):
    """What a caller (an agent loop, a test fake) can rely on, whatever the protocol."""
    provider: str
    protocol: str

    def stream(self, request: Request | str, messages: list[Message] | None = None, *,
               tools: list[Tool] | None = None, max_tokens: int | None = None,
               cancel: Cancel | None = None, **extra: Any) -> Iterator[Event]: ...

    def complete(self, request: Request | str, messages: list[Message] | None = None,
                 **kw: Any) -> Message: ...

    def info(self, model: str) -> catalog.ModelInfo | None: ...

    def context_window(self, model: str) -> int | None: ...

    def target(self, model: str) -> Target: ...


@dataclass(frozen=True)
class App:
    """The application making the calls. Some providers show it on their dashboards and
    rankings (OpenRouter); the compat table's `app_headers` says which header carries what."""
    name: str
    url: str | None = None


class HTTPProvider:
    protocol: ClassVar[str]
    path: ClassVar[str]                  # the streaming endpoint, after base_url
    models_path: ClassVar[str] = ""      # where GET .../models lives, after base_url
    id_pattern: ClassVar[str | None] = None

    def __init__(self, base_url: str, provider: str, headers: dict[str, str], *, vision: bool,
                 load_image: Callable[[str], str] | None, catalog: str | None, timeout: float,
                 client: httpx.Client | None, stall: float | None = None):
        self.base_url = base_url.rstrip("/")
        self.provider = provider
        self.vision = vision
        self.load_image = load_image
        self.catalog = catalog
        self._headers = {"Content-Type": "application/json", **headers}
        # `stall`: the longest wait for the next bytes of a stream, for providers known to send
        # something every few seconds (tokens, pings, keep-alive comments: the compat table
        # says which). A longer silence is a stalled connection: it fails as NETWORK (retried)
        # instead of hanging until `timeout`. None: only `timeout` applies
        read = min(stall, timeout) if stall else timeout
        self._client = client or httpx.Client(timeout=httpx.Timeout(timeout, connect=30, read=read))
        self._windows: dict[str, int] | None = None  # from the server's /models, when asked

    # --- what the protocol supplies ----------------------------------------------------------

    def body(self, req: Request) -> dict[str, Any]:
        raise NotImplementedError

    def _parse(self, model: str, lines: Iterator[str]) -> Iterator[Event]:
        raise NotImplementedError

    def _error(self, status: int, body: str, retry_after: str | None) -> errors.TarjumanError:
        return errors.from_http(status, body, retry_after)

    # --- models ------------------------------------------------------------------------------

    def info(self, model: str) -> catalog.ModelInfo | None:
        return catalog.lookup(self.catalog, model) if self.catalog else None

    def context_window(self, model: str) -> int | None:
        """The model's context window in tokens: the catalog's, else what the server's /models
        listing says (asked once), else None."""
        info = self.info(model)
        if info and info.context:
            return info.context
        if self._windows is None:
            self._windows = tokens.served_windows(self._client, self.base_url + self.models_path,
                                                  self._headers)
        return self._windows.get(model)

    def target(self, model: str) -> Target:
        info = self.info(model)
        vision = info.vision if info and info.vision is not None else self.vision
        return Target(self.provider, self.protocol, model, vision=vision,
                      id_pattern=self.id_pattern)

    # --- calls -------------------------------------------------------------------------------

    def stream(self, request: Request | str, messages: list[Message] | None = None, *,
               tools: list[Tool] | None = None, max_tokens: int | None = None,
               cancel: Cancel | None = None, **extra: Any) -> Iterator[Event]:
        """`stream(Request(...))`, or the shortcut `stream(model, messages, tools=...)`.
        With `cancel`, firing it stops the stream at once (see cancel.py)."""
        req = request if isinstance(request, Request) else Request(
            request, messages or [], tools, max_tokens=max_tokens, extra=extra or None)
        body = self.body(req)
        if cancel is None:
            yield from self._events(req, body, None)
        else:
            yield from cancellable(lambda token: self._events(req, body, token), cancel)

    def _events(self, req: Request, body: dict[str, Any], token: Cancel | None) -> Iterator[Event]:
        if token and token.cancelled:
            return
        try:
            with self._client.stream("POST", self.base_url + self.path,
                                     headers=self._headers, json=body) as r:
                if token:  # closing the response is what stops the provider generating
                    token.on_cancel(r.close)
                if r.status_code >= 400:
                    r.read()
                    raise self._error(r.status_code, r.text, r.headers.get("retry-after"))
                for ev in self._parse(req.model, r.iter_lines()):
                    if isinstance(ev, Finish) and self.catalog:
                        catalog.fill_cost(ev.message, self.catalog)
                    yield ev
        except httpx.TransportError as e:
            raise errors.TarjumanError(errors.NETWORK, str(e) or type(e).__name__) from e

    def complete(self, request: Request | str, messages: list[Message] | None = None,
                 **kw: Any) -> Message:
        """Non-streaming convenience: consume the stream, return the final message."""
        for ev in self.stream(request, messages, **kw):
            if isinstance(ev, Finish):
                return ev.message
        raise errors.TarjumanError(errors.SERVER_ERROR, "stream ended without a finish")
