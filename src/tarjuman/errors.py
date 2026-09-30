"""Stable error codes, whatever the provider said."""

from __future__ import annotations

RATE_LIMIT = "RATE_LIMIT"
OVERLOADED = "OVERLOADED"
SERVER_ERROR = "SERVER_ERROR"
NETWORK = "NETWORK"
CONTEXT_WINDOW_EXCEEDED = "CONTEXT_WINDOW_EXCEEDED"
INVALID_CREDENTIAL = "INVALID_CREDENTIAL"
INSUFFICIENT_CREDITS = "INSUFFICIENT_CREDITS"
INVALID_REQUEST = "INVALID_REQUEST"
CANCELLED = "CANCELLED"  # the caller stopped the request (tarjuman.Cancel); never retried

RETRYABLE = {RATE_LIMIT, OVERLOADED, SERVER_ERROR, NETWORK}


class TarjumanError(Exception):
    def __init__(self, code: str, message: str, *, status: int | None = None,
                 retry_after: float | None = None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.status = status
        self.retry_after = retry_after

    @property
    def retryable(self) -> bool:
        return self.code in RETRYABLE


def from_http(status: int, body: str, retry_after: str | None = None) -> TarjumanError:
    low = body.lower()
    wait = None
    if retry_after:
        try:
            wait = float(retry_after)
        except ValueError:
            pass
    if status in (401, 403):
        code = INVALID_CREDENTIAL
    elif status == 402:
        code = INSUFFICIENT_CREDITS
    elif status == 429:
        code = RATE_LIMIT
    elif status in (503, 529):
        code = OVERLOADED
    elif status >= 500 or status == 408:
        code = SERVER_ERROR
    elif "context" in low and ("length" in low or "window" in low or "too long" in low or "maximum" in low):
        code = CONTEXT_WINDOW_EXCEEDED
    else:
        code = INVALID_REQUEST
    return TarjumanError(code, body[:500] or f"HTTP {status}", status=status, retry_after=wait)
