"""Stopping a request from another thread, at once.

A `Cancel` is one stop signal the caller passes to everything a turn does (the stream, its
tools, retry waits), like an AbortSignal. A stream given one runs its HTTP read in a worker
thread: when the signal fires, the stream raises CANCELLED immediately, even while waiting
for the first byte, and the connection is closed so the provider stops generating."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterator
from typing import Any

from . import errors


class Cancel:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._callbacks: list[Callable[[], None]] = []

    def cancel(self) -> None:
        """Fire the signal (safe from any thread; only the first call does anything)."""
        with self._lock:
            if self._event.is_set():
                return
            self._event.set()
            callbacks, self._callbacks = self._callbacks, []
        for cb in callbacks:
            try:
                cb()
            except Exception:
                pass  # a failing cleanup must not stop the others

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Sleep up to `timeout`; True as soon as the signal fires."""
        return self._event.wait(timeout)

    def on_cancel(self, cb: Callable[[], None]) -> Callable[[], None]:
        """Call `cb` when the signal fires (now, if it already has). Returns an unregister."""
        with self._lock:
            if not self._event.is_set():
                self._callbacks.append(cb)
                return lambda: self._remove(cb)
        cb()
        return lambda: None

    def _remove(self, cb: Callable[[], None]) -> None:
        with self._lock:
            if cb in self._callbacks:
                self._callbacks.remove(cb)


def cancelled_error() -> errors.TarjumanError:
    return errors.TarjumanError(errors.CANCELLED, "cancelled by the caller")


def cancellable(events: Callable[[Cancel], Iterator[Any]], cancel: Cancel | None) -> Iterator[Any]:
    """Run `events(token)` in a worker thread and yield what it produces until it ends, fails
    or `cancel` fires. `token` fires when the caller cancels or stops iterating: the producer
    registers its cleanup on it (closing the HTTP response)."""
    token = Cancel()
    unhook = cancel.on_cancel(token.cancel) if cancel else (lambda: None)
    q: queue.Queue[tuple[str, Any]] = queue.Queue()
    token.on_cancel(lambda: q.put(("cancel", None)))

    def work() -> None:
        try:
            for ev in events(token):
                if token.cancelled:
                    return
                q.put(("event", ev))
            q.put(("done", None))
        except BaseException as e:  # handed to the consumer, who decides what it means
            q.put(("error", e))

    threading.Thread(target=work, daemon=True, name="tarjuman-stream").start()
    try:
        while True:
            try:
                kind, x = q.get(timeout=0.2)  # a timeout keeps Ctrl+C working on Windows
            except queue.Empty:
                continue
            if kind == "cancel" or (cancel is not None and cancel.cancelled):
                raise cancelled_error()
            if kind == "event":
                yield x
            elif kind == "done":
                return
            elif kind == "error":
                raise x
    finally:
        token.cancel()  # the caller stopped (cancel, break, error): close the connection
        unhook()
