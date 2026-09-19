from __future__ import annotations

import signal
import threading
import time
from contextlib import contextmanager
from collections.abc import Iterator


class DeadlineExceeded(TimeoutError):
    """Raised when an application operation exceeds its wall-clock deadline."""


def supports_deadlines() -> bool:
    return (
        hasattr(signal, "SIGALRM")
        and hasattr(signal, "setitimer")
        and threading.current_thread() is threading.main_thread()
    )


@contextmanager
def deadline(seconds: float, operation: str) -> Iterator[None]:
    """Interrupt a blocking operation in the Linux solo-worker main thread."""
    if seconds <= 0 or not supports_deadlines():
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()

    def handle_timeout(_signum, _frame) -> None:
        raise DeadlineExceeded(f"{operation} exceeded {seconds:g} seconds")

    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        previous_delay, previous_interval = previous_timer
        if previous_delay > 0:
            elapsed = time.monotonic() - started
            signal.setitimer(
                signal.ITIMER_REAL,
                max(0.000001, previous_delay - elapsed),
                previous_interval,
            )
