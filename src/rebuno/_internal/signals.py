from __future__ import annotations

import asyncio
import signal as signal_module
from collections.abc import Callable


def install_shutdown_handlers(handler: Callable[[], None]) -> None:
    """Wire SIGTERM and SIGINT to ``handler`` on the running loop.

    Uses ``loop.add_signal_handler`` where available; falls back to
    ``signal.signal`` on Windows.
    """
    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal_module.SIGTERM, handler)
        loop.add_signal_handler(signal_module.SIGINT, handler)
    except NotImplementedError:
        signal_module.signal(
            signal_module.SIGTERM,
            lambda s, f: loop.call_soon_threadsafe(handler),
        )
        signal_module.signal(
            signal_module.SIGINT,
            lambda s, f: loop.call_soon_threadsafe(handler),
        )
