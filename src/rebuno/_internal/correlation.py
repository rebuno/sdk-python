from __future__ import annotations

import asyncio
from typing import Any


class CorrelationMap:
    def __init__(self) -> None:
        self._futures: dict[tuple[str, str], asyncio.Future[Any]] = {}

    def future(self, kind: str, key: str) -> asyncio.Future[Any]:
        """Get or create a future for the given (kind, key) pair."""
        fut = self._futures.get((kind, key))
        if fut is None:
            fut = asyncio.get_event_loop().create_future()
            self._futures[(kind, key)] = fut
        return fut

    def resolve(self, kind: str, key: str, value: Any) -> None:
        """Set the future's result. No-op if already resolved or missing."""
        fut = self._futures.get((kind, key))
        if fut is None:
            fut = asyncio.get_event_loop().create_future()
            self._futures[(kind, key)] = fut
        if not fut.done():
            fut.set_result(value)

    def cancel_all(self) -> None:
        for fut in self._futures.values():
            if not fut.done():
                fut.cancel()
        self._futures.clear()
