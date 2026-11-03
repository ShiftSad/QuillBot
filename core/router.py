"""One ``on_message`` for the whole bot.

Several cogs used to listen to every message and coordinate through load
order and by peeking at each other's stores ("does DocsRAG already answer
this reply?"). Now the router owns the event:

1. **Observers** see every message (bots included — each filters itself) and
   never claim it: history indexing, log and Spark link detection. Each runs
   as its own task, so a slow log download never delays an answer.
2. **Handlers** run in priority order on human messages until one claims the
   message: the chat engine first, monitors after it — so a monitor never
   fires on a message the bot already answers.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

Observer = Callable[[Any], Awaitable[None]]
Handler = Callable[[Any], Awaitable[bool]]


class MessageRouter:
    def __init__(self) -> None:
        self._observers: dict[str, Observer] = {}
        self._handlers: list[tuple[int, str, Handler]] = []
        self._tasks: set[asyncio.Task] = set()

    def observe(self, name: str, fn: Observer) -> None:
        """Run *fn* on every message, concurrently with everything else."""
        self._observers[name] = fn

    def handle(self, name: str, fn: Handler, *, priority: int) -> None:
        """Offer human messages to *fn* (lower priority first) until one returns True."""
        self._handlers = [h for h in self._handlers if h[1] != name]
        self._handlers.append((priority, name, fn))
        self._handlers.sort(key=lambda h: h[0])

    def remove(self, name: str) -> None:
        self._observers.pop(name, None)
        self._handlers = [h for h in self._handlers if h[1] != name]

    async def _observe(self, name: str, fn: Observer, message: Any) -> None:
        try:
            await fn(message)
        except Exception:
            logger.exception('[router] observer %s failed on message %s', name, getattr(message, 'id', '?'))

    async def dispatch(self, message: Any) -> str | None:
        """Route one message; returns the name of the handler that claimed it."""
        for name, fn in list(self._observers.items()):
            task = asyncio.get_running_loop().create_task(self._observe(name, fn, message))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        if getattr(getattr(message, 'author', None), 'bot', False):
            return None
        for _priority, name, fn in list(self._handlers):
            try:
                if await fn(message):
                    return name
            except Exception:
                logger.exception('[router] handler %s failed on message %s', name, getattr(message, 'id', '?'))
                return name
        return None

    async def drain(self) -> None:
        """Wait for running observer tasks (tests, shutdown)."""
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)
