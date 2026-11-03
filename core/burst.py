"""Merge a person's quick consecutive messages into one turn.

People type in bursts: "ei" / "sabe aquele erro" / "de ontem?". The old
per-user cooldown answered the first and dropped the rest with "⏳ Aguarde".
Now a message aimed at the bot opens a *burst* for (channel, person); every
message that person sends there before going quiet for ``delay`` seconds joins
it, mention or not, and the whole burst is answered as one turn. Messages
sent while the bot is still answering start the next burst.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

BurstKey = tuple[int, int]  # (channel_id, author_id)


@dataclass
class Burst:
    key: BurstKey
    messages: list[Any] = field(default_factory=list)
    # Why each message reached the bot: 'reply', 'mention', 'thread', 'name' or 'join'.
    triggers: list[str] = field(default_factory=list)

    @property
    def first(self) -> Any:
        return self.messages[0]

    @property
    def last(self) -> Any:
        return self.messages[-1]

    def discard(self, message_id: int) -> bool:
        for i, m in enumerate(self.messages):
            if getattr(m, 'id', None) == message_id:
                del self.messages[i]
                del self.triggers[i]
                return True
        return False


class BurstCollector:
    """Debounce bot-directed messages per (channel, author) and hand bursts to *process*."""

    def __init__(self, delay: float, process: Callable[[Burst], Awaitable[None]]):
        self.delay = delay
        self.process = process
        self._open: dict[BurstKey, Burst] = {}
        self._timers: dict[BurstKey, asyncio.Task] = {}
        self._running: dict[BurstKey, asyncio.Event] = {}

    @staticmethod
    def key_for(message: Any) -> BurstKey:
        return (int(getattr(message.channel, 'id', 0) or 0), int(getattr(message.author, 'id', 0) or 0))

    def is_open(self, key: BurstKey) -> bool:
        """Whether a message from this person here would join a burst (collecting or being answered)."""
        return key in self._open or key in self._running

    async def add(self, message: Any, trigger: str | None) -> bool:
        """Add *message*; ``trigger=None`` only joins an open burst. Returns whether it was taken."""
        key = self.key_for(message)
        burst = self._open.get(key)
        if burst is None:
            if trigger is None and key not in self._running:
                return False
            burst = self._open[key] = Burst(key)
        burst.messages.append(message)
        burst.triggers.append(trigger or 'join')
        if self.delay <= 0:
            await self._flush(key)
        else:
            self._restart_timer(key)
        return True

    def discard(self, message_id: int) -> None:
        """Forget a deleted message that is still waiting in a burst."""
        for key, burst in list(self._open.items()):
            if burst.discard(message_id) and not burst.messages:
                self._open.pop(key, None)
                timer = self._timers.pop(key, None)
                if timer is not None:
                    timer.cancel()

    def _restart_timer(self, key: BurstKey) -> None:
        timer = self._timers.pop(key, None)
        if timer is not None:
            timer.cancel()
        self._timers[key] = asyncio.get_running_loop().create_task(self._wait_and_flush(key))

    async def _wait_and_flush(self, key: BurstKey) -> None:
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            return
        if self._timers.get(key) is asyncio.current_task():
            self._timers.pop(key, None)
        await self._flush(key)

    async def _flush(self, key: BurstKey) -> None:
        # One burst per person and channel at a time: messages that arrive
        # meanwhile wait for the answer and then go as the next turn.
        running = self._running.get(key)
        if running is not None:
            await running.wait()
            if key in self._running or key in self._timers:
                return  # another flush picked it up
        burst = self._open.pop(key, None)
        if burst is None or not burst.messages:
            return
        done = self._running[key] = asyncio.Event()
        try:
            await self.process(burst)
        except Exception:
            logger.exception('[burst] failed to answer burst %s', key)
        finally:
            self._running.pop(key, None)
            done.set()
        if key in self._open and key not in self._timers:
            if self.delay <= 0:
                await self._flush(key)
            else:
                self._restart_timer(key)

    async def close(self) -> None:
        for timer in self._timers.values():
            timer.cancel()
        self._timers.clear()
        self._open.clear()
