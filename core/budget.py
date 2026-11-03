"""Cost guard for conversational turns: sliding-window budgets per person and per server.

Replaces the per-message cooldown, which dropped messages. A turn (one burst
of messages) spends one unit of the person's budget and one of the server's;
when either is used up, the bot reacts with ⏳ instead of answering.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable


class _Window:
    def __init__(self, limit: int, seconds: float):
        self.limit = limit
        self.seconds = seconds
        self.hits: dict[int, deque[float]] = {}

    def _trim(self, key: int, now: float) -> deque[float]:
        dq = self.hits.setdefault(key, deque())
        while dq and now - dq[0] >= self.seconds:
            dq.popleft()
        return dq

    def full(self, key: int, now: float) -> bool:
        return self.limit > 0 and len(self._trim(key, now)) >= self.limit

    def spend(self, key: int, now: float) -> None:
        if self.limit > 0:
            self._trim(key, now).append(now)

    def retry_after(self, key: int, now: float) -> float:
        dq = self._trim(key, now)
        return max(0.0, self.seconds - (now - dq[0])) if dq else 0.0


class Budget:
    """``try_spend(user_id, guild_id)`` → whether this turn may run (and spends it)."""

    def __init__(
        self, *, user_limit: int, user_window: float, guild_limit: int, guild_window: float,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.users = _Window(user_limit, user_window)
        self.guilds = _Window(guild_limit, guild_window)
        self.clock = clock

    def try_spend(self, user_id: int | None, guild_id: int | None, *, per_user: bool = True) -> bool:
        now = self.clock()
        if per_user and user_id is not None and self.users.full(int(user_id), now):
            return False
        if guild_id is not None and self.guilds.full(int(guild_id), now):
            return False
        if per_user and user_id is not None:
            self.users.spend(int(user_id), now)
        if guild_id is not None:
            self.guilds.spend(int(guild_id), now)
        return True

    def retry_after(self, user_id: int | None, guild_id: int | None) -> float:
        now = self.clock()
        waits = [0.0]
        if user_id is not None and self.users.full(int(user_id), now):
            waits.append(self.users.retry_after(int(user_id), now))
        if guild_id is not None and self.guilds.full(int(guild_id), now):
            waits.append(self.guilds.retry_after(int(guild_id), now))
        return max(waits)
