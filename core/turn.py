"""Everything one turn of the bot needs, in one object.

:class:`TurnContext` replaces the fifteen keyword arguments that used to be
threaded through every chat function: who is talking, where, the stored
conversation, images, and the per-turn state tools write into (the thread
the model asked for, reactions left, tools used).
"""

from __future__ import annotations

import datetime
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from core.chat_threads import UNPROMPTED_HEADER
from core.tool_loop import EMPTY_ANSWER_FALLBACK

if TYPE_CHECKING:
    from core.profiles import Profile

__all__ = [
    'FOLLOW_HEADER', 'NAME_HEADER', 'NO_REPLY_TOKEN', 'UNPROMPTED_HEADER', 'TurnContext', 'TurnResult', 'is_silent',
]

# What the model answers to post nothing (it may still have reacted, saved a memory…).
NO_REPLY_TOKEN = '[NO_REPLY]'

# Header on turns where the bot was only named (UNPROMPTED_HEADER marks its threads);
# it may stay silent on both.
NAME_HEADER = '[Seu nome foi citado nesta mensagem, sem menção — pode ser com você ou sobre você]'
# Messages a person sent right after talking to the bot (while it answered), without addressing it.
FOLLOW_HEADER = '[Mensagem enviada logo depois de falar com você, sem menção — pode ou não ser para você]'


def is_silent(answer: str | None, *, unprompted: bool = True) -> bool:
    """Whether *answer* means "post nothing".

    ``[NO_REPLY]`` always does. An empty answer (or the fallback text) only
    does when nobody addressed the bot directly — someone who asked gets told
    the answer failed instead of hearing nothing.
    """
    text = (answer or '').strip()
    if 'NO_REPLY' in text.upper():
        return True
    return unprompted and (not text or text == EMPTY_ANSWER_FALLBACK)


@dataclass
class TurnContext:
    """One turn: the message being answered, its surroundings and per-turn state."""

    bot: Any
    profile: Profile
    question: str
    user: Any = None
    guild: Any = None
    channel: Any = None
    created_at: datetime.datetime | None = None
    # The message that triggered the turn (the last one of a burst): reactions
    # default to it and summaries stop before it.
    message: Any = None
    # The recent-channel window is read before this message (the first one of
    # a burst), so none of the burst shows up twice.
    window_before: Any = None
    interaction: Any = None
    history: list[dict] = field(default_factory=list)
    image_urls: list[str] = field(default_factory=list)
    context_image_urls: list[str] = field(default_factory=list)
    prior_context: list[str] | None = None
    reply_to: str | None = None
    participant_ids: set[str] = field(default_factory=set)
    origin: str | None = None
    # Why this answer can't move to a new thread; None when it can.
    thread_blocker: str | None = None
    spark_report: Any = None
    # Whether the model may answer [NO_REPLY] (not on slash commands or scheduled tasks).
    can_stay_silent: bool = True
    # Per-channel settings resolved for this turn (see core.settings).
    channel_settings: dict[str, Any] = field(default_factory=dict)
    on_status: Callable[[str], Awaitable[None]] | None = None

    # --- per-turn state written by tools ---
    thread_title: str | None = None
    used_tools: list[str] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)

    @property
    def services(self) -> Any:
        return getattr(self.bot, 'services', None)

    @property
    def actor_name(self) -> str:
        """How memory/scheduler audit logs name who acted: the bot, on someone's behalf."""
        name = getattr(self.user, 'display_name', None)
        return f'bot (via {name})' if name else 'bot'

    @property
    def guild_id(self) -> int | None:
        return getattr(self.guild, 'id', None)


@dataclass
class TurnResult:
    """What a turn produced; delivery decides how it goes out."""

    answer: str
    sources: list[dict]
    user_message: dict | None = None
    trajectory: list[dict] | None = None
    image_urls: list[str] = field(default_factory=list)
    context_image_urls: list[str] = field(default_factory=list)
    thread_title: str | None = None
    used_tools: list[str] = field(default_factory=list)
    structured: bool = False

    @property
    def capture(self) -> dict:
        """The fields a stored conversation turn keeps (see ``make_turn``)."""
        return {
            'user_message': self.user_message,
            'trajectory': self.trajectory,
            'image_urls': self.image_urls,
            'context_image_urls': self.context_image_urls,
            'thread_title': self.thread_title,
        }
