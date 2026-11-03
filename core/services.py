"""The bot's shared services, reachable as ``bot.services``.

Features talk to each other through these public objects instead of
``bot.get_cog('Name')._private_method(...)``:

- ``llm`` — the one LLM client (completions with retry, embeddings)
- ``tools`` — the tool registry every feature registers into
- ``router`` — the single ``on_message`` dispatcher
- ``settings`` — server and channel settings
- ``conversations`` — the conversation store
- ``chat`` — the chat engine (answer a message, a monitor hit, a scheduled
  task, a slash command; anchor a conversation on a message)
- ``memory``, ``history``, ``docs`` — set by those cogs when they load, so
  others can use them without knowing the cog
"""

from __future__ import annotations

from typing import Any

from config import (
    CONVERSATIONS_DB_PATH,
    CONVERSATIONS_MAX_STORED,
    CONVERSATIONS_TTL_SECONDS,
    EPISODES_ENABLED,
    MEMORY_ENABLED,
    SETTINGS_DB_PATH,
)
from core.conversations import ConversationStore
from core.llm import LLMService
from core.router import MessageRouter
from core.settings import SettingsStore
from core.tools import ToolRegistry


class Services:
    def __init__(
        self,
        *,
        llm: LLMService | None = None,
        settings: SettingsStore | None = None,
        conversations: ConversationStore | None = None,
    ):
        self.llm = llm if llm is not None else LLMService()
        self.tools = ToolRegistry()
        self.router = MessageRouter()
        self.settings = settings if settings is not None else SettingsStore(SETTINGS_DB_PATH)
        self.conversations = conversations if conversations is not None else ConversationStore(
            CONVERSATIONS_DB_PATH,
            kind='chat',
            ttl_seconds=CONVERSATIONS_TTL_SECONDS,
            max_stored=CONVERSATIONS_MAX_STORED,
            keep_expired=EPISODES_ENABLED and MEMORY_ENABLED,
        )
        self.chat: Any = None
        self.memory: Any = None
        self.history: Any = None
        self.docs: Any = None
