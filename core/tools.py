"""Tool registry: every feature registers its LLM tools in one place.

A :class:`Tool` bundles what used to be spread over if-chains in two cogs: the
JSON schema the model sees, the coroutine that runs it, and the status line
shown while it runs. Features register their tools when they load
(``bot.services.tools.register(...)``); a :class:`~core.profiles.Profile`
lists the tool names a kind of turn offers, in a fixed order so the prompt
prefix stays cacheable.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.turn import TurnContext

logger = logging.getLogger(__name__)

ToolOutput = tuple[str, list[dict]]
ToolRun = Callable[['TurnContext', dict], Awaitable[ToolOutput | str]]
ToolStatus = Callable[[dict, 'TurnContext'], str]


@dataclass
class Tool:
    """One LLM tool.

    ``run(ctx, args)`` returns ``(result_text, sources)`` or just the text.
    ``status(args, ctx)`` is the short line shown on a slash command while the
    tool runs. ``structured`` marks tools whose result is a finished document
    (a channel summary, a recap) that reads better in an embed.
    """

    name: str
    schema: dict
    run: ToolRun
    status: ToolStatus | str | None = None
    structured: bool = False

    def status_text(self, args: dict, ctx: TurnContext) -> str:
        if callable(self.status):
            try:
                return self.status(args, ctx)
            except Exception:
                logger.debug('status label failed for %s', self.name, exc_info=True)
        elif isinstance(self.status, str):
            return self.status
        return f'🔧 Executando: {self.name}'


def tool_schema(name: str, description: str, properties: dict | None = None, required: list[str] | None = None) -> dict:
    """OpenAI function-calling schema for a tool."""
    return {
        'type': 'function',
        'function': {
            'name': name,
            'description': description,
            'parameters': {
                'type': 'object',
                'properties': properties or {},
                'required': required or [],
            },
        },
    }


class ToolRegistry:
    """Name → :class:`Tool`, filled by the features as they load."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, *tools: Tool) -> None:
        for tool in tools:
            if tool.name in self._tools:
                logger.debug('tool %s re-registered', tool.name)
            self._tools[tool.name] = tool

    def unregister(self, *names: str) -> None:
        for name in names:
            self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return list(self._tools)

    def select(self, names: tuple[str, ...] | list[str]) -> list[Tool]:
        """The registered tools among *names*, in *names* order."""
        return [self._tools[n] for n in names if n in self._tools]

    async def execute(self, name: str, args: dict, ctx: TurnContext) -> ToolOutput:
        tool = self._tools.get(name)
        if tool is None:
            return f'Ferramenta desconhecida: {name}', []
        result: Any = await tool.run(ctx, args)
        if isinstance(result, tuple):
            text, sources = result
            return str(text), list(sources or [])
        return str(result), []
