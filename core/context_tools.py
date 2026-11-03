"""Core tools: channel history, server info, message context, reactions, threads, web.

Feature-specific tools (memory, history search, docs, scheduler, monitors,
summaries, Wrapped, Spark) are registered by their own cogs.
"""

from __future__ import annotations

import logging
from typing import Any

import discord

from config import TAVILY_AVAILABLE
from core import chat_threads, message_media, tavily_tools
from core.context import build_guild_context, fetch_channel_history, fetch_message_context
from core.tools import Tool, ToolRegistry, tool_schema
from core.turn import TurnContext

logger = logging.getLogger(__name__)

CHANNEL_HISTORY_TOOL = tool_schema(
    'get_channel_history',
    'Busca mensagens recentes do canal atual ou de outro canal do servidor. '
    'Use quando o usuário perguntar sobre conversas anteriores, contexto recente, '
    'ou quando precisar entender o que foi discutido antes. Cada linha segue o formato '
    '[data] Nome (@usuário) <author_id=…>: conteúdo ↩ reply_to=… [msg_id=…]. '
    'Use author_id/msg_id/reply_to com search_history, get_user_stats ou get_message_context.',
    {
        'limit': {'type': 'integer', 'description': 'Número de mensagens a buscar (padrão 20, máximo 50).'},
        'channel_id': {'type': 'string', 'description': 'ID do canal para buscar. Omita para usar o canal atual.'},
    },
)

GUILD_INFO_TOOL = tool_schema(
    'get_guild_info',
    'Retorna informações detalhadas sobre o servidor Discord atual: nome, '
    'quantidade de membros, canais, cargos e dono.',
)

GET_MESSAGE_CONTEXT_TOOL = tool_schema(
    'get_message_context',
    'Retorna o contexto local ao redor de uma mensagem específica (5 antes e 5 depois). '
    'Use após search_history para expandir o contexto de um resultado relevante. '
    'Aceita o msg_id ou reply_to visto em qualquer linha de mensagem, desde que combinados '
    'com o channel_id exibido nos cabeçalhos de resultado/blocos ou com o canal atual. '
    'Resultados próximos no tempo são agrupados pelo search_history — use esta ferramenta '
    'para ver as mensagens vizinhas de um cluster.',
    {
        'message_id': {'type': 'string', 'description': 'ID da mensagem central.'},
        'channel_id': {'type': 'string', 'description': 'ID do canal da mensagem.'},
        'window': {'type': 'integer', 'description': 'Janela de contexto (padrão 5, máximo 10).'},
    },
    ['message_id', 'channel_id'],
)


async def _channel_history(ctx: TurnContext, args: dict) -> str:
    return await fetch_channel_history(
        ctx.bot, ctx.channel, limit=args.get('limit', 20), channel_id=args.get('channel_id'),
    )


async def _guild_info(ctx: TurnContext, args: dict) -> str:
    guild = ctx.guild
    if not guild:
        return 'Fora de um servidor (DM).'
    text = build_guild_context(guild)
    try:
        channels = [f'#{c.name} ({c.id})' for c in guild.channels if isinstance(c, discord.TextChannel)][:30]
        if channels:
            text += '\nCanais de texto: ' + ', '.join(channels)
    except Exception:
        pass
    return text


async def _message_context(ctx: TurnContext, args: dict) -> str:
    try:
        window = max(1, min(10, int(args.get('window', 5))))
    except (TypeError, ValueError):
        window = 5
    channel_id = str(args.get('channel_id', ''))
    message_id = str(args.get('message_id', ''))
    text = await fetch_message_context(
        ctx.bot, channel_id=channel_id, message_id=message_id, window=window,
        guild_id=ctx.guild_id,
    )
    history = getattr(ctx.services, 'history', None)
    if text is None and history is not None and ctx.guild is not None:
        try:
            text = await history.get_message_context_from_index(ctx.guild.id, channel_id, message_id, window)
        except Exception:
            logger.exception('indexed get_message_context fallback failed')
    return text or f'Mensagem {message_id} não encontrada (nem via API, nem no índice).'


async def _add_reaction(ctx: TurnContext, args: dict) -> str:
    tool = ctx.state.get('reactions')
    if tool is None:
        tool = ctx.state['reactions'] = message_media.ReactionTool(
            ctx.channel, guild=ctx.guild, default_message=ctx.message,
        )
    return await tool(args)


async def _open_thread(ctx: TurnContext, args: dict) -> str:
    request = chat_threads.ThreadRequest(ctx.thread_blocker)
    result = request(args)
    if request.title:
        ctx.thread_title = request.title
    return result


def _channel_history_status(args: dict, ctx: Any) -> str:
    cid = args.get('channel_id')
    return f'📜 Lendo histórico ({args.get("limit", 20)} msgs)' + (f' canal {cid}' if cid else '')


async def _web(ctx: TurnContext, args: dict, *, name: str) -> tuple[str, list[dict]]:
    return await tavily_tools.exec_tool(name, args)


def register(registry: ToolRegistry) -> None:
    """Register the core tools (web tools only when Tavily is configured)."""
    if TAVILY_AVAILABLE:
        for schema in tavily_tools.TOOLS:
            name = schema['function']['name']
            registry.register(Tool(
                name, schema,
                run=lambda ctx, args, _n=name: _web(ctx, args, name=_n),
                status=lambda args, ctx, _n=name: tavily_tools.status_label(_n, args),
            ))
    registry.register(
        Tool('get_channel_history', CHANNEL_HISTORY_TOOL, _channel_history, _channel_history_status),
        Tool('get_guild_info', GUILD_INFO_TOOL, _guild_info, '🏰 Coletando informações do servidor'),
        Tool(
            'get_message_context', GET_MESSAGE_CONTEXT_TOOL, _message_context,
            lambda args, ctx: f'🧩 Contexto da mensagem {args.get("message_id", "")}…',
        ),
        Tool(
            'add_reaction', message_media.ADD_REACTION_TOOL, _add_reaction,
            lambda args, ctx: message_media.reaction_tool_status(args),
        ),
        Tool(
            'open_thread', chat_threads.OPEN_THREAD_TOOL, _open_thread,
            lambda args, ctx: chat_threads.thread_tool_status(args),
        ),
    )
