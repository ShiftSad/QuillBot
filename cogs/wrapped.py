"""Server Wrapped: recaps of what the server talked about.

``/wrapped`` (or asking the bot, through the ``server_wrapped`` tool) builds a
recap of a period from the history index: who talked most, busiest channels
and hours, who replied to whom, favorite emojis and the most-reacted
messages. With the ``wrapped`` setting (``/config servidor``) a recap is
posted automatically every Monday (weekly) or on the 1st (monthly) at
``WRAPPED_HOUR`` in the ``wrapped_channel``. People are named, never pinged.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import re
from typing import Any

import discord
from discord import app_commands
from discord.ext import commands

from config import BR_TZ, COOLDOWN_PER, COOLDOWN_RATE, SUMMARY_MODEL, WRAPPED_ENABLED, WRAPPED_HOUR
from core.tools import Tool, tool_schema

logger = logging.getLogger(__name__)

_PERIOD_DAYS = {'semana': 7, 'mes': 30, 'mês': 30, 'ano': 365}
_REL_RE = re.compile(r'^(\d{1,3})\s*d$')
_LOOP_SECONDS = 600
_NO_PING = discord.AllowedMentions.none()

WRAPPED_TOOL = tool_schema(
    'server_wrapped',
    'Retrospectiva do servidor num período: total de mensagens e membros ativos, quem mais falou, '
    'canais e horários mais movimentados, quem mais respondeu a quem, emojis favoritos e as '
    'mensagens mais reagidas. Use para "wrapped", "retrospectiva", "como foi o mês no servidor?". '
    'Apresente o resultado do seu jeito, citando pessoas sem @.',
    {
        'periodo': {
            'type': 'string',
            'description': 'semana (7 dias, padrão), mes (30 dias), ano (365 dias) ou "Nd" (ex.: 14d).',
        },
    },
)


def parse_period(value: str | None, now: datetime.datetime) -> tuple[datetime.datetime, datetime.datetime, str]:
    """(since, until, label) for a period name or ``Nd``; raises ValueError."""
    text = (value or 'semana').strip().lower()
    if text in _PERIOD_DAYS:
        days = _PERIOD_DAYS[text]
    else:
        m = _REL_RE.match(text)
        if not m or not 1 <= int(m.group(1)) <= 366:
            raise ValueError('Período inválido: use semana, mes, ano ou um número de dias como 14d.')
        days = int(m.group(1))
    label = {7: 'últimos 7 dias', 30: 'últimos 30 dias', 365: 'último ano'}.get(days, f'últimos {days} dias')
    return now - datetime.timedelta(days=days), now, label


def previous_period(mode: str, now: datetime.datetime) -> tuple[datetime.datetime, datetime.datetime, str, str]:
    """The last complete week (Mon–Sun) or month before *now* (BRT), its label and a key."""
    local = now.astimezone(BR_TZ)
    midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
    if mode == 'weekly':
        until = midnight - datetime.timedelta(days=local.weekday())
        since = until - datetime.timedelta(days=7)
        last_day = until - datetime.timedelta(days=1)
        return since, until, f'semana de {since:%d/%m} a {last_day:%d/%m}', f'{since:%G-W%V}'
    until = midnight.replace(day=1)
    since = (until - datetime.timedelta(days=1)).replace(day=1)
    months = ('janeiro', 'fevereiro', 'março', 'abril', 'maio', 'junho', 'julho', 'agosto',
              'setembro', 'outubro', 'novembro', 'dezembro')
    return since, until, f'{months[since.month - 1]} de {since.year}', f'{since:%Y-%m}'


def due_key(mode: str, now: datetime.datetime) -> str | None:
    """The period key an automatic recap is due for at *now*, or ``None`` when it isn't time."""
    local = now.astimezone(BR_TZ)
    if local.hour < WRAPPED_HOUR:
        return None
    if mode == 'weekly' and local.weekday() == 0:
        return previous_period(mode, now)[3]
    if mode == 'monthly' and local.day == 1:
        return previous_period(mode, now)[3]
    return None


def stats_text(stats: dict, label: str) -> str:
    """The recap as plain lines (tool result and LLM input)."""
    if not stats['messages']:
        return f'Retrospectiva ({label}): nenhuma mensagem registrada no período.'
    lines = [
        f"Retrospectiva ({label}): {stats['messages']} mensagens de {stats['active_members']} membros.",
        'Quem mais falou: ' + ', '.join(f'{n} ({c})' for n, c in stats['top_authors']),
        'Canais mais movimentados: ' + ', '.join(f'#{n} ({c})' for n, c in stats['top_channels']),
    ]
    if stats['busiest_hour']:
        hour, count = stats['busiest_hour']
        lines.append(f'Horário de pico: {hour}h ({count} mensagens)')
    if stats['busiest_weekday']:
        day, count = stats['busiest_weekday']
        lines.append(f'Dia mais agitado: {day} ({count} mensagens)')
    if stats['top_pairs']:
        lines.append('Quem mais respondeu a quem: ' + ', '.join(f'{a} → {b} ({n})' for a, b, n in stats['top_pairs']))
    if stats['top_emojis']:
        lines.append('Emojis favoritos: ' + ', '.join(f'{e} ({n})' for e, n in stats['top_emojis']))
    if stats['reactions']:
        lines.append(f"Reações no período: {stats['reactions']}")
    for m in stats['most_reacted']:
        lines.append(f"Mais reagida: \"{m['content']}\" — {m['author']} ({m['count']} reações) {m['jump_url']}")
    return '\n'.join(lines)


def build_embed(stats: dict, label: str, guild_name: str, intro: str = '') -> discord.Embed:
    embed = discord.Embed(title=f'📊 Wrapped — {guild_name}', color=discord.Color.purple())
    embed.description = (intro + '\n\n' if intro else '') + f'**{label}**'
    if not stats['messages']:
        embed.description += '\nNenhuma mensagem registrada no período.'
        return embed
    embed.add_field(name='💬 Mensagens', value=f"{stats['messages']:,}".replace(',', '.'), inline=True)
    embed.add_field(name='👥 Membros ativos', value=str(stats['active_members']), inline=True)
    peaks = []
    if stats['busiest_hour']:
        peaks.append(f"{stats['busiest_hour'][0]}h")
    if stats['busiest_weekday']:
        peaks.append(stats['busiest_weekday'][0])
    if peaks:
        embed.add_field(name='⏰ Pico', value=' · '.join(peaks), inline=True)
    embed.add_field(
        name='🗣️ Quem mais falou',
        value='\n'.join(f'{i}. {n} — {c}' for i, (n, c) in enumerate(stats['top_authors'], 1))[:1024],
        inline=False,
    )
    embed.add_field(
        name='📍 Canais',
        value='\n'.join(f'#{n} — {c}' for n, c in stats['top_channels'])[:1024],
        inline=True,
    )
    if stats['top_pairs']:
        embed.add_field(
            name='↩️ Quem mais respondeu a quem',
            value='\n'.join(f'{a} → {b} ({n})' for a, b, n in stats['top_pairs'])[:1024],
            inline=True,
        )
    if stats['top_emojis']:
        embed.add_field(name='😀 Emojis', value=' '.join(f'{e}×{n}' for e, n in stats['top_emojis'])[:1024], inline=False)
    if stats['most_reacted']:
        embed.add_field(
            name='⭐ Mais reagidas',
            value='\n'.join(
                f"[{m['content'][:80] or '(sem texto)'}]({m['jump_url']}) — {m['author']} ({m['count']})"
                for m in stats['most_reacted']
            )[:1024],
            inline=False,
        )
    embed.set_footer(text='Reações contam desde que o bot começou a registrá-las')
    return embed


class Wrapped(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.services = bot.services
        self._loop_task: asyncio.Task | None = None

    async def cog_load(self) -> None:
        self.services.tools.register(Tool(
            'server_wrapped', WRAPPED_TOOL, self._run_tool, '📊 Montando a retrospectiva do servidor…',
            structured=True,
        ))
        self._loop_task = asyncio.get_running_loop().create_task(self._auto_loop())

    async def cog_unload(self) -> None:
        self.services.tools.unregister('server_wrapped')
        if self._loop_task is not None:
            self._loop_task.cancel()

    async def stats(self, guild_id: int, since: datetime.datetime, until: datetime.datetime) -> dict | None:
        history = self.services.history
        if history is None:
            return None
        return await history.wrapped_stats(guild_id, since, until)

    async def _run_tool(self, ctx: Any, args: dict) -> str:
        if ctx.guild is None:
            return 'A retrospectiva só funciona dentro de um servidor.'
        try:
            since, until, label = parse_period(args.get('periodo'), datetime.datetime.now(datetime.timezone.utc))
        except ValueError as e:
            return str(e)
        stats = await self.stats(ctx.guild.id, since, until)
        if stats is None:
            return 'O histórico do servidor não está disponível (HISTORY_ENABLED).'
        return stats_text(stats, label)

    async def _intro(self, guild: discord.Guild, stats: dict, label: str) -> str:
        """Two or three lines presenting the recap in the bot's (the server's) voice."""
        llm = self.services.llm
        if not llm.available or not stats['messages']:
            return ''
        persona = ''
        if getattr(self.services, 'persona', None) is not None:
            persona = self.services.persona.block(guild.id)
        memory = self.services.memory
        if memory is not None:
            try:
                memories = await memory.build_memory_block(guild.id, 'retrospectiva do servidor, jeito do bot') or ''
                persona = f'{persona}\n\n{memories}'.strip()
            except Exception:
                logger.debug('memory block for wrapped failed', exc_info=True)
        try:
            text = await llm.text(
                model=SUMMARY_MODEL,
                system=(
                    'Você é o bot deste servidor do Discord. Escreva 2 ou 3 frases curtas, em português, '
                    'apresentando a retrospectiva abaixo com o seu jeito (veja sua personalidade e as '
                    'memórias abaixo, se houver). '
                    'Destaque uma ou duas curiosidades. Cite pessoas pelo nome, nunca com @. Sem títulos.\n\n'
                    + persona
                ),
                user=stats_text(stats, label),
                max_tokens=1024,
            )
        except Exception:
            logger.exception('[wrapped] intro failed')
            return ''
        return text.replace('@', '')[:1500]

    async def build(self, guild: discord.Guild, since: datetime.datetime, until: datetime.datetime, label: str) -> discord.Embed | None:
        stats = await self.stats(guild.id, since, until)
        if stats is None:
            return None
        return build_embed(stats, label, guild.name, await self._intro(guild, stats, label))

    @app_commands.command(name='wrapped', description='Retrospectiva do servidor: quem mais falou, canais, emojis…')
    @app_commands.guild_only()
    @app_commands.checks.cooldown(COOLDOWN_RATE, COOLDOWN_PER)
    @app_commands.describe(periodo='semana (padrão), mes, ano ou número de dias como 14d')
    async def wrapped(self, interaction: discord.Interaction, periodo: str | None = None):
        try:
            since, until, label = parse_period(periodo, datetime.datetime.now(datetime.timezone.utc))
        except ValueError as e:
            await interaction.response.send_message(str(e), ephemeral=True)
            return
        await interaction.response.defer(thinking=True)
        embed = await self.build(interaction.guild, since, until, label)
        if embed is None:
            await interaction.followup.send('O histórico do servidor não está disponível.')
            return
        await interaction.followup.send(embed=embed, allowed_mentions=_NO_PING)

    @wrapped.autocomplete('periodo')
    async def _periodo_autocomplete(self, interaction: discord.Interaction, current: str):
        options = [('Últimos 7 dias', 'semana'), ('Últimos 30 dias', 'mes'), ('Último ano', 'ano')]
        return [app_commands.Choice(name=n, value=v) for n, v in options if current.lower() in n.lower() or current in v]

    async def cog_app_command_error(self, interaction: discord.Interaction, error: app_commands.AppCommandError):
        if isinstance(error, app_commands.CommandOnCooldown):
            await interaction.response.send_message(f'⏳ Aguarde {error.retry_after:.0f}s.', ephemeral=True)
        else:
            raise error

    # --- Automatic recaps ---

    async def _auto_loop(self) -> None:
        await self.bot.wait_until_ready()
        while True:
            try:
                await self.post_due(datetime.datetime.now(datetime.timezone.utc))
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception('[wrapped] auto loop error')
            await asyncio.sleep(_LOOP_SECONDS)

    async def post_due(self, now: datetime.datetime) -> int:
        """Post every automatic recap due at *now*; returns how many were posted."""
        settings = self.services.settings
        posted = 0
        for guild_id, mode in settings.guilds_with('wrapped').items():
            key = due_key(mode, now)
            if key is None or settings.state(guild_id, 'wrapped_last') == key:
                continue
            guild = self.bot.get_guild(guild_id)
            channel_id = settings.get(guild_id, None, 'wrapped_channel')
            channel = guild.get_channel(int(channel_id)) if guild and channel_id else None
            if channel is None:
                continue
            since, until, label, key = previous_period(mode, now)
            embed = await self.build(guild, since, until, label)
            if embed is None:
                continue
            await asyncio.to_thread(settings.set_state, guild_id, 'wrapped_last', key)
            try:
                await channel.send(embed=embed, allowed_mentions=_NO_PING)
                posted += 1
            except discord.HTTPException:
                logger.exception('[wrapped] could not post in channel %s', channel_id)
        return posted


async def setup(bot: commands.Bot):
    if not WRAPPED_ENABLED:
        logger.info('Wrapped disabled via WRAPPED_ENABLED=false')
        return
    await bot.add_cog(Wrapped(bot))
