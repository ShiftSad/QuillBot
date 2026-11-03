"""``/config`` — server and channel settings (see :mod:`core.settings`).

Channel settings (when the bot talks, name calls, threads, reply format, log
detection, channel instructions) can be set per channel or as a server-wide
default; threads follow their parent channel. Server settings: the bot's
extra names and the automatic Wrapped. Needs Manage Server.
"""

from __future__ import annotations

import asyncio
import logging

import discord
from discord import app_commands
from discord.ext import commands

from core.settings import (
    GUILD_SCOPE,
    SETTINGS,
    SettingError,
    display_value,
    parse_value,
)

logger = logging.getLogger(__name__)

_CHANNEL_KEYS = [k for k, s in SETTINGS.items() if s.scope == 'channel']
_GUILD_KEYS = [k for k, s in SETTINGS.items() if s.scope == 'guild']
_CHANNEL_CHOICES = [app_commands.Choice(name=SETTINGS[k].label, value=k) for k in _CHANNEL_KEYS]
_ALL_CHOICES = [app_commands.Choice(name=SETTINGS[k].label, value=k) for k in (*_GUILD_KEYS, *_CHANNEL_KEYS)]


def _can_manage(member) -> bool:
    perms = getattr(member, 'guild_permissions', None)
    return bool(perms and (perms.administrator or perms.manage_guild))


class ServerConfig(commands.Cog, name='ServerConfig'):
    """/config ver | canal | servidor | limpar."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.settings = bot.services.settings

    config = app_commands.Group(
        name='config', description='Configurações do bot neste servidor',
        guild_only=True, default_permissions=discord.Permissions(manage_guild=True),
    )

    async def _guard(self, interaction: discord.Interaction) -> bool:
        if interaction.guild is None or not _can_manage(interaction.user):
            await interaction.response.send_message('Requer a permissão Gerenciar Servidor.', ephemeral=True)
            return False
        return True

    @config.command(name='ver', description='Mostra as configurações de um canal e do servidor')
    @app_commands.describe(canal='Canal (padrão: este)')
    async def ver(self, interaction: discord.Interaction, canal: discord.TextChannel | discord.Thread | None = None):
        if not await self._guard(interaction):
            return
        guild = interaction.guild
        channel = canal or interaction.channel
        lines = []
        for key in _CHANNEL_KEYS:
            setting = SETTINGS[key]
            value = self.settings.get(guild.id, channel, key)
            source = self.settings.source(guild.id, channel, key)
            lines.append(f'**{setting.label}** (`{key}`): {display_value(setting, value)} — *{source}*')
        guild_lines = [
            f'**{SETTINGS[k].label}** (`{k}`): {display_value(SETTINGS[k], self.settings.get(guild.id, None, k))}'
            for k in _GUILD_KEYS
        ]
        embed = discord.Embed(title=f'⚙️ Configurações — #{getattr(channel, "name", channel.id)}', color=discord.Color.blurple())
        embed.add_field(name='Neste canal', value='\n'.join(lines)[:1024], inline=False)
        embed.add_field(name='Servidor', value='\n'.join(guild_lines)[:1024], inline=False)
        embed.set_footer(text='/config canal para mudar um canal • /config servidor para o padrão do servidor')
        await interaction.response.send_message(embed=embed, ephemeral=True)

    async def _value_autocomplete(self, interaction: discord.Interaction, current: str) -> list[app_commands.Choice[str]]:
        key = getattr(interaction.namespace, 'opcao', None)
        setting = SETTINGS.get(key or '')
        if setting is None:
            return []
        if setting.kind == 'choice':
            options = list(setting.choices)
        elif setting.kind == 'bool':
            options = ['sim', 'não']
        else:
            return []
        return [app_commands.Choice(name=o, value=o) for o in options if current.lower() in o][:25]

    async def _store(
        self, interaction: discord.Interaction, key: str, raw: str | None, *, channel_id: int, where: str,
    ) -> None:
        setting = SETTINGS[key]
        try:
            value = None if raw is None else parse_value(setting, raw)
        except SettingError as e:
            await interaction.response.send_message(f'⚠️ {e}', ephemeral=True)
            return
        await asyncio.to_thread(
            self.settings.write, interaction.guild.id, channel_id, key, value,
            updated_by=str(interaction.user.id),
        )
        logger.info('[config] guild=%s channel=%s %s=%r by %s', interaction.guild.id, channel_id, key, value, interaction.user.id)
        target = interaction.guild.get_channel_or_thread(channel_id) if channel_id else None
        effective = display_value(setting, self.settings.get(interaction.guild.id, target, key))
        note = f'voltou ao padrão ({effective})' if value is None else effective
        await interaction.response.send_message(f'✅ **{setting.label}** {where}: {note}', ephemeral=True)

    @config.command(name='canal', description='Muda uma configuração de um canal')
    @app_commands.describe(opcao='O que configurar', valor='Novo valor', canal='Canal (padrão: este)')
    @app_commands.choices(opcao=_CHANNEL_CHOICES)
    @app_commands.autocomplete(valor=_value_autocomplete)
    async def canal(
        self, interaction: discord.Interaction, opcao: str, valor: str,
        canal: discord.TextChannel | discord.Thread | None = None,
    ):
        if not await self._guard(interaction):
            return
        channel = canal or interaction.channel
        await self._store(interaction, opcao, valor, channel_id=channel.id, where=f'em {channel.mention}')

    @config.command(name='servidor', description='Muda uma configuração do servidor (ou o padrão de todos os canais)')
    @app_commands.describe(opcao='O que configurar', valor='Novo valor')
    @app_commands.choices(opcao=_ALL_CHOICES)
    @app_commands.autocomplete(valor=_value_autocomplete)
    async def servidor(self, interaction: discord.Interaction, opcao: str, valor: str):
        if not await self._guard(interaction):
            return
        await self._store(interaction, opcao, valor, channel_id=GUILD_SCOPE, where='no servidor')

    @config.command(name='limpar', description='Volta uma configuração ao padrão')
    @app_commands.describe(opcao='O que limpar', canal='Canal (omita para limpar o padrão do servidor)')
    @app_commands.choices(opcao=_ALL_CHOICES)
    async def limpar(
        self, interaction: discord.Interaction, opcao: str,
        canal: discord.TextChannel | discord.Thread | None = None,
    ):
        if not await self._guard(interaction):
            return
        if canal is not None and SETTINGS[opcao].scope == 'guild':
            await interaction.response.send_message('Esta configuração é do servidor, não de um canal.', ephemeral=True)
            return
        await self._store(
            interaction, opcao, None, channel_id=canal.id if canal else GUILD_SCOPE,
            where=f'em {canal.mention}' if canal else 'no servidor',
        )


async def setup(bot: commands.Bot):
    await bot.add_cog(ServerConfig(bot))
