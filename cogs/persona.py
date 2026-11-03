"""``/persona`` and the ``persona_update`` tool — the bot's personality on each server.

See :mod:`core.persona`. Admins write the base with ``/persona editar``; the
bot rewrites its own notes through ``persona_update`` (and the base only when
someone with Manage Server asks). Every change is posted to the log channel
with a Revert button, and ``/persona historico`` / ``reverter`` cover the rest.
``/persona importar`` moves personality memories from the memory system into
the base.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

import discord
from discord import app_commands
from discord.ext import commands

from config import PERSONA_BOT_EDITS_PER_DAY, PERSONA_DB_PATH, PERSONA_ENABLED, PERSONA_LOG_CHANNEL_ID
from core.persona import SECTIONS, PersonaError, PersonaStore, Version, can_edit_base
from core.tools import Tool, tool_schema

logger = logging.getLogger(__name__)

_SECTION_CHOICES = [app_commands.Choice(name=s['label'], value=key) for key, s in SECTIONS.items()]

PERSONA_TOOL = tool_schema(
    'persona_update',
    'Reescreve uma seção da sua personalidade neste servidor (bloco <personalidade>). '
    'section=notes: suas notas — piadas internas, apelidos, bordões, o jeito que você foi pegando aqui; '
    'você mesmo mantém, poucas vezes por dia. section=base: quem você é; só quando alguém com '
    'Gerenciar Servidor pedir. Mande a seção INTEIRA já reescrita (substitui a atual), curta, sem '
    'perder o que ainda vale. Vale a partir da próxima mensagem.',
    {
        'section': {'type': 'string', 'enum': list(SECTIONS), 'description': 'notes (suas notas) ou base.'},
        'content': {'type': 'string', 'description': 'O novo texto completo da seção.'},
        'reason': {'type': 'string', 'description': 'Por que mudou, em uma linha.'},
    },
    ['section', 'content', 'reason'],
)


def _snippet(text: str, n: int) -> str:
    text = (text or '').strip()
    return '—' if not text else text if len(text) <= n else text[:n] + '…'


class PersonaModal(discord.ui.Modal):
    """Edit one section in a text box pre-filled with the current text."""

    def __init__(self, cog: 'Persona', guild_id: int, section: str):
        super().__init__(title=f'Personalidade — {SECTIONS[section]["label"]}')
        self.cog = cog
        self.guild_id = guild_id
        self.section = section
        self.text = discord.ui.TextInput(
            label='Texto (vazio apaga a seção)',
            style=discord.TextStyle.paragraph,
            default=cog.store.get(guild_id, section) or None,
            max_length=SECTIONS[section]['max'],
            required=False,
        )
        self.reason = discord.ui.TextInput(label='Motivo (opcional)', max_length=200, required=False)
        self.add_item(self.text)
        self.add_item(self.reason)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        try:
            version = await self.cog.change(
                self.guild_id, self.section, str(self.text.value),
                actor=str(interaction.user.id), actor_name=interaction.user.display_name,
                reason=str(self.reason.value) or 'Edição manual',
            )
        except PersonaError as e:
            await interaction.response.send_message(f'⚠️ {e}', ephemeral=True)
            return
        await interaction.response.send_message(
            f'✅ {SECTIONS[self.section]["label"]} atualizada (versão {version.version}). '
            'Vale a partir da próxima mensagem.', ephemeral=True,
        )


class PersonaLogView(discord.ui.View):
    """Revert button on log-channel posts (admins of the same server)."""

    def __init__(self, cog: 'Persona', guild_id: int, history_id: int):
        # Not persistent across restarts; /persona reverter works any time.
        super().__init__(timeout=86400)
        self.cog = cog
        self.guild_id = guild_id
        self.history_id = history_id

    @discord.ui.button(label='Reverter', emoji='↩️', style=discord.ButtonStyle.secondary)
    async def revert_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.guild_id != self.guild_id or not can_edit_base(interaction.user):
            await interaction.response.send_message('Só admins deste servidor podem reverter.', ephemeral=True)
            return
        try:
            version = await self.cog.revert(self.guild_id, self.history_id, actor=interaction.user)
        except PersonaError as e:
            await interaction.response.send_message(f'⚠️ {e}', ephemeral=True)
            return
        button.disabled = True
        await interaction.response.edit_message(view=self)
        await interaction.followup.send(f'↩️ Revertido (versão {version.version}).', ephemeral=True)


class Persona(commands.Cog):
    """The bot's personality per server."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.services = bot.services
        self.store = PersonaStore(PERSONA_DB_PATH)
        self._log_tasks: set[asyncio.Task] = set()

    async def cog_load(self) -> None:
        await asyncio.to_thread(self.store.load)
        self.services.persona = self
        self.services.tools.register(Tool(
            'persona_update', PERSONA_TOOL, self._run_tool, '🎭 Atualizando minha personalidade…',
        ))

    async def cog_unload(self) -> None:
        self.services.tools.unregister('persona_update')
        if self.services.persona is self:
            self.services.persona = None

    # --- Public API (bot.services.persona) ---

    def block(self, guild_id: int | None) -> str:
        """The ``<personalidade>`` block for the system prompt."""
        return self.store.block(guild_id)

    async def change(
        self, guild_id: int, section: str, content: str, *, actor: str, actor_name: str, reason: str,
    ) -> Version:
        """Write a section, then post the change to the log channel."""
        version = await asyncio.to_thread(
            self.store.write, guild_id, section, content, actor=actor, actor_name=actor_name, reason=reason,
        )
        self._schedule_log(version)
        return version

    async def revert(self, guild_id: int, history_id: int, *, actor: Any) -> Version:
        """Undo one change: its section goes back to what it was before it."""
        target = await asyncio.to_thread(self.store.get_version, guild_id, history_id)
        if target is None:
            raise PersonaError(f'Mudança #{history_id} não encontrada.')
        return await self.change(
            guild_id, target.section, target.before,
            actor=str(actor.id), actor_name=actor.display_name,
            reason=f'Reversão da mudança #{history_id}',
        )

    # --- The persona_update tool ---

    async def _run_tool(self, ctx: Any, args: dict) -> str:
        guild = ctx.guild
        if guild is None:
            return 'Personalidade só existe dentro de um servidor.'
        section = str(args.get('section') or '')
        if section not in SECTIONS:
            return 'section deve ser notes ou base.'
        reason = str(args.get('reason') or '').strip()
        if not reason:
            return '`reason` é obrigatório (uma linha).'
        asked_by_admin = can_edit_base(ctx.user)
        if section == 'base' and not asked_by_admin:
            return (
                'Só alguém com Gerenciar Servidor pode mudar a base da sua personalidade. Não mude; '
                'se fizer sentido, sugira a quem pediu que fale com um admin (/persona editar).'
            )
        if asked_by_admin:
            actor = str(ctx.user.id)
            actor_name = f'bot (a pedido de {ctx.user.display_name})'
        else:
            if self.store.notes_locked(guild.id):
                return 'Suas notas estão travadas pelos admins deste servidor; não dá para mudá-las agora.'
            done = await asyncio.to_thread(self.store.bot_edits_since, guild.id, 'notes')
            if done >= PERSONA_BOT_EDITS_PER_DAY:
                return (
                    f'Você já mudou suas notas {done} vez(es) nas últimas 24h (limite '
                    f'{PERSONA_BOT_EDITS_PER_DAY}). Deixe para depois.'
                )
            actor, actor_name = 'bot', ctx.actor_name
        try:
            version = await self.change(
                guild.id, section, str(args.get('content') or ''), actor=actor, actor_name=actor_name, reason=reason,
            )
        except PersonaError as e:
            return f'⚠️ {e}'
        return (
            f'{SECTIONS[section]["label"]} atualizada (versão {version.version}); vale a partir da próxima '
            'mensagem. Os admins veem a mudança no canal de log e podem reverter.'
        )

    # --- Log channel ---

    def _schedule_log(self, version: Version) -> None:
        if not PERSONA_LOG_CHANNEL_ID:
            return
        task = asyncio.get_running_loop().create_task(self._post_log(version))
        self._log_tasks.add(task)
        task.add_done_callback(self._log_tasks.discard)

    async def _post_log(self, version: Version) -> None:
        try:
            channel = self.bot.get_channel(PERSONA_LOG_CHANNEL_ID) or await self.bot.fetch_channel(PERSONA_LOG_CHANNEL_ID)
            embed = discord.Embed(
                title=f'🎭 Personalidade — {SECTIONS[version.section]["label"]} v{version.version}',
                description=f'Por **{version.actor_name}** — {version.reason or "sem motivo"}',
                color=discord.Color.purple(),
            )
            embed.add_field(name='Antes', value=_snippet(version.before, 1000), inline=False)
            embed.add_field(name='Depois', value=_snippet(version.after, 1000), inline=False)
            embed.set_footer(text=f'mudança #{version.id} • servidor {version.guild_id}')
            await channel.send(embed=embed, view=PersonaLogView(self, version.guild_id, version.id))
        except Exception:
            logger.exception('[persona] failed to post log for change #%s', version.id)

    # --- Slash commands ---

    persona = app_commands.Group(name='persona', description='Personalidade do bot neste servidor', guild_only=True)

    async def _admin(self, interaction: discord.Interaction) -> bool:
        if not can_edit_base(interaction.user):
            await interaction.response.send_message('Requer a permissão Gerenciar Servidor.', ephemeral=True)
            return False
        return True

    @persona.command(name='ver', description='Mostra a personalidade do bot neste servidor')
    async def ver(self, interaction: discord.Interaction):
        gid = interaction.guild.id
        embed = discord.Embed(title='🎭 Personalidade', color=discord.Color.purple())
        for key, s in SECTIONS.items():
            info = self.store.info(gid, key)
            name = s['label'] + (f' (v{info["version"]}, por {info["updated_by_name"]})' if info else '')
            embed.add_field(name=name, value=_snippet(info['content'] if info else '', 1000), inline=False)
        if self.store.notes_locked(gid):
            embed.set_footer(text='🔒 Notas travadas: o bot não pode mudá-las sozinho')
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @persona.command(name='editar', description='Edita a personalidade (Admin)')
    @app_commands.describe(secao='Base (quem o bot é) ou Notas (o que ele foi pegando)')
    @app_commands.choices(secao=_SECTION_CHOICES)
    async def editar(self, interaction: discord.Interaction, secao: str = 'base'):
        if not await self._admin(interaction):
            return
        await interaction.response.send_modal(PersonaModal(self, interaction.guild.id, secao))

    @persona.command(name='historico', description='Últimas mudanças da personalidade (Admin)')
    @app_commands.choices(secao=_SECTION_CHOICES)
    async def historico(self, interaction: discord.Interaction, secao: str | None = None):
        if not await self._admin(interaction):
            return
        versions = await asyncio.to_thread(self.store.history, interaction.guild.id, secao, 10)
        if not versions:
            await interaction.response.send_message('Nenhuma mudança ainda.', ephemeral=True)
            return
        lines = [
            f'`#{v.id}` {SECTIONS[v.section]["label"]} v{v.version} — {v.actor_name} — {v.reason or "sem motivo"}'
            for v in versions
        ]
        embed = discord.Embed(title='🎭 Histórico', description='\n'.join(lines)[:4000], color=discord.Color.purple())
        embed.set_footer(text='/persona reverter <id> desfaz uma mudança')
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @persona.command(name='reverter', description='Desfaz uma mudança da personalidade (Admin)')
    @app_commands.describe(id='Número da mudança (veja /persona historico)')
    async def reverter(self, interaction: discord.Interaction, id: int):
        if not await self._admin(interaction):
            return
        try:
            version = await self.revert(interaction.guild.id, id, actor=interaction.user)
        except PersonaError as e:
            await interaction.response.send_message(f'⚠️ {e}', ephemeral=True)
            return
        await interaction.response.send_message(
            f'↩️ {SECTIONS[version.section]["label"]} voltou ao que era antes da mudança #{id} (versão {version.version}).',
            ephemeral=True,
        )

    @persona.command(name='travar', description='Impede (ou volta a permitir) que o bot mude as próprias notas (Admin)')
    @app_commands.describe(travar='sim: o bot não muda mais as notas sozinho; não: volta a poder')
    async def travar(self, interaction: discord.Interaction, travar: bool):
        if not await self._admin(interaction):
            return
        await asyncio.to_thread(self.store.set_notes_locked, interaction.guild.id, travar)
        await interaction.response.send_message(
            '🔒 Notas travadas.' if travar else '🔓 O bot volta a poder mudar as próprias notas.', ephemeral=True,
        )

    @persona.command(name='importar', description='Move memórias de personalidade para a base (Admin)')
    @app_commands.describe(ids='IDs das memórias do servidor, separados por vírgula (ex.: 3, 12)')
    async def importar(self, interaction: discord.Interaction, ids: str):
        if not await self._admin(interaction):
            return
        memory = self.services.memory
        if memory is None:
            await interaction.response.send_message('A memória está desativada.', ephemeral=True)
            return
        guild = interaction.guild
        wanted = [int(x) for x in re.findall(r'\d+', ids)]
        entries, problems = [], []
        for mid in wanted:
            entry = await memory.get(guild.id, mid)
            if entry is None or entry['status'] != 'active':
                problems.append(f'#{mid} não existe ou está arquivada')
            elif entry['subject']:
                problems.append(f'#{mid} é sobre uma pessoa, não sobre o servidor')
            else:
                entries.append(entry)
        if not entries:
            await interaction.response.send_message('⚠️ Nada importado: ' + '; '.join(problems or ['nenhum ID']), ephemeral=True)
            return
        base = self.store.get(guild.id, 'base')
        added = '\n'.join(f"- {e['content']}" for e in entries)
        try:
            version = await self.change(
                guild.id, 'base', f'{base}\n{added}'.strip(),
                actor=str(interaction.user.id), actor_name=interaction.user.display_name,
                reason='Importado das memórias ' + ', '.join(f"#{e['id']}" for e in entries),
            )
        except PersonaError as e:
            await interaction.response.send_message(f'⚠️ {e}', ephemeral=True)
            return
        unpinned = []
        for e in entries:
            if e['pinned']:
                try:
                    await memory.set_pinned(
                        guild.id, e['id'], pinned=False,
                        actor_id=str(interaction.user.id), actor_name=interaction.user.display_name,
                    )
                    unpinned.append(f"#{e['id']}")
                except Exception:
                    logger.exception('[persona] could not unpin memory %s', e['id'])
        text = f'✅ {len(entries)} memória(s) copiada(s) para a base (versão {version.version}).'
        if unpinned:
            text += f" Desafixadas: {', '.join(unpinned)} (arquive-as com /memory delete se quiser)."
        if problems:
            text += ' Ignoradas: ' + '; '.join(problems) + '.'
        await interaction.response.send_message(text, ephemeral=True)


async def setup(bot: commands.Bot):
    if not PERSONA_ENABLED:
        logger.info('Personality disabled via PERSONA_ENABLED=false')
        return
    await bot.add_cog(Persona(bot))
