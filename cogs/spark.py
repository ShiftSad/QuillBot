"""SparkAnalyzer cog — Spark profiler report analysis for the Discord bot.

Responsibilities
----------------
- ``/spark <url>`` slash command: fetch the report and run the chat engine's
  Spark analysis (``bot.services.chat.run_spark``); replies continue it.
- The ``get_spark_detail`` / ``get_config_key`` tools, offered while a report
  is loaded in the conversation.
- Passive URL detection (a router observer, off where the channel's
  ``log_detection`` setting is off): ``spark.lucko.me`` links get an
  "Analyze with AI" button via ``SparkAnalyzeView``.
- Manages its own ``aiohttp.ClientSession`` with a 30-second timeout (Spark
  JSON service can be slow on first fetch).
"""

from __future__ import annotations

import json
import logging

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands

from cogs.spark_parser import (
    AVAILABLE_SECTIONS as _SPARK_SECTIONS,
    SPARK_URL_PATTERN,
    SparkReport,
    build_detail,
    fetch_report,
)
from config import COOLDOWN_PER, COOLDOWN_RATE
from core.tools import Tool

logger = logging.getLogger(__name__)

OBSERVER_NAME = 'spark-links'

_SPARK_SECTIONS_STR = ', '.join(f'"{s}"' for s in _SPARK_SECTIONS)
SPARK_TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'get_spark_detail',
            'description': (
                'Retorna dados detalhados de uma seção específica do relatório Spark '
                'atualmente carregado. Use para obter informações além do resumo inicial. '
                f'Seções disponíveis: {_SPARK_SECTIONS_STR}. '
                'Use "hotspots" para identificar gargalos na árvore de chamadas completa. '
                'Use "jvm" para verificar GC e flags da JVM. '
                'Use "profiler" para estatísticas TPS/MSPT por janela de tempo. '
                'Use "configs:<arquivo>" para obter um arquivo de configuração específico '
                '(ex: "configs:server.properties", "configs:paper/").'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'section': {
                        'type': 'string',
                        'description': (
                            'Nome da seção a retornar. Use "configs:<arquivo>" para '
                            'configurações por arquivo (ex: "configs:spigot.yml").'
                        ),
                    },
                },
                'required': ['section'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_config_key',
            'description': (
                'Lê uma única chave de configuração do servidor sem buscar o arquivo inteiro. '
                'Use quando só precisar verificar um valor específico antes de recomendar uma mudança. '
                'Mais eficiente que get_spark_detail para consultas pontuais.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'file': {
                        'type': 'string',
                        'description': (
                            'Nome do arquivo de configuração '
                            '(ex: "server.properties", "spigot.yml", "paper/").'
                        ),
                    },
                    'key': {
                        'type': 'string',
                        'description': 'Nome da chave de configuração a consultar.',
                    },
                },
                'required': ['file', 'key'],
            },
        },
    },
]

_SECTION_NAMES = {
    'hotspots': 'árvore de hotspots CPU',
    'jvm': 'flags JVM e GC',
    'profiler': 'estatísticas TPS/MSPT',
    'configs': 'arquivos de configuração',
    'game_rules': 'game rules',
    'world': 'dados de mundo',
    'plugins': 'plugins',
}


async def _spark_detail(ctx, args: dict) -> str:
    if ctx.spark_report is None:
        return 'Nenhum relatório Spark carregado para esta conversa.'
    return build_detail(ctx.spark_report, args.get('section', ''))


async def _config_key(ctx, args: dict) -> str:
    report = ctx.spark_report
    if report is None:
        return 'Nenhum relatório Spark carregado para esta conversa.'
    file_name = args.get('file', '')
    key = args.get('key', '')
    cfg = report.configs.get(file_name)
    if cfg is None:
        available = ', '.join(report.configs.keys()) or 'none'
        return f'Arquivo "{file_name}" não encontrado. Disponíveis: {available}'
    if isinstance(cfg, dict):
        if key not in cfg:
            return f'Chave "{key}" não encontrada em "{file_name}".'
        return f'{file_name}/{key} = {json.dumps(cfg[key])}'
    return f'{file_name} = {cfg}'


def _detail_status(args: dict, ctx) -> str:
    section = args.get('section', '')
    if section.startswith('configs:'):
        return f"📂 Lendo configuração: `{section[len('configs:'):]}`"
    return f'📊 Analisando relatório Spark — {_SECTION_NAMES.get(section, section)}'


def _config_status(args: dict, ctx) -> str:
    return f"⚙️ Verificando `{args.get('key', '')}` em `{args.get('file', '')}`"


# ---------------------------------------------------------------------------
# Button view — offered when a Spark URL is detected passively in chat
# ---------------------------------------------------------------------------

class SparkAnalyzeView(discord.ui.View):
    """One-time button that triggers AI analysis of a detected Spark report."""

    def __init__(self, cog: SparkAnalyzer, code: str):
        super().__init__(timeout=300)
        self.cog = cog
        self.code = code
        self.message: discord.Message | None = None
        self._used = False

    @discord.ui.button(
        label='Analisar com IA',
        style=discord.ButtonStyle.primary,
        emoji='🔥',
    )
    async def analyze(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        # Set before the first await: concurrent clicks must not each start
        # an expensive SPARK_MODEL analysis before the button is disabled.
        if self._used:
            await interaction.response.send_message(
                'Esta análise já foi solicitada.', ephemeral=True
            )
            return
        self._used = True
        button.disabled = True
        await interaction.response.edit_message(view=self)
        await self.cog._do_spark_analysis(interaction, self.code, followup=True)

    async def on_timeout(self) -> None:
        for item in self.children:
            item.disabled = True  # type: ignore[attr-defined]
        if self.message is not None:
            try:
                await self.message.edit(view=self)
            except discord.HTTPException:
                pass


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class SparkAnalyzer(commands.Cog, name='SparkAnalyzer'):
    """Spark profiler report analysis commands and passive detection."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.services = bot.services
        self.session: aiohttp.ClientSession | None = None

    async def cog_load(self) -> None:
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
        )
        self.services.tools.register(
            Tool('get_spark_detail', SPARK_TOOLS[0], _spark_detail, _detail_status),
            Tool('get_config_key', SPARK_TOOLS[1], _config_key, _config_status),
        )
        self.services.router.observe(OBSERVER_NAME, self.detect_links)

    async def cog_unload(self) -> None:
        self.services.router.remove(OBSERVER_NAME)
        self.services.tools.unregister('get_spark_detail', 'get_config_key')
        if self.session:
            await self.session.close()
            self.session = None

    # --- /spark command ---

    @app_commands.command(
        name='spark',
        description='Analisa um relatório do Spark Profiler com IA',
    )
    @app_commands.checks.cooldown(COOLDOWN_RATE, COOLDOWN_PER)
    @app_commands.describe(
        url='URL ou código do relatório (ex: https://spark.lucko.me/ABC ou apenas ABC)',
    )
    async def spark(self, interaction: discord.Interaction, url: str) -> None:
        await interaction.response.defer(thinking=True)
        try:
            await interaction.edit_original_response(content='🔥 Carregando relatório Spark…')
        except discord.HTTPException:
            pass
        logger.info(
            "Processing /spark user=%s guild=%s url=%r",
            interaction.user.id, interaction.guild_id, url[:80],
        )
        await self._do_spark_analysis(interaction, url, followup=True)

    # --- Core analysis flow ---

    async def _do_spark_analysis(
        self,
        interaction: discord.Interaction,
        url_or_code: str,
        *,
        followup: bool = False,
    ) -> None:
        """Fetch the Spark report and delegate LLM analysis to DocsRAG.

        Parameters
        ----------
        interaction:
            The active Discord interaction (must already be deferred or
            responded to before calling with ``followup=True``).
        url_or_code:
            Full ``spark.lucko.me`` URL or a bare report code.
        followup:
            When ``True``, uses ``interaction.followup.send`` instead of
            ``interaction.response.send_message``.
        """
        async def _send(content: str, *, ephemeral: bool = False) -> None:
            if followup:
                try:
                    await interaction.edit_original_response(content=None)
                except discord.HTTPException:
                    pass
                await interaction.followup.send(content, ephemeral=ephemeral)
            else:
                await interaction.response.send_message(content, ephemeral=ephemeral)

        # Fetch report -------------------------------------------------------
        if followup:
            try:
                await interaction.edit_original_response(content='🔥 Baixando relatório Spark…')
            except discord.HTTPException:
                pass
        try:
            report: SparkReport = await fetch_report(url_or_code, session=self.session)
        except ValueError as exc:
            await _send(f'❌ {exc}', ephemeral=True)
            return
        except RuntimeError as exc:
            await _send(
                f'❌ Falha ao carregar o relatório Spark: {exc}', ephemeral=True
            )
            return
        except Exception:
            logger.exception("Unexpected error fetching Spark report for %r", url_or_code)
            await _send(
                '❌ Erro inesperado ao buscar o relatório. Tente novamente.',
                ephemeral=True,
            )
            return

        chat = self.services.chat
        if chat is None or not self.services.llm.available:
            await _send(
                '⚠️ A análise com IA não está disponível agora.',
                ephemeral=True,
            )
            return

        if followup:
            try:
                await interaction.edit_original_response(content='🤖 Analisando desempenho com IA…')
            except discord.HTTPException:
                pass
        await chat.run_spark(interaction, report)

    # --- Passive URL detection (router observer) ---

    async def detect_links(self, message: discord.Message) -> None:
        """Detect Spark URLs in chat and offer an analysis button."""
        if message.author.bot:
            return
        match = SPARK_URL_PATTERN.search(message.content or '')
        if not match:
            return
        if not self.services.settings.get(getattr(message.guild, 'id', None), message.channel, 'log_detection'):
            return
        code = match.group(1)
        view = SparkAnalyzeView(self, code)
        reply = await message.reply(
            '🔥 Relatório Spark detectado! Deseja analisar com IA?',
            view=view,
            mention_author=False,
        )
        view.message = reply


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(SparkAnalyzer(bot))
