"""Sending answers: plain messages, embeds or a thread — decided in one place.

The bot used to answer everything with a teal embed, even "valeu!". Now
answers go out as ordinary messages, like any member's, and embeds are kept
for what reads better framed: Spark reports, channel summaries and recaps,
and answers too long for one message. A channel can force one or the other
(``reply_style`` in ``/config``).

Pings: ``@Name`` written by the model becomes a real ``<@id>`` mention, and
only those human members may be pinged — never roles, ``@everyone`` or bots.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import discord

from config import CHAT_PLAIN_MAX_CHARS
from core.discord_text import resolve_mentions, split_response
from core.ui import PaginatedEmbedView, build_source_pages

logger = logging.getLogger(__name__)

MESSAGE_LIMIT = 2000
_EMBED_PAGE = 3500
_MAX_PLAIN_SOURCES = 5
_FENCE_RE = re.compile(r'^\s*```')


@dataclass
class Outgoing:
    """An answer ready to be sent, with how it should look."""

    answer: str
    sources: list[dict] = field(default_factory=list)
    style: str = 'plain'  # 'plain' | 'embed'
    title: str | None = None
    color: int = 0x1ABC9C
    footer: str = ''
    sources_title: str = '🌐 Fontes'


def choose_style(answer: str, *, preference: str = 'auto', structured: bool = False) -> str:
    """``plain`` or ``embed`` for *answer* under a channel's ``reply_style``."""
    if preference in ('plain', 'embed'):
        return preference
    if structured or len(answer) > CHAT_PLAIN_MAX_CHARS:
        return 'embed'
    return 'plain'


def split_plain(text: str, limit: int = MESSAGE_LIMIT) -> list[str]:
    """Split *text* into messages of at most *limit* chars at line breaks.

    A code block cut in half is closed at the end of one message and reopened
    (same language) at the start of the next, so both render.
    """
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    fence: str | None = None  # opening line of the code block we're inside

    def flush() -> None:
        nonlocal current, size
        if not current:
            return
        body = '\n'.join(current)
        if fence is not None:
            body += '\n```'
        chunks.append(body)
        current = [fence] if fence is not None else []
        size = len(fence) + 1 if fence is not None else 0

    for line in text.split('\n'):
        reserve = 4 if fence is not None else 0
        while len(line) > limit - 8:
            if current:
                flush()
            head, line = line[:limit - 8], line[limit - 8:]
            current.append(head)
            size += len(head) + 1
            flush()
        if current and size + len(line) + 1 + reserve > limit:
            flush()
        current.append(line)
        size += len(line) + 1
        if _FENCE_RE.match(line):
            fence = None if fence is not None else line.strip()
    if current and any(part.strip() for part in current):
        chunks.append('\n'.join(current))
    return [c for c in chunks if c.strip()]


def sources_line(sources: list[dict]) -> str:
    """Compact ``-# Fontes:`` subtext with link previews suppressed."""
    links = []
    for s in sources[:_MAX_PLAIN_SOURCES]:
        title = str(s.get('title') or s.get('url') or '').replace('[', '(').replace(']', ')')[:60]
        links.append(f"[{title}](<{s['url']}>)")
    return '-# Fontes: ' + ' · '.join(links) if links else ''


def _allowed(ids: list[int], *, mention_author: bool | None = None) -> discord.AllowedMentions:
    kwargs: dict[str, Any] = {
        'users': [discord.Object(id=i) for i in ids],
        'roles': False,
        'everyone': False,
    }
    if mention_author is not None:
        kwargs['replied_user'] = mention_author
    return discord.AllowedMentions(**kwargs)


def render(out: Outgoing, guild: Any, *, prefix: str = '', extra_ping: int | None = None) -> list[dict]:
    """Send kwargs for each message carrying *out* (``prefix`` goes on the first)."""
    text, ping_ids = resolve_mentions(out.answer, guild)
    if extra_ping and extra_ping not in ping_ids:
        ping_ids = [extra_ping, *ping_ids]
    if out.style == 'embed':
        pages = split_response(text, _EMBED_PAGE) or ['']
        embeds = []
        for i, page in enumerate(pages):
            e = discord.Embed(description=page, color=out.color)
            if i == 0 and out.title:
                e.title = out.title[:256]
            footer = out.footer
            if len(pages) > 1:
                footer = f'Página {i + 1}/{len(pages)}' + (f' • {footer}' if footer else '')
            if footer:
                e.set_footer(text=footer[:2048])
            embeds.append(e)
        source_lines = [
            f"• [{s['title']}]({s['url']})" + (f" — {s['source']}" if s.get('source') else '')
            for s in out.sources
        ]
        embeds.extend(build_source_pages(
            source_lines, title=out.sources_title, color=discord.Color(out.color), footer_base=out.footer,
        ))
        # Mentions inside embeds render but never notify: the pings ride as content.
        content = ' '.join([prefix.strip(), *(f'<@{i}>' for i in ping_ids if f'<@{i}>' not in prefix)]).strip()
        kwargs: dict[str, Any] = {'embed': embeds[0], 'allowed_mentions': _allowed(ping_ids)}
        if content:
            kwargs['content'] = content
        if len(embeds) > 1:
            kwargs['view'] = PaginatedEmbedView(embeds)
        return [kwargs]

    body = f'{prefix}{text}' if prefix else text
    footer = sources_line(out.sources)
    if footer and len(body) + len(footer) + 1 <= MESSAGE_LIMIT:
        body = f'{body}\n{footer}'
        footer = ''
    chunks = split_plain(body) or ['…']
    if footer:
        chunks.append(footer)
    return [{'content': c, 'allowed_mentions': _allowed(ping_ids), 'suppress_embeds': True} for c in chunks]


async def deliver(
    out: Outgoing,
    *,
    guild: Any,
    reply_to: Any = None,
    channel: Any = None,
    thread: Any = None,
    asker: Any = None,
    interaction: Any = None,
    mention_author: bool | None = None,
) -> list[Any]:
    """Send *out* and return the messages sent.

    Where it goes: a *thread* (pinging *asker*, which also adds them to it),
    a slash-command *interaction*, a reply to *reply_to*, or *channel*. Long
    plain answers become several messages; only the first is a reply.
    """
    sent: list[Any] = []
    if thread is not None:
        prefix = f'{asker.mention} ' if asker is not None else ''
        for kwargs in render(out, guild, prefix=prefix, extra_ping=getattr(asker, 'id', None)):
            sent.append(await thread.send(**kwargs))
        return sent
    messages = render(out, guild)
    if interaction is not None:
        for kwargs in messages:
            sent.append(await interaction.followup.send(wait=True, **kwargs))
        return sent
    target_channel = channel or getattr(reply_to, 'channel', None)
    for i, kwargs in enumerate(messages):
        if i == 0 and reply_to is not None:
            if mention_author is not None:
                kwargs['mention_author'] = mention_author
            sent.append(await reply_to.reply(**kwargs))
        else:
            sent.append(await target_channel.send(**kwargs))
    return sent
