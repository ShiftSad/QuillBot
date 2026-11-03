"""Discord text helpers: mentions and pings, the bot's identity, message lines.

Everything that turns Discord messages into text for the model (the canonical
message line) or turns model text into something safe to post (pings) lives
here.
"""

from __future__ import annotations

import datetime
import logging
import re
import unicodedata
from typing import Final

import discord

from config import BR_TZ
from core import message_media

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ping-safe user mention extraction
# ---------------------------------------------------------------------------

_MENTION_USER_RE = re.compile(r'<@!?(\d{15,25})>')
# '@' not preceded by a word char or another '@' (skips emails, @@) — single
# token; multi-word display names are resolved by extending the match.
_AT_MENTION_RE = re.compile(r'(?<![\w@])@([\w][\w.\-]{0,31})')
# Never pingable as @tokens: Discord pings and PT-BR "everyone" shorthand.
_PING_BLACKLIST = {'here', 'everyone', 'all', 'todos', 'todo'}


def strip_bot_mention(text: str, bot_user: discord.abc.User, guild: discord.Guild | None = None) -> str:
    """Remove the bot's *addressing* mention from *text*, keeping in-sentence ones.

    A leading mention ("@Bot como...") and a trailing one that follows a
    sentence end ("...? @Bot") only address the bot and are dropped. Any other
    mention is part of the sentence ("pode usar o @Bot") and becomes
    ``@<name>`` so the model still sees the word instead of a hole.
    """
    mention = rf'<@!?{bot_user.id}>'
    text = re.sub(rf'\A\s*(?:{mention}[\s,;:]*)+', '', text)
    text = re.sub(rf'(?<=[.!?…,;:])\s*(?:{mention}\s*)+\Z', '', text)
    name = (guild.me.display_name if guild and guild.me else None) or bot_user.display_name
    return re.sub(mention, f'@{name}', text)


def build_identity_block(bot_user: discord.abc.User | None, guild: discord.Guild | None = None) -> str:
    """System-prompt block telling the model who *it* is and where it is.

    Without it the bot's own earlier replies show up in channel context as a
    third-party ``Nome (@handle) <author_id=…>`` line and ``@<nome>`` in a
    question reads like another participant.  Only stable fields (no member
    counts, no clock) so the system prompt stays prefix-cacheable.
    """
    if bot_user is None:
        return ''
    name = (guild.me.display_name if guild and guild.me else None) or bot_user.display_name
    handle = getattr(bot_user, 'name', None)
    who = f'{name} (@{handle})' if handle and handle != name else name
    lines = [
        '<identidade>',
        f'Você é o bot do Discord "{who}", id={bot_user.id}. Quando alguém diz "{name}" ou "@{name}", '
        f'ou menciona <@{bot_user.id}>, está falando com você ou sobre você.',
        f'Mensagens com author_id={bot_user.id} (nos blocos de mensagens do canal e no histórico) '
        'são SUAS respostas anteriores — não as trate como de outro usuário ou outro bot.',
    ]
    if guild is not None:
        lines.append(f'Servidor atual: {guild.name} (id={guild.id}).')
        description = _clip(_flatten(getattr(guild, 'description', None) or ''), 300)
        if description:
            lines.append(f'Descrição do servidor: {description}')
        owner_id = getattr(guild, 'owner_id', None)
        if owner_id:
            lines.append(f'Dono do servidor: <@{owner_id}> (id={owner_id}).')
    else:
        lines.append('Conversa em mensagem direta (sem servidor).')
    lines.append('</identidade>')
    return '\n'.join(lines)


def _norm_name(text: str) -> str:
    text = unicodedata.normalize('NFKD', text or '')
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return re.sub(r'\s+', ' ', text).strip().lower()


def extract_pingable_mentions(text: str, guild: discord.Guild | None, *, limit: int = 10) -> str:
    """Collect ping-safe user mentions from LLM output as a content string.

    Discord only notifies for mentions in the message *content* — mentions
    inside embeds render but never ping.  This helper gathers the users the
    model referenced and returns them as ``'<@id> <@id>'`` for use as
    ``content=`` alongside the embed.

    Accepts both ``<@user_id>`` mentions and ``@username`` / ``@display name``
    tokens (resolved against guild members, accent/case-insensitive, extending
    up to 3 words for names with spaces).  Only real human members ping:
    roles, ``@here``/``@everyone``, bots (including this bot — prevents
    self-ping loops) and unresolvable names are silently dropped.
    """
    if not text or guild is None:
        return ''
    picked: list[str] = []

    def _add(member: discord.Member | None) -> None:
        if member is None or member.bot:
            return
        sid = str(member.id)
        if sid not in picked:
            picked.append(sid)

    def _resolve(name: str) -> discord.Member | None:
        member = guild.get_member_named(name)
        if member is not None:
            return member
        target = _norm_name(name)
        return next(
            (
                m for m in guild.members
                if not m.bot and target in (_norm_name(m.display_name), _norm_name(m.name))
            ),
            None,
        )

    for uid in _MENTION_USER_RE.findall(text):
        _add(guild.get_member(int(uid)))
        if len(picked) >= limit:
            return ' '.join(f'<@{i}>' for i in picked)

    for match in _AT_MENTION_RE.finditer(text):
        if len(picked) >= limit:
            break
        raw = match.group(1)
        if raw.lower() in _PING_BLACKLIST:
            continue
        member = _resolve(raw)
        if member is None:
            # Multi-word display name: greedily extend with following words,
            # stripping trailing punctuation ("@John Doe," -> "John Doe").
            extended = raw
            for word in text[match.end():].lstrip().split()[:2]:
                extended = f'{extended} {word.rstrip(".,;:!?…")}'
                member = _resolve(extended)
                if member is not None:
                    break
        _add(member)
    return ' '.join(f'<@{i}>' for i in picked)


_CODE_SPAN_RE = re.compile(r'```.*?(?:```|\Z)|`[^`\n]*`', re.S)
_TRAILING_PUNCT = '.,;:!?…)'


def _member_resolver(guild: discord.Guild):
    def _resolve(name: str) -> discord.Member | None:
        member = guild.get_member_named(name)
        if member is not None:
            return member
        target = _norm_name(name)
        return next(
            (
                m for m in guild.members
                if not m.bot and target in (_norm_name(m.display_name), _norm_name(m.name))
            ),
            None,
        )
    return _resolve


def resolve_mentions(text: str, guild: discord.Guild | None, *, limit: int = 10) -> tuple[str, list[int]]:
    """Turn ``@Name`` in model text into real ``<@id>`` mentions; return who to ping.

    Like :func:`extract_pingable_mentions`, but for plain messages: the
    resolved ``@name`` spans are rewritten in place (so the name renders as a
    mention), and the returned ids — human members only, at most *limit* —
    are the ones allowed to ping. Code spans are left alone; roles,
    ``@everyone``/``@here`` and bots never ping.
    """
    if not text or guild is None:
        return text, []
    resolve = _member_resolver(guild)
    code = [m.span() for m in _CODE_SPAN_RE.finditer(text)]

    def in_code(pos: int) -> bool:
        return any(a <= pos < b for a, b in code)

    picked: list[int] = []

    def add(member: discord.Member | None) -> bool:
        if member is None or member.bot:
            return False
        if member.id not in picked and len(picked) < limit:
            picked.append(member.id)
        return member.id in picked

    for match in _MENTION_USER_RE.finditer(text):
        if not in_code(match.start()):
            add(guild.get_member(int(match.group(1))))

    out: list[str] = []
    last = 0
    for match in _AT_MENTION_RE.finditer(text):
        if match.start() < last or in_code(match.start()):
            continue
        raw = match.group(1)
        if raw.lower() in _PING_BLACKLIST:
            continue
        # Candidate names: "@Ana", "@Ana Paula", "@Ana Paula Souza" — longest wins,
        # so a multi-word display name isn't cut at its first word.
        candidates = [(raw, match.end())]
        name, pos = raw, match.end()
        for word in re.finditer(r'[ \t]+(\S+)', text[pos:]):
            if len(candidates) >= 3 or match.end() + word.start() != pos:
                break
            stripped = word.group(1).rstrip(_TRAILING_PUNCT)
            if not stripped or stripped.startswith(('@', '<')):
                break
            name = f'{name} {stripped}'
            candidates.append((name, match.end() + word.start(1) + len(stripped)))
            if stripped != word.group(1):
                break
            pos = match.end() + word.end()
        member, end = None, match.end()
        for candidate, cand_end in reversed(candidates):
            member = resolve(candidate)
            if member is not None:
                end = cand_end
                break
        if member is None or not add(member):
            continue
        out.append(text[last:match.start()])
        out.append(f'<@{member.id}>')
        last = end
    out.append(text[last:])
    return ''.join(out), picked


def ping_send_kwargs(text: str, guild: discord.Guild | None) -> dict:
    """Send kwargs that turn mentions in LLM output *text* into real pings.

    Mentions inside embeds never notify (Discord platform behavior) — the
    extracted user pings ride as message ``content``.  everyone/roles stay
    unparsable so no code path using this can ever ping @everyone.
    """
    kwargs: dict = {
        'allowed_mentions': discord.AllowedMentions(
            users=True, roles=False, everyone=False,
        ),
    }
    mentions = extract_pingable_mentions(text, guild)
    if mentions:
        kwargs['content'] = mentions
    return kwargs


def _fmt_dt(dt: datetime.datetime | None) -> str:
    if not dt:
        return "—"
    try:
        return dt.astimezone(BR_TZ).strftime("%d/%m/%Y %H:%M BRT")
    except Exception:
        return dt.isoformat()


def _fmt_dt_line(dt: datetime.datetime | None) -> str:
    if not dt:
        return "—"
    try:
        return dt.astimezone(BR_TZ).strftime("%d/%m/%Y %H:%M")
    except Exception:
        return dt.isoformat()

# ---------------------------------------------------------------------------
# Canonical message-line format
#
# Every renderer of Discord messages (channel history, message context,
# search results, history chunks, conversation turns) emits lines in this
# exact shape so the LLM can cross-reference messages between tools:
#
#   [dd/mm/aaaa HH:MM] Display (@handle) <author_id=123>: content ↩ reply_to=456 (Alvo) [msg_id=789]
#
# - ``author_id`` feeds search_history/get_user_stats filters;
# - ``reply_to`` (only for replies) feeds get_message_context;
# - ``msg_id`` always comes LAST so content truncation never eats it.
# ---------------------------------------------------------------------------

MESSAGE_LINE_MAX_CONTENT: Final[int] = 800

def _flatten(s: str) -> str:
    return ' '.join(s.split())


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + '…'


def message_content_text(msg: discord.Message, *, max_length: int = MESSAGE_LINE_MAX_CONTENT) -> str:
    """Flattened message content with attachment/sticker/embed markers, never empty.

    Custom emojis are shown as ``:name:`` (the raw ``<:name:id>`` markup is
    noise to the model and to keyword search).
    """
    c = message_media.emoji_text(msg.content or "")
    if msg.attachments:
        c += " " + " ".join(f"[anexo:{a.filename}]" for a in msg.attachments)
    markers = message_media.sticker_markers(msg) + message_media.embed_markers(msg)
    if markers:
        c += " " + " ".join(markers)
    if not c.strip() and msg.embeds:
        try:
            c = f"[embed: {msg.embeds[0].title or msg.embeds[0].description[:100]}]"
        except Exception:
            c = "[embed]"
    c = _flatten(c)
    if not c:
        c = "[sem texto]"
    return _clip(c, max_length)


def format_message_line(msg: discord.Message, *, reactions: str = '') -> str:
    """Render a Discord message in the canonical line format.

    *reactions* (from :func:`core.message_media.reaction_summaries`) goes right
    before ``msg_id``, outside the clipped content.
    """
    display = _flatten(getattr(msg.author, 'display_name', None) or str(msg.author))
    handle = _flatten(getattr(msg.author, 'name', None) or '')
    author = f"{display} (@{handle})" if handle else display
    author_id = getattr(msg.author, 'id', '')
    line = f"[{_fmt_dt_line(msg.created_at)}] {author} <author_id={author_id}>: {message_content_text(msg)}"
    ref = getattr(msg, 'reference', None)
    ref_id = getattr(ref, 'message_id', None)
    if ref_id:
        target = ""
        resolved = getattr(ref, 'resolved', None)
        if resolved is not None and not isinstance(resolved, discord.DeletedReferencedMessage):
            rdisplay = _flatten(getattr(resolved.author, 'display_name', '') or '')
            if rdisplay:
                target = f" ({rdisplay})"
        line += f" ↩ reply_to={ref_id}{target}"
    if reactions:
        line += f" {reactions}"
    line += f" [msg_id={msg.id}]"
    return line


async def format_message_lines(messages: list[discord.Message]) -> list[str]:
    """Canonical lines for *messages* (same order), with their reactions."""
    reactions = await message_media.reaction_summaries(messages)
    return [format_message_line(m, reactions=reactions.get(m.id, '')) for m in messages]


def format_chunk_line(ch: dict) -> str:
    """Render a stored history chunk (cogs/history_rag.py) in the canonical format."""
    ts_raw = str(ch.get('ts', ''))
    try:
        dt = datetime.datetime.fromisoformat(ts_raw.replace('Z', '+00:00'))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        ts = dt.astimezone(BR_TZ).strftime("%d/%m/%Y %H:%M")
    except Exception:
        ts = ts_raw[:19]
    author = _flatten(str(ch.get('author_full') or ch.get('author_name') or '?'))
    content = _clip(_flatten(str(ch.get('content', ''))), MESSAGE_LINE_MAX_CONTENT)
    if not content:
        content = '[sem texto]'
    line = f"[{ts}] {author} <author_id={ch.get('author_id', '?')}>: {content}"
    if ch.get('reply_to'):
        line += f" ↩ reply_to={ch['reply_to']}"
    return f"{line} [msg_id={ch.get('msg_id', '')}]"


def render_search_results(results: list[dict], *, window_chars: int = 1200, include_msg_id: bool = True) -> str:
    """Render search_history results (shared by commands.py and docs_rag.py)."""
    parts: list[str] = []
    for r in results:
        jump = r.get('jump_url', '')
        link = f"[ver]({jump})" if jump else ''
        header = (
            f"**{r.get('author_full', '?')}** em #{r.get('channel_name', '?')} "
            f"(channel_id={r.get('channel_id', '?')}) — {str(r.get('ts', ''))[:19]} "
            f"{link} (score {r.get('_score', 0):.2f})"
        )
        raw_lines = str(r.get('chunk_text', r.get('content', ''))).replace('```', 'ˋˋˋ').split('\n')
        kept: list[str] = []
        used = 0
        dropped = 0
        for line in raw_lines:
            cost = len(line) + (1 if kept else 0)
            if used + cost > window_chars:
                dropped = len(raw_lines) - len(kept)
                break
            kept.append(line)
            used += cost
        if dropped:
            kept.append(f"… (+{dropped} linhas)")
        body = f"{header}\n```\n" + '\n'.join(kept) + "\n```"
        if include_msg_id:
            body += f"\n`msg_id={r.get('msg_id')}`"
        parts.append(body)
    return "\n\n---\n\n".join(parts)


def truncate_safe(text: str, limit: int = 3800, suffix: str = '\n\n...') -> str:
    """Truncate *text* at the last newline before *limit* to avoid breaking markdown."""
    if len(text) <= limit:
        return text
    idx = text.rfind('\n', 0, limit)
    if idx == -1:
        idx = limit
    return text[:idx] + suffix


def split_response(text: str, chunk_size: int = 3500) -> list[str]:
    """Split *text* into chunks of at most *chunk_size* chars, breaking at newlines."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    while text:
        if len(text) <= chunk_size:
            chunks.append(text)
            break
        idx = text.rfind('\n', 0, chunk_size)
        if idx == -1:
            idx = chunk_size
        chunks.append(text[:idx])
        text = text[idx:].lstrip('\n')
    return chunks
