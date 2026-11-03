"""Episodes: what the bot remembers of conversations that ended.

Conversations expire after ``CONVERSATIONS_TTL_SECONDS`` without activity,
and memory only kept one-line facts, so the bot forgot what it talked about.
Now an expired conversation is summarized by a cheap model into one memory of
kind ``episode`` per participant ("Conversa de 03/11/2026 com Ana: …; Ana ia
testar o backup"). Memories are injected when that person talks again and the
subject comes up, so the bot can pick up where it left off.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

from config import BR_TZ, EPISODES_MODEL

logger = logging.getLogger(__name__)

NOTHING = 'NADA'
_TRANSCRIPT_CHARS = 12000
_TURN_CHARS = 1500

EPISODE_SYSTEM_PROMPT = (
    'Você resume conversas que um bot do Discord teve com membros de um servidor, para o bot '
    'se lembrar delas depois. Escreva uma ou duas frases em português, em terceira pessoa, '
    'citando as pessoas pelo nome: do que falaram, o que foi resolvido e o que ficou pendente '
    'ou combinado (ex.: alguém ia testar algo, voltar com um log). Não invente nada e não '
    'repita detalhes técnicos longos. Se a conversa não tiver nada que valha lembrar (só '
    f'cumprimentos, risadas, uma pergunta trivial já respondida), responda exatamente {NOTHING}.'
)


def _strip_header(question: str) -> str:
    """Drop the bracketed routing header (monitor, thread, name) from a stored question."""
    if question.startswith('[') and '\n---\n' in question:
        return question.split('\n---\n', 1)[1]
    return question


def transcript(data: dict) -> str:
    """The conversation as plain lines for the summarizer."""
    lines: list[str] = []
    for turn in data.get('turns', []):
        author = turn.get('author') or {}
        name = author.get('display') or author.get('name') or 'alguém'
        when = ''
        if turn.get('ts'):
            when = datetime.datetime.fromtimestamp(float(turn['ts']), BR_TZ).strftime('%d/%m %H:%M') + ' '
        question = _strip_header(str(turn.get('question') or ''))[:_TURN_CHARS]
        answer = str(turn.get('answer') or '')[:_TURN_CHARS]
        lines.append(f'{when}{name}: {question}\nBot: {answer}')
    text = '\n\n'.join(lines)
    return text[-_TRANSCRIPT_CHARS:]


class EpisodeWriter:
    """Summarizes expired conversations into episode memories."""

    def __init__(self, services: Any, *, model: str = EPISODES_MODEL):
        self.services = services
        self.model = model

    async def run_once(self, limit: int = 10) -> int:
        """Summarize up to *limit* expired conversations; returns how many were handled."""
        store = self.services.conversations
        handled = 0
        for conv in await store.expired_pending(limit):
            try:
                state = await self.summarize(conv)
            except Exception:
                logger.exception('[episodes] failed to summarize conversation %s', conv['conv_id'])
                continue
            await store.mark_episode(conv['conv_id'], state)
            handled += 1
        return handled

    async def summarize(self, conv: dict) -> str:
        """Write the episode of *conv*; returns ``done`` or ``skipped``."""
        memory = self.services.memory
        data = conv.get('data') or {}
        guild_id = conv.get('guild_id') or (data.get('origin') or {}).get('guild_id')
        people = [p for p in data.get('participants', []) if isinstance(p, dict) and str(p.get('id', '')).isdigit()]
        if memory is None or not guild_id or not people or not data.get('turns'):
            return 'skipped'
        names = ', '.join(p.get('display') or p.get('name') or p['id'] for p in people)
        text = await self.services.llm.text(
            model=self.model,
            system=EPISODE_SYSTEM_PROMPT,
            user=f'Participantes: {names}\n\nConversa:\n{transcript(data)}',
            max_tokens=1024,
        )
        text = ' '.join(text.split())
        if not text or text.upper().strip(' .!') == NOTHING or text.upper().startswith(NOTHING + ' '):
            return 'skipped'
        started = data.get('started_ts') or (data['turns'][0].get('ts'))
        date = datetime.datetime.fromtimestamp(float(started), BR_TZ).strftime('%d/%m/%Y') if started else 'data desconhecida'
        content = f'Conversa de {date} com {names}: {text}'
        origin = ''
        thread_id = data.get('thread_id')
        if thread_id:
            origin = f'https://discord.com/channels/{guild_id}/{thread_id}'
        await memory.record_episode(int(guild_id), [str(p['id']) for p in people], content, origin=origin)
        logger.info('[episodes] conversation %s -> %r', conv['conv_id'], content[:120])
        return 'done'
