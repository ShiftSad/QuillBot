"""Server Wrapped: the numbers, the schedule and automatic posting."""

import datetime
from types import SimpleNamespace as NS

from cogs import wrapped as wrapped_mod
from cogs.history_rag import compute_wrapped
from config import BR_TZ
from core.settings import SettingsStore

from helpers import check

UTC = datetime.timezone.utc


def chunk(mid, author, ts, content='', channel='geral', reply_to=None):
    return {
        'msg_id': str(mid), 'author_id': str(author), 'author_full': {1: 'Ana (@ana)', 2: 'Bob (@bob)', 3: 'Caio (@caio)'}[author],
        'channel_name': channel, 'content': content, 'reply_to': str(reply_to) if reply_to else None,
        'ts': ts.isoformat(), 'jump_url': f'https://discord.com/channels/1/10/{mid}',
    }


def test_compute_wrapped_counts():
    base = datetime.datetime(2026, 10, 26, 22, 0, tzinfo=UTC)  # 19h BRT, a Monday
    chunks = [
        chunk(1, 1, base, 'bom dia :kek: 😂'),
        chunk(2, 2, base + datetime.timedelta(minutes=1), 'oi ana', reply_to=1),
        chunk(3, 1, base + datetime.timedelta(minutes=2), 'oi bob :kek:', channel='off', reply_to=2),
        chunk(4, 2, base + datetime.timedelta(minutes=3), 'e aí', reply_to=1),
        chunk(5, 3, base - datetime.timedelta(days=30), 'velha'),
    ]
    reactions = [('2', '👍', '1'), ('2', '👍', '3'), ('2', ':kek:', '3'), ('1', '👍', '2'), ('999', '👍', '1')]
    stats = compute_wrapped(chunks, reactions, base - datetime.timedelta(days=7), base + datetime.timedelta(days=1))
    check('messages in range only', stats['messages'] == 4 and stats['active_members'] == 2, stats)
    check('top authors', stats['top_authors'] == [('Ana (@ana)', 2), ('Bob (@bob)', 2)], stats['top_authors'])
    check('channels', stats['top_channels'][0] == ('geral', 3))
    check('busiest hour in BRT', stats['busiest_hour'] == (19, 4), stats['busiest_hour'])
    check('weekday in Portuguese', stats['busiest_weekday'] == ('segunda', 4), stats['busiest_weekday'])
    check('who replies to whom', stats['top_pairs'][0] == ('Bob (@bob)', 'Ana (@ana)', 2), stats['top_pairs'])
    check('emojis from text and reactions', dict(stats['top_emojis'])[':kek:'] == 3 and dict(stats['top_emojis'])['👍'] == 4, stats['top_emojis'])
    check('most reacted message first, unknown ids skipped',
          [m['count'] for m in stats['most_reacted']] == [3, 1] and stats['most_reacted'][0]['author'] == 'Bob (@bob)', stats['most_reacted'])
    text = wrapped_mod.stats_text(stats, 'semana')
    check('names, never pings', 'Ana (@ana)' in text and '<@' not in text)


def test_periods_and_schedule():
    monday = datetime.datetime(2026, 11, 2, 10, 30, tzinfo=BR_TZ)
    since, until, label, key = wrapped_mod.previous_period('weekly', monday)
    check('last full week', since == datetime.datetime(2026, 10, 26, tzinfo=BR_TZ) and until == datetime.datetime(2026, 11, 2, tzinfo=BR_TZ), (since, until))
    check('labelled', label == 'semana de 26/10 a 01/11', label)
    check('due on Monday after the hour', wrapped_mod.due_key('weekly', monday) == key)
    check('not before the hour', wrapped_mod.due_key('weekly', monday.replace(hour=8)) is None)
    check('not on Tuesday', wrapped_mod.due_key('weekly', monday + datetime.timedelta(days=1)) is None)
    first = datetime.datetime(2026, 11, 1, 12, tzinfo=BR_TZ)
    since, until, label, key = wrapped_mod.previous_period('monthly', first)
    check('last month', since.month == 10 and until.month == 11 and label == 'outubro de 2026' and key == '2026-10', (since, until, label))
    check('monthly due on the 1st', wrapped_mod.due_key('monthly', first) == '2026-10')
    s, u, lbl = wrapped_mod.parse_period('14d', first)
    check('relative days', (u - s).days == 14 and lbl == 'últimos 14 dias')


async def test_automatic_recap_posts_once_per_period(tmp_path):
    settings = SettingsStore(str(tmp_path / 's.db'))
    settings.load()
    settings.write(1, 0, 'wrapped', 'weekly')
    settings.write(1, 0, 'wrapped_channel', '55')
    posted = []
    channel = NS(send=lambda **kw: _record(posted, kw))
    guild = NS(id=1, name='Miners', get_channel=lambda cid: channel if cid == 55 else None)

    async def stats(guild_id, since, until):
        return compute_wrapped([], [], since, until)

    services = NS(settings=settings, history=NS(wrapped_stats=stats), llm=NS(available=False), memory=None, tools=None)
    cog = wrapped_mod.Wrapped.__new__(wrapped_mod.Wrapped)
    cog.bot = NS(get_guild=lambda gid: guild if gid == 1 else None)
    cog.services = services
    monday = datetime.datetime(2026, 11, 2, 11, 0, tzinfo=BR_TZ)
    check('posted', await cog.post_due(monday) == 1 and posted)
    check('as an embed that pings nobody', posted[0]['embed'].title == '📊 Wrapped — Miners' and not posted[0]['allowed_mentions'].users)
    check('not twice for the same week', await cog.post_due(monday + datetime.timedelta(hours=3)) == 0)


async def _record(posted, kwargs):
    posted.append(kwargs)
    return NS(id=1)


async def test_tool_returns_the_recap_text():
    async def stats(guild_id, since, until):
        return compute_wrapped([chunk(1, 1, until - datetime.timedelta(hours=1), 'oi')], [], since, until)

    cog = wrapped_mod.Wrapped.__new__(wrapped_mod.Wrapped)
    cog.services = NS(history=NS(wrapped_stats=stats))
    text = await cog._run_tool(NS(guild=NS(id=1)), {'periodo': 'semana'})
    check('recap text', text.startswith('Retrospectiva (últimos 7 dias): 1 mensagens de 1 membros.'), text)
    check('bad period explained', 'Período inválido' in await cog._run_tool(NS(guild=NS(id=1)), {'periodo': 'ontem'}))
    cog.services = NS(history=None)
    check('history off', 'não está disponível' in await cog._run_tool(NS(guild=NS(id=1)), {}))
