"""Monitors: validation, matching, the store, firing and the bot's monitor tools."""

from types import SimpleNamespace as NS

import pytest

from cogs import monitors as mon_mod
from cogs.monitors import (
    MonitorError,
    Monitors,
    MonitorStore,
    build_trigger_header,
    is_silent,
    monitor_matches,
    normalize_fields,
)
from core.tool_loop import EMPTY_ANSWER_FALLBACK

from helpers import check

BOT_ID = 900
GENERAL, STAFF, THREAD = 10, 20, 21


class FakeGuild:
    id = 1

    def __init__(self):
        self.channels = [NS(id=GENERAL, name='geral'), NS(id=STAFF, name='staff')]
        self.threads = [NS(id=THREAD, name='ajuda', parent_id=STAFF)]
        self.members = {111: NS(id=111, name='nyuu', display_name='Nyuu')}

    def get_channel_or_thread(self, cid):
        return next((c for c in self.channels + self.threads if c.id == cid), None)

    def get_member_named(self, name):
        return next((m for m in self.members.values() if name in (m.name, m.display_name)), None)


def perms(manage):
    return NS(administrator=False, manage_guild=manage)


ADMIN = NS(id=111, display_name='Nyuu', guild_permissions=perms(True))
MEMBER = NS(id=222, display_name='John', guild_permissions=perms(False))


def message(content='', *, channel=GENERAL, parent=None, author=222, bot=False, mentions=(), reference=None, mid=500):
    return NS(
        id=mid,
        content=content,
        channel=NS(id=channel, parent_id=parent),
        author=NS(id=author, bot=bot),
        mentions=[NS(id=u) for u in mentions],
        reference=reference,
        guild=FakeGuild(),
    )


# --- Validation ---------------------------------------------------------------

def test_normalize_resolves_ids_names_and_mentions():
    g = FakeGuild()
    out = normalize_fields({
        'prompt': ' ajude ', 'channels': ['<#10>', '#staff', '10'],
        'authors': '<@!111>', 'mentions': ['@nyuu', 333], 'regex': 'lag',
    }, g, partial=False)
    check('prompt trimmed', out['prompt'] == 'ajude', out)
    check('channels deduped, names resolved', out['channels'] == [GENERAL, STAFF], out)
    check('comma/space string splits', out['authors'] == [111], out)
    check('names and raw ids', out['mentions'] == [111, 333], out)


@pytest.mark.parametrize('args, fragment', [
    ({}, 'prompt'),
    ({'prompt': 'x' * 10_000}, 'prompt excede'),
    ({'prompt': 'x', 'channels': ['999']}, 'não existe'),
    ({'prompt': 'x', 'channels': ['#nope']}, 'não encontrado'),
    ({'prompt': 'x', 'authors': ['@ghost']}, 'não encontrado'),
    ({'prompt': 'x', 'regex': '(unclosed'}, 'regex inválida'),
    ({'prompt': 'x', 'regex': r'(\w+)+$'}, 'aninhado'),
    ({'prompt': 'x', 'regex': 'a' * 500}, 'excede'),
    ({'prompt': 'x', 'cooldown_seconds': 'soon'}, 'inteiro'),
])
def test_normalize_rejects(args, fragment):
    with pytest.raises(MonitorError, match=fragment):
        normalize_fields(args, FakeGuild(), partial=False)


def test_cooldown_is_clamped_and_partial_keeps_only_given_keys():
    out = normalize_fields({'cooldown_seconds': 0, 'regex': '', 'prompt': None}, FakeGuild(), partial=True)
    check('cooldown raised to the minimum', out['cooldown_seconds'] == mon_mod.MONITORS_MIN_COOLDOWN, out)
    check('empty regex clears, missing prompt untouched', out == {'cooldown_seconds': mon_mod.MONITORS_MIN_COOLDOWN, 'regex': ''}, out)


# --- Matching -----------------------------------------------------------------

def test_filters_combine_with_and():
    mon = {'channels': [STAFF], 'authors': [222], 'mentions': [111], 'regex': r'\blag\b'}
    check('all filters pass', monitor_matches(mon, message('LAG aqui', channel=STAFF, mentions=[111])))
    check('thread under a watched channel counts', monitor_matches(mon, message('lag', channel=THREAD, parent=STAFF, mentions=[111])))
    check('other channel', not monitor_matches(mon, message('lag', channel=GENERAL, mentions=[111])))
    check('other author', not monitor_matches(mon, message('lag', channel=STAFF, author=333, mentions=[111])))
    check('no mention', not monitor_matches(mon, message('lag', channel=STAFF)))
    check('regex miss', not monitor_matches(mon, message('flag', channel=STAFF, mentions=[111])))


def test_single_filters():
    check('mention anywhere', monitor_matches({'mentions': [111]}, message('oi', mentions=[333, 111])))
    check('author sends a message', monitor_matches({'authors': [222]}, message('')))
    check('channel only fires on everything there', monitor_matches({'channels': [GENERAL]}, message('x')))
    check('no filter never matches', not monitor_matches({'channels': [], 'regex': ''}, message('x')))


def test_silence_detection():
    for answer in ('[NO_REPLY]', ' no_reply ', 'Nada a fazer. [NO_REPLY]', '', EMPTY_ANSWER_FALLBACK):
        check(f'silent: {answer!r}', is_silent(answer))
    check('a real answer is sent', not is_silent('Tenta reduzir o view-distance.'))
    check('someone who asked hears about a failed answer', not is_silent(EMPTY_ANSWER_FALLBACK, unprompted=False))
    check('but [NO_REPLY] is silent on any turn', is_silent('[NO_REPLY]', unprompted=False))


def test_header_lists_monitors_and_the_way_out():
    header = build_trigger_header([
        {'id': 3, 'name': 'lag', 'prompt': 'ajude com lag', 'created_by_name': 'Nyuu', 'regex': 'lag'},
        {'id': 4, 'name': '', 'prompt': 'avise a staff', 'created_by_name': 'Nyuu', 'mentions': [111]},
    ])
    for part in ('[Monitor disparado', 'Monitor #3 "lag"', 'ajude com lag', 'Monitor #4', '<@111>', '[NO_REPLY]'):
        check(f'header has {part}', part in header, header)


# --- Store --------------------------------------------------------------------

def test_store_roundtrip(tmp_path):
    store = MonitorStore(str(tmp_path / 'monitors.db'))
    store.ensure()
    mon = store.create(
        guild_id=1, fields={'prompt': 'p', 'channels': [GENERAL], 'regex': 'x', 'cooldown_seconds': 30},
        created_by=111, created_by_name='Nyuu',
    )
    check('lists round-trip', mon['channels'] == [GENERAL] and mon['authors'] == [], mon)
    updated = store.update(1, mon['id'], {'channels': [], 'mentions': [111], 'status': 'paused'})
    check('update replaces lists and status', updated['channels'] == [] and updated['mentions'] == [111] and updated['status'] == 'paused', updated)
    check('paused is not active', store.list(1, active_only=True) == [])
    store.mark_fired(1, [mon['id']])
    check('fire count', store.get(1, mon['id'])['fire_count'] == 1)
    store.update(1, mon['id'], {'status': 'deleted'})
    check('deleted is gone', store.get(1, mon['id']) is None and store.count(1) == 0)


# --- Cog ----------------------------------------------------------------------

class FakeChat:
    def __init__(self):
        self.calls = []

    async def answer_monitor(self, message, header):
        self.calls.append((message.id, header))
        return False


def make_cog(tmp_path, chat=None):
    cog = Monitors.__new__(Monitors)
    cog.store = MonitorStore(str(tmp_path / 'monitors.db'))
    cog.store.ensure()
    cog._cache = {}
    cog._last_fired = {}
    cog.services = NS(chat=chat or FakeChat(), llm=NS(available=True))
    cog.bot = NS(user=NS(id=BOT_ID))
    return cog


async def test_bot_manages_monitors_through_tools(tmp_path):
    cog = make_cog(tmp_path)
    g = FakeGuild()
    text, _ = await cog.exec_tool('monitor_create', {'prompt': 'p', 'regex': 'lag'}, guild=g, requester=MEMBER)
    check('members without Manage Server are refused', 'Gerenciar Servidor' in text, text)
    text, _ = await cog.exec_tool('monitor_create', {'prompt': 'p'}, guild=g, requester=ADMIN)
    check('a filter is required', 'ao menos um filtro' in text, text)
    text, _ = await cog.exec_tool('monitor_create', {'prompt': 'ajude', 'name': 'lag', 'regex': 'lag', 'channels': ['#staff']}, guild=g, requester=ADMIN)
    check('created', text.startswith('✅ Monitor #1 criado'), text)
    text, _ = await cog.exec_tool('monitor_update', {'id': 1, 'regex': ''}, guild=g, requester=ADMIN)
    check('clearing one filter keeps the other', 'atualizado' in text and 'regex' not in text.split('—')[1], text)
    text, _ = await cog.exec_tool('monitor_update', {'id': 1, 'channels': []}, guild=g, requester=ADMIN)
    check('the last filter cannot be cleared', 'ao menos um filtro' in text, text)
    text, _ = await cog.exec_tool('monitor_update', {'id': 1, 'status': 'paused'}, guild=g, requester=ADMIN)
    check('paused', '⏸️' in text, text)
    text, _ = await cog.exec_tool('monitor_list', {}, guild=g, requester=MEMBER)
    check('anyone can list', '#1' in text and 'ajude' in text, text)
    text, _ = await cog.exec_tool('monitor_delete', {'id': 1}, guild=g, requester=ADMIN)
    check('deleted', 'removido' in text, text)
    text, _ = await cog.exec_tool('monitor_update', {'id': 1, 'name': 'x'}, guild=g, requester=ADMIN)
    check('deleted monitor is gone', 'não encontrado' in text, text)


async def test_firing_combines_monitors_and_respects_cooldown(tmp_path):
    commands_cog = FakeChat()
    cog = make_cog(tmp_path, commands_cog)
    g = FakeGuild()
    await cog.create(g, ADMIN, {'prompt': 'ajude com lag', 'regex': 'lag'})
    await cog.create(g, ADMIN, {'prompt': 'avise a staff', 'mentions': ['111'], 'cooldown_seconds': 3600})
    await cog.create(g, ADMIN, {'prompt': 'nunca', 'channels': [str(STAFF)]})

    await cog.handle_message(message('lag com <@111>', mentions=[111], mid=1))
    check('one run for both matching monitors', len(commands_cog.calls) == 1, commands_cog.calls)
    header = commands_cog.calls[0][1]
    check('header names both, not the third', 'Monitor #1' in header and 'Monitor #2' in header and 'Monitor #3' not in header, header)
    check('fire counts recorded', [m['fire_count'] for m in cog.store.list(1)] == [1, 1, 0])

    await cog.handle_message(message('lag de novo', mid=2))
    check('cooldown holds the next message', len(commands_cog.calls) == 1, commands_cog.calls)
    cog._last_fired[1] -= 3600
    await cog.handle_message(message('lag de novo', mentions=[111], mid=3))
    check('after the cooldown only the cooled-down monitor fires', 'Monitor #1' in commands_cog.calls[-1][1] and 'Monitor #2' not in commands_cog.calls[-1][1], commands_cog.calls)


async def test_messages_the_bot_already_answers_are_skipped(tmp_path, monkeypatch):
    """The router offers a message to the chat engine first; monitors only see the rest."""
    from core.router import MessageRouter
    chat = FakeChat()
    cog = make_cog(tmp_path, chat)
    await cog.create(FakeGuild(), ADMIN, {'prompt': 'p', 'channels': [str(GENERAL)]})
    router = MessageRouter()

    async def chat_handler(msg):
        return msg.id in (2, 3)  # a mention and a reply to a conversation: the engine takes them

    router.handle('chat', chat_handler, priority=10)
    router.handle('monitors', cog.handle_message, priority=20)
    check('bots never reach handlers', await router.dispatch(message('bot talking', bot=True, mid=1)) is None)
    check('claimed by chat', await router.dispatch(message(f'<@{BOT_ID}> oi', mentions=[BOT_ID], mid=2)) == 'chat')
    await router.dispatch(message('seguindo', reference=NS(message_id=77), mid=3))
    check('mentions and conversation replies never fire', chat.calls == [], chat.calls)
    check('anything else fires', await router.dispatch(message('respondendo outra pessoa', mid=4)) == 'monitors')
    check('one run', len(chat.calls) == 1, chat.calls)


async def test_paused_monitor_stops_firing(tmp_path):
    commands_cog = FakeChat()
    cog = make_cog(tmp_path, commands_cog)
    g = FakeGuild()
    mon = await cog.create(g, ADMIN, {'prompt': 'p', 'regex': 'x'})
    await cog.update(g, ADMIN, mon['id'], {'status': 'paused'})
    await cog.handle_message(message('x'))
    check('paused monitor is ignored', commands_cog.calls == [], commands_cog.calls)


# --- ChatEngine.answer_monitor --------------------------------------------------

async def _answer(tmp_path, monkeypatch, answer):
    from chat_fakes import Guild, Message, TextChannel, make_engine
    engine = make_engine(tmp_path, monkeypatch, answers=(answer,))
    msg = Message(500, TextChannel(Guild()), 'o servidor ta com lag')
    sent = await engine.answer_monitor(msg, '[Monitor disparado — ninguém chamou você]')
    return engine, msg, sent


async def test_answer_monitor_can_stay_silent(tmp_path, monkeypatch):
    engine, msg, sent = await _answer(tmp_path, monkeypatch, '[NO_REPLY]')
    check('ran the chat pipeline on the message', engine.turns and engine.turns[0].message is msg, engine.turns)
    check('the model sees it came from a monitor', engine.turns[0].question.startswith('[Monitor disparado'), engine.turns[0].question)
    check('nothing sent', not sent and msg.replies == [], msg.replies)
    check('no conversation stored', await engine.store.get_by_handle(501) is None)


async def test_answer_monitor_replies_and_starts_a_conversation(tmp_path, monkeypatch):
    engine, msg, sent = await _answer(tmp_path, monkeypatch, 'Reduz o view-distance.')
    check('reply sent without pinging the author', sent and msg.replies and msg.replies[0]['mention_author'] is False, msg.replies)
    conv = await engine.store.get_by_handle(501)
    check('reply is a conversation handle users can continue', conv is not None, conv)
    check('stored question keeps the monitor header', conv['data']['turns'][0]['question'].startswith('[Monitor'), conv)
