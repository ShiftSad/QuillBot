"""Server/channel settings and the single message router."""

import asyncio
from types import SimpleNamespace as NS

import pytest

from core import settings as settings_mod
from core.router import MessageRouter
from core.settings import SETTINGS, SettingError, SettingsStore, parse_value

from helpers import check


def make_store(tmp_path):
    store = SettingsStore(str(tmp_path / 'settings.db'))
    store.load()
    return store


def test_values_are_validated():
    check('bool words', parse_value(SETTINGS['name_trigger'], 'Sim') == 'true' and parse_value(SETTINGS['log_detection'], 'off') == 'false')
    check('choices lowercased', parse_value(SETTINGS['threads'], 'ALWAYS') == 'always')
    check('channel mention to id', parse_value(SETTINGS['wrapped_channel'], '<#123456>') == '123456')
    check('names deduplicated', parse_value(SETTINGS['aliases'], 'Pena, pena2, Pena') == 'Pena, pena2')
    for key, raw in (('threads', 'sometimes'), ('name_trigger', 'talvez'), ('aliases', 'ab'), ('instructions', 'x' * 600)):
        with pytest.raises(SettingError):
            parse_value(SETTINGS[key], raw)


def test_lookup_order_channel_parent_server_default(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    channel = NS(id=10, parent_id=None)
    thread = NS(id=11, parent_id=10)
    check('built-in default', store.get(1, channel, 'reply_style') == 'auto')
    store.write(1, 0, 'reply_style', 'embed')
    check('server default', store.get(1, channel, 'reply_style') == 'embed' and store.source(1, channel, 'reply_style') == 'servidor')
    store.write(1, 10, 'reply_style', 'plain')
    check('channel wins', store.get(1, channel, 'reply_style') == 'plain')
    check('threads follow their parent', store.get(1, thread, 'reply_style') == 'plain' and store.source(1, thread, 'reply_style') == 'canal pai')
    store.write(1, 11, 'reply_style', 'embed')
    check('a thread can override', store.get(1, thread, 'reply_style') == 'embed')
    check('other servers unaffected', store.get(2, channel, 'reply_style') == 'auto')
    check('bools coerced', store.get(1, channel, 'log_detection') is True)
    store.write(1, 10, 'log_detection', 'false')
    check('stored bool read back', store.get(1, channel, 'log_detection') is False)
    store.write(1, 10, 'reply_style', None)
    check('removing a value falls back', store.get(1, channel, 'reply_style') == 'embed')

    monkeypatch.setattr(settings_mod, 'CHAT_THREAD_CHANNELS', frozenset({20}))
    check('CHAT_THREAD_CHANNELS still means always', store.get(1, NS(id=20, parent_id=None), 'threads') == 'always')

    reloaded = make_store(tmp_path)
    check('persisted', reloaded.get(1, thread, 'reply_style') == 'embed' and reloaded.get(1, channel, 'log_detection') is False)


def test_internal_state_and_listing(tmp_path):
    store = make_store(tmp_path)
    store.write(1, 0, 'wrapped', 'weekly')
    store.write(2, 0, 'wrapped', 'monthly')
    store.write(1, 10, 'threads', 'never')
    check('servers with a setting', store.guilds_with('wrapped') == {1: 'weekly', 2: 'monthly'})
    store.set_state(1, 'wrapped_last', '2026-W44')
    check('state kept apart from settings', store.state(1, 'wrapped_last') == '2026-W44' and store.guilds_with('wrapped_last') == {})
    check('names are a list', store.get(1, None, 'aliases') == [])


async def test_router_observes_everything_and_claims_in_priority_order():
    router = MessageRouter()
    seen, order = [], []

    async def observer(msg):
        seen.append(msg.id)

    async def chat(msg):
        order.append(('chat', msg.id))
        return msg.content == 'para o bot'

    async def monitors(msg):
        order.append(('monitors', msg.id))
        return True

    router.observe('history', observer)
    router.handle('monitors', monitors, priority=20)
    router.handle('chat', chat, priority=10)

    def human(mid, content):
        return NS(id=mid, content=content, author=NS(bot=False))

    check('chat claims first', await router.dispatch(human(1, 'para o bot')) == 'chat')
    check('monitors get the rest', await router.dispatch(human(2, 'outra coisa')) == 'monitors')
    check('bots are observed but never handled', await router.dispatch(NS(id=3, content='x', author=NS(bot=True))) is None)
    await router.drain()
    check('observers saw all three', sorted(seen) == [1, 2, 3], seen)
    check('handler order', order == [('chat', 1), ('chat', 2), ('monitors', 2)], order)


async def test_a_failing_observer_never_blocks_handlers():
    router = MessageRouter()
    handled = []

    async def broken(msg):
        raise RuntimeError('boom')

    async def slow(msg):
        await asyncio.sleep(0.05)

    async def chat(msg):
        handled.append(msg.id)
        return True

    router.observe('broken', broken)
    router.observe('slow', slow)
    router.handle('chat', chat, priority=10)
    await router.dispatch(NS(id=1, author=NS(bot=False)))
    check('handled before the slow observer finished', handled == [1])
    await router.drain()
