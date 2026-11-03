"""An @mention continues the bot's recent conversation instead of starting a new one."""

from types import SimpleNamespace

from core.conversations import ConversationStore, find_recent_conversation

import chat_fakes
from chat_fakes import Guild, Message, TextChannel, make_engine
from helpers import check

BOT_ID = 900


def msg(mid, author_id, content='', *, reference=None):
    return SimpleNamespace(
        id=mid,
        author=SimpleNamespace(id=author_id, bot=author_id == BOT_ID, name=f'u{author_id}', display_name=f'U{author_id}'),
        content=content,
        reference=reference,
        attachments=[],
        mentions=[],
        guild=None,
    )


class Channel:
    """Channel whose history is given newest first, like discord.py's."""

    def __init__(self, messages):
        self.messages = messages
        self.calls = []
        self.id = 1

    def history(self, *, limit, before):
        self.calls.append((limit, before))
        items = self.messages[:limit]

        async def gen():
            for m in items:
                yield m
        return gen()


async def make_store(tmp_path):
    store = ConversationStore(str(tmp_path / 'conv.db'), kind='chat')
    await store.create(50, channel_id='1', data={'turns': [{'answer': 'old'}]})
    await store.update(50, {'turns': [{'answer': 'old'}, {'answer': 'new'}]}, new_handle_msg_id=60)
    return store


async def test_finds_newest_conversation_in_window(tmp_path):
    store = await make_store(tmp_path)
    channel = Channel([
        msg(70, 1, 'oi'),
        msg(65, BOT_ID, 'not a conversation'),
        msg(60, BOT_ID, 'latest answer'),
        msg(55, 2, 'chatter'),
        msg(50, BOT_ID, 'first answer'),
    ])
    trigger = msg(80, 1)
    found = await find_recent_conversation(store, channel, BOT_ID, before=trigger, limit=10)
    check('picks the newest bot message that is a conversation handle', found and found[0] == 60, found)
    check('returns that conversation', found[1]['conv_id'] == '50', found)
    check('window is the channel-context size, before the mention', channel.calls == [(10, trigger)], channel.calls)


async def test_no_conversation_outside_window(tmp_path):
    store = await make_store(tmp_path)
    channel = Channel([msg(i, 1, 'chatter') for i in range(79, 70, -1)] + [msg(60, BOT_ID, 'answer')])
    found = await find_recent_conversation(store, channel, BOT_ID, before=msg(80, 1), limit=5)
    check('bot message older than the window is ignored', found is None, found)
    found = await find_recent_conversation(store, channel, BOT_ID, before=msg(80, 1), limit=0)
    check('a zero window disables the lookup', found is None, found)


async def test_ignores_other_users_messages_with_handle_ids(tmp_path):
    store = await make_store(tmp_path)
    channel = Channel([msg(60, 1, 'same id but not from the bot')])
    found = await find_recent_conversation(store, channel, BOT_ID, before=msg(80, 1), limit=10)
    check('only the bot own messages count', found is None, found)


async def engine_with_conversation(tmp_path, monkeypatch, *, continue_enabled=True):
    engine = make_engine(tmp_path, monkeypatch)
    monkeypatch.setattr(chat_fakes.chat_engine, 'CHAT_MENTION_CONTINUE_ENABLED', continue_enabled)
    await engine.store.create(50, channel_id='10', data={'turns': [{'answer': 'old', 'message_id': '45', 'channel_id': '10'}]})
    await engine.store.update(50, {'turns': [{'answer': 'old'}, {'answer': 'new', 'message_id': '58', 'channel_id': '10'}]}, new_handle_msg_id=60)
    return engine


def channel_with(*messages):
    channel = TextChannel(Guild())
    channel.messages = list(messages)
    return channel


async def test_mention_continues_recent_conversation(tmp_path, monkeypatch):
    engine = await engine_with_conversation(tmp_path, monkeypatch)
    channel = channel_with(msg(70, 2, 'chatter'), msg(60, BOT_ID, 'answer'))
    await engine.handle_message(Message(80, channel, f'<@{BOT_ID}> e agora?'))
    conv = await engine.store.get_by_trigger(80)
    check('mention merged into the conversation', conv and conv['conv_id'] == '50' and len(conv['data']['turns']) == 3, conv)
    check('with no reply target', engine.turns[0].reply_to is None, engine.turns[0].reply_to)


async def test_mention_without_recent_conversation_starts_new(tmp_path, monkeypatch):
    engine = await engine_with_conversation(tmp_path, monkeypatch)
    channel = channel_with(msg(70, 2, 'chatter'))
    await engine.handle_message(Message(80, channel, f'<@{BOT_ID}> oi'))
    conv = await engine.store.get_by_handle(81)
    check('no conversation in the window starts a new one', conv and conv['conv_id'] == '81' and not engine.turns[0].history, conv)


async def test_continuation_can_be_disabled(tmp_path, monkeypatch):
    engine = await engine_with_conversation(tmp_path, monkeypatch, continue_enabled=False)
    channel = channel_with(msg(60, BOT_ID, 'answer'))
    await engine.handle_message(Message(80, channel, f'<@{BOT_ID}> e agora?'))
    conv = await engine.store.get_by_handle(81)
    check('disabled flag always starts a new conversation', conv and conv['conv_id'] == '81', conv)


async def test_mention_replying_to_someone_continues_with_context(tmp_path, monkeypatch):
    engine = await engine_with_conversation(tmp_path, monkeypatch)
    quoted = msg(70, 2, 'o servidor caiu de novo')
    quoted.embeds = []
    channel = channel_with(quoted, msg(60, BOT_ID, 'answer'))
    mention = Message(80, channel, f'<@{BOT_ID}> e isso?', reference=SimpleNamespace(message_id=70, resolved=quoted))
    await engine.handle_message(mention)
    check('reply to another user still merges, replying to that message',
          engine.turns[0].reply_to == '70' and len(engine.turns[0].history) == 2, engine.turns[0].reply_to)
    check('quoted message rides along as context', 'o servidor caiu de novo' in engine.turns[0].question, engine.turns[0].question)
