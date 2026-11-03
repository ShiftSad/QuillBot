"""Who told the bot a fact, and episodes: what it remembers of ended conversations."""

import time
from types import SimpleNamespace as NS

from cogs import memory as memory_mod
from cogs.memory import Memory, MemoryStore
from core.conversations import ConversationStore
from core.episodes import EpisodeWriter, transcript

from helpers import check

ALICE = NS(id=111, name='alice_h', display_name='Alice', nick=None, global_name=None)
BOB = NS(id=222, name='bob_h', display_name='Bob', nick=None, global_name=None)


def memory_cog(tmp_path):
    members = {111: ALICE, 222: BOB}
    guild = NS(id=1, get_member=lambda uid: members.get(uid), get_member_named=lambda n: None, members=list(members.values()))
    cog = Memory.__new__(Memory)
    cog.bot = NS(get_guild=lambda gid: guild)
    cog.services = None
    cog.store = MemoryStore(str(tmp_path / 'memory.db'))
    cog.store.ensure()
    cog._bot_writes = {}
    cog._log_tasks = set()

    async def no_embeddings(texts):
        return None

    cog._embed_texts = no_embeddings
    return cog, guild


async def test_facts_about_someone_told_by_someone_else_are_reports(tmp_path):
    cog, guild = memory_cog(tmp_path)
    participants = {'111', '222'}
    text, _ = await cog.exec_tool(
        'memory_write',
        {'action': 'create', 'about_user': '222', 'kind': 'preference', 'content': 'Bob odeia servidores Fabric', 'reason': 'dito'},
        guild=guild, requester=ALICE, participant_ids=participants,
    )
    check('the model is told it is a report', 'segundo Alice' in text and 'não confirmada' in text, text)
    hearsay = cog.store.list_memories(1, subject='222')[0]
    check('told_by recorded', hearsay['told_by'] == '111', hearsay)

    await cog.exec_tool(
        'memory_write',
        {'action': 'create', 'about_user': '111', 'kind': 'preference', 'content': 'Alice prefere Paper a Fabric', 'reason': 'disse'},
        guild=guild, requester=ALICE, participant_ids=participants,
    )
    block = await cog.build_memory_block(1, 'fabric', speaker_id='111', participant_ids=participants)
    check('report marked in the block', 'Bob odeia servidores Fabric (segundo Alice)' in block, block)
    check("someone's own words are not marked", 'Alice prefere Paper a Fabric\n' in block + '\n' and 'Paper a Fabric (segundo' not in block, block)

    text, _ = await cog.exec_tool('memory_search', {'query': 'Fabric', 'about_user': '222'}, guild=guild, requester=ALICE, participant_ids=participants)
    check('search shows it too', '(segundo Alice)' in text, text)

    await cog.exec_tool(
        'memory_write',
        {'action': 'update', 'memory_id': hearsay['id'], 'content': 'Bob odeia servidores Fabric, confirmou ele', 'reason': 'confirmou'},
        guild=guild, requester=BOB, participant_ids=participants,
    )
    updated = cog.store.get_memory(1, hearsay['id'])
    check('confirmed by the person: no longer a report', updated['told_by'] == '222' and cog.told_by_note(1, updated) == '', updated)
    history = cog.store.history_for_guild(1)
    check('told_by is part of the audited snapshot', '"told_by": "111"' in history[-1]['before'], history[-1]['before'])


def test_episodes_are_not_writable_by_the_model():
    enum = memory_mod.MEMORY_WRITE_TOOL['function']['parameters']['properties']['kind']['enum']
    check('memory_write cannot create episodes', 'episode' not in enum, enum)
    check('but search can filter them', 'episode' in memory_mod.MEMORY_SEARCH_TOOL['function']['parameters']['properties']['kind']['enum'])


async def test_record_episode_per_participant_with_a_cap(tmp_path, monkeypatch):
    cog, _ = memory_cog(tmp_path)
    monkeypatch.setattr(memory_mod, 'EPISODES_PER_USER', 2)
    for i in range(3):
        await cog.record_episode(1, ['111', '222'], f'Conversa de 0{i}/11/2026 com Alice, Bob: assunto {i}')
    for subject in ('111', '222'):
        active = [e for e in cog.store.list_memories(1, subject=subject) if e['status'] == 'active']
        check(f'only the newest two for {subject}', sorted(e['content'][-1] for e in active) == ['1', '2'], active)
        check('kind and low importance', all(e['kind'] == 'episode' and e['importance'] == 2 for e in active))


async def test_expired_conversations_wait_for_their_episode(tmp_path):
    store = ConversationStore(str(tmp_path / 'conv.db'), kind='chat', ttl_seconds=60, keep_expired=True)
    await store.create(1, guild_id='5', channel_id='10', data={'turns': [{'question': 'q', 'answer': 'a'}], 'participants': []}, trigger_ids=[99])
    con = store._connect()
    con.execute('UPDATE conversations SET last_active_ts=?', (time.time() - 120,))
    con.commit()
    con.close()
    check('expired: not reachable by reply', await store.get_by_handle(1) is None)
    pending = await store.expired_pending()
    check('but waiting for its summary', [c['conv_id'] for c in pending] == ['1'], pending)
    by_trigger = await store.get_by_trigger(99)
    check('deletes still find it, flagged expired', by_trigger and by_trigger['conv_id'] == '1' and by_trigger['expired'], by_trigger)
    await store.mark_episode(1, 'done')
    check('summarized ones are pruned', await store.expired_pending() == [])
    con = store._connect()
    left = con.execute('SELECT COUNT(*) FROM conversations').fetchone()[0]
    con.close()
    check('row removed', left == 0, left)


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    async def text(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply


class FakeMemory:
    def __init__(self):
        self.episodes = []

    async def record_episode(self, guild_id, subjects, content, *, origin=''):
        self.episodes.append((guild_id, subjects, content, origin))


def conversation(conv_id='1'):
    return {
        'conv_id': conv_id, 'guild_id': '5',
        'data': {
            'started_ts': 1793188800.0,
            'participants': [{'id': '111', 'display': 'Alice'}, {'id': '222', 'display': 'Bob'}],
            'turns': [
                {'question': '[Seu nome foi citado nesta mensagem]\n---\nquill, faço backup como?', 'answer': 'Use o plugin X.',
                 'author': {'display': 'Alice'}, 'ts': 1793188800.0},
            ],
            'thread_id': '700',
        },
    }


async def test_episode_writer_summarizes_or_skips():
    memory = FakeMemory()
    llm = FakeLLM('Alice perguntou como fazer backup; ia testar o plugin X.')
    writer = EpisodeWriter(NS(memory=memory, llm=llm, conversations=None), model='m')
    check('summarized', await writer.summarize(conversation()) == 'done')
    guild_id, subjects, content, origin = memory.episodes[0]
    check('one memory per participant, naming them', guild_id == 5 and subjects == ['111', '222'] and 'com Alice, Bob' in content, memory.episodes)
    check('dated', content.startswith('Conversa de 28/10/2026'), content)
    check('links the thread', origin == 'https://discord.com/channels/5/700', origin)
    check('routing headers stripped from the transcript', '[Seu nome' not in llm.calls[0]['user'] and 'faço backup' in llm.calls[0]['user'])

    nothing = EpisodeWriter(NS(memory=FakeMemory(), llm=FakeLLM('NADA'), conversations=None), model='m')
    check('nothing worth remembering', await nothing.summarize(conversation()) == 'skipped')
    no_memory = EpisodeWriter(NS(memory=None, llm=llm, conversations=None), model='m')
    check('memory disabled', await no_memory.summarize(conversation()) == 'skipped')


async def test_episode_writer_marks_handled_conversations(tmp_path):
    store = ConversationStore(str(tmp_path / 'conv.db'), kind='chat', ttl_seconds=60, keep_expired=True)
    await store.create(1, guild_id='5', data=conversation()['data'])
    con = store._connect()
    con.execute('UPDATE conversations SET last_active_ts=?', (time.time() - 120,))
    con.commit()
    con.close()
    memory = FakeMemory()
    writer = EpisodeWriter(NS(memory=memory, llm=FakeLLM('Alice ia testar o backup.'), conversations=store), model='m')
    check('one handled', await writer.run_once() == 1)
    check('episode written', len(memory.episodes) == 1)
    check('not summarized twice', await writer.run_once() == 0)


def test_transcript_keeps_the_end_of_long_conversations():
    data = {'turns': [{'question': f'q{i}' + 'x' * 1000, 'answer': 'a', 'author': {'display': 'A'}} for i in range(40)]}
    text = transcript(data)
    check('bounded', len(text) <= 12000)
    check('newest turns kept', 'q39' in text and 'q0x' not in text)
