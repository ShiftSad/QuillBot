"""The bot's personality per server: store, prompt block, the tool's rules, revert and import."""

import asyncio
from types import SimpleNamespace as NS

import pytest

import cogs.persona as persona_mod
import core.runner as runner_mod
from cogs.persona import Persona
from core.persona import PersonaError, PersonaStore, render_block
from core.profiles import ASK, CHAT
from core.prompts import CHAT_SYSTEM_PROMPT
from core.tools import ToolRegistry

from helpers import as_stream, check

ADMIN = NS(id=1, display_name='Dona', guild_permissions=NS(administrator=False, manage_guild=True))
MEMBER = NS(id=2, display_name='Ana', guild_permissions=NS(administrator=False, manage_guild=False))
GUILD = NS(id=10)


def make_store(tmp_path):
    store = PersonaStore(str(tmp_path / 'persona.db'))
    store.load()
    return store


def make_cog(tmp_path, memory=None):
    services = NS(tools=ToolRegistry(), persona=None, memory=memory)
    cog = Persona.__new__(Persona)
    cog.bot = NS(services=services, get_channel=lambda cid: None)
    cog.services = services
    cog.store = make_store(tmp_path)
    cog._log_tasks = set()
    return cog


def tool_ctx(user):
    return NS(guild=GUILD, user=user, actor_name=f'bot (via {user.display_name})')


# --- Store ----------------------------------------------------------------------

def test_versions_history_and_reload(tmp_path):
    store = make_store(tmp_path)
    v1 = store.write(10, 'base', 'Você é o Quill, um pirata gente boa.', actor='1', actor_name='Dona', reason='primeira')
    v2 = store.write(10, 'base', 'Você é o Quill, um pirata gente boa que ama Paper.', actor='bot', actor_name='bot', reason='ajuste')
    check('versions count up', (v1.version, v2.version) == (1, 2))
    check('history keeps before and after', v2.before == v1.after and v2.after.endswith('ama Paper.'))
    check('newest first', [v.version for v in store.history(10, 'base')] == [2, 1])
    check('found by id', store.get_version(10, v1.id).after == v1.after)
    check('other servers untouched', store.get(11, 'base') == '')
    with pytest.raises(PersonaError):
        store.write(10, 'base', v2.after, actor='1', actor_name='Dona')
    with pytest.raises(PersonaError):
        store.write(10, 'notes', 'x' * 5000, actor='bot', actor_name='bot')
    store.set_notes_locked(10, True)
    reloaded = make_store(tmp_path)
    check('persisted', reloaded.get(10, 'base') == v2.after and reloaded.info(10, 'base')['version'] == 2)
    check('lock persisted', reloaded.notes_locked(10) and not reloaded.notes_locked(11))
    check('only the bot own edits count toward its limit', reloaded.bot_edits_since(10, 'base') == 1)


def test_block_rendering(tmp_path):
    check('nothing set, no block', render_block('', '  ') == '')
    block = render_block('Fala como pirata.', 'Chamam você de capitão.')
    check('both sections', block.startswith('<personalidade>') and 'Fala como pirata.' in block
          and 'Chamam você de capitão.' in block and block.endswith('</personalidade>'), block)
    check('notes alone', 'Base' not in render_block('', 'só notas'))
    store = make_store(tmp_path)
    check('no guild, no block', store.block(None) == '')


# --- In the prompt --------------------------------------------------------------

async def test_runner_puts_the_personality_in_the_system_prompt(tmp_path, monkeypatch):
    from core.runner import run_turn
    from core.turn import TurnContext
    from chat_fakes import make_services

    monkeypatch.setattr(runner_mod, 'CHANNEL_CONTEXT_MESSAGES', 0)
    services = make_services(tmp_path)
    calls = []

    class Completions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            msg = NS(role='assistant', content='Arr!', tool_calls=None)
            return as_stream(NS(choices=[NS(message=msg, finish_reason='stop')], usage=None))

    services.llm.client = NS(chat=NS(completions=Completions()))
    store = make_store(tmp_path)
    store.write(10, 'base', 'Você fala como um pirata.', actor='1', actor_name='Dona')
    services.persona = NS(block=store.block)
    guild = NS(id=10, name='Miners', me=None, description=None, owner_id=None, member_count=1,
               channels=[], roles=[], created_at=None, owner=None)
    bot = NS(user=NS(id=900, name='quill', display_name='Quill'), services=services)
    await run_turn(TurnContext(bot=bot, profile=CHAT, question='oi', guild=guild))
    system = calls[0]['messages'][0]['content']
    system = system[0]['text'] if isinstance(system, list) else system
    check('block in the system prompt', '<personalidade>' in system and 'fala como um pirata' in system, system[-600:])
    check('before the identity block', system.index('<personalidade>') < system.index('<identidade>'))


def test_tool_and_instructions_are_offered_in_chat_only():
    check('chat can update its personality', 'persona_update' in CHAT.tools)
    check('/ask stays on docs', 'persona_update' not in ASK.tools)
    check('the model knows the rules', 'persona_update' in CHAT_SYSTEM_PROMPT and 'Gerenciar Servidor' in CHAT_SYSTEM_PROMPT)


# --- The persona_update tool ------------------------------------------------------

async def test_who_can_change_what(tmp_path, monkeypatch):
    monkeypatch.setattr(persona_mod, 'PERSONA_BOT_EDITS_PER_DAY', 2)
    cog = make_cog(tmp_path)
    run = cog._run_tool

    text = await run(tool_ctx(MEMBER), {'section': 'base', 'content': 'Você é grosso com todo mundo.', 'reason': 'pediram'})
    check('a member cannot change the base', 'Gerenciar Servidor' in text and cog.store.get(10, 'base') == '', text)

    text = await run(tool_ctx(ADMIN), {'section': 'base', 'content': 'Você é um pirata gente boa.', 'reason': 'admin pediu'})
    check('an admin can, through the bot', 'versão 1' in text and cog.store.get(10, 'base') == 'Você é um pirata gente boa.', text)
    check('recorded as asked by the admin', cog.store.info(10, 'base')['updated_by'] == '1')

    text = await run(tool_ctx(MEMBER), {'section': 'notes', 'content': 'Te chamam de capitão.', 'reason': 'apelido novo'})
    check('the bot keeps its own notes', 'versão 1' in text and cog.store.info(10, 'notes')['updated_by'] == 'bot', text)
    await run(tool_ctx(MEMBER), {'section': 'notes', 'content': 'Te chamam de capitão. Bordão: arr!', 'reason': 'bordão'})
    text = await run(tool_ctx(MEMBER), {'section': 'notes', 'content': 'outra coisa', 'reason': 'mais uma'})
    check('a few changes a day at most', 'limite 2' in text and 'arr!' in cog.store.get(10, 'notes'), text)
    text = await run(tool_ctx(ADMIN), {'section': 'notes', 'content': 'Te chamam de capitão.', 'reason': 'admin limpou'})
    check('an admin request is not limited', 'versão 3' in text, text)

    await asyncio.to_thread(cog.store.set_notes_locked, 10, True)
    text = await run(tool_ctx(MEMBER), {'section': 'notes', 'content': 'nova nota', 'reason': 'x'})
    check('locked notes', 'travadas' in text, text)
    text = await run(tool_ctx(MEMBER), {'section': 'notes', 'content': 'y' * 900, 'reason': 'x'})
    check('still locked whatever the size', 'travadas' in text)

    await asyncio.to_thread(cog.store.set_notes_locked, 10, False)
    monkeypatch.setattr(persona_mod, 'PERSONA_BOT_EDITS_PER_DAY', 99)
    text = await run(tool_ctx(MEMBER), {'section': 'notes', 'content': 'y' * 900, 'reason': 'x'})
    check('too long is explained', 'máximo' in text, text)
    check('reason required', 'obrigatório' in await run(tool_ctx(MEMBER), {'section': 'notes', 'content': 'z'}))
    check('needs a server', 'servidor' in await run(NS(guild=None, user=MEMBER, actor_name='bot'), {'section': 'notes', 'content': 'z', 'reason': 'r'}))


async def test_revert_and_log(tmp_path, monkeypatch):
    sent = []

    async def send(**kwargs):
        sent.append(kwargs)

    monkeypatch.setattr(persona_mod, 'PERSONA_LOG_CHANNEL_ID', 55)
    cog = make_cog(tmp_path)
    cog.bot = NS(services=cog.services, get_channel=lambda cid: NS(send=send) if cid == 55 else None)
    first = await cog.change(10, 'base', 'Versão boa.', actor='1', actor_name='Dona', reason='inicial')
    bad = await cog.change(10, 'base', 'Versão ruim.', actor='bot', actor_name='bot (via Ana)', reason='pediram')
    reverted = await cog.revert(10, bad.id, actor=ADMIN)
    check('back to what it was', cog.store.get(10, 'base') == first.after and reverted.version == 3)
    check('the revert is itself a change', 'Reversão da mudança' in reverted.reason)
    await asyncio.gather(*list(cog._log_tasks))
    check('every change posted', len(sent) == 3, sent)
    embed, view = sent[1]['embed'], sent[1]['view']
    check('log shows before/after', [f.name for f in embed.fields] == ['Antes', 'Depois'] and 'Versão ruim.' in embed.fields[1].value)
    check('with a revert button for that change', view.history_id == bad.id)
    with pytest.raises(PersonaError):
        await cog.revert(10, 999, actor=ADMIN)


# --- /persona importar --------------------------------------------------------------

class FakeMemory:
    def __init__(self, entries):
        self.entries = entries
        self.unpinned = []

    async def get(self, guild_id, memory_id):
        return self.entries.get(memory_id)

    async def set_pinned(self, guild_id, memory_id, *, pinned, actor_id, actor_name):
        self.unpinned.append(memory_id)


async def test_import_moves_memories_into_the_base(tmp_path):
    memory = FakeMemory({
        3: {'id': 3, 'status': 'active', 'subject': '', 'pinned': True, 'content': 'O bot fala como pirata.'},
        4: {'id': 4, 'status': 'active', 'subject': '222', 'pinned': False, 'content': 'Ana gosta de Paper.'},
        5: {'id': 5, 'status': 'archived', 'subject': '', 'pinned': False, 'content': 'velha'},
        7: {'id': 7, 'status': 'active', 'subject': '', 'pinned': False, 'content': 'O bot nunca usa palavrão.'},
    })
    cog = make_cog(tmp_path, memory=memory)
    replies = []

    async def send_message(text, **kw):
        replies.append(text)

    interaction = NS(guild=GUILD, user=ADMIN, response=NS(send_message=send_message))
    await cog.importar.callback(cog, interaction, '3, 4, 5, 7')
    base = cog.store.get(10, 'base')
    check('server memories copied as a list', base == '- O bot fala como pirata.\n- O bot nunca usa palavrão.', base)
    check('pinned ones unpinned', memory.unpinned == [3])
    check('the rest explained', 'sobre uma pessoa' in replies[0] and 'arquivada' in replies[0], replies)

    replies.clear()
    member_itx = NS(guild=GUILD, user=MEMBER, response=NS(send_message=send_message))
    await cog.importar.callback(cog, member_itx, '3')
    check('admins only', 'Gerenciar Servidor' in replies[0])


async def test_edit_modal_saves_the_section(tmp_path):
    cog = make_cog(tmp_path)
    await cog.change(10, 'base', 'Antes.', actor='1', actor_name='Dona', reason='inicial')
    modal = persona_mod.PersonaModal(cog, 10, 'base')
    check('pre-filled with the current text', modal.text.default == 'Antes.')
    check('limited to the section size', modal.text.max_length == 3000)
    modal.text._value = 'Depois.'  # what Discord fills in on submit
    modal.reason._value = ''
    replies = []

    async def send_message(text, **kw):
        replies.append(text)

    await modal.on_submit(NS(user=ADMIN, response=NS(send_message=send_message)))
    check('saved', cog.store.get(10, 'base') == 'Depois.' and 'versão 2' in replies[0], replies)
    check('default reason', cog.store.history(10, 'base')[0].reason == 'Edição manual')
