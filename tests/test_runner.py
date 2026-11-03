"""The one runner: profiles, the tool registry, per-turn context, sources and LLM retries."""

from types import SimpleNamespace as NS

import openai
import httpx

import core.llm as llm_mod
import core.runner as runner_mod
from core.context_tools import register as register_core_tools
from core.profiles import ASK, CHAT, SPARK, profile_for
from core.runner import run_turn
from core.tools import Tool, tool_schema
from core.turn import TurnContext

from chat_fakes import ASKER, BOT_ID, make_services
from helpers import as_stream, check


class Completions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return as_stream(response)


def completion(content=None, tool_calls=None):
    tcs = [
        NS(id=f'call_{i}', type='function', function=NS(name=name, arguments=args))
        for i, (name, args) in enumerate(tool_calls or [])
    ] or None
    return NS(choices=[NS(message=NS(role='assistant', content=content, tool_calls=tcs), finish_reason='stop')], usage=None)


def services_with(tmp_path, responses):
    services = make_services(tmp_path)
    completions = Completions(responses)
    services.llm.client = NS(chat=NS(completions=completions))
    register_core_tools(services.tools)
    return services, completions


def ctx_for(services, profile=CHAT, **kw):
    return TurnContext(
        bot=NS(user=NS(id=BOT_ID, name='quillbot', display_name='QuillBot'), services=services),
        profile=profile, question=kw.pop('question', 'como reduzo o lag?'), user=ASKER, **kw,
    )


def docs_tool(calls):
    async def run(ctx, args):
        calls.append(args)
        return 'trecho da doc', [{'title': 'View distance', 'url': 'https://docs.papermc.io/vd', 'source': 'PaperMC', 'path': 'vd.md'}]
    return Tool('search_docs', tool_schema('search_docs', 'docs', {'query': {'type': 'string'}}, ['query']), run, 'docs')


async def test_chat_turns_can_search_the_docs(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, 'CHANNEL_CONTEXT_MESSAGES', 0)
    services, completions = services_with(tmp_path, [
        completion(tool_calls=[('search_docs', '{"query": "view distance"}')]),
        completion('Use view-distance 6.'),
    ])
    calls = []
    services.tools.register(docs_tool(calls))
    result = await run_turn(ctx_for(services))
    offered = [t['function']['name'] for t in completions.calls[0]['tools']]
    check('search_docs offered in normal chat', 'search_docs' in offered, offered)
    check('tools in profile order (cache-stable)', offered == [n for n in CHAT.tools if n in offered], offered)
    check('tool ran', calls == [{'query': 'view distance'}])
    check('answer', result.answer == 'Use view-distance 6.')
    check('sources normalized with their url', result.sources == [{'title': 'View distance', 'url': 'https://docs.papermc.io/vd', 'source': 'PaperMC'}], result.sources)
    check('tools used recorded', result.used_tools == ['search_docs'])


async def test_prompt_follows_the_server_and_channel(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, 'CHANNEL_CONTEXT_MESSAGES', 0)
    services, completions = services_with(tmp_path, [completion('oi!')])

    class Memory:
        async def build_memory_block(self, guild_id, query, **kw):
            return '<memory>\n- [mem #1 📌] O bot fala como um pirata.\n</memory>'

    services.memory = Memory()
    guild = NS(id=1, name='Miners', me=NS(display_name='QuillBot'), description=None, owner_id=5,
               member_count=10, channels=[], roles=[], created_at=None, owner=None)
    await run_turn(ctx_for(
        services, guild=guild, channel=NS(id=10, name='geral', topic=None, parent=None),
        channel_settings={'instructions': 'aqui é papo livre'},
    ))
    messages = completions.calls[0]['messages']
    system = messages[0]['content'][0]['text'] if isinstance(messages[0]['content'], list) else messages[0]['content']
    check('personality comes from the server memories', 'definidos pelo' in system and 'próprio servidor' in system, system[:400])
    check('one voice: no separate "assistente" persona', 'assistente de propósito geral' not in system and 'assistente oficial' not in system)
    user = [m for m in messages if isinstance(m, dict) and m.get('role') == 'user'][-1]
    current = user['content'] if isinstance(user['content'], str) else user['content'][0]['text']
    check('channel instructions ride on the current message', '<canal>' in current and 'aqui é papo livre' in current, current[:600])
    check('memory block injected', 'fala como um pirata' in current)


async def test_structured_tools_and_profiles(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, 'CHANNEL_CONTEXT_MESSAGES', 0)
    services, completions = services_with(tmp_path, [
        completion(tool_calls=[('summarize_channel', '{}')]),
        completion('**Tópicos**\n- algo'),
    ])

    async def summarize(ctx, args):
        return 'resumo pronto'

    services.tools.register(Tool('summarize_channel', tool_schema('summarize_channel', 's'), summarize, 'resumo', structured=True))
    result = await run_turn(ctx_for(services))
    check('a summary goes out framed (embed)', result.structured)
    check('/ask has no web or scheduler tools', not {'web_search', 'schedule_create', 'open_thread'} & set(ASK.tools))
    check('spark adds its tools', SPARK.tools[-2:] == ('get_spark_detail', 'get_config_key') and SPARK.structured)
    check('a spark conversation without its report continues as /ask', profile_for('spark', has_spark_report=False) is ASK)
    check('unknown profile is chat', profile_for(None) is CHAT)


async def test_transient_llm_errors_are_retried(monkeypatch):
    monkeypatch.setattr(llm_mod, 'LLM_RETRY_BASE_DELAY', 0)
    request = httpx.Request('POST', 'https://example.test/v1/chat/completions')
    completions = Completions([
        openai.APIConnectionError(request=request),
        openai.InternalServerError('boom', response=httpx.Response(502, request=request), body=None),
        completion('ok'),
    ])
    response = await llm_mod.create_chat_completion(NS(chat=NS(completions=completions)), model='m', messages=[])
    check('third attempt answered', response.choices[0].message.content == 'ok' and len(completions.calls) == 3)

    bad = Completions([openai.BadRequestError('nope', response=httpx.Response(400, request=request), body=None), completion('x')])
    try:
        await llm_mod.create_chat_completion(NS(chat=NS(completions=bad)), model='m', messages=[])
        raised = False
    except openai.BadRequestError:
        raised = True
    check('client errors are not retried', raised and len(bad.calls) == 1)


async def test_status_lines_reach_the_slash_command(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, 'CHANNEL_CONTEXT_MESSAGES', 0)
    services, _ = services_with(tmp_path, [
        completion(tool_calls=[('get_guild_info', '{}')]),
        completion('pronto'),
    ])
    statuses = []

    async def on_status(text):
        statuses.append(text)

    await run_turn(ctx_for(services, on_status=on_status))
    check('the tool status was shown', statuses == ['🏰 Coletando informações do servidor'], statuses)
