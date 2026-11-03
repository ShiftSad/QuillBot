"""Seeing what people point at: the wider channel window, waiting for the image after
"ai oh @bot", sharper shown images labelled as such, and doing an asked-for search."""

import asyncio
from types import SimpleNamespace as NS

import core.context as context_mod
from core import chat_engine, conversations as cs
from core.context import ChannelContext, _trim_oldest
from core.runner import SEARCH_HINT, search_requested

from chat_fakes import BOT_ID, Guild, Message, TextChannel, make_engine, reply_to
from helpers import check


def test_window_keeps_the_newest_lines_within_the_budget():
    lines = [f'[15:{i:02d}] Pessoa <author_id=1>: mensagem {i} ' + 'x' * 40 for i in range(30)]
    context = ChannelContext(text='Histórico de #geral (channel_id=1) — últimas 30 mensagens, cronológica:\n' + '\n'.join(lines), lines=lines)
    trimmed = _trim_oldest(context, 1000)
    check('fits the budget', sum(len(line) + 1 for line in trimmed.lines) <= 1000)
    check('newest kept, oldest dropped', trimmed.lines[-1] == lines[-1] and trimmed.lines[0] != lines[0])
    check('header says how many', f'últimas {len(trimmed.lines)} mensagens' in trimmed.text.split('\n')[0], trimmed.text[:90])
    check('a small window is untouched', _trim_oldest(context, 100000) is context)
    check('0 disables trimming', _trim_oldest(context, 0) is context)
    check('wider default window', context_mod.CHANNEL_CONTEXT_MESSAGES >= 30)


async def test_bare_pointer_waits_for_what_comes_next(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch)
    monkeypatch.setattr(chat_engine, 'CHAT_BURST_SHORT_SECONDS', 0.3)
    engine.bursts.delay = 0.05
    channel = TextChannel(Guild())
    pointer = Message(80, channel, f'Ai oh <@{BOT_ID}>')
    check('"ai oh @bot" waits longer', engine.burst_delay(NS(messages=[pointer])) == 0.3)
    question = Message(81, channel, f'<@{BOT_ID}> pesquise esse sinal misterioso')
    check('a real question does not', engine.burst_delay(NS(messages=[question])) == 0.05)
    replying = Message(82, channel, f'<@{BOT_ID}> e isso?', reference=reply_to(5))
    check('nor a short reply (it points at the replied message)', engine.burst_delay(NS(messages=[replying])) == 0.05)

    await engine.handle_message(pointer)
    await asyncio.sleep(0.15)
    image = Message(83, channel, '')
    image.attachments = [NS(id=9, filename='pr.png', content_type='image/png', url='https://cdn/x.png', size=1)]
    check('the image sent seconds later joins', await engine.handle_message(image))
    await asyncio.sleep(0.25)
    check('nothing answered before the image', len(engine.turns) == 1 and engine.turns[0].message is image, engine.turns)
    check('one turn, replying to the image message', image.replies and not pointer.replies)


def test_shown_images_are_labelled_apart_from_channel_images(monkeypatch):
    monkeypatch.setattr(cs.image_store, 'is_image_ref', lambda value: value.startswith('img:'))
    monkeypatch.setattr(cs.image_store, 'image_part', lambda ref: {'type': 'image_url', 'image_url': {'url': ref}})
    message = cs.build_current_message(
        'Ai oh @Quill', author=None, ts=None,
        image_urls=['img:pr.jpg'], context_image_urls=['img:clip.jpg'],
    )
    text = message['content'][0]['text']
    check('the shown image is named as what the person is showing',
          'a primeira imagem abaixo veio com esta mensagem' in text and 'é o que a pessoa está mostrando' in text, text)
    check('channel images still announced after it', 'depois das enviadas pelo usuário' in text)
    check('order: shown first', [p['image_url']['url'] for p in message['content'][1:]] == ['img:pr.jpg', 'img:clip.jpg'])
    plain = cs.build_current_message('oi', author=None, ts=None, context_image_urls=['img:clip.jpg'])
    check('no shown-image label without one', 'pessoa está mostrando' not in plain['content'][0]['text'])


async def test_shown_images_are_kept_sharper(tmp_path, monkeypatch):
    from PIL import Image
    import io
    from core import image_store

    monkeypatch.setattr(image_store, 'IMAGES_DIR', str(tmp_path))
    buf = io.BytesIO()
    Image.new('RGB', (1920, 1080), 'white').save(buf, format='PNG')
    data = buf.getvalue()

    class Attachment:
        id = 77

        async def read(self):
            return data

    image_store._attachment_refs.clear()
    context_ref = await image_store.persist_attachment(Attachment())
    shown_ref = (await image_store.persist_shown([Attachment()]))[0]
    sizes = [Image.open(image_store.image_path(r)).size for r in (context_ref, shown_ref)]
    check('channel context at 640, shown at 1280', sizes == [(640, 360), (1280, 720)], sizes)
    check('both cached separately', len(image_store._attachment_refs) == 2)
    image_store._attachment_refs.clear()


def test_asked_for_search_is_flagged():
    for text in ('pesquise esse sinal misterioso', 'procura aí pra mim', 'busca isso', 'dá um google'):
        check(f'search request: {text!r}', search_requested(text))
    for text in ('qual o melhor plugin?', 'boa tarde'):
        check(f'not a search request: {text!r}', not search_requested(text))
    quoted = '[Mensagem respondida — Ana (@ana)]\nConteúdo: fiz uma pesquisa ontem\n---\nconcorda?'
    check('a quoted message does not count', not search_requested(quoted))
    check('a routing header does not hide it', search_requested('[Seu nome foi citado]\n---\nquill pesquisa isso'))
    check('hint names the tool', 'web_search' in SEARCH_HINT)


async def test_search_hint_reaches_the_model(tmp_path, monkeypatch):
    import core.runner as runner_mod
    from core.context_tools import register
    from core.profiles import CHAT
    from core.runner import run_turn
    from core.tools import Tool, tool_schema
    from core.turn import TurnContext
    from chat_fakes import make_services
    from helpers import as_stream

    monkeypatch.setattr(runner_mod, 'CHANNEL_CONTEXT_MESSAGES', 0)
    services = make_services(tmp_path)
    calls = []

    class Completions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            msg = NS(role='assistant', content='ok', tool_calls=None)
            return as_stream(NS(choices=[NS(message=msg, finish_reason='stop')], usage=None))

    services.llm.client = NS(chat=NS(completions=Completions()))
    register(services.tools)

    async def search(ctx, args):
        return 'x'

    services.tools.register(Tool('web_search', tool_schema('web_search', 'w'), search))
    bot = NS(user=NS(id=900, name='quill', display_name='Quill'), services=services)
    await run_turn(TurnContext(bot=bot, profile=CHAT, question='pesquise esse sinal misterioso'))
    user = [m for m in calls[0]['messages'] if isinstance(m, dict) and m.get('role') == 'user'][-1]
    check('hint in the current message', SEARCH_HINT in user['content'], user['content'][-300:])
    calls.clear()
    await run_turn(TurnContext(bot=bot, profile=CHAT, question='bom dia'))
    user = [m for m in calls[0]['messages'] if isinstance(m, dict) and m.get('role') == 'user'][-1]
    check('no hint otherwise', SEARCH_HINT not in user['content'])
