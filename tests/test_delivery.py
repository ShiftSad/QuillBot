"""Delivery: plain messages vs embeds, splitting, pings and sources."""

from types import SimpleNamespace as NS

import discord

from core import delivery
from core.delivery import Outgoing, choose_style, render, split_plain, sources_line
from core.discord_text import resolve_mentions

from helpers import check

ANA = NS(id=1, bot=False, display_name='Ana Paula', name='ana')
BOB = NS(id=2, bot=False, display_name='Bob', name='bob')
BOT = NS(id=3, bot=True, display_name='Quill', name='quill')
GUILD = NS(
    members=[ANA, BOB, BOT],
    get_member_named=lambda n: None,
    get_member=lambda i: {1: ANA, 2: BOB, 3: BOT}.get(i),
)


def test_style_follows_the_channel_setting_and_the_answer():
    check('short chat answers are plain', choose_style('valeu, de nada!') == 'plain')
    check('summaries and reports are embeds', choose_style('resumo', structured=True) == 'embed')
    check('answers longer than one message are embeds', choose_style('x' * 2500) == 'embed')
    check('a channel can force plain', choose_style('x' * 2500, preference='plain') == 'plain')
    check('or embeds', choose_style('oi', preference='embed') == 'embed')


def test_mentions_become_real_pings_for_humans_only():
    text, ids = resolve_mentions('oi @Ana Paula, e @bob! fala @Quill e @everyone `@bob` <@2>', GUILD)
    check('multi-word name and handle rewritten', text.startswith('oi <@1>, e <@2>!'), text)
    check('bots, everyone and code untouched', '@Quill' in text and '@everyone' in text and '`@bob`' in text, text)
    check('only humans ping, once each', ids == [1, 2], ids)
    check('no guild: nothing changes', resolve_mentions('@Ana', None) == ('@Ana', []))


def test_plain_render_allows_only_resolved_pings():
    [kwargs] = render(Outgoing('valeu @bob!'), GUILD)
    check('content is the text', kwargs['content'] == 'valeu <@2>!', kwargs)
    allowed = kwargs['allowed_mentions']
    check('only that user may be pinged', [u.id for u in allowed.users] == [2] and not allowed.everyone and not allowed.roles, allowed)
    check('no embed', 'embed' not in kwargs)


def test_plain_sources_ride_as_subtext():
    sources = [{'title': 'Paper [docs]', 'url': 'https://docs.papermc.io/x'}]
    [kwargs] = render(Outgoing('Use view-distance 6.', sources=sources), GUILD)
    check('sources line appended', kwargs['content'].endswith('-# Fontes: [Paper (docs)](<https://docs.papermc.io/x>)'), kwargs['content'])
    check('empty without sources', sources_line([]) == '')


def test_long_plain_answers_split_and_keep_code_blocks_valid():
    text = 'intro\n' + '\n'.join(f'linha {i} ' + 'x' * 50 for i in range(30)) + '\n```yaml\n' + \
        '\n'.join(f'k{i}: ' + 'v' * 60 for i in range(40)) + '\n```\nfim'
    parts = split_plain(text)
    check('every part fits a message', all(len(p) <= 2000 for p in parts), [len(p) for p in parts])
    check('every part has balanced fences', all(p.count('```') % 2 == 0 for p in parts), [p.count('```') for p in parts])
    check('reopened with its language', any(p.startswith('```yaml') for p in parts[1:]))
    check('nothing lost', parts[-1].endswith('fim') and parts[0].startswith('intro'))
    check('a single huge line is cut', all(len(p) <= 2000 for p in split_plain('a' * 4500)))


def test_embed_render_puts_pings_in_content_and_pages_sources():
    out = Outgoing('### Diagnóstico\nfale com @bob', sources=[{'title': 'T', 'url': 'https://u', 'source': 'Spark'}],
                   style='embed', title='🔥 Spark', color=0xE67E22, footer='rodapé', sources_title='📄 Fontes')
    [kwargs] = render(out, GUILD)
    check('embed with title', kwargs['embed'].title == '🔥 Spark' and '<@2>' in kwargs['embed'].description, kwargs)
    check('pings go in content (embeds never notify)', kwargs['content'] == '<@2>', kwargs.get('content'))
    check('sources page behind the paginator', isinstance(kwargs['view'], discord.ui.View) and len(kwargs['view'].embeds) == 2)


async def test_deliver_threads_and_replies():
    sent = []

    class Target:
        def __init__(self, mid):
            self.mid = mid

        async def reply(self, **kwargs):
            sent.append(('reply', kwargs))
            return NS(id=self.mid)

        async def send(self, **kwargs):
            sent.append(('send', kwargs))
            return NS(id=self.mid + len(sent))

    message = Target(10)
    message.channel = Target(20)
    msgs = await delivery.deliver(Outgoing('a' * 2500 + '\nfim', style='plain'), guild=GUILD, reply_to=message, mention_author=False)
    check('first part replies, the rest go to the channel', [k for k, _ in sent] == ['reply', 'send'], sent)
    check('mention_author passed to the reply', sent[0][1]['mention_author'] is False)
    check('every message returned', len(msgs) == 2)

    sent.clear()
    thread = Target(30)
    await delivery.deliver(Outgoing('oi'), guild=GUILD, thread=thread, asker=NS(id=1, mention='<@1>'))
    kind, kwargs = sent[0]
    check('thread answer pings the asker', kind == 'send' and kwargs['content'] == '<@1> oi' and [u.id for u in kwargs['allowed_mentions'].users] == [1], kwargs)
