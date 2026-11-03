"""The agentic tool loop shared by every kind of turn.

:func:`run_tool_loop` calls the model, runs the tools it asks for, feeds the
results back and repeats until it answers. It also recovers tool calls that
some providers return as raw text, retries an empty answer once, and returns
the JSON-safe trajectory that conversation replay stores with the turn.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any

from config import LLM_MAX_TOKENS
from core.discord_text import truncate_safe
from core.llm import _get, cached_tokens, create_chat_completion, usage_summary as _usage_summary

logger = logging.getLogger(__name__)


def _accumulate_usage(totals: dict[str, int], response: Any) -> None:
    """Add one completion's usage into *totals* (prompt/completion/cached)."""
    usage = _get(response, 'usage', None)
    if usage is None:
        return
    for key in ('prompt_tokens', 'completion_tokens'):
        value = _get(usage, key, None)
        if value:
            totals[key] += value
    cached = cached_tokens(usage)
    if cached:
        totals['cached_tokens'] += cached


def serialize_trajectory(messages: list[Any]) -> list[dict]:
    """Convert appended loop messages (SDK objects or dicts) to JSON-safe dicts.

    Only a whitelist of fields is kept (role/content/tool_calls/tool_call_id):
    provider extras such as ``reasoning``/thinking payloads are dropped because
    most APIs reject them on input, and the result is what gets persisted in the
    conversation store and replayed verbatim on follow-ups.
    """
    out: list[dict] = []
    for msg in messages:
        role = _get(msg, 'role')
        if role == 'assistant':
            content = _get(msg, 'content')
            entry: dict = {
                'role': 'assistant',
                'content': content if isinstance(content, str) else None,
            }
            calls: list[dict] = []
            for tc in _get(msg, 'tool_calls') or []:
                fn = _get(tc, 'function') or {}
                name = _get(fn, 'name')
                tc_id = _get(tc, 'id')
                if not name or not tc_id:
                    continue
                calls.append({
                    'id': tc_id,
                    'type': 'function',
                    'function': {
                        'name': name,
                        'arguments': _get(fn, 'arguments') or '{}',
                    },
                })
            if calls:
                entry['tool_calls'] = calls
            out.append(entry)
        elif role == 'tool':
            tc_id = _get(msg, 'tool_call_id')
            content = _get(msg, 'content')
            if not tc_id or not isinstance(content, str):
                continue
            out.append({'role': 'tool', 'tool_call_id': tc_id, 'content': content})
        elif role == 'user':
            # A leading user entry can only be the empty-answer retry nudge
            # (emitted when the model returned no tool calls AND no content in
            # round 1).  Keeping it would replay as two consecutive user
            # messages — providers enforcing strict user/assistant alternation
            # (Anthropic) reject that — so it is dropped from the capture.
            if not out:
                continue
            content = _get(msg, 'content')
            out.append({'role': 'user', 'content': content if isinstance(content, str) else ''})
    return out


# Tool calls some providers hand back as raw text in ``content`` instead of
# ``tool_calls``: always when the request carries no ``tools`` (the forced
# final round), and whenever their parser misses the model's native format.
# Bodies: GLM (``name<arg_key>k</arg_key><arg_value>v</arg_value>``),
# Qwen3-Coder (``<function=name><parameter=k>v</parameter></function>``) and
# Hermes/Qwen JSON (``{"name": ..., "arguments": {...}}``).  An unterminated
# block (output cut at max_tokens) runs to the end of the text and is only
# stripped: its arguments are incomplete.
_TEXT_TOOL_CALL_RE = re.compile(r'<tool_call>(.*?)(</tool_call>|\Z)', re.S)
_GLM_ARG_RE = re.compile(r'<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>', re.S)
_XML_FUNCTION_RE = re.compile(r'<function=([^>\s]+)>(.*?)(?:</function>|\Z)', re.S)
_XML_PARAM_RE = re.compile(r'<parameter=([^>\s]+)>(.*?)</parameter>', re.S)


def _parse_text_tool_call(body: str, schemas: dict[str, dict]) -> tuple[str, dict] | None:
    body = body.strip()
    if body.startswith('{'):
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict):
            return None
        name = obj.get('name')
        args = obj.get('arguments', obj.get('parameters', {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return None
        if name not in schemas or not isinstance(args, dict):
            return None
        return name, args
    fn = _XML_FUNCTION_RE.search(body)
    if fn:
        name, pairs = fn.group(1).strip(), _XML_PARAM_RE.findall(fn.group(2))
    else:
        name, pairs = body.split('<', 1)[0].strip(), _GLM_ARG_RE.findall(body)
    if name not in schemas:
        return None
    args: dict[str, Any] = {}
    for key, value in pairs:
        key, value = key.strip(), value.strip()
        if schemas[name].get(key, {}).get('type', 'string') != 'string':
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        args[key] = value
    return name, args


def extract_text_tool_calls(
    content: str | None, tools: list[dict] | None = None,
) -> tuple[str, list[tuple[str, dict]], bool]:
    """Split raw-text tool-call markup out of an assistant *content*.

    Returns ``(text_without_markup, calls, had_markup)``.  *calls* holds the
    ``(name, args)`` of blocks that parse as a call to one of *tools*; every
    ``<tool_call>`` block is removed from the text either way, so the markup
    never reaches the user.
    """
    if not content or '<tool_call>' not in content:
        return content or '', [], False
    schemas: dict[str, dict] = {}
    for tool in tools or []:
        fn = tool.get('function') or {}
        if fn.get('name'):
            schemas[fn['name']] = (fn.get('parameters') or {}).get('properties') or {}
    calls = [
        parsed
        for m in _TEXT_TOOL_CALL_RE.finditer(content)
        if m.group(2) and (parsed := _parse_text_tool_call(m.group(1), schemas))
    ]
    return _TEXT_TOOL_CALL_RE.sub('', content).strip(), calls, True


# What run_tool_loop answers when the model produced nothing usable.
EMPTY_ANSWER_FALLBACK = 'Não foi possível gerar uma resposta.'


async def run_tool_loop(
    client: Any,
    model: str,
    messages: list[dict],
    tools: list[dict] | None,
    exec_tool: Callable[[str, dict[str, Any]], Awaitable[tuple[str, list[dict]]]],
    status_label: Callable[[str, dict[str, Any]], str] | None = None,
    on_status: Callable[[str], Awaitable[None]] | None = None,
    max_rounds: int = 6,
    dedup_key: Callable[[dict], str] | None = None,
) -> tuple[str, list[dict], list[dict]]:
    """Run an agentic tool-calling loop until the LLM produces a final answer.

    Parameters
    ----------
    client:
        The OpenAI-compatible async client.
    model:
        The model ID to use for all chat completions.
    messages:
        The message list (system prompt, history, current question).  Mutated
        in place with tool calls and results.
    tools:
        The tool definitions to offer, or ``None`` to skip the loop entirely.
    exec_tool:
        Async callable ``(name, args) -> (result_text, sources)``.
    status_label:
        Optional callable ``(name, args) -> status_text`` describing a tool
        while it runs.
    on_status:
        Optional async callable receiving that text (e.g. to show it on a
        deferred slash command). Failures are ignored.
    max_rounds:
        Maximum iterations of the tool-calling loop.
    dedup_key:
        Optional callable ``(source_dict) -> str`` that returns a unique
        identifier for a source.  When provided, duplicate sources across
        rounds are filtered out.  When ``None``, no deduplication is applied.

    Returns
    -------
    ``(answer_text, all_sources, trajectory)`` where *all_sources* is the
    deduplicated list of source dicts aggregated across all tool rounds and
    *trajectory* is the JSON-safe list of messages appended during the loop
    (assistant tool calls, tool results, the final answer — plus the
    empty-answer retry nudge when it fired).  It is persisted with the
    conversation turn and replayed verbatim on follow-ups, so later turns see
    the full internal steps of earlier ones and the replayed prefix stays
    byte-identical to what was actually sent (keeping provider prefix caches
    effective).
    """
    all_sources: list[dict] = []
    seen_keys: set[str] = set()
    rounds_used = 0
    finish_reasons: list[str] = []
    usage_totals: dict[str, int] = {'prompt_tokens': 0, 'completion_tokens': 0, 'cached_tokens': 0}
    capture_start = len(messages)

    for round_num in range(1, max_rounds + 1):
        rounds_used = round_num
        response = await create_chat_completion(
            client,
            model=model,
            messages=messages,
            max_tokens=LLM_MAX_TOKENS,
            tools=tools,
        )
        _accumulate_usage(usage_totals, response)

        choice = response.choices[0]
        finish_reason = getattr(choice, 'finish_reason', None) or 'unknown'
        finish_reasons.append(finish_reason)
        logger.debug(
            "Tool loop round %d/%d: model=%s finish_reason=%s tool_calls=%d content_chars=%d usage=[%s]",
            round_num,
            max_rounds,
            model,
            finish_reason,
            len(choice.message.tool_calls or []),
            len(choice.message.content or ''),
            _usage_summary(response),
        )

        tool_calls = list(choice.message.tool_calls or [])
        assistant_msg: Any = choice.message
        if not tool_calls and tools:
            text, parsed, _ = extract_text_tool_calls(choice.message.content, tools)
            if parsed:
                logger.warning(
                    "Recovered %d tool call(s) returned as text (model=%s finish_reason=%s): %s",
                    len(parsed), model, finish_reason, [name for name, _ in parsed],
                )
                tool_calls = [
                    SimpleNamespace(
                        id=f'call_{uuid.uuid4().hex[:24]}',
                        type='function',
                        function=SimpleNamespace(name=name, arguments=json.dumps(args, ensure_ascii=False)),
                    )
                    for name, args in parsed
                ]
                assistant_msg = {
                    'role': 'assistant',
                    'content': text or None,
                    'tool_calls': [
                        {
                            'id': tc.id,
                            'type': 'function',
                            'function': {'name': tc.function.name, 'arguments': tc.function.arguments},
                        }
                        for tc in tool_calls
                    ],
                }
        if not tool_calls:
            break

        messages.append(assistant_msg)

        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "Malformed tool call arguments (model=%s finish_reason=%s): %r",
                    model,
                    finish_reason,
                    (tc.function.arguments or '')[:200],
                )
                args = {}

            if on_status is not None and status_label is not None:
                try:
                    await on_status(status_label(tc.function.name, args))
                except Exception:
                    logger.debug('status update failed', exc_info=True)

            try:
                result_text, sources = await exec_tool(tc.function.name, args)
            except Exception as e:
                # One failing tool (bad model-supplied args, API hiccup) must
                # not abort the whole answer: report it back to the model,
                # which can retry with other arguments or answer without it.
                logger.exception(
                    "Tool %s failed (model=%s args=%r)",
                    tc.function.name, model, str(args)[:200],
                )
                result_text = (
                    f'Erro ao executar a ferramenta {tc.function.name}: '
                    f'{type(e).__name__}: {str(e)[:200]}. '
                    'Corrija os argumentos ou responda sem esta ferramenta.'
                )
                sources = []

            if sources:
                if dedup_key is not None:
                    new_sources = [
                        s for s in sources if dedup_key(s) not in seen_keys
                    ]
                    if len(new_sources) < len(sources):
                        logger.info(
                            "Filtered %d duplicate source(s) from %s tool result",
                            len(sources) - len(new_sources),
                            tc.function.name,
                        )
                    for s in sources:
                        k = dedup_key(s)
                        if k:
                            seen_keys.add(k)
                    all_sources.extend(new_sources)

                    if not new_sources and sources:
                        result_text = (
                            'Os resultados desta busca já foram retornados '
                            'em uma rodada anterior. Use as informações já '
                            'fornecidas para formular sua resposta.'
                        )
                else:
                    all_sources.extend(sources)

            messages.append({
                'role': 'tool',
                'tool_call_id': tc.id,
                'content': truncate_safe(result_text, limit=6000),
            })
    else:
        logger.info(
            "Tool loop exhausted max_rounds=%d without a final answer (finish_reasons=%s); "
            "forcing one completion without tools",
            max_rounds,
            finish_reasons,
        )
        response = await create_chat_completion(
            client,
            model=model,
            messages=messages,
            max_tokens=LLM_MAX_TOKENS,
        )
        _accumulate_usage(usage_totals, response)

    final_choice = response.choices[0]
    # A reply that still tries to call a tool (as text: no tools were sent, or
    # none of its calls parsed) is not an answer — only its preamble would be
    # left once the markup is stripped, so it gets the same retry as an empty one.
    answer, _, had_markup = extract_text_tool_calls(final_choice.message.content)
    if not answer.strip() or had_markup:
        logger.warning(
            "LLM returned %s (finish_reason=%s usage=[%s]); "
            "retrying once without tools and with a conciseness nudge",
            'a tool call as text instead of an answer' if had_markup else 'an empty answer',
            getattr(final_choice, 'finish_reason', None),
            _usage_summary(response),
        )
        # The nudge is appended to `messages` itself (payload-identical to the
        # previous copy) so the captured trajectory includes exactly what was
        # sent and the follow-up prefix keeps matching the provider cache.
        messages.append({
            'role': 'user',
            'content': (
                'Responda agora, em português, de forma direta e concisa, sem '
                'usar nenhuma ferramenta. Use apenas as informações já coletadas.'
            ),
        })
        response = await create_chat_completion(
            client,
            model=model,
            messages=messages,
            max_tokens=LLM_MAX_TOKENS,
        )
        _accumulate_usage(usage_totals, response)
        final_choice = response.choices[0]
        answer, _, _ = extract_text_tool_calls(final_choice.message.content)
    if not answer.strip():
        logger.warning(
            "LLM returned an empty answer (user got the fallback message): "
            "model=%s rounds=%d final_finish_reason=%s finish_reasons=%s usage=[%s]",
            model,
            rounds_used,
            getattr(final_choice, 'finish_reason', None),
            finish_reasons,
            _usage_summary(response),
        )
        answer = EMPTY_ANSWER_FALLBACK
    messages.append({'role': 'assistant', 'content': answer})
    trajectory = serialize_trajectory(messages[capture_start:])
    prompt_total = usage_totals['prompt_tokens']
    if prompt_total:
        cached_total = usage_totals['cached_tokens']
        logger.info(
            "Turn usage: model=%s rounds=%d prompt=%d cached=%d hit=%.0f%% completion=%d",
            model,
            rounds_used,
            prompt_total,
            cached_total,
            100.0 * cached_total / prompt_total,
            usage_totals['completion_tokens'],
        )
    return answer, all_sources, trajectory
