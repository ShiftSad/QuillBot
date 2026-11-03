"""Documentation RAG: indexing the doc sources and searching them.

Indexes Markdown from the configured GitHub repos (``DOC_SOURCES``) into an
embedding matrix, keeps it fresh, and serves searches. The ``search_docs`` and
``search_plugins`` tools it registers are offered to every conversation (not
only ``/ask``), so the bot answers Minecraft questions from the docs wherever
people ask them.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import time

import aiohttp
import discord
import numpy as np
from discord import app_commands
from discord.ext import commands, tasks

from cogs.plugin_apis import HTTP_HEADERS as _HTTP_HEADERS
from cogs.plugin_apis import search_all as _search_plugins_all
from config import (
    DOC_SOURCES,
    DOCS_BASE_URL,
    DOCS_BRANCH,
    EMBEDDING_PROVIDER,
    GITHUB_API,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    REINDEX_INTERVAL_HOURS,
    RERANK_AVAILABLE,
    RERANK_MODEL,
    VECTOR_STORE_PATH,
)
from core.tools import Tool

logger = logging.getLogger(__name__)

# --- Tool definitions for the LLM (OpenAI function-calling format) ---
_SOURCE_LABELS = [src['label'] for src in DOC_SOURCES]
_SOURCE_LABELS_STR = ', '.join(f'"{s}"' for s in _SOURCE_LABELS)

TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'search_docs',
            'description': (
                'Pesquisa na documentação indexada. '
                f'Fontes disponíveis: {_SOURCE_LABELS_STR}. '
                'Use para qualquer pergunta sobre configuração, administração ou setup de servidores Minecraft. '
                'Use o parâmetro `source` para restringir a busca a uma fonte específica quando a pergunta '
                'claramente pertence a um projeto concreto (ex: Spark para profiling, PaperMC para configurações '
                'Paper). Omita `source` para perguntas gerais ou que cruzam múltiplas documentações. '
                'Não use para perguntas sobre plugins específicos — use search_plugins para isso.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'A consulta de busca em linguagem natural.',
                    },
                    'max_results': {
                        'type': 'integer',
                        'description': (
                            'Número máximo de resultados (padrão: 5, intervalo: 1-12). '
                            'Use 3-5 para perguntas focadas, 8-12 para tópicos amplos.'
                        ),
                    },
                    'source': {
                        'type': 'string',
                        'enum': _SOURCE_LABELS,
                        'description': (
                            f'Filtrar por fonte de documentação específica ({_SOURCE_LABELS_STR}). '
                            'Use apenas quando a pergunta é claramente específica de um projeto. '
                            'Se omitido, pesquisa em todas as fontes.'
                        ),
                    },
                },
                'required': ['query'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'search_plugins',
            'description': (
                'Pesquisa plugins no Modrinth, Hangar e SpigotMC. '
                'Use quando o usuário perguntar sobre plugins, recomendações de plugins ou alternativas. '
                'Não use para perguntas gerais de configuração de servidor — use search_docs para isso.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'Nome ou descrição do plugin a pesquisar.',
                    },
                },
                'required': ['query'],
            },
        },
    },
]


def _search_docs_status(args: dict, ctx) -> str:
    label = f"🔍 Pesquisando documentação: *{str(args.get('query', ''))[:60]}*"
    if args.get('source'):
        label += f" ({args['source']})"
    return label


def _search_plugins_status(args: dict, ctx) -> str:
    return f"🔌 Pesquisando plugins: *{str(args.get('query', ''))[:60]}*"


def _parse_frontmatter(content: str) -> dict:
    """Extract key-value pairs from YAML frontmatter (between --- delimiters)."""
    if not content.startswith('---'):
        return {}
    end = content.find('\n---', 3)
    if end == -1:
        return {}
    fm_text = content[3:end]
    result = {}
    for line in fm_text.splitlines():
        m = re.match(r'^(\w+)\s*:\s*(.+)$', line)
        if m:
            result[m.group(1)] = m.group(2).strip().strip('"\'')
    return result


def _title_from_path(path: str) -> str:
    """Derive a human-readable title from a file path as a last resort."""
    name = path.rsplit('/', 1)[-1]
    name = re.sub(r'\.(md|mdx)$', '', name)
    name = re.sub(r'[-_]', ' ', name)
    return name.title()


def _compute_doc_url(path: str, base_url: str, url_strip_prefix: str = '') -> str:
    """Convert a repo file path to its docs website URL."""
    base_url = base_url.rstrip('/')
    if url_strip_prefix and path.startswith(url_strip_prefix):
        path = path[len(url_strip_prefix):]
    # Strip extension (suffix only, longest first to avoid .mdx -> x)
    for ext in ('.mdx', '.md'):
        if path.endswith(ext):
            path = path[:-len(ext)]
            break
    # Strip README and index from the final path segment (mdBook / MkDocs conventions)
    for index_name in ('/README', '/index'):
        if path.endswith(index_name):
            path = path[:-len(index_name)]
            break
    if path in ('README', 'index'):
        path = ''
    url = path.rstrip('/')
    return f'{base_url}/{url}' if url else base_url


def path_to_docs_url(path: str) -> str:
    """Convert a MinersRefuge repo file path to its docs URL (backward compat)."""
    return _compute_doc_url(path, DOCS_BASE_URL)



class DocsRAG(commands.Cog):
    """RAG-powered documentation search with vector storage and reranking."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.services = bot.services
        self.session: aiohttp.ClientSession | None = None
        self.chunks: list[dict] = []  # {content, path, title, source, doc_url}; embeddings live only in _emb_matrix
        self._last_commit_sha: str | None = None
        self._indexing: bool = False
        self._emb_matrix: np.ndarray | None = None  # (N, dim) float32 for vectorized search
        # Per-source SHA tracking for granular reindex
        self._source_shas: dict[str, str] = {}  # label -> latest commit SHA
        self._source_last_index: dict[str, float] = {}  # label -> timestamp

    async def cog_load(self):
        self.session = aiohttp.ClientSession(
            headers=_HTTP_HEADERS,
            timeout=aiohttp.ClientTimeout(total=15),
        )
        self.services.docs = self
        self.services.tools.register(
            Tool('search_docs', TOOLS[0], self._tool_search_docs, _search_docs_status),
            Tool('search_plugins', TOOLS[1], self._tool_search_plugins, _search_plugins_status),
        )
        await self.services.llm.warm_up()
        loaded = await asyncio.to_thread(self._load_vectors)
        if not loaded:
            await self.index_docs()
        self.periodic_reindex.start()

    async def cog_unload(self):
        self.periodic_reindex.cancel()
        self.services.tools.unregister('search_docs', 'search_plugins')
        if self.services.docs is self:
            self.services.docs = None
        if self.session:
            await self.session.close()

    # --- Public state (used by /docs and /health) ---

    @property
    def indexing(self) -> bool:
        return self._indexing

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def last_commit_sha(self) -> str | None:
        return self._last_commit_sha

    @property
    def source_last_index(self) -> dict[str, float]:
        return dict(self._source_last_index)

    # --- Vector Storage ---

    def _save_vectors(self):
        """Persist chunk metadata to JSON and embeddings to a numpy binary file."""
        os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
        meta = {
            'commit_sha': self._last_commit_sha,
            'chunks': [
                {
                    'content': c['content'],
                    'path': c['path'],
                    'title': c['title'],
                    'source': c.get('source', "Miners' Refuge"),
                    'doc_url': c.get('doc_url', path_to_docs_url(c['path'])),
                    # embeddings stored separately in .npy
                }
                for c in self.chunks
            ],
        }
        with open(VECTOR_STORE_PATH, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False)
        npy_path = os.path.splitext(VECTOR_STORE_PATH)[0] + '.npy'
        np.save(npy_path, self._emb_matrix)
        logger.info("Saved %d vectors to %s + %s", len(self.chunks), VECTOR_STORE_PATH, npy_path)

    def _load_vectors(self) -> bool:
        """Load chunk metadata from JSON and embeddings from numpy binary file."""
        if not os.path.exists(VECTOR_STORE_PATH):
            return False
        npy_path = os.path.splitext(VECTOR_STORE_PATH)[0] + '.npy'
        try:
            with open(VECTOR_STORE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            chunks = data.get('chunks', [])

            if os.path.exists(npy_path):
                # New binary format
                embeddings = np.load(npy_path)
                if len(embeddings) != len(chunks):
                    logger.warning(
                        "Embedding count mismatch (%d vs %d chunks), will reindex",
                        len(embeddings), len(chunks),
                    )
                    return False
                local_dim = self.services.llm.local_dim
                if EMBEDDING_PROVIDER == 'local' and local_dim and embeddings.shape[1] != local_dim:
                    logger.warning(
                        "Embedding dim mismatch (stored %d vs local model %d), will reindex",
                        embeddings.shape[1], local_dim,
                    )
                    return False
            elif chunks and 'embedding' in chunks[0]:
                # Old JSON-with-embeddings format — migrate on load
                logger.info("Migrating vector store from JSON to binary format...")
                embeddings = np.array([c.pop('embedding') for c in chunks], dtype=np.float32)
            else:
                logger.warning("No embeddings found in vector store, will reindex")
                return False

            for chunk in chunks:
                chunk.setdefault('source', "Miners' Refuge")
                chunk.setdefault('doc_url', path_to_docs_url(chunk['path']))
            self.chunks = chunks
            self._emb_matrix = embeddings
            self._last_commit_sha = data.get('commit_sha')
            logger.info(
                "Loaded %d vectors from disk (commit: %s)",
                len(self.chunks),
                self._last_commit_sha,
            )
            # If loaded from old format, immediately save in new binary format
            if not os.path.exists(npy_path):
                self._save_vectors()
            return bool(self.chunks)
        except Exception:
            logger.exception("Failed to load vectors from %s", VECTOR_STORE_PATH)
            return False

    # --- Periodic Reindexing ---

    @tasks.loop(hours=REINDEX_INTERVAL_HOURS)
    async def periodic_reindex(self):
        """Check for doc updates across all sources and reindex if anything changed."""
        try:
            latest_sha = await self._get_composite_sha()
            if latest_sha and latest_sha != self._last_commit_sha:
                logger.info(
                    "Doc source changes detected (%s -> %s), reindexing...",
                    self._last_commit_sha,
                    latest_sha,
                )
                await self.index_docs()
            else:
                logger.info("All doc sources up to date (composite: %s)", self._last_commit_sha)
        except Exception:
            logger.exception("Error during periodic reindex check")

    @periodic_reindex.before_loop
    async def _wait_for_bot(self):
        await self.bot.wait_until_ready()

    async def _get_composite_sha(self) -> str | None:
        """Fetch latest commit SHAs from all configured doc sources and return a composite hash."""
        shas = []
        for source in DOC_SOURCES:
            url = f'https://api.github.com/repos/{source["repo"]}/commits/{source["branch"]}'
            try:
                async with self.session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        sha = data.get('sha', '')
                        if sha:
                            shas.append(f'{source["repo"]}:{sha}')
                    else:
                        logger.warning(
                            "Could not fetch commit SHA for %s (status %d)",
                            source['repo'], resp.status,
                        )
            except Exception:
                logger.exception("Failed to fetch commit SHA for %s", source['repo'])
        if not shas:
            return None
        return hashlib.md5(':'.join(sorted(shas)).encode()).hexdigest()

    async def _get_latest_commit_sha(self) -> str | None:
        """Fetch the latest commit SHA from the primary docs repo (MinersRefuge/docs).

        Kept for backward compatibility; prefer _get_composite_sha for reindex checks.
        """
        url = f'{GITHUB_API}/commits/{DOCS_BRANCH}'
        try:
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get('sha')
        except Exception:
            logger.exception("Failed to fetch latest commit SHA")
        return None

    # --- Indexing ---

    async def _fetch(self, url: str) -> str | None:
        try:
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    return await resp.text()
        except Exception:
            logger.exception("Failed to fetch %s", url)
        return None

    def _extract_paths(self, summary: str) -> list[str]:
        # Exclude external URLs (http/https) -- only match relative .md paths
        paths = re.findall(r'\((?!https?://)([^)]+\.md)\)', summary)
        return list(dict.fromkeys(paths))  # deduplicate preserving order

    def _extract_title(self, content: str) -> str:
        fm = _parse_frontmatter(content)
        if fm.get('title'):
            return fm['title']
        match = re.search(r'^#\s+(.+)', content, re.MULTILINE)
        return match.group(1).strip() if match else ''

    def _chunk_text(self, text: str, path: str, chunk_size: int = 1500) -> list[dict]:
        """Split markdown into chunks by headings, falling back to size-based splits."""
        title = self._extract_title(text)
        sections = re.split(r'(?=^##?\s)', text, flags=re.MULTILINE)
        result = []

        for section in sections:
            section = section.strip()
            if not section:
                continue
            if len(section) <= chunk_size:
                result.append({
                    'content': section,
                    'path': path,
                    'title': title,
                })
            else:
                paragraphs = section.split('\n\n')
                current = []
                current_size = 0
                for para in paragraphs:
                    if current_size + len(para) > chunk_size and current:
                        result.append({
                            'content': '\n\n'.join(current),
                            'path': path,
                            'title': title,
                        })
                        current = []
                        current_size = 0
                    current.append(para)
                    current_size += len(para)
                if current:
                    result.append({
                        'content': '\n\n'.join(current),
                        'path': path,
                        'title': title,
                    })
        return result

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        return await self.services.llm.embed(texts)

    async def _fetch_doc(
        self, path: str, github_raw: str, semaphore: asyncio.Semaphore
    ) -> tuple[str, str | None]:
        """Fetch a single doc file from the given GitHub raw base URL."""
        async with semaphore:
            content = await self._fetch(f'{github_raw}/{path}')
        return path, content

    async def _get_paths_from_tree(
        self, repo: str, branch: str, path_prefix: str = ''
    ) -> list[str]:
        """Use the GitHub tree API to discover .md/.mdx files under an optional prefix."""
        url = f'https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1'
        try:
            async with self.session.get(url) as resp:
                if resp.status != 200:
                    logger.warning("Tree API returned %d for %s", resp.status, repo)
                    return []
                data = await resp.json()
                tree = data.get('tree', [])
                return [
                    entry['path'] for entry in tree
                    if entry.get('type') == 'blob'
                    and (entry['path'].endswith('.md') or entry['path'].endswith('.mdx'))
                    and (not path_prefix or entry['path'].startswith(path_prefix))
                ]
        except Exception:
            logger.exception("Failed to get file tree for %s", repo)
            return []

    async def _index_source(
        self, source: dict, semaphore: asyncio.Semaphore
    ) -> list[dict]:
        """Index one documentation source and return its text chunks (no embeddings yet)."""
        repo = source['repo']
        branch = source['branch']
        base_url = source['base_url']
        label = source['label']
        github_raw = f'https://raw.githubusercontent.com/{repo}/{branch}'
        url_strip_prefix = source.get('url_strip_prefix', '')
        max_files = source.get('max_files', 200)

        # Discover file paths
        summary_path = source.get('summary')
        if summary_path:
            summary = await self._fetch(f'{github_raw}/{summary_path}')
            if not summary:
                logger.error(
                    "Failed to fetch %s from %s -- skipping source", summary_path, repo
                )
                return []
            paths = self._extract_paths(summary)
            paths.insert(0, 'README.md')
        else:
            path_prefix = source.get('path_prefix', '')
            paths = await self._get_paths_from_tree(repo, branch, path_prefix)

        paths = paths[:max_files]
        if not paths:
            logger.warning("No paths found for source '%s'", label)
            return []

        # Fetch documents concurrently (bounded by the shared semaphore)
        fetch_tasks = [self._fetch_doc(p, github_raw, semaphore) for p in paths]
        results = await asyncio.gather(*fetch_tasks)

        chunks = []
        fetched = 0
        for path, content in results:
            if content:
                fm = _parse_frontmatter(content)
                slug = fm.get('slug')
                if slug:
                    doc_url = f'{base_url}/{slug}'
                else:
                    doc_url = _compute_doc_url(path, base_url, url_strip_prefix)
                for chunk in self._chunk_text(content, path):
                    chunk['source'] = label
                    chunk['doc_url'] = doc_url
                    chunks.append(chunk)
                fetched += 1

        logger.info(
            "Source '%s': fetched %d/%d docs, %d chunks",
            label, fetched, len(paths), len(chunks),
        )
        return chunks

    async def index_docs(self, sources: list[dict] | None = None):
        """Fetch docs from all configured sources (or only *sources* if provided) and create embeddings."""
        if self._indexing:
            logger.info("index_docs() called while already indexing; skipping")
            return
        self._indexing = True
        try:
            await self._index_docs_inner(sources)
        finally:
            self._indexing = False

    async def _index_docs_inner(self, sources: list[dict] | None = None):
        partial = sources is not None
        sources_to_index = sources if partial else DOC_SOURCES
        logger.info(
            "Indexing %d documentation source(s)...",
            len(sources_to_index),
        )

        # All sources share a semaphore to cap total concurrent HTTP fetches
        semaphore = asyncio.Semaphore(5)
        source_tasks = [self._index_source(src, semaphore) for src in sources_to_index]
        source_results = await asyncio.gather(*source_tasks, return_exceptions=True)

        all_chunks = []
        failed_labels: set[str] = set()
        for src, result in zip(sources_to_index, source_results, strict=True):
            if isinstance(result, BaseException):
                logger.error("Error indexing source '%s': %s", src['label'], result)
                failed_labels.add(src['label'])
            elif not result:
                # Nothing fetched (GitHub outage, rate limit): treat as a
                # failure so the source keeps its previous chunks.
                failed_labels.add(src['label'])
            else:
                all_chunks.extend(result)

        logger.info("Total chunks before embedding: %d", len(all_chunks))

        # Generate embeddings in batches
        batch_size = 20
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            texts = [c['content'] for c in batch]
            try:
                embeddings = await self._embed_batch(texts)
                if len(embeddings) != len(batch):
                    raise ValueError(f'{len(embeddings)} embeddings for {len(batch)} chunks')
                for chunk, emb in zip(batch, embeddings):
                    chunk['embedding'] = emb
            except Exception:
                logger.exception("Failed to embed batch %d", i // batch_size)
                failed_labels.update(c['source'] for c in batch)

        # A source is replaced only when every one of its chunks embedded;
        # otherwise its previous chunks stay, so a flaky run (or a single-source
        # /reindex) never shrinks the rest of the index.
        replaced = {src['label'] for src in sources_to_index} - failed_labels
        if failed_labels:
            logger.warning(
                "Doc source(s) failed to index, keeping their previous chunks: %s",
                ', '.join(sorted(failed_labels)),
            )
        new_chunks = [c for c in all_chunks if c['source'] in replaced]
        if not new_chunks:
            logger.warning("Reindex produced no usable chunks; existing index left untouched")
            return
        # The float32 matrix is the single source of truth for embeddings:
        # pop() the per-chunk copies (Python-float lists cost ~8x the matrix).
        new_matrix = np.array([c.pop('embedding') for c in new_chunks], dtype=np.float32)

        # Carry over every chunk whose source was not replaced — except, on a
        # full reindex, sources that were removed from DOC_SOURCES.
        configured = {src['label'] for src in DOC_SOURCES}
        keep_idx = [
            i for i, c in enumerate(self.chunks)
            if c.get('source') not in replaced and (partial or c.get('source') in configured)
        ]
        kept_chunks = [self.chunks[i] for i in keep_idx]
        kept_matrix: np.ndarray | None = None
        if keep_idx and self._emb_matrix is not None:
            kept_matrix = self._emb_matrix[np.array(keep_idx)]
            if kept_matrix.ndim != 2 or kept_matrix.shape[1] != new_matrix.shape[1]:
                logger.warning(
                    "Dropping %d carried-over chunks: embedding width changed (%s -> %s)",
                    len(kept_chunks), kept_matrix.shape, new_matrix.shape,
                )
                kept_chunks, kept_matrix = [], None
        self.chunks = kept_chunks + new_chunks
        self._emb_matrix = new_matrix if kept_matrix is None else np.vstack([kept_matrix, new_matrix])
        logger.info(
            "Documentation indexed: %d new chunks (%s), %d carried over, %d total",
            len(new_chunks), ', '.join(sorted(replaced)), len(kept_chunks), len(self.chunks),
        )

        # Track per-source SHAs and wall-clock index times (shown by /health)
        now = time.time()
        for src in sources_to_index:
            label = src['label']
            if label not in replaced:
                continue
            url = f'https://api.github.com/repos/{src["repo"]}/commits/{src["branch"]}'
            try:
                async with self.session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self._source_shas[label] = data.get('sha', '')
            except Exception:
                logger.exception("Failed to fetch commit SHA for %s", label)
            self._source_last_index[label] = now
        if not partial:
            if failed_labels:
                # Leave the composite SHA stale so the periodic check retries.
                logger.warning("Composite SHA not advanced: failed sources will be retried")
            else:
                self._last_commit_sha = await self._get_composite_sha()

        await asyncio.to_thread(self._save_vectors)

    # --- Search with Reranking ---

    async def _rerank(self, query: str, documents: list[dict], top_n: int = 5) -> list[dict]:
        """Rerank documents using the reranking model via OpenAI-compatible API."""
        if not RERANK_AVAILABLE:
            return documents[:top_n]
        try:
            headers = {
                'Authorization': f'Bearer {OPENAI_API_KEY}',
                'Content-Type': 'application/json',
            }
            payload = {
                'model': RERANK_MODEL,
                'query': query,
                'documents': [d['content'] for d in documents],
                'top_n': top_n,
            }
            async with self.session.post(
                f'{OPENAI_BASE_URL}/rerank',
                headers=headers,
                json=payload,
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    results = data.get('results', [])
                    results.sort(key=lambda r: r.get('relevance_score', 0), reverse=True)
                    return [documents[r['index']] for r in results[:top_n]]
                else:
                    body = await resp.text()
                    logger.warning("Rerank failed (status %d): %s", resp.status, body)
        except Exception:
            logger.exception("Rerank request failed, falling back to embedding scores")
        # Fallback: return documents as-is (already sorted by cosine similarity)
        return documents[:top_n]

    async def search(self, query: str, top_k: int = 12, source_filter: str | None = None) -> list[dict]:
        if not self.chunks or self._emb_matrix is None:
            return []

        # Apply optional source filter
        if source_filter:
            indices = [i for i, c in enumerate(self.chunks) if c.get('source') == source_filter]
            if not indices:
                return []
            idx_arr = np.array(indices)
            chunks = [self.chunks[i] for i in indices]
            emb_matrix = self._emb_matrix[idx_arr]
        else:
            chunks = self.chunks
            emb_matrix = self._emb_matrix

        try:
            query_emb = (await self._embed_batch([query]))[0]
        except Exception:
            logger.exception("Failed to embed query")
            return []

        query_arr = np.array(query_emb, dtype=np.float32)
        dots = emb_matrix @ query_arr
        norms = np.linalg.norm(emb_matrix, axis=1) * np.linalg.norm(query_arr)
        with np.errstate(invalid='ignore', divide='ignore'):
            scores = np.where(norms > 0, dots / norms, 0.0)

        top_indices = np.argsort(scores)[::-1][:top_k * 3]
        candidates = [chunks[i] for i in top_indices]

        # Rerank for better precision
        reranked = await self._rerank(query, candidates, top_n=top_k)
        return reranked


    # --- LLM tools ---

    async def _tool_search_docs(self, ctx, args: dict) -> tuple[str, list[dict]]:
        if self._indexing and not self.chunks:
            return 'A documentação está sendo indexada; tente de novo em instantes.', []
        query = args.get('query', '')
        try:
            top_k = max(1, min(12, int(args.get('max_results', 5))))
        except (TypeError, ValueError):
            top_k = 5
        results = await self.search(query, top_k=top_k, source_filter=args.get('source') or None)
        if not results:
            return 'Nenhum resultado encontrado na documentação.', []
        parts = []
        sources = []
        for r in results:
            doc_url = r.get('doc_url', path_to_docs_url(r['path']))
            parts.append(f"[Fonte: {r['title']} — {doc_url}]\n{r['content']}")
            sources.append({
                'title': r['title'] or _title_from_path(r['path']),
                'url': doc_url,
                'source': r.get('source', "Miners' Refuge"),
                'path': r['path'],
            })
        return '\n\n---\n\n'.join(parts), sources

    async def _tool_search_plugins(self, ctx, args: dict) -> tuple[str, list[dict]]:
        results = await _search_plugins_all(self.session, args.get('query', ''))
        if not results:
            return 'Nenhum plugin encontrado.', []
        lines = []
        for r in results[:6]:
            versions = ', '.join(str(v) for v in r.get('versions', [])[:3]) or '—'
            lines.append(
                f"**{r['name']}** ({r['source']}) — {r.get('description', '')}\n"
                f"Downloads: {r.get('downloads', 0):,} | Versões: {versions}\n"
                f"URL: {r['url']}"
            )
        return '\n\n'.join(lines), []

    @app_commands.command(
        name='reindex',
        description='Re-indexar a documentação (Admin)',
    )
    @app_commands.checks.has_permissions(administrator=True)
    @app_commands.describe(
        source='Fonte específica para reindexar (opcional; omitir para reindexar tudo)',
    )
    async def reindex(
        self,
        interaction: discord.Interaction,
        source: str | None = None,
    ):
        if self._indexing:
            await interaction.response.send_message(
                '📚 Já há uma indexação em andamento, aguarde a conclusão.',
                ephemeral=True,
            )
            return

        # Validate source name if provided
        if source:
            source_labels = [s['label'] for s in DOC_SOURCES]
            matching = [s for s in DOC_SOURCES if s['label'] == source]
            if not matching:
                await interaction.response.send_message(
                    f'Fonte "{source}" não encontrada. Fontes disponíveis: '
                    f'{", ".join(source_labels)}',
                    ephemeral=True,
                )
                return
            sources_to_index = matching
            description = f'Somente a fonte "{source}"'
        else:
            sources_to_index = None
            description = 'Toda a documentação'

        await interaction.response.defer(thinking=True)
        target_msg = f'📚 Reindexando {description}…' if description else '📚 Reindexando documentação…'
        try:
            await interaction.edit_original_response(content=target_msg)
        except discord.HTTPException:
            pass
        await self.index_docs(sources_to_index)
        try:
            await interaction.edit_original_response(content=None)
        except discord.HTTPException:
            pass
        await interaction.followup.send(
            f'✅ {description} re-indexada! '
            f'({len(self.chunks)} chunks totais)'
        )




async def setup(bot: commands.Bot):
    await bot.add_cog(DocsRAG(bot))
