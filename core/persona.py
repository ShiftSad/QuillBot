"""The bot's personality on each server — a ``<personalidade>`` block in the system prompt.

Two sections per server:

- **base** — who the bot is: voice, humor, values, hard rules. Written by
  admins (``/persona editar``) or by the bot when someone with Manage Server
  asks it to.
- **notes** — what it picked up on that server: running jokes, nicknames,
  catchphrases. The bot rewrites this section itself (``persona_update``),
  a few times a day at most, and admins can freeze it.

Every change is a new version kept in ``persona_history`` (before and after),
so any of them can be reverted. Unlike memories, the block lives in the system
prompt: it carries instruction weight, and since it rarely changes it stays
in the provider's prompt cache. Reads are served from memory (the table is
loaded at startup and every write updates both).
"""

from __future__ import annotations

import datetime
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from typing import Any

from config import PERSONA_BASE_MAX_CHARS, PERSONA_NOTES_MAX_CHARS

logger = logging.getLogger(__name__)

SECTIONS: dict[str, dict[str, Any]] = {
    'base': {'label': 'Base', 'max': PERSONA_BASE_MAX_CHARS},
    'notes': {'label': 'Notas', 'max': PERSONA_NOTES_MAX_CHARS},
}


class PersonaError(ValueError):
    """User-facing error (too long, locked, not allowed…)."""


def can_edit_base(member: Any) -> bool:
    """Whether *member* may change the base personality (Manage Server or admin)."""
    perms = getattr(member, 'guild_permissions', None)
    return bool(perms and (perms.administrator or perms.manage_guild))


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


@dataclass(frozen=True)
class Version:
    """One change of one section."""

    id: int
    guild_id: int
    section: str
    version: int
    before: str
    after: str
    actor: str
    actor_name: str
    reason: str
    created_at: str


def render_block(base: str, notes: str) -> str:
    """The ``<personalidade>`` block for the system prompt ('' when both are empty)."""
    base, notes = base.strip(), notes.strip()
    if not base and not notes:
        return ''
    parts = ['<personalidade>', 'Sua personalidade neste servidor. Siga-a em todas as respostas.']
    if base:
        parts += ['## Base (definida pelo servidor)', base]
    if notes:
        parts += ['## Suas notas (o que você foi pegando aqui — você mesmo mantém)', notes]
    parts.append('</personalidade>')
    return '\n'.join(parts)


class PersonaStore:
    """SQLite-backed personality sections with full version history."""

    def __init__(self, path: str):
        self.path = path
        self._current: dict[tuple[int, str], dict] = {}
        self._locked: set[int] = set()
        self._lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=30)
        con.row_factory = sqlite3.Row
        return con

    def load(self) -> None:
        """Create the tables if needed and load the current sections into memory."""
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        con = self._connect()
        try:
            con.execute("""
                CREATE TABLE IF NOT EXISTS persona (
                    guild_id INTEGER NOT NULL,
                    section TEXT NOT NULL,
                    content TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    updated_by TEXT NOT NULL DEFAULT '',
                    updated_by_name TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (guild_id, section)
                )""")
            con.execute("""
                CREATE TABLE IF NOT EXISTS persona_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    guild_id INTEGER NOT NULL,
                    section TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    before TEXT NOT NULL,
                    after TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    actor_name TEXT NOT NULL DEFAULT '',
                    reason TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )""")
            con.execute('CREATE INDEX IF NOT EXISTS idx_persona_hist ON persona_history(guild_id, section, id)')
            con.execute("""
                CREATE TABLE IF NOT EXISTS persona_flags (
                    guild_id INTEGER PRIMARY KEY,
                    notes_locked INTEGER NOT NULL DEFAULT 0
                )""")
            con.commit()
            rows = con.execute('SELECT * FROM persona').fetchall()
            locked = con.execute('SELECT guild_id FROM persona_flags WHERE notes_locked=1').fetchall()
        finally:
            con.close()
        with self._lock:
            self._current = {(r['guild_id'], r['section']): dict(r) for r in rows}
            self._locked = {r['guild_id'] for r in locked}

    # --- Reads (from memory) ---

    def get(self, guild_id: int, section: str) -> str:
        row = self._current.get((int(guild_id), section))
        return row['content'] if row else ''

    def info(self, guild_id: int, section: str) -> dict | None:
        """Current row of a section (content, version, updated_by…) or ``None``."""
        row = self._current.get((int(guild_id), section))
        return dict(row) if row else None

    def block(self, guild_id: int | None) -> str:
        if guild_id is None:
            return ''
        return render_block(self.get(guild_id, 'base'), self.get(guild_id, 'notes'))

    def notes_locked(self, guild_id: int) -> bool:
        return int(guild_id) in self._locked

    # --- Writes (synchronous; call through a thread) ---

    def write(
        self, guild_id: int, section: str, content: str, *, actor: str, actor_name: str, reason: str = '',
    ) -> Version:
        """Replace *section* with *content* as a new version and record it."""
        if section not in SECTIONS:
            raise PersonaError(f'Seção desconhecida: {section}.')
        content = (content or '').strip()
        limit = SECTIONS[section]['max']
        if len(content) > limit:
            raise PersonaError(
                f'{SECTIONS[section]["label"]} tem {len(content)} caracteres; o máximo é {limit}. Resuma.'
            )
        before = self.get(guild_id, section)
        if content == before:
            raise PersonaError('Nada mudou: o texto é igual ao atual.')
        now = _now().isoformat()
        con = self._connect()
        try:
            con.execute('BEGIN IMMEDIATE')
            row = con.execute(
                'SELECT version FROM persona WHERE guild_id=? AND section=?', (guild_id, section),
            ).fetchone()
            version = (row['version'] if row else 0) + 1
            con.execute(
                'INSERT OR REPLACE INTO persona (guild_id, section, content, version, updated_by, updated_by_name, updated_at) '
                'VALUES (?,?,?,?,?,?,?)',
                (guild_id, section, content, version, actor, actor_name, now),
            )
            cur = con.execute(
                'INSERT INTO persona_history (guild_id, section, version, before, after, actor, actor_name, reason, created_at) '
                'VALUES (?,?,?,?,?,?,?,?,?)',
                (guild_id, section, version, before, content, actor, actor_name, reason[:200], now),
            )
            con.commit()
            history_id = cur.lastrowid or 0
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()
        with self._lock:
            self._current[(int(guild_id), section)] = {
                'guild_id': int(guild_id), 'section': section, 'content': content, 'version': version,
                'updated_by': actor, 'updated_by_name': actor_name, 'updated_at': now,
            }
        logger.info('[persona] guild=%s %s v%d by %s: %s', guild_id, section, version, actor_name, reason[:80])
        return Version(history_id, int(guild_id), section, version, before, content, actor, actor_name, reason, now)

    def set_notes_locked(self, guild_id: int, locked: bool) -> None:
        con = self._connect()
        try:
            con.execute(
                'INSERT INTO persona_flags (guild_id, notes_locked) VALUES (?,?) '
                'ON CONFLICT(guild_id) DO UPDATE SET notes_locked=excluded.notes_locked',
                (guild_id, int(locked)),
            )
            con.commit()
        finally:
            con.close()
        with self._lock:
            (self._locked.add if locked else self._locked.discard)(int(guild_id))

    # --- History ---

    @staticmethod
    def _version(row: sqlite3.Row) -> Version:
        return Version(
            row['id'], row['guild_id'], row['section'], row['version'], row['before'], row['after'],
            row['actor'], row['actor_name'], row['reason'], row['created_at'],
        )

    def history(self, guild_id: int, section: str | None = None, limit: int = 10) -> list[Version]:
        con = self._connect()
        try:
            if section:
                rows = con.execute(
                    'SELECT * FROM persona_history WHERE guild_id=? AND section=? ORDER BY id DESC LIMIT ?',
                    (guild_id, section, limit),
                ).fetchall()
            else:
                rows = con.execute(
                    'SELECT * FROM persona_history WHERE guild_id=? ORDER BY id DESC LIMIT ?',
                    (guild_id, limit),
                ).fetchall()
        finally:
            con.close()
        return [self._version(r) for r in rows]

    def get_version(self, guild_id: int, history_id: int) -> Version | None:
        con = self._connect()
        try:
            row = con.execute(
                'SELECT * FROM persona_history WHERE guild_id=? AND id=?', (guild_id, history_id),
            ).fetchone()
        finally:
            con.close()
        return self._version(row) if row else None

    def bot_edits_since(self, guild_id: int, section: str, hours: float = 24) -> int:
        """How many changes the bot made to *section* on its own in the last *hours*."""
        since = (_now() - datetime.timedelta(hours=hours)).isoformat()
        con = self._connect()
        try:
            row = con.execute(
                "SELECT COUNT(*) AS n FROM persona_history WHERE guild_id=? AND section=? "
                "AND actor='bot' AND created_at >= ?",
                (guild_id, section, since),
            ).fetchone()
        finally:
            con.close()
        return int(row['n'])
