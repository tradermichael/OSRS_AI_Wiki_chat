from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..core.config import settings


@dataclass(frozen=True)
class PublicChatRecord:
    id: int
    created_at: str
    session_id: str | None
    user_message: str
    bot_answer: str
    sources: list[dict]


class PublicChatStore:
    """SQLite-backed public chat log.

    Intentionally stores minimal data: message text + answer + sources.
    """

    def __init__(self) -> None:
        db_path = Path(settings.history_db_path or settings.rag_db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(db_path)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS public_chat (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  created_at TEXT NOT NULL,
                  session_id TEXT,
                  user_message TEXT NOT NULL,
                  bot_answer TEXT NOT NULL,
                  sources_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_public_chat_created_at ON public_chat(created_at DESC)"
            )
            conn.commit()

    def add(
        self,
        *,
        session_id: str | None,
        user_message: str,
        bot_answer: str,
        sources: list[dict],
    ) -> int:
        created_at = datetime.now(timezone.utc).isoformat()
        sources_json = json.dumps(sources or [], ensure_ascii=False)

        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO public_chat (created_at, session_id, user_message, bot_answer, sources_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (created_at, session_id, user_message, bot_answer, sources_json),
            )
            conn.commit()
            return int(cur.lastrowid)

    def list(self, *, limit: int = 50, offset: int = 0) -> list[PublicChatRecord]:
        limit = max(1, min(int(limit), 200))
        offset = max(0, int(offset))

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, created_at, session_id, user_message, bot_answer, sources_json
                FROM public_chat
                ORDER BY id DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()

        out: list[PublicChatRecord] = []
        for r in rows:
            try:
                sources = json.loads(r["sources_json"] or "[]")
            except Exception:
                sources = []
            out.append(
                PublicChatRecord(
                    id=int(r["id"]),
                    created_at=str(r["created_at"]),
                    session_id=r["session_id"],
                    user_message=str(r["user_message"]),
                    bot_answer=str(r["bot_answer"]),
                    sources=list(sources) if isinstance(sources, list) else [],
                )
            )
        return out

    def get(self, record_id: int) -> PublicChatRecord | None:
        rid = int(record_id)
        with self._connect() as conn:
            r = conn.execute(
                """
                SELECT id, created_at, session_id, user_message, bot_answer, sources_json
                FROM public_chat
                WHERE id = ?
                """,
                (rid,),
            ).fetchone()

        if not r:
            return None

        try:
            sources = json.loads(r["sources_json"] or "[]")
        except Exception:
            sources = []

        return PublicChatRecord(
            id=int(r["id"]),
            created_at=str(r["created_at"]),
            session_id=r["session_id"],
            user_message=str(r["user_message"]),
            bot_answer=str(r["bot_answer"]),
            sources=list(sources) if isinstance(sources, list) else [],
        )
