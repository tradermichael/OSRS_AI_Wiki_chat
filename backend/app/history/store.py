from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..core.config import settings


@dataclass(frozen=True)
class PublicChatRecord:
    id: str
    created_at: str
    session_id: str | None
    user_message: str
    bot_answer: str
    sources: list[dict]
    videos: list[dict]


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
                  sources_json TEXT NOT NULL,
                  videos_json TEXT NOT NULL DEFAULT '[]'
                )
                """
            )
            # Backwards-compatible migration for existing DBs.
            try:
                conn.execute("ALTER TABLE public_chat ADD COLUMN videos_json TEXT NOT NULL DEFAULT '[]'")
            except Exception:
                pass
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_public_chat_created_at ON public_chat(created_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_public_chat_session_id ON public_chat(session_id, id DESC)"
            )
            conn.commit()

    def add(
        self,
        *,
        session_id: str | None,
        user_message: str,
        bot_answer: str,
        sources: list[dict],
        videos: list[dict] | None = None,
    ) -> str:
        created_at = datetime.now(timezone.utc).isoformat()
        sources_json = json.dumps(sources or [], ensure_ascii=False)
        videos_json = json.dumps(videos or [], ensure_ascii=False)

        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO public_chat (created_at, session_id, user_message, bot_answer, sources_json, videos_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (created_at, session_id, user_message, bot_answer, sources_json, videos_json),
            )
            conn.commit()
            lastrowid = cur.lastrowid
            if lastrowid is None:
                raise RuntimeError("sqlite did not return lastrowid")
            return str(int(lastrowid))

    def list(self, *, limit: int = 50, offset: int = 0) -> list[PublicChatRecord]:
        limit = max(1, min(int(limit), 200))
        offset = max(0, int(offset))

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, created_at, session_id, user_message, bot_answer, sources_json, videos_json
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
            try:
                videos = json.loads(r["videos_json"] or "[]")
            except Exception:
                videos = []
            out.append(
                PublicChatRecord(
                    id=str(int(r["id"])),
                    created_at=str(r["created_at"]),
                    session_id=r["session_id"],
                    user_message=str(r["user_message"]),
                    bot_answer=str(r["bot_answer"]),
                    sources=list(sources) if isinstance(sources, list) else [],
                    videos=list(videos) if isinstance(videos, list) else [],
                )
            )
        return out

    def list_by_session(self, *, session_id: str, limit: int = 10) -> list[PublicChatRecord]:
        session_id = str(session_id or "").strip()
        if not session_id:
            return []
        limit = max(1, min(int(limit), 50))

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, created_at, session_id, user_message, bot_answer, sources_json, videos_json
                FROM public_chat
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()

        out: list[PublicChatRecord] = []
        for r in rows:
            try:
                sources = json.loads(r["sources_json"] or "[]")
            except Exception:
                sources = []
            try:
                videos = json.loads(r["videos_json"] or "[]")
            except Exception:
                videos = []

            out.append(
                PublicChatRecord(
                    id=str(int(r["id"])),
                    created_at=str(r["created_at"]),
                    session_id=r["session_id"],
                    user_message=str(r["user_message"]),
                    bot_answer=str(r["bot_answer"]),
                    sources=list(sources) if isinstance(sources, list) else [],
                    videos=list(videos) if isinstance(videos, list) else [],
                )
            )
        return out

    def get(self, record_id: str) -> PublicChatRecord | None:
        rid = int(str(record_id))
        with self._connect() as conn:
            r = conn.execute(
                """
                SELECT id, created_at, session_id, user_message, bot_answer, sources_json, videos_json
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

        try:
            videos = json.loads(r["videos_json"] or "[]")
        except Exception:
            videos = []

        return PublicChatRecord(
            id=str(int(r["id"])),
            created_at=str(r["created_at"]),
            session_id=r["session_id"],
            user_message=str(r["user_message"]),
            bot_answer=str(r["bot_answer"]),
            sources=list(sources) if isinstance(sources, list) else [],
            videos=list(videos) if isinstance(videos, list) else [],
        )


def get_public_chat_store():
    if (settings.history_backend or "sqlite").lower() == "firestore":
        from .firestore_store import FirestorePublicChatStore

        return FirestorePublicChatStore()
    return PublicChatStore()
