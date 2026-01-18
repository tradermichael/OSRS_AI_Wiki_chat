from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..core.config import settings


@dataclass(frozen=True)
class FeedbackRecord:
    id: str
    created_at: str
    history_id: str
    rating: int
    session_id: str | None


class FeedbackStore:
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
                CREATE TABLE IF NOT EXISTS feedback (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  created_at TEXT NOT NULL,
                  history_id TEXT NOT NULL,
                  rating INTEGER NOT NULL,
                  session_id TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_history_id ON feedback(history_id)")
            conn.commit()

    def add(self, *, history_id: str, rating: int, session_id: str | None) -> str:
        rating = int(rating)
        if rating not in (-1, 1):
            raise ValueError("rating must be -1 or 1")

        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO feedback (created_at, history_id, rating, session_id)
                VALUES (?, ?, ?, ?)
                """,
                (created_at, str(history_id), rating, session_id),
            )
            conn.commit()
            lastrowid = cur.lastrowid
            if lastrowid is None:
                raise RuntimeError("sqlite did not return lastrowid")
            return str(int(lastrowid))

    def summary(self, *, history_id: str) -> dict:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT rating, COUNT(*) as c FROM feedback WHERE history_id = ? GROUP BY rating",
                (str(history_id),),
            ).fetchall()
        up = 0
        down = 0
        for r in rows:
            if int(r["rating"]) == 1:
                up = int(r["c"])
            elif int(r["rating"]) == -1:
                down = int(r["c"])
        return {"thumbs_up": up, "thumbs_down": down, "net": up - down}


def get_feedback_store():
    if (settings.history_backend or "sqlite").lower() == "firestore":
        from .firestore_store import FirestoreFeedbackStore

        return FirestoreFeedbackStore()
    return FeedbackStore()
