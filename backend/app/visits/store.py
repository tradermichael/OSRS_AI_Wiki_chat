from __future__ import annotations

import sqlite3
from pathlib import Path

from ..core.config import settings


class VisitsStore:
    """Tiny SQLite-backed store for a global site visit counter.

    Note: On Cloud Run, the default DB path may be on an ephemeral filesystem.
    Use Firestore for persistence across deploys/instances.
    """

    def __init__(self) -> None:
        db_path = Path(settings.visits_db_path or settings.rag_db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(db_path)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS site_state (
                  k TEXT PRIMARY KEY,
                  v TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                INSERT INTO site_state (k, v)
                VALUES ('visits_total', '0')
                ON CONFLICT(k) DO NOTHING
                """
            )
            conn.commit()

    def get_total(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT v FROM site_state WHERE k='visits_total'").fetchone()
            if not row:
                return 0
            try:
                return int(row["v"])
            except Exception:
                return 0

    def increment(self, amount: int = 1) -> int:
        if amount <= 0:
            raise ValueError("amount must be > 0")

        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT v FROM site_state WHERE k='visits_total'").fetchone()
            current = int(row["v"]) if row else 0
            new_total = current + int(amount)
            conn.execute(
                "UPDATE site_state SET v=? WHERE k='visits_total'",
                (str(new_total),),
            )
            conn.commit()
            return int(new_total)


def get_visits_store():
    if (settings.visits_backend or "sqlite").lower() == "firestore":
        from .firestore_store import FirestoreVisitsStore

        return FirestoreVisitsStore()
    return VisitsStore()
