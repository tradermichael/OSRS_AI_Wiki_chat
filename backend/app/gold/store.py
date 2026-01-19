from __future__ import annotations

import sqlite3
from pathlib import Path

from ..core.config import settings


class GoldStore:
    """Tiny SQLite-backed store for fake "gold" donations.

    Note: On Cloud Run, the default DB path may be on an ephemeral filesystem.
    That's fine for a demo "fake" counter.
    """

    def __init__(self) -> None:
        db_path = Path(settings.gold_db_path or settings.rag_db_path)
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
                VALUES ('gold_total', '0')
                ON CONFLICT(k) DO NOTHING
                """
            )
            conn.commit()

    def get_total(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT v FROM site_state WHERE k='gold_total'").fetchone()
            if not row:
                return 0
            try:
                return int(row["v"])
            except Exception:
                return 0

    def add(self, amount_gold: int) -> int:
        if amount_gold <= 0:
            raise ValueError("amount_gold must be > 0")

        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT v FROM site_state WHERE k='gold_total'").fetchone()
            current = int(row["v"]) if row else 0
            new_total = current + int(amount_gold)
            conn.execute(
                "UPDATE site_state SET v=? WHERE k='gold_total'",
                (str(new_total),),
            )
            conn.commit()
            return new_total


def get_gold_store():
    if (settings.gold_backend or "sqlite").lower() == "firestore":
        from .firestore_store import FirestoreGoldStore

        return FirestoreGoldStore()
    return GoldStore()
