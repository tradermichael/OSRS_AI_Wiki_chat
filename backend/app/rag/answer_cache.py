from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..core.config import settings


@dataclass(frozen=True)
class CachedAnswer:
    id: int
    created_at: str
    question: str
    answer: str
    sources: list[dict]


_WORD_RE = re.compile(r"[a-zA-Z0-9_']+")

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "can",
    "do",
    "does",
    "for",
    "from",
    "get",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "who",
    "why",
    "with",
    "you",
    "your",
}


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def _keywords(text: str) -> set[str]:
    toks = [t for t in _tokenize(text) if t and t not in _STOPWORDS]
    return set(toks)


class AnswerCacheStore:
    """SQLite-backed cache of prior Q/A pairs with citations.

    Goal: speed up answering similar questions and preserve a locally stored,
    cited response (answer + source URLs/titles) for reuse.

    We intentionally store only:
    - user question
    - model answer text
    - sources (urls/titles and short snippet already limited elsewhere)

    We do *not* store full wiki pages.
    """

    def __init__(self) -> None:
        db_path = Path(settings.rag_db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS answer_cache (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              created_at TEXT NOT NULL,
              question TEXT NOT NULL,
              question_norm TEXT,
              answer TEXT NOT NULL,
              sources_json TEXT NOT NULL
            )
            """
        )

        # Lightweight migration for older DBs.
        cols = [r["name"] for r in cur.execute("PRAGMA table_info(answer_cache)").fetchall()]
        if "question_norm" not in cols:
            cur.execute("ALTER TABLE answer_cache ADD COLUMN question_norm TEXT")

        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_answer_cache_created_at ON answer_cache(created_at DESC)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_answer_cache_question_norm ON answer_cache(question_norm)"
        )
        self._conn.commit()

    @staticmethod
    def _normalize_question(question: str) -> str:
        q = (question or "").strip().lower()
        q = re.sub(r"\s+", " ", q)
        # Keep alphanumerics and spaces only.
        q = re.sub(r"[^a-z0-9 ]+", "", q)
        q = re.sub(r"\s+", " ", q).strip()
        return q

    def add(self, *, question: str, answer: str, sources: list[dict]) -> int:
        question = (question or "").strip()
        answer = (answer or "").strip()
        if not question or not answer:
            raise ValueError("question/answer required")

        question_norm = self._normalize_question(question)

        created_at = datetime.now(timezone.utc).isoformat()
        sources_json = json.dumps(sources or [], ensure_ascii=False)

        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO answer_cache (created_at, question, question_norm, answer, sources_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (created_at, question, question_norm, answer, sources_json),
        )
        self._conn.commit()
        lastrowid = cur.lastrowid
        if lastrowid is None:
            raise RuntimeError("sqlite did not return lastrowid")
        return int(lastrowid)

    def _load_all(self) -> list[CachedAnswer]:
        rows = self._conn.execute(
            """
            SELECT id, created_at, question, answer, sources_json
            FROM answer_cache
            ORDER BY id DESC
            LIMIT 500
            """
        ).fetchall()

        out: list[CachedAnswer] = []
        for r in rows:
            try:
                sources = json.loads(r["sources_json"] or "[]")
            except Exception:
                sources = []
            out.append(
                CachedAnswer(
                    id=int(r["id"]),
                    created_at=str(r["created_at"]),
                    question=str(r["question"]),
                    answer=str(r["answer"]),
                    sources=list(sources) if isinstance(sources, list) else [],
                )
            )
        return out

    def find_similar(
        self,
        *,
        question: str,
        min_score: float = 0.35,
        allowed_url_prefixes: list[str] | None = None,
    ) -> CachedAnswer | None:
        """Return a cached answer for a similar question.

        Uses a keyword-based Jaccard similarity over cached questions.
        """
        q_norm = self._normalize_question(question)
        if q_norm:
            row = self._conn.execute(
                """
                SELECT id, created_at, question, answer, sources_json
                FROM answer_cache
                WHERE question_norm = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (q_norm,),
            ).fetchone()
            if row:
                try:
                    sources = json.loads(row["sources_json"] or "[]")
                except Exception:
                    sources = []
                return CachedAnswer(
                    id=int(row["id"]),
                    created_at=str(row["created_at"]),
                    question=str(row["question"]),
                    answer=str(row["answer"]),
                    sources=list(sources) if isinstance(sources, list) else [],
                )

        items = self._load_all()
        if not items:
            return None

        prefixes = tuple(p for p in (allowed_url_prefixes or []) if p)
        if prefixes:
            items = [
                it
                for it in items
                if any(str((s or {}).get("url") or "").startswith(prefixes) for s in (it.sources or []))
            ]
            if not items:
                return None

        q_keys = _keywords(question)
        if len(q_keys) < 2:
            return None

        best_item: CachedAnswer | None = None
        best_score = -1.0

        for it in items:
            it_keys = _keywords(it.question)
            if not it_keys:
                continue
            inter = len(q_keys & it_keys)
            if inter < 2:
                continue
            union = len(q_keys | it_keys)
            score = (inter / union) if union else 0.0
            if score > best_score:
                best_score = score
                best_item = it

        if not best_item or best_score < float(min_score):
            return None
        return best_item
