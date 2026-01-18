from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from ..core.config import settings


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    url: str
    title: str | None


_WORD_RE = re.compile(r"[a-zA-Z0-9_']+")


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


class RAGStore:
    """SQLite-backed chunk store with BM25 retrieval.

    This is intentionally lightweight (pure Python) so it runs on Windows
    without native build toolchains.
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
            CREATE TABLE IF NOT EXISTS chunks (
              id TEXT PRIMARY KEY,
              url TEXT NOT NULL,
              title TEXT,
              text TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def add_documents(self, *, texts: list[str], metadatas: list[dict], ids: list[str]) -> None:
        if not (len(texts) == len(metadatas) == len(ids)):
            raise ValueError("texts/metadatas/ids length mismatch")

        cur = self._conn.cursor()
        for doc_text, meta, doc_id in zip(texts, metadatas, ids, strict=False):
            url = (meta or {}).get("url") or ""
            title = (meta or {}).get("title")
            if not url:
                continue
            cur.execute(
                """
                INSERT INTO chunks (id, url, title, text)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  url=excluded.url,
                  title=excluded.title,
                  text=excluded.text
                """,
                (doc_id, url, title, doc_text),
            )
        self._conn.commit()

    def query(self, *, text: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = int(top_k or settings.rag_top_k)

        cur = self._conn.cursor()
        rows = cur.execute("SELECT id, url, title, text FROM chunks").fetchall()
        if not rows:
            return []

        corpus_texts = [r["text"] for r in rows]
        tokenized_corpus = [_tokenize(t) for t in corpus_texts]
        bm25 = BM25Okapi(tokenized_corpus)

        query_tokens = _tokenize(text)
        scores = bm25.get_scores(query_tokens)

        ranked = sorted(range(len(rows)), key=lambda i: scores[i], reverse=True)
        out: list[RetrievedChunk] = []
        for idx in ranked[:top_k]:
            r = rows[idx]
            out.append(RetrievedChunk(text=r["text"], url=r["url"], title=r["title"]))
        return out


def make_chunk_id(url: str, chunk_index: int) -> str:
    raw = f"{url}#{chunk_index}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
