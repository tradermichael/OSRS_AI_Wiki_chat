from __future__ import annotations

import asyncio

from ..core.rag_sources import load_rag_sources
from .ingest_mediawiki import chunk_page, search_and_fetch
from .store import RetrievedChunk


async def live_query_chunks(
    query: str,
    *,
    max_pages_per_source: int = 2,
    max_chunks_total: int = 8,
    allowed_url_prefixes: list[str] | None = None,
) -> list[RetrievedChunk]:
    """Live-query configured MediaWiki sources and return chunks.

    This is used as a fallback when the local SQLite/BM25 store has no matches.
    """

    sources = load_rag_sources()
    if not sources:
        return []

    prefixes = tuple(p for p in (allowed_url_prefixes or []) if p)

    async def fetch_for_source(api: str) -> list[RetrievedChunk]:
        try:
            pages = await search_and_fetch(api, query, limit=max_pages_per_source)
        except Exception:
            return []

        out: list[RetrievedChunk] = []
        for page in pages:
            if prefixes and page.url and not page.url.startswith(prefixes):
                continue
            for text, meta in chunk_page(page):
                out.append(
                    RetrievedChunk(
                        text=text,
                        url=str((meta or {}).get("url") or ""),
                        title=(meta or {}).get("title"),
                    )
                )
        return out

    tasks = [fetch_for_source(s.mediawiki_api) for s in sources if s.mediawiki_api]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    merged: list[RetrievedChunk] = []
    for lst in results:
        merged.extend(lst)
        if len(merged) >= max_chunks_total:
            break

    return merged[:max_chunks_total]
