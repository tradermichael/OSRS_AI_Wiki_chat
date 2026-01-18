from __future__ import annotations

import asyncio

from ..core.rag_sources import load_rag_sources
from ..core.config import settings
from .ingest_mediawiki import chunk_page, search_and_fetch
from .ingest_mediawiki import mediawiki_fetch_plaintext
from .google_cse import google_cse_search, url_to_title
from .query_expansion import extract_keywords
from .store import RetrievedChunk


def _chunk_score(text: str, terms: list[str]) -> int:
    hay = (text or "").lower()
    return sum(1 for t in (terms or []) if t and t.lower() in hay)


def _select_best_page_chunks(
    page_chunks: list[tuple[str, dict]],
    *,
    query: str,
    max_chunks: int = 3,
) -> list[tuple[str, dict]]:
    if not page_chunks:
        return []
    max_chunks = max(1, int(max_chunks))

    terms = extract_keywords(query, max_terms=10)
    if not terms:
        return page_chunks[:max_chunks]

    scored = sorted(page_chunks, key=lambda p: _chunk_score(p[0], terms), reverse=True)
    top = scored[:max_chunks]

    # If nothing matches at all, keep the lead chunk for basic context.
    if top and _chunk_score(top[0][0], terms) == 0:
        return page_chunks[:max_chunks]
    return top


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
            page_chunks = _select_best_page_chunks(chunk_page(page), query=query, max_chunks=3)
            for text, meta in page_chunks:
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


async def live_search_web_and_fetch_chunks(
    query: str,
    *,
    max_results: int = 5,
    max_chunks_total: int = 8,
    allowed_url_prefixes: list[str] | None = None,
) -> list[RetrievedChunk]:
    """Live search via Google PSE, then fetch plaintext from matched wiki pages.

    This keeps citations on the allowed wiki sites (your PSE config should also restrict sites).
    """

    api_key = settings.google_cse_api_key
    cx = settings.google_cse_cx
    if not api_key or not cx:
        return []

    prefixes = tuple(p for p in (allowed_url_prefixes or []) if p)
    sources = load_rag_sources()

    # Map allowed prefixes -> MediaWiki API
    prefix_to_api: list[tuple[str, str]] = []
    for s in sources:
        for p in (s.allowed_url_prefixes or []):
            prefix_to_api.append((p, s.mediawiki_api))

    results = await google_cse_search(api_key=api_key, cx=cx, query=query, num=max_results)
    urls = [r.url for r in results if r.url]
    if prefixes:
        urls = [u for u in urls if u.startswith(prefixes)]
    if not urls:
        return []

    out: list[RetrievedChunk] = []
    seen_url: set[str] = set()

    for url in urls:
        if url in seen_url:
            continue
        seen_url.add(url)

        title = url_to_title(url)
        if not title:
            continue

        api = None
        for p, a in prefix_to_api:
            if p and url.startswith(p):
                api = a
                break
        if not api:
            continue

        try:
            page = await mediawiki_fetch_plaintext(api, title)
        except Exception:
            continue

        if prefixes and page.url and not page.url.startswith(prefixes):
            continue

        page_chunks = _select_best_page_chunks(chunk_page(page), query=query, max_chunks=3)
        for text, meta in page_chunks:
            out.append(
                RetrievedChunk(
                    text=text,
                    url=str((meta or {}).get("url") or ""),
                    title=(meta or {}).get("title"),
                )
            )
            if len(out) >= max_chunks_total:
                return out[:max_chunks_total]

    return out[:max_chunks_total]
