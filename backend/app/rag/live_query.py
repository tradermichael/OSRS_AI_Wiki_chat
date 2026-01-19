from __future__ import annotations

import asyncio
import re

from ..core.rag_sources import load_rag_sources
from ..core.config import settings
from .ingest_mediawiki import chunk_page
from .ingest_mediawiki import mediawiki_search, mediawiki_fetch_plaintext
from .google_cse import SearchResult, google_cse_search, url_to_title
from .query_expansion import extract_keywords
from .store import RetrievedChunk


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _query_title_hint(query: str) -> str:
    """Try to derive a likely page title from a query."""
    q = (query or "").strip()

    # If the query already contains an explicit /Strategies title, prefer that.
    m_strat = re.search(r"^\s*([^/]{2,120})\s*/\s*strategies\b", q, flags=re.IGNORECASE)
    if m_strat:
        left = re.sub(r"\s+", " ", (m_strat.group(1) or "").strip()).strip(" ?!.\"'")
        if 1 <= len(left) <= 80:
            return f"{left}/Strategies"

    m_quick = re.search(r"^\s*([^/]{2,120})\s*/\s*quick\s*_?\s*guide\b", q, flags=re.IGNORECASE)
    if m_quick:
        left = re.sub(r"\s+", " ", (m_quick.group(1) or "").strip()).strip(" ?!.\"'")
        if 1 <= len(left) <= 80:
            return f"{left}/Quick_guide"

    m_walk = re.search(r"^\s*([^/]{2,120})\s*/\s*walkthrough\b", q, flags=re.IGNORECASE)
    if m_walk:
        left = re.sub(r"\s+", " ", (m_walk.group(1) or "").strip()).strip(" ?!.\"'")
        if 1 <= len(left) <= 80:
            return f"{left}/Walkthrough"

    # Drop common qualifiers.
    q = re.sub(
        r"^(how do i|how to|what is|what's|tell me about|best way to)\s+",
        "",
        q,
        flags=re.IGNORECASE,
    )
    q = re.sub(
        r"\b(complete|finish|start|beat|defeat|kill|survive|requirements|reqs|steps)\b",
        "",
        q,
        flags=re.IGNORECASE,
    )
    q = re.sub(r"\bosrs\b", "", q, flags=re.IGNORECASE)
    q = re.sub(
        r"\bguide\b|\bquick\b|\bwalkthrough\b|\bstrateg(?:y|ies)\b|\bboss fight\b",
        "",
        q,
        flags=re.IGNORECASE,
    )
    q = re.sub(r"\s+", " ", q).strip(" ?!.\"'")
    # Keep it short so we don't try fetching long conversational strings.
    if len(q) > 60:
        return ""
    # Heuristic: looks like a title if it has at least one letter and isn't too long.
    if not re.search(r"[A-Za-z]", q):
        return ""
    if len(q.split()) > 8:
        return ""
    return q


def _title_score(*, title: str, query: str) -> int:
    t = _norm(title)
    qn = _norm(query)

    score = 0
    hint = _norm(_query_title_hint(query))
    if hint:
        if t == hint:
            score += 200
        elif t.startswith(hint + " ") or (hint and hint in t):
            score += 80
        if t == hint + "/strategies" or t.endswith("/strategies") and hint and hint in t:
            score += 120

    # Keyword overlap
    terms = extract_keywords(query, max_terms=10)
    score += 6 * _chunk_score(title, terms)

    # Penalize generic index pages for specific entity-like queries.
    if hint:
        if t in {"boss", "quest", "quests"}:
            score -= 120
        if t.startswith("combat achievements"):
            score -= 120

    # Small preference for shorter titles when scores tie.
    score -= max(0, len(t) - 30) // 10
    return score


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
        # If the query looks like a specific page title, bias towards fetching it (and /Strategies).
        hint = _query_title_hint(query)

        # If the hint explicitly points at a known subpage, fetch it directly.
        subpage_suffixes = ("/strategies", "/quick_guide", "/walkthrough")
        if hint and hint.lower().endswith(subpage_suffixes):
            base = hint.rsplit("/", 1)[0]
            titles = [hint]
            if base and base != hint:
                titles.append(base)
        else:
            titles = []

        if not titles:
            try:
                # Pull more candidates, then rank by title relevance.
                titles = await mediawiki_search(api, query, limit=max(8, max_pages_per_source * 4))
            except Exception:
                titles = []

        if hint and not hint.lower().endswith(subpage_suffixes):
            preferred = [hint, f"{hint}/Strategies"]
            for p in reversed(preferred):
                if p and p not in titles:
                    titles.insert(0, p)

        # If search yielded nothing (or search failed), still try direct fetch of the hinted title(s).
        if not titles and hint:
            if hint.lower().endswith(subpage_suffixes):
                base = hint.rsplit("/", 1)[0]
                titles = [hint]
                if base and base != hint:
                    titles.append(base)
            else:
                titles = [hint, f"{hint}/Strategies"]

        if not titles:
            return []

        ranked_titles = sorted(
            list(dict.fromkeys(titles)),
            key=lambda t: _title_score(title=t, query=query),
            reverse=True,
        )

        picked_titles = ranked_titles[: max(2, int(max_pages_per_source))]

        pages = []
        try:
            pages = await asyncio.gather(
                *[mediawiki_fetch_plaintext(api, t) for t in picked_titles],
                return_exceptions=False,
            )
        except Exception:
            # Fall back to sequential best-effort
            pages = []
            for t in picked_titles:
                try:
                    pages.append(await mediawiki_fetch_plaintext(api, t))
                except Exception:
                    continue

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
) -> tuple[list[RetrievedChunk], list[SearchResult]]:
    """Live search via Google PSE, then fetch plaintext from matched wiki pages.

    This keeps citations on the allowed wiki sites (your PSE config should also restrict sites).
    """

    api_key = settings.google_cse_api_key
    cx = settings.google_cse_cx
    if not api_key or not cx:
        return ([], [])

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
        return ([], results)

    out: list[RetrievedChunk] = []
    seen_url: set[str] = set()

    ql = (query or "").lower()
    combat_intent = any(w in ql for w in ("strateg", "beat", "defeat", "kill", "boss", "gear", "prayer", "phase"))
    quest_intent = any(w in ql for w in ("quest", "walkthrough", "quick guide", "requirements", "reqs", "steps"))

    for url in urls:
        if url in seen_url:
            continue
        seen_url.add(url)

        title = url_to_title(url)
        if not title:
            continue

        candidates: list[str] = []
        if combat_intent:
            candidates.append(f"{title}/Strategies")
        if quest_intent:
            candidates.append(f"{title}/Quick_guide")
            candidates.append(f"{title}/Walkthrough")
        candidates.append(title)
        # de-dupe while preserving order
        seen_t: set[str] = set()
        candidates = [t for t in candidates if t and not (t.lower() in seen_t or seen_t.add(t.lower()))]

        api = None
        for p, a in prefix_to_api:
            if p and url.startswith(p):
                api = a
                break
        if not api:
            continue

        page = None
        for cand in candidates:
            try:
                page = await mediawiki_fetch_plaintext(api, cand)
                break
            except Exception:
                continue
        if not page:
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
                return (out[:max_chunks_total], results)

    return (out[:max_chunks_total], results)
