from __future__ import annotations

import re

from ..core.config import settings
from .google_cse import google_cse_search
from .store import RetrievedChunk


def _normalize_query(q: str) -> str:
    q = re.sub(r"\s+", " ", (q or "").strip())
    return q[:200]


def _reddit_site_filter() -> str:
    # Keep this narrow to stay OSRS-relevant and avoid random subreddit noise.
    # Add ironscape as a secondary common community.
    return "(site:reddit.com/r/2007scape OR site:reddit.com/r/ironscape)"


async def reddit_search_chunks(*, query: str, max_results: int = 4) -> list[RetrievedChunk]:
    """Fetch Reddit search snippets as citable chunks.

    Uses Google CSE, either via a dedicated community CX or the primary CX.
    Does not scrape Reddit pages (snippets only).
    """

    api_key = settings.google_cse_api_key
    cx = settings.google_cse_community_cx or settings.google_cse_cx
    if not api_key or not cx:
        return []

    q = _normalize_query(query)
    if not q:
        return []

    # Bias toward OSRS phrasing.
    final_q = f"osrs {q} {_reddit_site_filter()}"

    hits = await google_cse_search(api_key=api_key, cx=cx, query=final_q, num=max_results)

    out: list[RetrievedChunk] = []
    for h in hits:
        if not h.url:
            continue
        text = (h.snippet or "").strip()
        if not text:
            # Keep a minimal chunk so the model can cite the link.
            text = f"Reddit thread: {h.title}" if h.title else "Reddit thread"

        out.append(
            RetrievedChunk(
                text=text[:800],
                url=str(h.url),
                title=(str(h.title)[:120] if h.title else "Reddit"),
            )
        )

    return out
