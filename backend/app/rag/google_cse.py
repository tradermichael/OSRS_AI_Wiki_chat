from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urlparse

import httpx


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str | None = None


def _clean_title_from_url_path(path: str) -> str | None:
    # /w/Foo_bar or /wiki/Foo_bar
    path = path or ""
    path = path.strip("/")
    if not path:
        return None

    parts = path.split("/")
    if len(parts) < 2:
        return None

    slug = parts[-1]
    slug = unquote(slug)
    slug = slug.replace("_", " ")
    slug = re.sub(r"\s+", " ", slug).strip()
    return slug or None


def url_to_title(url: str) -> str | None:
    """Best-effort conversion from a wiki URL to a MediaWiki page title."""
    try:
        u = urlparse(url)
    except Exception:
        return None

    if not u.path:
        return None

    # Some wiki links use index.php?title=Page_title
    if u.query:
        try:
            qs = parse_qs(u.query)
            title_q = (qs.get("title") or [None])[0]
            if title_q:
                title_q = unquote(str(title_q))
                title_q = title_q.replace("_", " ")
                title_q = re.sub(r"\s+", " ", title_q).strip()
                if title_q:
                    return title_q
        except Exception:
            pass

    # Old School wiki: /w/Title
    if "/w/" in u.path:
        return _clean_title_from_url_path(u.path)

    # Fandom: /wiki/Title
    if "/wiki/" in u.path:
        return _clean_title_from_url_path(u.path)

    return None


async def google_cse_search(*, api_key: str, cx: str, query: str, num: int = 5) -> list[SearchResult]:
    """Search via Google Programmable Search Engine (Custom Search JSON API)."""

    q = (query or "").strip()
    if not q:
        return []

    params = {
        "key": api_key,
        "cx": cx,
        "q": q,
        "num": str(max(1, min(int(num), 10))),
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get("https://www.googleapis.com/customsearch/v1", params=params)
        r.raise_for_status()
        data = r.json()

    items = (data or {}).get("items") or []
    out: list[SearchResult] = []
    for it in items:
        link = (it or {}).get("link")
        title = (it or {}).get("title") or link or ""
        snippet = (it or {}).get("snippet")
        if link:
            out.append(SearchResult(title=str(title), url=str(link), snippet=str(snippet) if snippet else None))
    return out
