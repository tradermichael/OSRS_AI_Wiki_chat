from __future__ import annotations

import re
from dataclasses import dataclass

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass(frozen=True)
class WikiPage:
    title: str
    url: str
    text: str


def _chunk_text(text: str, *, max_chars: int = 900, overlap: int = 120) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not text:
        return []

    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        parts.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return parts


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=0.5, max=4))
async def mediawiki_search(base_api: str, query: str, *, limit: int = 5) -> list[str]:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": str(limit),
    }
    async with httpx.AsyncClient(timeout=30.0, headers={"User-Agent": "OSRS-AI-Wiki-Chat/0.1"}) as client:
        r = await client.get(base_api, params=params)
        r.raise_for_status()
        data = r.json()

    items = (((data or {}).get("query") or {}).get("search") or [])
    return [i.get("title") for i in items if i.get("title")]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=0.5, max=4))
async def mediawiki_fetch_plaintext(base_api: str, title: str) -> WikiPage:
    params = {
        "action": "query",
        "prop": "extracts|info",
        "inprop": "url",
        "explaintext": "1",
        "exsectionformat": "plain",
        "titles": title,
        "format": "json",
    }
    async with httpx.AsyncClient(timeout=30.0, headers={"User-Agent": "OSRS-AI-Wiki-Chat/0.1"}) as client:
        r = await client.get(base_api, params=params)
        r.raise_for_status()
        data = r.json()

    pages = (((data or {}).get("query") or {}).get("pages") or {})
    page = next(iter(pages.values()), {})
    extract = (page or {}).get("extract") or ""
    fullurl = (page or {}).get("fullurl") or ""

    return WikiPage(title=title, url=fullurl, text=extract)


async def search_and_fetch(base_api: str, query: str, *, limit: int = 3) -> list[WikiPage]:
    titles = await mediawiki_search(base_api, query, limit=limit)
    pages: list[WikiPage] = []
    for t in titles:
        pages.append(await mediawiki_fetch_plaintext(base_api, t))
    return pages


def chunk_page(page: WikiPage, *, max_chars: int = 900, overlap: int = 120) -> list[tuple[str, dict]]:
    chunks = _chunk_text(page.text, max_chars=max_chars, overlap=overlap)
    return [(c, {"url": page.url, "title": page.title}) for c in chunks]
