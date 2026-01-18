from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class WikiPreview:
    title: str
    url: str
    extract: str
    thumbnail_url: str | None = None


async def fetch_wiki_preview(*, mediawiki_api: str, title: str, thumb_size: int = 360) -> WikiPreview:
    """Fetch a wiki preview: title, canonical url, plaintext extract, and a thumbnail if available."""

    params = {
        "action": "query",
        "prop": "extracts|info|pageimages",
        "inprop": "url",
        "explaintext": "1",
        "exsectionformat": "plain",
        "exintro": "1",
        "piprop": "thumbnail",
        "pithumbsize": str(int(thumb_size)),
        "titles": title,
        "format": "json",
    }

    async with httpx.AsyncClient(timeout=25.0, headers={"User-Agent": "OSRS-AI-Wiki-Chat/0.1"}) as client:
        r = await client.get(mediawiki_api, params=params)
        r.raise_for_status()
        data = r.json()

    pages = (((data or {}).get("query") or {}).get("pages") or {})
    page = next(iter(pages.values()), {})

    extract = (page or {}).get("extract") or ""
    fullurl = (page or {}).get("fullurl") or ""
    thumb = ((page or {}).get("thumbnail") or {}).get("source")

    return WikiPreview(title=title, url=str(fullurl), extract=str(extract), thumbnail_url=str(thumb) if thumb else None)
