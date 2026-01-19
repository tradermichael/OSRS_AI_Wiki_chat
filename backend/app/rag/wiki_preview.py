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
        # Don't force exintro: many /Strategies pages start with templates/tables, yielding an empty intro.
        # We'll fetch the full extract and take the first non-empty slice.
        "piprop": "thumbnail",
        "pithumbsize": str(int(thumb_size)),
        "redirects": "1",
        "titles": title,
        "exchars": "2200",
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

    # Normalize and keep the preview short for the UI.
    extract = str(extract)
    extract = "\n".join([ln.strip() for ln in extract.splitlines()]).strip()
    extract = "\n".join([ln for ln in extract.split("\n") if ln])
    if len(extract) > 900:
        extract = extract[:900].rstrip() + "…"

    return WikiPreview(title=title, url=str(fullurl), extract=str(extract), thumbnail_url=str(thumb) if thumb else None)
