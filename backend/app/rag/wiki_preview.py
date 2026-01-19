from __future__ import annotations

from dataclasses import dataclass

import re

import httpx


@dataclass(frozen=True)
class WikiPreview:
    title: str
    url: str
    extract: str
    thumbnail_url: str | None = None


async def fetch_wiki_preview(*, mediawiki_api: str, title: str, thumb_size: int = 360) -> WikiPreview:
    """Fetch a wiki preview: title, canonical url, plaintext extract, and a thumbnail if available."""

    base_params = {
        "action": "query",
        "prop": "extracts|info|pageimages",
        "inprop": "url",
        "explaintext": "1",
        "exsectionformat": "plain",
        "piprop": "thumbnail",
        "pithumbsize": str(int(thumb_size)),
        "redirects": "1",
        "titles": title,
        "format": "json",
    }

    async def _fetch(*, intro: bool) -> dict:
        params = dict(base_params)
        # Prefer short, readable previews in the UI.
        # For many pages, intro-only is perfect; for some subpages (/Strategies, /Quick guide) the intro can be empty.
        if intro:
            params["exintro"] = "1"
            params["exchars"] = "1400"
        else:
            params["exchars"] = "2200"

        async with httpx.AsyncClient(timeout=25.0, headers={"User-Agent": "OSRS-AI-Wiki-Chat/0.1"}) as client:
            r = await client.get(mediawiki_api, params=params)
            r.raise_for_status()
            return r.json()

    # Try intro-only first for a cleaner preview; fall back to full extract if the intro is empty.
    data = await _fetch(intro=True)
    pages = (((data or {}).get("query") or {}).get("pages") or {})
    page = next(iter(pages.values()), {})
    extract = (page or {}).get("extract") or ""
    if len(str(extract).strip()) < 80:
        data = await _fetch(intro=False)

    pages = (((data or {}).get("query") or {}).get("pages") or {})
    page = next(iter(pages.values()), {})

    extract = (page or {}).get("extract") or ""
    fullurl = (page or {}).get("fullurl") or ""
    thumb = ((page or {}).get("thumbnail") or {}).get("source")

    # Normalize and keep the preview short for the UI.
    # MediaWiki plaintext extracts for /Quick guide pages can look like raw tables/lists with lots of short lines.
    # Convert runs of whitespace/newlines into a readable paragraph.
    extract = str(extract)
    extract = "\n".join([ln.strip() for ln in extract.splitlines()]).strip()
    extract = "\n".join([ln for ln in extract.split("\n") if ln])

    # If the extract is very line-y, join with separators; otherwise join as a paragraph.
    lines = [ln.strip() for ln in extract.split("\n") if ln.strip()]
    if len(lines) >= 8 and (sum(1 for ln in lines[:12] if len(ln) <= 32) >= 8):
        extract = " • ".join(lines)
    else:
        extract = re.sub(r"\s+", " ", " ".join(lines)).strip()

    if len(extract) > 900:
        extract = extract[:900].rstrip() + "…"

    return WikiPreview(title=title, url=str(fullurl), extract=str(extract), thumbnail_url=str(thumb) if thumb else None)
