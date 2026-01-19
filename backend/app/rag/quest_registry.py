from __future__ import annotations

import re
from typing import Optional

import httpx


_DEFAULT_MEDIAWIKI_API = "https://oldschool.runescape.wiki/api.php"

# Loaded once at startup (best-effort). If it fails, the heuristics in main.py still help.
_QUEST_TITLES: set[str] = set()
_QUEST_ALIASES_NORM_TO_TITLE: dict[str, str] = {}


def _norm_phrase(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _build_aliases(titles: set[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}

    # Filter out clearly non-quest meta pages.
    blacklist_exact = {
        "quests",
        "quest points",
        "quest difficulties",
        "quest item rewards",
        "quest experience rewards",
        "quest point hood",
        "optimal quest guide",
        "quest list",
    }

    for title in sorted(titles):
        if not title:
            continue

        # Skip meta pages/subpages.
        if "/" in title:
            continue

        tnorm = _norm_phrase(title)
        if not tnorm:
            continue

        if tnorm in blacklist_exact:
            continue

        aliases[tnorm] = title

        # Common player shorthand: drop trailing " I" (Roman numeral one).
        # Examples: "Dragon Slayer I" -> "Dragon Slayer".
        if title.endswith(" I"):
            short = title[:-2].strip()
            sn = _norm_phrase(short)
            if sn and sn not in aliases:
                aliases[sn] = title

    return aliases


async def load_osrs_quest_titles(*, mediawiki_api: str | None = None) -> None:
    """Fetch OSRS quest titles from the wiki (best-effort) and cache in memory.

    Uses MediaWiki categorymembers for Category:Quests.
    """

    api = (mediawiki_api or _DEFAULT_MEDIAWIKI_API).strip() or _DEFAULT_MEDIAWIKI_API

    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": "Category:Quests",
        "cmnamespace": "0",
        "cmlimit": "500",
        "format": "json",
    }

    titles: set[str] = set()
    try:
        async with httpx.AsyncClient(timeout=30.0, headers={"User-Agent": "OSRS-AI-Wiki-Chat/0.1"}) as client:
            r = await client.get(api, params=params)
            r.raise_for_status()
            data = r.json()
    except Exception:
        return

    items = (((data or {}).get("query") or {}).get("categorymembers") or [])
    for it in items:
        t = (it or {}).get("title")
        if t:
            titles.add(str(t))

    global _QUEST_TITLES, _QUEST_ALIASES_NORM_TO_TITLE
    if titles:
        _QUEST_TITLES = titles
        _QUEST_ALIASES_NORM_TO_TITLE = _build_aliases(titles)


def find_quest_title_in_text(text: str) -> Optional[str]:
    """Return the canonical quest title if the message contains a known quest name."""

    if not text:
        return None

    # Fast path: if we haven't loaded, we can't match reliably.
    if not _QUEST_ALIASES_NORM_TO_TITLE:
        return None

    hay = " " + _norm_phrase(text) + " "
    if hay.strip() == "":
        return None

    best_title: str | None = None
    best_len = 0

    # Prefer longest match to avoid partial overlaps.
    for key, title in _QUEST_ALIASES_NORM_TO_TITLE.items():
        if not key:
            continue
        needle = f" {key} "
        if needle in hay:
            if len(key) > best_len:
                best_len = len(key)
                best_title = title

    return best_title
