from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

from ..core.config import settings


@dataclass(frozen=True)
class YouTubeVideo:
    video_id: str
    title: str
    channel: str | None
    url: str
    description: str | None = None


def _looks_like_quest_question(message: str) -> bool:
    msg = (message or "").lower()
    if not msg:
        return False
    if "quest" in msg:
        return True
    # common phrasing for quests
    if any(p in msg for p in ("quest guide", "how do i start", "how to start", "quest req", "quest requirements")):
        return True
    return False


def _normalize_query(message: str) -> str:
    msg = re.sub(r"\s+", " ", (message or "").strip())
    return msg[:200]


async def youtube_search_videos(*, api_key: str, query: str, max_results: int = 3) -> list[YouTubeVideo]:
    q = _normalize_query(query)
    if not q:
        return []

    params = {
        "part": "snippet",
        "q": q,
        "type": "video",
        "maxResults": str(max(1, min(int(max_results), 10))),
        "safeSearch": "moderate",
        "relevanceLanguage": "en",
        "key": api_key,
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get("https://www.googleapis.com/youtube/v3/search", params=params)
        r.raise_for_status()
        data = r.json()

    out: list[YouTubeVideo] = []
    for it in (data or {}).get("items") or []:
        vid = (((it or {}).get("id") or {}).get("videoId"))
        sn = (it or {}).get("snippet") or {}
        title = str(sn.get("title") or "").strip()
        channel = str(sn.get("channelTitle") or "").strip() or None
        desc = str(sn.get("description") or "").strip() or None
        if not vid:
            continue
        url = f"https://www.youtube.com/watch?v={vid}"
        out.append(YouTubeVideo(video_id=str(vid), title=title or url, channel=channel, url=url, description=desc))

    return out


async def youtube_fetch_transcript(video_id: str) -> str | None:
    """Best-effort transcript fetch.

    Uses youtube-transcript-api if available. This may fail if captions are unavailable.
    """

    try:
        from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore
    except Exception:
        return None

    try:
        parts = YouTubeTranscriptApi.get_transcript(video_id)
    except Exception:
        return None

    text = " ".join(str(p.get("text") or "") for p in (parts or [])).strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return None
    # Hard cap to keep prompts small.
    return text[:4500]


async def maybe_get_quest_videos(*, user_message: str) -> list[YouTubeVideo]:
    api_key = settings.youtube_api_key
    if not api_key:
        return []

    if not _looks_like_quest_question(user_message):
        return []

    # A slightly OSRS-quest-focused query.
    q = _normalize_query(user_message)
    if "quest" not in q.lower():
        q = f"{q} quest"
    q = f"osrs {q} guide"

    return await youtube_search_videos(
        api_key=api_key,
        query=q,
        max_results=settings.youtube_max_results,
    )
