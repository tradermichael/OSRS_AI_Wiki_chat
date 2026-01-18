from __future__ import annotations

import json
import re
from dataclasses import dataclass

import httpx

from ..core.config import settings
from ..llm.gemini_vertex import GeminiVertexClient


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


def looks_like_quest_question(message: str) -> bool:
    """Heuristic quest intent detection."""

    return _looks_like_quest_question(message)


def build_osrs_quest_guide_query(user_message: str) -> str:
    q = _normalize_query(user_message)
    if not q:
        return ""
    if "quest" not in q.lower():
        q = f"{q} quest"
    return f"osrs {q} guide"


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

    if not looks_like_quest_question(user_message):
        return []

    q = build_osrs_quest_guide_query(user_message)
    if not q:
        return []

    return await youtube_search_videos(
        api_key=api_key,
        query=q,
        max_results=settings.youtube_max_results,
    )


def _extract_json_array(text: str) -> list | None:
    raw = (text or "").strip()
    if not raw:
        return None
    # Best-effort: Gemini sometimes wraps JSON with commentary.
    start = raw.find("[")
    end = raw.rfind("]")
    if start < 0 or end < 0 or end <= start:
        return None
    try:
        obj = json.loads(raw[start : end + 1])
    except Exception:
        return None
    return obj if isinstance(obj, list) else None


async def quest_youtube_videos_with_summaries(*, user_message: str) -> list[dict]:
    """Quest-only YouTube results + short Gemini summaries (best-effort).

    Returns a list of dicts compatible with schemas.VideoItem.
    """

    if not settings.youtube_api_key:
        return []

    # Gate YouTube lookup: either heuristic quest detection, or (best-effort) LLM quest detection.
    if not looks_like_quest_question(user_message):
        # Only attempt the LLM check if Vertex is configured; otherwise be conservative.
        if not settings.google_cloud_project:
            return []
        try:
            det = GeminiVertexClient().generate(
                "Return ONLY 'YES' or 'NO'.\n"
                "Does the user question ask about an OSRS quest (requirements, starting, walkthrough, bosses, items, rewards)?\n\n"
                f"Question: {str(user_message or '').strip()}\n"
            ).text
            if "YES" not in (det or "").upper():
                return []
        except Exception:
            return []

    videos = await youtube_search_videos(
        api_key=settings.youtube_api_key,
        query=build_osrs_quest_guide_query(user_message),
        max_results=settings.youtube_max_results,
    )
    if not videos:
        return []

    # Summarize only a couple to keep latency/cost bounded.
    max_summaries = max(0, min(int(settings.youtube_max_summaries or 0), len(videos)))
    to_summarize = videos[:max_summaries]

    url_to_summary: dict[str, str] = {}

    # If Vertex isn't configured, fall back to short descriptions.
    if not settings.google_cloud_project or not to_summarize:
        out: list[dict] = []
        for v in videos:
            summary = (v.description or "").strip()
            if not summary:
                summary = "Quest guide video."
            out.append({"title": v.title, "url": v.url, "channel": v.channel, "summary": summary[:400]})
        return out

    # Fetch transcripts (best-effort) before prompting.
    contexts: list[dict[str, str]] = []
    for v in to_summarize:
        transcript = await youtube_fetch_transcript(v.video_id)
        contexts.append(
            {
                "url": v.url,
                "title": v.title,
                "channel": v.channel or "",
                "description": v.description or "",
                "transcript": transcript or "",
            }
        )

    prompt = (
        "You summarize OSRS quest guide videos for a player.\n"
        "Given the following candidate YouTube videos and (optional) transcript excerpts, "
        "return ONLY a JSON array.\n\n"
        "Each array item MUST be an object with:\n"
        "- url: the exact video url provided\n"
        "- summary: 2-4 short sentences focusing on quest steps, key requirements, and common pitfalls\n\n"
        "Do not include markdown fences or extra keys.\n\n"
        f"User question: {user_message.strip()}\n\n"
        "Videos:\n"
    )

    for i, c in enumerate(contexts, start=1):
        prompt += (
            f"\n[{i}] url: {c['url']}\n"
            f"title: {c['title']}\n"
            f"channel: {c['channel']}\n"
            f"description: {c['description'][:600]}\n"
        )
        if c["transcript"]:
            prompt += f"transcript_excerpt: {c['transcript']}\n"

    try:
        res = GeminiVertexClient().generate(prompt)
        arr = _extract_json_array(res.text)
        if arr:
            for it in arr:
                if not isinstance(it, dict):
                    continue
                url = str(it.get("url") or "").strip()
                summ = str(it.get("summary") or "").strip()
                if url and summ:
                    url_to_summary[url] = summ
    except Exception:
        url_to_summary = {}

    out: list[dict] = []
    for v in videos:
        summary = (url_to_summary.get(v.url) or (v.description or "")).strip()
        if not summary:
            summary = "Quest guide video."
        out.append({"title": v.title, "url": v.url, "channel": v.channel, "summary": summary[:600]})
    return out
