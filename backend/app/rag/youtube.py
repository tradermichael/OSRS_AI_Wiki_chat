from __future__ import annotations

import json
import re
from dataclasses import dataclass

import httpx

from ..core.config import settings
from ..llm.gemini_vertex import GeminiVertexClient
from .quest_registry import find_quest_title_in_text
from .store import RetrievedChunk


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
    # Meta/community questions about quests shouldn't trigger the quest-guide video widget.
    meta_needles = (
        "quest cape",
        "quest point cape",
        "questpoint cape",
        "most people",
        "most common",
        "popular",
        "tier list",
        "best quests",
        "most important quests",
        "last quest",
        "final quest",
    )
    if any(n in msg for n in meta_needles):
        return False

    # If we can identify a specific quest title, treat it as quest intent.
    # (This enables videos for questions like "How do I complete Desert Treasure II?".)
    try:
        if find_quest_title_in_text(message):
            return True
    except Exception:
        pass

    # Otherwise require both the word "quest" and explicit help phrasing.
    if "quest" not in msg:
        return False

    help_needles = (
        "guide",
        "walkthrough",
        "quick guide",
        "requirements",
        "reqs",
        "quest req",
        "quest requirements",
        "how do i",
        "how to",
        "where do i start",
        "how do i start",
        "how to start",
        "start",
        "complete",
        "finish",
        "steps",
    )
    return any(p in msg for p in help_needles)


def looks_like_quest_question(message: str) -> bool:
    """Heuristic quest intent detection."""

    return _looks_like_quest_question(message)


def build_osrs_quest_guide_query(user_message: str) -> str:
    q = _normalize_query(user_message)
    if not q:
        return ""
    if "quest" not in q.lower():
        q = f"{q} quest"
    # Bias strongly toward OSRS results (and away from RS3) at query time.
    return f"osrs \"old school runescape\" {q} guide"


_OSRS_POSITIVE_MARKERS = (
    "osrs",
    "old school runescape",
    "oldschool runescape",
    "old school rs",
    "oldschool rs",
    "07scape",
)


_OSRS_NEGATIVE_MARKERS = (
    "rs3",
    "runescape 3",
    "rune scape 3",
    "eoc",
)


def _is_osrs_relevant(video: YouTubeVideo) -> bool:
    """Strict relevance filter.

    YouTube search can return loosely-related results even with an OSRS-biased query.
    This filter ensures we only return videos that are very likely about OSRS.
    """

    haystack = " ".join(
        [
            (video.title or ""),
            (video.channel or ""),
            (video.description or ""),
        ]
    ).lower()

    if any(bad in haystack for bad in _OSRS_NEGATIVE_MARKERS):
        return False

    # Require an explicit OSRS marker. This is intentionally strict to prevent
    # showing unrelated videos in the UI.
    return any(good in haystack for good in _OSRS_POSITIVE_MARKERS)


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
        # Gaming category to reduce unrelated results.
        "videoCategoryId": "20",
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

    videos = await youtube_search_videos(
        api_key=api_key,
        query=q,
        max_results=settings.youtube_max_results,
    )

    # Keep YouTube results OSRS-only.
    return [v for v in videos if _is_osrs_relevant(v)]


async def quest_youtube_insight_chunks(*, user_message: str, max_videos: int = 3) -> list[RetrievedChunk]:
    """Turn top YouTube quest videos into short paraphrased insight chunks.

    This is designed for *citations* (Source N) and therefore avoids long verbatim transcript excerpts.
    Requires YouTube API key. If Vertex isn't configured, it falls back to video descriptions.
    """

    if not settings.youtube_api_key:
        return []

    max_videos = max(1, min(int(max_videos), 5))

    videos = await maybe_get_quest_videos(user_message=user_message)
    if not videos:
        return []

    picked = videos[:max_videos]
    out: list[RetrievedChunk] = []

    # If Vertex isn't configured, we can still cite short descriptions (best-effort).
    vertex_ok = bool(settings.google_cloud_project)

    for v in picked:
        title = v.title or "YouTube video"
        channel = v.channel or ""
        label = f"[YouTube] {title}" + (f" — {channel}" if channel else "")

        base_desc = (v.description or "").strip()
        if not base_desc:
            base_desc = "OSRS quest guide video."

        insights_text = base_desc

        if vertex_ok:
            transcript = await youtube_fetch_transcript(v.video_id)
            # Even with Vertex, don't fail the whole run if transcript is unavailable.
            if transcript:
                prompt = (
                    "You extract helpful gameplay insights from OSRS quest guide videos.\n"
                    "Return ONLY a short plain-text list of 4-8 bullet points.\n"
                    "Rules:\n"
                    "- Paraphrase. Do NOT quote the transcript verbatim.\n"
                    "- Focus on steps, requirements, common failure points, and tips.\n"
                    "- Avoid filler and avoid community/meta opinions.\n\n"
                    f"User question: {str(user_message or '').strip()}\n\n"
                    f"Video title: {title}\n"
                    f"Channel: {channel}\n"
                    f"Description: {base_desc[:700]}\n\n"
                    f"Transcript excerpt (for context only; do not quote): {transcript}\n"
                )
                try:
                    insights = GeminiVertexClient().generate(prompt).text
                    insights = re.sub(r"\s+", " ", (insights or "").strip())
                    # Keep it compact for the source display.
                    if insights:
                        insights_text = insights[:1000]
                except Exception:
                    insights_text = base_desc

        out.append(
            RetrievedChunk(
                text=insights_text,
                url=v.url,
                title=label,
            )
        )

    return out


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
    videos = [v for v in videos if _is_osrs_relevant(v)]
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
