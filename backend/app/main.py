from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Awaitable, Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .core.config import settings
from .core.rag_sources import allowed_url_prefixes, load_rag_sources
from .gold.store import get_gold_store
from .feedback.store import get_feedback_store
from .history.store import get_public_chat_store
from .llm.gemini_vertex import GeminiVertexClient
from .payments.paypal import PayPalClient
from .rag.answer_cache import AnswerCacheStore
from .rag.live_query import live_query_chunks, live_search_web_and_fetch_chunks
from .rag.prompting import build_rag_prompt
from .rag.query_expansion import derive_search_queries, extract_keywords
from .rag.store import RAGStore
from .rag.store import make_chunk_id
from .rag.wiki_preview import fetch_wiki_preview
from .rag.google_cse import url_to_title
from .rag.youtube import quest_youtube_videos_with_summaries
from .schemas import (
    CapturePayPalOrderResponse,
    ChatRequest,
    ChatResponse,
    CreatePayPalOrderRequest,
    CreatePayPalOrderResponse,
    GoldDonateRequest,
    GoldTotalResponse,
    FeedbackRequest,
    FeedbackResponse,
    WikiPreviewResponse,
    PublicHistoryItem,
    PublicHistoryListResponse,
    SourceChunk,
)

app = FastAPI(title="OSRS AI Wiki Chat")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/donate")
def donate_page():
    return FileResponse(STATIC_DIR / "donate.html")


@app.get("/donate/success")
def donate_success():
    return FileResponse(STATIC_DIR / "donate_success.html")


@app.get("/donate/cancel")
def donate_cancel():
    return FileResponse(STATIC_DIR / "donate_cancel.html")


@app.get("/robots.txt")
def robots():
    return FileResponse(STATIC_DIR / "robots.txt")


@app.get("/api/health")
def health():
    return {"ok": True, "env": settings.app_env}


@app.get("/api/wiki/preview", response_model=WikiPreviewResponse)
async def wiki_preview(url: str):
    prefixes = allowed_url_prefixes()
    if not url:
        raise HTTPException(status_code=400, detail="url is required")
    if prefixes and not url.startswith(tuple(prefixes)):
        raise HTTPException(status_code=400, detail="url not allowed")

    # Find the matching source config so we know which MediaWiki API to use.
    matched_api = None
    for s in load_rag_sources():
        for p in (s.allowed_url_prefixes or []):
            if p and url.startswith(p):
                matched_api = s.mediawiki_api
                break
        if matched_api:
            break

    if not matched_api:
        raise HTTPException(status_code=400, detail="No matching source config for url")

    # Convert url -> title
    from .rag.google_cse import url_to_title

    title = url_to_title(url)
    if not title:
        raise HTTPException(status_code=400, detail="Could not derive page title")

    prev = await fetch_wiki_preview(mediawiki_api=matched_api, title=title)
    # Ensure the returned fullurl still matches allowlist
    if prefixes and prev.url and not prev.url.startswith(tuple(prefixes)):
        raise HTTPException(status_code=400, detail="preview url not allowed")
    return WikiPreviewResponse(
        title=prev.title,
        url=prev.url or url,
        extract=prev.extract,
        thumbnail_url=prev.thumbnail_url,
    )


@app.get("/api/gold/total", response_model=GoldTotalResponse)
def gold_total():
    store = get_gold_store()
    return GoldTotalResponse(total_gold=store.get_total())


@app.post("/api/gold/donate", response_model=GoldTotalResponse)
def gold_donate(req: GoldDonateRequest):
    store = get_gold_store()
    try:
        total = store.add(req.amount_gold)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return GoldTotalResponse(total_gold=total)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    return await _chat_impl(req)


async def _chat_impl(
    req: ChatRequest,
    status_cb: Callable[[str], Awaitable[None]] | None = None,
) -> ChatResponse:
    prefixes = allowed_url_prefixes()
    history_store = get_public_chat_store()

    actions: list[str] = []
    web_query: str | None = None
    web_results: list[dict] = []

    async def _status(msg: str) -> None:
        if status_cb:
            await status_cb(msg)

    def _needs_pronoun_resolution(message: str) -> bool:
        m = (message or "").strip().lower()
        if not m:
            return False
        if len(m) > 140:
            return False
        pronouns = (
            " it ",
            " this ",
            " that ",
            " these ",
            " those ",
            " he ",
            " she ",
            " her ",
            " him ",
            " they ",
            " them ",
            " their ",
            " there ",
        )
        padded = f" {m} "
        return any(p in padded for p in pronouns)

    def _topic_hint_from_text(text: str) -> str:
        t = (text or "").strip()
        if not t:
            return ""
        # Prefer explicit "about X" phrasing.
        import re

        m = re.search(r"\babout\s+(.+)$", t, flags=re.IGNORECASE)
        if m:
            topic = m.group(1).strip().strip("?!.\"")
            if 1 <= len(topic) <= 60:
                return topic

        # Fall back: capture sequences of Title-Cased words, allowing Roman numerals and hyphenated subtitles.
        # Examples: "The Whisperer", "Desert Treasure II", "Desert Treasure II - The Fallen Empire".
        title_token = r"(?:\b[A-Z][a-z0-9']+\b|\b[IVX]{1,6}\b|\b\d+\b)"
        pattern = rf"{title_token}(?:\s+{title_token}){{0,7}}(?:\s*-\s*{title_token}(?:\s+{title_token}){{0,7}})?"

        caps = re.findall(pattern, t)
        if caps:
            cand = str(caps[-1] or "").strip().strip("?!.\"")
            cand = re.sub(r"\s+", " ", cand)
            if 1 <= len(cand) <= 80:
                return cand

        def _smart_title_case(s: str) -> str:
            parts = re.split(r"\s+", (s or "").strip())
            out: list[str] = []
            for p in parts:
                if not p:
                    continue
                if re.fullmatch(r"[IVX]{1,8}", p, flags=re.IGNORECASE):
                    out.append(p.upper())
                    continue
                if p.isdigit():
                    out.append(p)
                    continue
                out.append(p[:1].upper() + p[1:])
            return " ".join(out)

        # Lowercase-friendly fallback: pull entity after common combat verbs.
        # e.g. "how do I beat the whisperer?" -> "The Whisperer"
        m2 = re.search(
            r"\b(?:beat|defeat|kill|fight|handle|survive)\s+(?:the\s+)?([a-z0-9][a-z0-9'\- ]{2,80})",
            t,
            flags=re.IGNORECASE,
        )
        if m2:
            tail = (m2.group(1) or "").strip()
            tail = re.sub(r"\s+", " ", tail).strip("?!.\"'")
            # Keep it reasonably short.
            words = tail.split()
            tail = " ".join(words[:5])
            if tail:
                # Bosses/quests that use a leading "The" benefit from adding it.
                titled = _smart_title_case(tail)
                if not titled.lower().startswith("the ") and (" the " in f" {t.lower()} "):
                    titled = "The " + titled
                if 1 <= len(titled) <= 80:
                    return titled
        return ""

    def _looks_like_strategy_question(message: str) -> bool:
        m = (message or "").lower()
        return any(
            w in m
            for w in (
                "best way",
                "how do i",
                "how to",
                "beat",
                "defeat",
                "kill",
                "strategy",
                "strategies",
                "guide",
                "tips",
                "gear",
                "inventory",
                "mechanic",
                "mechanics",
                "phase",
                "prayer",
                "pray",
            )
        )

    def _looks_like_quest_help_question(message: str) -> bool:
        m = (message or "").lower()
        return any(
            w in m
            for w in (
                "quest",
                "complete",
                "finish",
                "walkthrough",
                "quick guide",
                "start",
                "requirements",
                "reqs",
                "steps",
            )
        )

    await _status("Opening the public logbook of our last words...")

    # Pull a small amount of per-session context so follow-ups like "how do I beat her" work.
    prior_turns = []
    if req.session_id:
        try:
            prior_turns = history_store.list_by_session(session_id=req.session_id, limit=4)
        except Exception:
            prior_turns = []

    if prior_turns:
        actions.append("Consulted our recent conversation for context.")

    conversation_context = ""
    if prior_turns:
        # list_by_session returns newest-first; prompt wants oldest-first.
        lines: list[str] = []
        for t in reversed(prior_turns[-3:]):
            um = (t.user_message or "").strip()
            ba = (t.bot_answer or "").strip()
            if um:
                lines.append(f"User: {um}")
            if ba:
                lines.append(f"Assistant: {ba}")
        conversation_context = "\n".join(lines).strip()

    prev_user_message = (prior_turns[0].user_message if prior_turns else "") or ""
    prev_topic_hint = _topic_hint_from_text(prev_user_message)
    cur_topic_hint = _topic_hint_from_text(req.message)

    pronoun_followup = bool(prev_user_message and _needs_pronoun_resolution(req.message))

    # Prefer current explicit topic when present; otherwise fall back to prior turn.
    topic_hint = (cur_topic_hint or prev_topic_hint).strip()

    strategy_followup = bool(topic_hint and _looks_like_strategy_question(req.message))
    quest_help = bool(topic_hint and _looks_like_quest_help_question(req.message))

    # If the user asks for "latest"/"current" info, skip the answer cache.
    msg_l = (req.message or "").lower()
    force_fresh = any(w in msg_l for w in ("latest", "today", "current", "new", "update"))

    # Context-dependent follow-ups ("her", "that", "it") should not be served from cache.
    if pronoun_followup:
        force_fresh = True
        actions.append("Follow-up detected; refreshed my search.")

    # For quest walkthrough/requirements questions, prefer live retrieval so we pull the actual quest guide pages.
    if quest_help:
        force_fresh = True
        actions.append("Quest help detected; favored fresh wiki lookups.")

    # Fast path: reuse a previously-generated, cited answer for similar questions.
    await _status("Leafing through my old scrolls for a matching answer...")
    cached = None if force_fresh else AnswerCacheStore().find_similar(
        question=req.message, allowed_url_prefixes=prefixes
    )
    if cached:
        # If the cached answer is explicitly low-confidence, force a fresh retrieval.
        ans_l = (cached.answer or "").lower()
        if any(p in ans_l for p in ("do not know", "don't know", "no mention", "naught of", "cannot find")):
            cached = None
    if cached:
        actions.append("Returned a previously-prepared scroll (with citations).")
        videos = []
        try:
            await _status("Peering into the crystal screen for quest videos...")
            videos = await quest_youtube_videos_with_summaries(user_message=req.message)
        except Exception:
            videos = []

        sources = [
            SourceChunk(
                title=(s or {}).get("title"),
                url=(s or {}).get("url") or "",
                text=((s or {}).get("text") or "")[:500],
                thumbnail_url=(s or {}).get("thumbnail_url"),
            )
            for s in (cached.sources or [])
            if (s or {}).get("url")
        ]

        history_id = history_store.add(
            session_id=req.session_id,
            user_message=req.message,
            bot_answer=cached.answer,
            sources=[s.model_dump() for s in sources],
            videos=videos,
        )
        return ChatResponse(
            answer=cached.answer,
            sources=sources,
            history_id=str(history_id),
            videos=videos,
            actions=actions,
        )

    store = RAGStore()

    # For short pronoun follow-ups, augment retrieval with the prior topic.
    retrieval_seed = req.message
    if pronoun_followup and prev_topic_hint:
        retrieval_seed = f"{req.message} {prev_topic_hint}".strip()
    elif pronoun_followup:
        retrieval_seed = f"{req.message} {prev_user_message}".strip()

    # If we can infer a specific boss/topic from the prior turn and the user is
    # asking for tactics, search the boss's /Strategies page first.
    queries = derive_search_queries(retrieval_seed)
    if strategy_followup:
        strat = f"{topic_hint}/Strategies"
        boosted = [strat, f"{topic_hint} strategies", topic_hint]
        for q in reversed(boosted):
            if q and q.lower() not in {x.lower() for x in queries}:
                queries.insert(0, q)

    # If the user asks about completing a quest, the OSRS wiki often has /Quick_guide and /Walkthrough.
    if quest_help:
        boosted = [
            # Prefer search-style queries because quest subpage titles vary ("Quick guide" vs "Quick_guide",
            # and many quests include subtitles like "- The Fallen Empire").
            f"{topic_hint} quick guide",
            f"{topic_hint} walkthrough",
            f"{topic_hint} guide",
            topic_hint,
        ]
        for q in reversed(boosted):
            if q and q.lower() not in {x.lower() for x in queries}:
                queries.insert(0, q)

    await _status("Rummaging through my local tomes and index cards...")

    # Merge top chunks across a few derived queries.
    chunks = []
    seen_chunk_keys: set[tuple[str, str]] = set()
    for q in queries or [req.message]:
        for c in store.query(text=q, top_k=5, allowed_url_prefixes=prefixes):
            key = (c.url or "", (c.text or "")[:80])
            if key in seen_chunk_keys:
                continue
            seen_chunk_keys.add(key)
            chunks.append(c)
        if len(chunks) >= 8:
            break
    chunks = chunks[:8]
    if chunks:
        actions.append("Searched my local archive.")

    # If we got chunks but they don't mention the core topic terms, treat as weak.
    # This also checks the *leading* terms so we don't miss specific concepts (e.g., "shadow realm")
    # just because generic words like "desert" or "treasure" appear.
    # Prefer the inferred topic terms (boss name) over generic strategy words like "best".
    topic_terms: list[str] = []
    if topic_hint:
        topic_terms = extract_keywords(topic_hint, max_terms=6)
        topic_terms = [t for t in topic_terms if len(t) >= 4]

    key_terms = extract_keywords(retrieval_seed, max_terms=10)
    key_terms = [t for t in key_terms if len(t) >= 4]
    weak_local = False
    if chunks and (key_terms or topic_terms):
        hay = "\n".join(((c.title or "") + "\n" + (c.text or "")) for c in chunks).lower()
        hits_all = sum(1 for t in key_terms if t in hay)

        # When we know the topic (e.g., a boss name), require it to appear.
        hits_topic = sum(1 for t in topic_terms[:2] if t in hay) if topic_terms else 0

        # Otherwise, use the first couple of extracted terms as a fallback.
        primary_terms = key_terms[:2]
        hits_primary = sum(1 for t in primary_terms if t in hay)

        # Require at least one primary term AND at least one meaningful term overall.
        weak_local = (hits_all == 0) or (hits_primary == 0)

        if topic_terms:
            weak_local = weak_local or (hits_topic == 0)

        # If we have a specific topic (boss/quest name), drop chunks that never mention it.
        # This reduces "random" sources from the local DB when the keyword overlap is weak.
        if chunks and topic_terms:
            tt = [t for t in topic_terms[:2] if t]
            if tt:
                def _mentions_topic(c) -> bool:
                    hay = (((getattr(c, "title", "") or "") + "\n" + (getattr(c, "text", "") or "")).lower())
                    return any(t in hay for t in tt)

                filtered = [c for c in chunks if _mentions_topic(c)]
                if filtered:
                    chunks = filtered

    # If the local SQLite/BM25 store has nothing (common on fresh deploys),
    # live-query the configured wiki sources and cache the chunks.
    if not chunks or weak_local or force_fresh:
        await _status("Consulting the OSRS Wiki's enchanted index...")
        live_chunks: list = []
        # Prefer the most focused query (first derived query).
        primary_q = (queries[0] if queries else retrieval_seed)
        live_chunks.extend(await live_query_chunks(primary_q, allowed_url_prefixes=prefixes))
        live_chunks = live_chunks[:8]
        if live_chunks:
            actions.append("Pulled fresh excerpts from the wiki.")
            # Merge live chunks in; prefer live if local was weak.
            chunks = live_chunks if (not chunks or weak_local or force_fresh) else (chunks + live_chunks)
            try:
                texts: list[str] = []
                metadatas: list[dict] = []
                ids: list[str] = []
                for i, c in enumerate(live_chunks):
                    if not c.url:
                        continue
                    texts.append(c.text)
                    metadatas.append({"url": c.url, "title": c.title})
                    ids.append(make_chunk_id(c.url, i))
                if texts:
                    store.add_documents(texts=texts, metadatas=metadatas, ids=ids)
            except Exception:
                # Cache is best-effort; don't break chat if it fails.
                pass

        # If MediaWiki search still didn't produce relevant context, fall back to Google PSE.
        if not chunks:
            await _status("Casting a net into the wider web for leads...")
            try:
                web_chunks, web_hits = await live_search_web_and_fetch_chunks(
                    primary_q,
                    allowed_url_prefixes=prefixes,
                    max_results=5,
                    max_chunks_total=8,
                )
            except Exception:
                web_chunks, web_hits = ([], [])

            if web_chunks:
                actions.append("Used a web search fallback to find the right wiki page.")
                web_query = primary_q
                # Include a tiny sample so the UI can show what the web search returned.
                web_results = [
                    {
                        "title": str((r.title or "")[:120]),
                        "url": str(r.url),
                        "snippet": (str(r.snippet)[:240] if r.snippet else None),
                    }
                    for r in (web_hits or [])
                    if getattr(r, "url", None)
                ][:5]
                chunks = web_chunks
                # best-effort cache
                try:
                    texts: list[str] = []
                    metadatas: list[dict] = []
                    ids: list[str] = []
                    for i, c in enumerate(web_chunks):
                        if not c.url:
                            continue
                        texts.append(c.text)
                        metadatas.append({"url": c.url, "title": c.title})
                        ids.append(make_chunk_id(c.url, i))
                    if texts:
                        store.add_documents(texts=texts, metadatas=metadatas, ids=ids)
                except Exception:
                    pass

    # De-dupe by URL before prompting so citation numbers match what we display.
    # BUT: include multiple relevant excerpts from the same page so we don't miss
    # terms that appear later in the article (e.g., user asks about "shadow realm").
    # Prefer topic terms first when scoring chunks for prompting.
    key_terms_for_prompt = []
    if topic_hint:
        key_terms_for_prompt.extend(extract_keywords(topic_hint, max_terms=6))
    key_terms_for_prompt.extend(extract_keywords(retrieval_seed, max_terms=10))
    # De-dupe while preserving order.
    seen_k: set[str] = set()
    key_terms_for_prompt = [t for t in key_terms_for_prompt if t and not (t in seen_k or seen_k.add(t))]

    url_to_chunks: dict[str, list] = {}
    for c in chunks:
        u = (c.url or "").strip()
        if not u:
            continue
        if prefixes and not u.startswith(tuple(prefixes)):
            continue
        url_to_chunks.setdefault(u, []).append(c)

    def _chunk_score(text: str) -> int:
        hay = (text or "").lower()
        return sum(1 for t in key_terms_for_prompt if t and t.lower() in hay)

    # Rank pages by their best matching chunk score.
    ranked_pairs: list[tuple[str, int]] = []
    for u, cs in url_to_chunks.items():
        best = max((_chunk_score((c.text or "")) for c in (cs or [])), default=0)
        ranked_pairs.append((u, int(best)))

    ranked_pairs.sort(key=lambda p: p[1], reverse=True)
    # If we have at least one relevant page, drop zero-score URLs so we don't cite random pages.
    nonzero = [u for (u, s) in ranked_pairs if s > 0]
    ranked_urls = nonzero if nonzero else [u for (u, _s) in ranked_pairs]

    prompt_chunks = []
    for u in ranked_urls[:6]:
        cs = url_to_chunks.get(u) or []
        if not cs:
            continue

        # Pick a few best chunks from this page.
        ranked_chunks = sorted(cs, key=lambda c: _chunk_score((c.text or "")), reverse=True)
        picked = ranked_chunks[:3]
        merged_text = "\n\n".join(((c.text or "").strip()) for c in picked if (c.text or "").strip())
        merged_text = merged_text.strip()

        # Cap merged context per URL to keep prompts bounded.
        if len(merged_text) > 2200:
            merged_text = merged_text[:2200] + "..."

        prompt_chunks.append(
            type(cs[0])(
                text=merged_text or (cs[0].text or ""),
                url=u,
                title=cs[0].title,
            )
        )

    await _status("Arranging citations and weaving your answer...")
    prompt = build_rag_prompt(
        user_message=req.message,
        conversation_context=conversation_context,
        chunks=prompt_chunks,
        allowed_url_prefixes=prefixes,
    )

    client = GeminiVertexClient()
    try:
        await _status("Consulting my crystal ball (Gemini)...")
        res = client.generate(prompt)
    except Exception as exc:
        # For local development, keep the endpoint usable even if Vertex isn't configured.
        fallback = (
            "Gemini (Vertex AI) is not configured yet. "
            "Set GOOGLE_CLOUD_PROJECT (and authenticate with gcloud) to enable LLM answers."
        )

        videos = []
        try:
            videos = await quest_youtube_videos_with_summaries(user_message=req.message)
        except Exception:
            videos = []

        sources = []
        for c in prompt_chunks[:5]:
            if not c.url:
                continue
            sources.append(SourceChunk(title=c.title, url=c.url, text=c.text[:500]))
        history_id = history_store.add(
            session_id=req.session_id,
            user_message=req.message,
            bot_answer=f"{fallback}\n\nError: {exc}",
            sources=[s.model_dump() for s in sources],
            videos=videos,
        )
        return ChatResponse(
            answer=f"{fallback}\n\nError: {exc}",
            sources=sources,
            history_id=str(history_id),
            videos=videos,
            actions=actions + ["Could not reach Gemini; returned a friendly fallback."],
            web_query=web_query,
            web_results=web_results,
        )

    actions.append("Composed the reply with citations.")

    # Attach a small thumbnail per cited URL (best-effort) so the UI can show relevant images.
    # This is intentionally limited to a few sources to keep latency reasonable.
    url_to_thumb: dict[str, str | None] = {}
    try:
        await _status("Fetching a few illustrations for the sources...")
        sources_cfg = load_rag_sources()
        prefix_to_api: list[tuple[str, str]] = []
        for s in sources_cfg:
            for p in (s.allowed_url_prefixes or []):
                if p and s.mediawiki_api:
                    prefix_to_api.append((p, s.mediawiki_api))

        async def _thumb_for_url(u: str) -> tuple[str, str | None]:
            api = None
            for p, a in prefix_to_api:
                if u.startswith(p):
                    api = a
                    break
            if not api:
                return (u, None)
            title = url_to_title(u)
            if not title:
                return (u, None)
            try:
                prev = await fetch_wiki_preview(mediawiki_api=api, title=title)
                return (u, prev.thumbnail_url)
            except Exception:
                return (u, None)

        import asyncio

        tasks = [_thumb_for_url((c.url or "").strip()) for c in prompt_chunks if (c.url or "").strip()]
        if tasks:
            pairs = await asyncio.gather(*tasks, return_exceptions=False)
            url_to_thumb = {u: t for (u, t) in pairs}
    except Exception:
        url_to_thumb = {}

    sources: list[SourceChunk] = []
    for c in prompt_chunks[:5]:
        if not c.url:
            continue
        u = c.url
        sources.append(
            SourceChunk(
                title=c.title,
                url=u,
                text=c.text[:500],
                thumbnail_url=url_to_thumb.get(u),
            )
        )

    # Store a cited copy for faster future retrieval.
    # Avoid caching explicit "I can't find it" answers so future runs will re-research.
    if sources:
        try:
            ans_l = (res.text or "").lower()
            if not any(p in ans_l for p in ("do not know", "don't know", "no mention", "naught of", "cannot find")):
                AnswerCacheStore().add(
                    question=req.message,
                    answer=res.text,
                    sources=[s.model_dump() for s in sources],
                )
        except Exception:
            pass

    # Quest-only YouTube results (best-effort; doesn't block the core answer).
    videos = []
    try:
        await _status("Scrying the crystal screen for quest videos...")
        videos = await quest_youtube_videos_with_summaries(user_message=req.message)
        if videos:
            actions.append("Found quest videos and summarized them.")
    except Exception:
        videos = []

    history_id = history_store.add(
        session_id=req.session_id,
        user_message=req.message,
        bot_answer=res.text,
        sources=[s.model_dump() for s in sources],
        videos=videos,
    )

    return ChatResponse(
        answer=res.text,
        sources=sources,
        history_id=str(history_id),
        videos=videos,
        actions=actions,
        web_query=web_query,
        web_results=web_results,
    )


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    async def gen():
        queue: asyncio.Queue[dict] = asyncio.Queue()
        done = asyncio.Event()

        async def status_cb(msg: str) -> None:
            await queue.put({"type": "status", "text": str(msg)})

        async def run() -> None:
            try:
                resp = await _chat_impl(req, status_cb=status_cb)
                await queue.put({"type": "final", "data": resp.model_dump()})
            except HTTPException as exc:
                await queue.put({"type": "error", "status": exc.status_code, "detail": str(exc.detail)})
            except Exception as exc:
                await queue.put({"type": "error", "status": 500, "detail": str(exc)})
            finally:
                done.set()

        task = asyncio.create_task(run())
        try:
            while True:
                if done.is_set() and queue.empty():
                    break
                item = await queue.get()
                yield json.dumps(item, ensure_ascii=False) + "\n"
        finally:
            await task

    return StreamingResponse(
        gen(),
        media_type="application/x-ndjson",
        headers={"Content-Type": "application/x-ndjson; charset=utf-8"},
    )


@app.get("/api/history", response_model=PublicHistoryListResponse)
def public_history(limit: int = 50, offset: int = 0):
    items = get_public_chat_store().list(limit=limit, offset=offset)
    return PublicHistoryListResponse(
        items=[
            PublicHistoryItem(
                id=i.id,
                created_at=i.created_at,
                user_message=i.user_message,
                bot_answer=i.bot_answer,
                sources=i.sources,
                videos=getattr(i, "videos", []) or [],
            )
            for i in items
        ]
    )


@app.get("/api/history/{item_id}", response_model=PublicHistoryItem)
def public_history_item(item_id: str):
    rec = get_public_chat_store().get(item_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Not found")
    return PublicHistoryItem(
        id=rec.id,
        created_at=rec.created_at,
        user_message=rec.user_message,
        bot_answer=rec.bot_answer,
        sources=rec.sources,
        videos=getattr(rec, "videos", []) or [],
    )


@app.post("/api/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    store = get_feedback_store()
    try:
        fid = store.add(history_id=req.history_id, rating=req.rating, session_id=req.session_id)
        summary = store.summary(history_id=req.history_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FeedbackResponse(ok=True, feedback_id=fid, summary=summary)


@app.post("/api/paypal/create-order", response_model=CreatePayPalOrderResponse)
async def paypal_create_order(req: CreatePayPalOrderRequest):
    paypal = PayPalClient()
    try:
        order_id, approve_url = await paypal.create_order(amount_usd=req.amount_usd, note=req.note)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return CreatePayPalOrderResponse(order_id=order_id, approve_url=approve_url)


@app.post("/api/paypal/capture-order", response_model=CapturePayPalOrderResponse)
async def paypal_capture_order(order_id: str):
    paypal = PayPalClient()
    try:
        data = await paypal.capture_order(order_id=order_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    payer_email = (((data or {}).get("payer") or {}).get("email_address"))
    status = data.get("status") or "UNKNOWN"

    return CapturePayPalOrderResponse(order_id=order_id, status=status, payer_email=payer_email)
