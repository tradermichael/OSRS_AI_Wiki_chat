from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .core.config import settings
from .core.rag_sources import allowed_url_prefixes, load_rag_sources
from .gold.store import GoldStore
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
    store = GoldStore()
    return GoldTotalResponse(total_gold=store.get_total())


@app.post("/api/gold/donate", response_model=GoldTotalResponse)
def gold_donate(req: GoldDonateRequest):
    store = GoldStore()
    try:
        total = store.add(req.amount_gold)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return GoldTotalResponse(total_gold=total)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    prefixes = allowed_url_prefixes()
    history_store = get_public_chat_store()

    # If the user asks for "latest"/"current" info, skip the answer cache.
    msg_l = (req.message or "").lower()
    force_fresh = any(w in msg_l for w in ("latest", "today", "current", "new", "update"))

    # Fast path: reuse a previously-generated, cited answer for similar questions.
    cached = None if force_fresh else AnswerCacheStore().find_similar(
        question=req.message, allowed_url_prefixes=prefixes
    )
    if cached:
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
            videos=[],
        )
        return ChatResponse(answer=cached.answer, sources=sources, history_id=str(history_id), videos=[])

    store = RAGStore()
    queries = derive_search_queries(req.message)

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

    # If we got chunks but they don't mention the core topic terms, treat as weak.
    key_terms = extract_keywords(req.message)
    key_terms = [t for t in key_terms if len(t) >= 4]
    weak_local = False
    if chunks and key_terms:
        hay = "\n".join(((c.title or "") + "\n" + (c.text or "")) for c in chunks).lower()
        hits = sum(1 for t in key_terms if t in hay)
        # Require at least one meaningful term to appear in the retrieved context.
        weak_local = hits == 0

    # If the local SQLite/BM25 store has nothing (common on fresh deploys),
    # live-query the configured wiki sources and cache the chunks.
    if not chunks or weak_local or force_fresh:
        live_chunks: list = []
        # Prefer the most focused query (first derived query).
        primary_q = (queries[0] if queries else req.message)
        live_chunks.extend(await live_query_chunks(primary_q, allowed_url_prefixes=prefixes))
        live_chunks = live_chunks[:8]
        if live_chunks:
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
            try:
                web_chunks = await live_search_web_and_fetch_chunks(
                    primary_q,
                    allowed_url_prefixes=prefixes,
                    max_results=5,
                    max_chunks_total=8,
                )
            except Exception:
                web_chunks = []

            if web_chunks:
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
    prompt_chunks = []
    prompt_seen: set[str] = set()
    for c in chunks:
        u = (c.url or "").strip()
        if not u:
            continue
        if prefixes and not u.startswith(tuple(prefixes)):
            continue
        if u in prompt_seen:
            continue
        prompt_seen.add(u)
        prompt_chunks.append(c)
        if len(prompt_chunks) >= 6:
            break

    prompt = build_rag_prompt(user_message=req.message, chunks=prompt_chunks, allowed_url_prefixes=prefixes)

    client = GeminiVertexClient()
    try:
        res = client.generate(prompt)
    except Exception as exc:
        # For local development, keep the endpoint usable even if Vertex isn't configured.
        fallback = (
            "Gemini (Vertex AI) is not configured yet. "
            "Set GOOGLE_CLOUD_PROJECT (and authenticate with gcloud) to enable LLM answers."
        )
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
            videos=[],
        )
        return ChatResponse(answer=f"{fallback}\n\nError: {exc}", sources=sources, history_id=str(history_id), videos=[])

    # Attach a small thumbnail per cited URL (best-effort) so the UI can show relevant images.
    # This is intentionally limited to a few sources to keep latency reasonable.
    url_to_thumb: dict[str, str | None] = {}
    try:
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
    if sources:
        try:
            AnswerCacheStore().add(
                question=req.message,
                answer=res.text,
                sources=[s.model_dump() for s in sources],
            )
        except Exception:
            pass

    history_id = history_store.add(
        session_id=req.session_id,
        user_message=req.message,
        bot_answer=res.text,
        sources=[s.model_dump() for s in sources],
        videos=[],
    )

    return ChatResponse(answer=res.text, sources=sources, history_id=str(history_id), videos=[])


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
