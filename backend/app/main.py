from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .core.config import settings
from .core.rag_sources import allowed_url_prefixes
from .gold.store import GoldStore
from .history.store import PublicChatStore
from .llm.gemini_vertex import GeminiVertexClient
from .payments.paypal import PayPalClient
from .rag.answer_cache import AnswerCacheStore
from .rag.live_query import live_query_chunks
from .rag.prompting import build_rag_prompt
from .rag.store import RAGStore
from .rag.store import make_chunk_id
from .schemas import (
    CapturePayPalOrderResponse,
    ChatRequest,
    ChatResponse,
    CreatePayPalOrderRequest,
    CreatePayPalOrderResponse,
    GoldDonateRequest,
    GoldTotalResponse,
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

    # Fast path: reuse a previously-generated, cited answer for similar questions.
    cached = AnswerCacheStore().find_similar(question=req.message, allowed_url_prefixes=prefixes)
    if cached:
        sources = [
            SourceChunk(
                title=(s or {}).get("title"),
                url=(s or {}).get("url") or "",
                text=((s or {}).get("text") or "")[:500],
            )
            for s in (cached.sources or [])
            if (s or {}).get("url")
        ]

        PublicChatStore().add(
            session_id=req.session_id,
            user_message=req.message,
            bot_answer=cached.answer,
            sources=[s.model_dump() for s in sources],
        )
        return ChatResponse(answer=cached.answer, sources=sources)

    store = RAGStore()
    chunks = store.query(text=req.message, top_k=5, allowed_url_prefixes=prefixes)

    # If the local SQLite/BM25 store has nothing (common on fresh deploys),
    # live-query the configured wiki sources and cache the chunks.
    if not chunks:
        live_chunks = await live_query_chunks(req.message, allowed_url_prefixes=prefixes)
        if live_chunks:
            chunks = live_chunks
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

    prompt = build_rag_prompt(user_message=req.message, chunks=chunks, allowed_url_prefixes=prefixes)

    client = GeminiVertexClient()
    try:
        res = client.generate(prompt)
    except Exception as exc:
        # For local development, keep the endpoint usable even if Vertex isn't configured.
        fallback = (
            "Gemini (Vertex AI) is not configured yet. "
            "Set GOOGLE_CLOUD_PROJECT (and authenticate with gcloud) to enable LLM answers."
        )
        sources = [
            SourceChunk(title=c.title, url=c.url, text=c.text[:500])
            for c in chunks
            if c.url and (not prefixes or c.url.startswith(tuple(prefixes)))
        ]
        PublicChatStore().add(
            session_id=req.session_id,
            user_message=req.message,
            bot_answer=f"{fallback}\n\nError: {exc}",
            sources=[s.model_dump() for s in sources],
        )
        return ChatResponse(answer=f"{fallback}\n\nError: {exc}", sources=sources)

    sources = [
        SourceChunk(title=c.title, url=c.url, text=c.text[:500])
        for c in chunks
        if c.url and (not prefixes or c.url.startswith(tuple(prefixes)))
    ]

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

    PublicChatStore().add(
        session_id=req.session_id,
        user_message=req.message,
        bot_answer=res.text,
        sources=[s.model_dump() for s in sources],
    )

    return ChatResponse(answer=res.text, sources=sources)


@app.get("/api/history", response_model=PublicHistoryListResponse)
def public_history(limit: int = 50, offset: int = 0):
    items = PublicChatStore().list(limit=limit, offset=offset)
    return PublicHistoryListResponse(
        items=[
            PublicHistoryItem(
                id=i.id,
                created_at=i.created_at,
                user_message=i.user_message,
                bot_answer=i.bot_answer,
                sources=i.sources,
            )
            for i in items
        ]
    )


@app.get("/api/history/{item_id}", response_model=PublicHistoryItem)
def public_history_item(item_id: int):
    rec = PublicChatStore().get(item_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Not found")
    return PublicHistoryItem(
        id=rec.id,
        created_at=rec.created_at,
        user_message=rec.user_message,
        bot_answer=rec.bot_answer,
        sources=rec.sources,
    )


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
