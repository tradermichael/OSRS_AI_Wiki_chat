from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .core.config import settings
from .llm.gemini_vertex import GeminiVertexClient
from .payments.paypal import PayPalClient
from .rag.prompting import build_rag_prompt
from .rag.store import RAGStore
from .schemas import (
    CapturePayPalOrderResponse,
    ChatRequest,
    ChatResponse,
    CreatePayPalOrderRequest,
    CreatePayPalOrderResponse,
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


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    store = RAGStore()
    chunks = store.query(text=req.message, top_k=5)

    prompt = build_rag_prompt(user_message=req.message, chunks=chunks)

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
            if c.url
        ]
        return ChatResponse(answer=f"{fallback}\n\nError: {exc}", sources=sources)

    sources = [
        SourceChunk(title=c.title, url=c.url, text=c.text[:500])
        for c in chunks
        if c.url
    ]

    return ChatResponse(answer=res.text, sources=sources)


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
