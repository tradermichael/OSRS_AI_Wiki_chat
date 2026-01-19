from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=128)


class SourceChunk(BaseModel):
    title: str | None = None
    url: str
    text: str
    thumbnail_url: str | None = None


class VideoItem(BaseModel):
    title: str
    url: str
    channel: str | None = None
    summary: str


class WebSearchResult(BaseModel):
    title: str
    url: str
    snippet: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk] = []
    history_id: str | None = None
    videos: list[VideoItem] = []
    actions: list[str] = []
    web_query: str | None = None
    web_results: list[WebSearchResult] = []


class CreatePayPalOrderRequest(BaseModel):
    amount_usd: str = Field(
        ..., pattern=r"^\d+(?:\.\d{1,2})?$", description="Amount like 5 or 5.00"
    )
    note: str | None = Field(default=None, max_length=140)


class CreatePayPalOrderResponse(BaseModel):
    order_id: str
    approve_url: str


class CapturePayPalOrderResponse(BaseModel):
    order_id: str
    status: str
    payer_email: str | None = None


class GoldDonateRequest(BaseModel):
    amount_gold: int = Field(..., ge=1, le=2_147_483_647)


class GoldTotalResponse(BaseModel):
    total_gold: int


class PublicHistoryItem(BaseModel):
    id: str
    created_at: str
    user_message: str
    bot_answer: str
    sources: list[dict] = []
    videos: list[dict] = []


class FeedbackRequest(BaseModel):
    history_id: str = Field(min_length=1, max_length=128)
    rating: int = Field(..., description="1 for thumbs up, -1 for thumbs down")
    session_id: str | None = Field(default=None, max_length=128)


class FeedbackResponse(BaseModel):
    ok: bool = True
    feedback_id: str
    summary: dict = {}


class WikiPreviewResponse(BaseModel):
    title: str
    url: str
    extract: str
    thumbnail_url: str | None = None


class PublicHistoryListResponse(BaseModel):
    items: list[PublicHistoryItem]
