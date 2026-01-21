from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
from pathlib import Path
from typing import Awaitable, Callable

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi import File, Form, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .core.config import settings
from .core.rag_sources import allowed_url_prefixes, load_rag_sources
from .gold.store import get_gold_store
from .feedback.store import get_feedback_store
from .history.store import get_public_chat_store
from .llm.gemini_vertex import GeminiVertexClient
from .llm.answer_judge import judge_answer_confidence
from .payments.paypal import PayPalClient
from .visits.store import get_visits_store
from .rag.answer_cache import AnswerCacheStore
from .rag.live_query import live_query_chunks, live_search_web_and_fetch_chunks, live_search_web_and_scrape_chunks
from .rag.prompting import build_rag_prompt
from .rag.query_expansion import derive_search_queries, extract_keywords
from .rag.quest_registry import find_quest_title_in_text, load_osrs_quest_titles
from .rag.store import RAGStore
from .rag.store import make_chunk_id
from .rag.store import RetrievedChunk
from .rag.wiki_preview import fetch_wiki_preview
from .rag.google_cse import url_to_title
from .rag.youtube import (
    quest_youtube_videos_with_summaries,
    quest_youtube_insight_chunks,
    opinion_youtube_insight_chunks,
    combat_youtube_insight_chunks,
    general_youtube_fallback_chunks,
)
from .rag.reddit import reddit_search_chunks

from typing import Any
from .schemas import (
    CapturePayPalOrderResponse,
    ChatRequest,
    ChatResponse,
    CreatePayPalOrderRequest,
    CreatePayPalOrderResponse,
    GoldDonateRequest,
    GoldTotalResponse,
    VisitsTotalResponse,
    FeedbackRequest,
    FeedbackResponse,
    WikiPreviewResponse,
    SpeechTranscribeResponse,
    PublicHistoryItem,
    PublicHistoryListResponse,
    SourceChunk,
    VideoItem,
    WebSearchResult,
)

app = FastAPI(title="OSRS AI Wiki Chat")

logger = logging.getLogger("osrs_ai_wiki_chat")


async def _auto_retry_targeted_retrieval(
    *,
    raw_user_message: str,
    user_message: str,
    retrieval_seed: str,
    topic_hint: str,
    msg_l: str,
    difficulty_question: bool,
    queries: list[str],
    prefixes: list[str] | None,
    existing_web_query: str | None,
    existing_web_results: list[WebSearchResult] | None,
) -> tuple[list[RetrievedChunk], str | None, list[WebSearchResult] | None, list[str]]:
    """Best-effort retry when first-pass retrieval yields no usable prompt chunks.

    Returns: (chunks, web_query, web_results, actions_to_add)
    """

    actions: list[str] = []

    retry_candidates: list[str] = []
    for q in (queries or []):
        qq = (q or "").strip()
        if qq:
            retry_candidates.append(qq)

    # If this reads like a boss difficulty question, try boss-focused variants.
    if topic_hint and (difficulty_question or ("boss" in msg_l) or ("fight" in msg_l)):
        boss_map = {
            "Song of the Elves": ["Fragment of Seren", "Fragment of Seren/Strategies"],
        }
        for extra in boss_map.get(topic_hint, []):
            retry_candidates.insert(0, extra)
        retry_candidates.insert(0, f"{topic_hint} toughest boss")
        retry_candidates.insert(0, f"{topic_hint} final boss")
        retry_candidates.insert(0, f"{topic_hint} boss")

    for q in (retrieval_seed, raw_user_message, user_message):
        qq = (q or "").strip()
        if qq:
            retry_candidates.append(qq)

    # De-dupe while preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for q in retry_candidates:
        qn = (q or "").strip()
        if not qn:
            continue
        key = qn.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(qn)
    retry_candidates = deduped

    # 1) Try MediaWiki first.
    for q in retry_candidates[:6]:
        try:
            live_chunks = await live_query_chunks(q, allowed_url_prefixes=prefixes)
        except Exception:
            live_chunks = []
        if live_chunks:
            actions.append("Auto-retry: performed a targeted wiki lookup.")
            return live_chunks, existing_web_query, existing_web_results, actions

    # 2) If CSE is configured, try one last web fetch.
    if settings.google_cse_api_key and (settings.google_cse_cx or settings.google_cse_community_cx):
        q = (retry_candidates[0] if retry_candidates else "") or (raw_user_message or user_message or retrieval_seed)
        try:
            web_chunks, web_hits = await live_search_web_and_fetch_chunks(
                q,
                allowed_url_prefixes=prefixes,
                max_results=5,
                max_chunks_total=8,
            )
        except Exception:
            web_chunks, web_hits = ([], [])

        web_query = existing_web_query
        web_results = existing_web_results

        if web_hits and not web_results:
            web_query = q
            web_results = [
                WebSearchResult(
                    title=str((r.title or "")[:120]),
                    url=str(r.url),
                    snippet=(str(r.snippet)[:240] if r.snippet else None),
                )
                for r in (web_hits or [])
                if getattr(r, "url", None)
            ][:5]
            actions.append(f"Google CSE returned {len(web_results)} lead(s).")

        if web_chunks:
            actions.append("Auto-retry: used web search to fetch wiki excerpts.")
            return web_chunks, web_query, web_results, actions

    return [], existing_web_query, existing_web_results, actions

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
async def _startup_prefetch() -> None:
    # Best-effort: load a list of quests so we can recognize quest names even when the user types them in lowercase.
    # If it fails, the app still works (we fall back to heuristic topic extraction).
    try:
        await load_osrs_quest_titles()
    except Exception:
        pass


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


@app.post("/api/speech/transcribe", response_model=SpeechTranscribeResponse)
async def speech_transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
):
    """Transcribe short audio snippets via Google Cloud Speech-to-Text.

    Intended as a fallback when browser SpeechRecognition isn't available.
    Uses Application Default Credentials (gcloud auth/service account) and must be enabled
    via GCP_SPEECH_ENABLED=true.
    """

    if not bool(settings.gcp_speech_enabled):
        raise HTTPException(status_code=400, detail="Speech-to-text is disabled (set GCP_SPEECH_ENABLED=true).")

    ct = (file.content_type or "").lower().strip()
    if ct not in {
        "audio/webm",
        "audio/webm;codecs=opus",
        "audio/ogg",
        "audio/ogg;codecs=opus",
        "audio/wav",
        "audio/x-wav",
        "audio/mpeg",
        "audio/mp4",
    }:
        raise HTTPException(status_code=400, detail=f"Unsupported audio type: {file.content_type!r}")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio upload")

    max_bytes = int(getattr(settings, "gcp_speech_max_bytes", 3_000_000) or 3_000_000)
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Audio too large (max {max_bytes} bytes)")

    lang = (language or settings.gcp_speech_language or "en-US").strip() or "en-US"

    try:
        from google.cloud import speech
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"google-cloud-speech not installed: {exc}") from exc

    enc = speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED
    if "webm" in ct:
        enc = speech.RecognitionConfig.AudioEncoding.WEBM_OPUS
    elif "ogg" in ct:
        enc = speech.RecognitionConfig.AudioEncoding.OGG_OPUS
    elif "wav" in ct:
        enc = speech.RecognitionConfig.AudioEncoding.LINEAR16
    elif "mpeg" in ct:
        enc = speech.RecognitionConfig.AudioEncoding.MP3

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=enc,
        language_code=lang,
        enable_automatic_punctuation=True,
        model="latest_short",
    )
    audio = speech.RecognitionAudio(content=data)

    try:
        resp = client.recognize(config=config, audio=audio)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Speech recognition failed: {exc}") from exc

    parts: list[str] = []
    for r in (resp.results or []):
        alts = getattr(r, "alternatives", None) or []
        if not alts:
            continue
        t = (alts[0].transcript or "").strip()
        if t:
            parts.append(t)
    text = " ".join(parts).strip()

    return SpeechTranscribeResponse(text=text)


@app.websocket("/api/live/ws")
async def gemini_live_websocket(ws: WebSocket):
    """Realtime voice chat proxy for Gemini Live API.

    Gemini Live is designed for server-to-server auth, so the browser connects here,
    and this server holds the authenticated Live API session.

    Protocol (browser <-> server):
    - First message: JSON text {"type":"start", "systemInstruction"?: str, "voiceName"?: str}
    - Audio frames: raw bytes (PCM16LE @ 16kHz) sent as binary websocket messages
    - Control: JSON text {"type":"audio_stream_end"} to flush/trigger response
    """

    await ws.accept()

    if not bool(getattr(settings, "gemini_live_enabled", False)):
        await ws.send_json({"type": "error", "detail": "Gemini Live is disabled (set GEMINI_LIVE_ENABLED=true)."})
        # 1008 shows as "policy violation" in browsers, which is confusing for a config flag.
        await ws.close(code=1011)
        return

    try:
        from google import genai
        from google.genai import types
    except Exception as exc:
        await ws.send_json({"type": "error", "detail": f"google-genai not installed: {exc}"})
        await ws.close(code=1011)
        return

    project = settings.google_cloud_project
    api_key = getattr(settings, "gemini_live_api_key", None)
    auth_mode = "api_key" if api_key else "vertex"

    default_live_location = (getattr(settings, "vertex_location", None) or "us-central1").strip() or "us-central1"
    location = (getattr(settings, "gemini_live_location", None) or default_live_location).strip() or default_live_location
    model = (getattr(settings, "gemini_live_model", None) or "gemini-live-2.5-flash-native-audio").strip()
    default_voice = getattr(settings, "gemini_live_voice_name", None)

    if auth_mode == "vertex" and not project:
        await ws.send_json({"type": "error", "detail": "GOOGLE_CLOUD_PROJECT is required for Gemini Live (Vertex mode)."})
        await ws.close(code=1011)
        return
    if auth_mode == "api_key" and not api_key:
        await ws.send_json({"type": "error", "detail": "GEMINI_LIVE_API_KEY is required for Gemini Live (API key mode)."})
        await ws.close(code=1011)
        return

    def _normalize_model_name(m: str) -> str:
        m = (m or "").strip()
        # Accept full resource paths like "projects/.../publishers/google/models/<id>"
        if "/models/" in m:
            m = m.rsplit("/models/", 1)[-1].strip()
        return m

    allowed_models = {
        "gemini-live-2.5-flash-native-audio",
        "gemini-live-2.5-flash-exp-native-audio",
    }

    # Normalize server-configured model as well (env vars sometimes contain full resource paths).
    model = _normalize_model_name(model)

    # Guard against a common misconfiguration: truncated model names.
    # The intended model is e.g. "gemini-live-2.5-flash-native-audio".
    if model in {"gemini-live-2.5-flash-native-a", "gemini-live-2.5-flash-native"} or model.endswith("-native-a"):
        await ws.send_json(
            {
                "type": "error",
                "detail": f"Invalid GEMINI_LIVE_MODEL '{model}'.",
                "hint": "Did you mean 'gemini-live-2.5-flash-native-audio'?",
                "model": model,
                "location": location,
            }
        )
        await ws.close(code=1011)
        return

    # If the server is configured with an unsupported model ID, fail fast with a clear error.
    # NOTE: In API-key mode, model IDs can differ; don't hard-block unless it's clearly malformed.
    if auth_mode == "vertex" and model and model not in allowed_models:
        await ws.send_json(
            {
                "type": "error",
                "detail": f"Unsupported GEMINI_LIVE_MODEL '{model}'.",
                "hint": "Use 'gemini-live-2.5-flash-native-audio'.",
                "model": model,
                "location": location,
            }
        )
        await ws.close(code=1011)
        return

    try:
        start_raw = await ws.receive_text()
        start_msg = json.loads(start_raw)
    except Exception:
        await ws.send_json({"type": "error", "detail": "Expected initial JSON start message."})
        await ws.close(code=1003)
        return

    if (start_msg.get("type") or "start") != "start":
        await ws.send_json({"type": "error", "detail": "First message must be {type:'start', ...}."})
        await ws.close(code=1003)
        return

    # Allow the client to request a specific live model (still validated on the server).
    requested_model = _normalize_model_name(str(start_msg.get("model") or start_msg.get("modelId") or ""))
    if requested_model:
        if auth_mode == "vertex" and requested_model not in allowed_models:
            await ws.send_json(
                {
                    "type": "error",
                    "detail": f"Unsupported live model '{requested_model}'.",
                    "hint": "Use 'gemini-live-2.5-flash-native-audio'.",
                    "model": requested_model,
                    "location": location,
                }
            )
            await ws.close(code=1011)
            return
        model = requested_model

    system_instruction = (start_msg.get("systemInstruction") or "").strip() or (
        "You are Wise Old AI, an ancient wizard from the world of Gielinor (Old School RuneScape). "
        "For this entire conversation, adopt the persona of an ancient, wise wizard. "
        "Use a slow, deliberate pace with long pauses for dramatic effect. "
        "Use archaic language, metaphors about the stars and elements, and a raspy, weathered tone. "
        "Speak with great gravitas as if you are imparting ancient secrets and guidance. "
        "You help adventurers with quests, skills, monsters, items, and all matters of the realm. "
        "Keep your knowledge accurate to Old School RuneScape. "
        "IMPORTANT: You have access to the search_osrs_wiki tool. Use it to look up accurate information "
        "about quests, items, monsters, bosses, skills, drop rates, requirements, and strategies. "
        "Always use the tool when the adventurer asks about specific game mechanics, stats, or guides. "
        "After receiving search results, summarize the key information in your mystical wizard voice."
    )

    voice_name = (start_msg.get("voiceName") or default_voice or "").strip() or None

    # Configure session.
    speech_config = None
    if voice_name:
        speech_config = types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
            )
        )

    # Define tools for function calling - enables RAG for voice chat
    # The wiki search tool lets Gemini look up accurate OSRS information
    wiki_search_tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="search_osrs_wiki",
                description=(
                    "Search the Old School RuneScape Wiki for information about quests, items, monsters, "
                    "skills, bosses, NPCs, locations, or any other OSRS topic. Use this tool whenever "
                    "you need accurate, up-to-date information about OSRS that you're not certain about. "
                    "Always search before answering questions about specific stats, requirements, drop rates, "
                    "quest guides, boss strategies, or game mechanics."
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "query": types.Schema(
                            type=types.Type.STRING,
                            description=(
                                "The search query. Be specific - use exact item names, monster names, "
                                "quest names, or skill names. For strategies, include '/Strategies' suffix. "
                                "Examples: 'Abyssal whip', 'Zulrah/Strategies', 'Dragon Slayer II requirements'"
                            ),
                        ),
                    },
                    required=["query"],
                ),
            ),
        ]
    )

    # Configure realtime input with automatic Voice Activity Detection (VAD)
    # This makes it feel like a real phone call - Gemini detects when you stop talking
    realtime_input_config = types.RealtimeInputConfig(
        automaticActivityDetection=types.AutomaticActivityDetection(
            disabled=False,  # Enable VAD
            silenceDurationMs=1500,  # Wait 1.5s of silence before considering speech ended
        ),
        # CRITICAL for multi-turn: include ALL input, not just first activity
        turnCoverage=types.TurnCoverage.TURN_INCLUDES_ALL_INPUT,
        # Allow user to interrupt the AI while it's speaking
        activityHandling=types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
    )

    # Gemini Live currently allows at most one response modality in the setup request.
    # We use AUDIO for voice chat, and enable transcripts via input/output_audio_transcription.
    connect_config = types.LiveConnectConfig(
        responseModalities=[types.Modality.AUDIO],
        systemInstruction=system_instruction,
        inputAudioTranscription=types.AudioTranscriptionConfig(),
        outputAudioTranscription=types.AudioTranscriptionConfig(),
        speechConfig=speech_config,
        realtimeInputConfig=realtime_input_config,
        # Enable transparent session resumption for multi-turn conversation
        sessionResumption=types.SessionResumptionConfig(transparent=True),
        # Enable tool use for RAG - wiki search
        tools=[wiki_search_tool],
    )

    # Create client and connect.
    # Vertex mode uses ADC + project/location.
    # API-key mode uses the Gemini API key and ignores project/location.
    if auth_mode == "vertex":
        client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            http_options=types.HttpOptions(api_version="v1beta1"),
        )
    else:
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(api_version="v1beta1"),
        )

    async def _send_server_content(server_content: object) -> None:
        # server_content is a types.LiveServerContent.
        interrupted = bool(getattr(server_content, "interrupted", False))
        if interrupted:
            await ws.send_json({"type": "interrupted"})

        input_tx = getattr(server_content, "input_transcription", None)
        if input_tx and getattr(input_tx, "text", None):
            await ws.send_json(
                {
                    "type": "input_transcript",
                    "text": input_tx.text,
                    "finished": bool(getattr(input_tx, "finished", False)),
                }
            )

        output_tx = getattr(server_content, "output_transcription", None)
        if output_tx and getattr(output_tx, "text", None):
            await ws.send_json(
                {
                    "type": "output_transcript",
                    "text": output_tx.text,
                    "finished": bool(getattr(output_tx, "finished", False)),
                }
            )

        model_turn = getattr(server_content, "model_turn", None)
        if model_turn:
            parts = getattr(model_turn, "parts", None) or []
            for part in parts:
                text = getattr(part, "text", None)
                if text:
                    await ws.send_json({"type": "model_text", "text": text})

                inline = getattr(part, "inline_data", None)
                if inline and getattr(inline, "data", None) and getattr(inline, "mime_type", None):
                    mime = str(inline.mime_type)
                    if mime.startswith("audio/"):
                        b64 = base64.b64encode(inline.data).decode("ascii")
                        logger.info("Sending audio to client: %s, %d bytes encoded", mime, len(b64))
                        await ws.send_json({"type": "audio", "mime": mime, "data": b64})

        if bool(getattr(server_content, "turn_complete", False)):
            logger.info("Gemini Live turn_complete received")
            await ws.send_json({"type": "turn_complete"})

    async def _execute_wiki_search(query: str) -> str:
        """Execute a wiki search for the tool call and return formatted results."""
        query = (query or "").strip()
        if not query:
            return "No query provided. Please specify what to search for."
        
        logger.info("Executing wiki search for voice chat: %s", query)
        
        prefixes = allowed_url_prefixes()
        
        try:
            # First try the local/fast wiki lookup
            chunks = await live_query_chunks(
                query,
                allowed_url_prefixes=prefixes,
                max_pages_per_source=2,
                max_chunks_total=4,  # Keep it concise for voice
            )
            
            # If no results from wiki, try web search
            if not chunks and settings.google_cse_api_key and settings.google_cse_cx:
                web_chunks, _ = await live_search_web_and_fetch_chunks(
                    query,
                    allowed_url_prefixes=prefixes,
                    max_results=3,
                    max_chunks_total=4,
                )
                chunks = web_chunks
            
            if not chunks:
                return f"No information found for '{query}'. The topic may not exist in the OSRS Wiki, or try rephrasing your search."
            
            # Format results for Gemini to speak
            result_parts = []
            for i, chunk in enumerate(chunks[:4], 1):  # Max 4 chunks
                title = getattr(chunk, "title", None) or "OSRS Wiki"
                text = getattr(chunk, "text", "") or ""
                # Truncate long chunks for voice (keep it speakable)
                if len(text) > 800:
                    text = text[:800] + "..."
                result_parts.append(f"--- {title} ---\n{text}")
            
            return "\n\n".join(result_parts)
            
        except Exception as exc:
            logger.exception("Wiki search failed for voice chat: %s", exc)
            return f"Wiki search failed: {exc}. Please try again or ask in a different way."

    async def _pump_from_gemini(session) -> None:
        logger.info("Starting to pump messages from Gemini Live")
        message_count = 0
        last_turn_complete_at = 0
        try:
            async for message in session.receive():
                message_count += 1
                msg_type = type(message).__name__
                
                # Log all message types to understand session lifecycle
                logger.info("Gemini Live message #%d: type=%s", message_count, msg_type)
                
                # Check for go_away or session termination signals
                go_away = getattr(message, "go_away", None)
                if go_away is not None:
                    logger.warning("Gemini Live GO_AWAY received: %s", go_away)
                    # Send to client so we know why session is ending
                    with contextlib.suppress(Exception):
                        await ws.send_json({"type": "go_away", "detail": str(go_away)})
                
                # Check for setup_complete (indicates session is ready for multi-turn)
                setup_complete = getattr(message, "setup_complete", None)
                if setup_complete is not None:
                    logger.info("Gemini Live SETUP_COMPLETE received - session ready for multi-turn")
                    with contextlib.suppress(Exception):
                        await ws.send_json({"type": "setup_complete"})
                
                # Check for session resumption updates
                session_resumption_update = getattr(message, "session_resumption_update", None)
                if session_resumption_update is not None:
                    logger.info("Gemini Live SESSION_RESUMPTION_UPDATE: %s", session_resumption_update)
                
                sc = getattr(message, "server_content", None)
                if sc is not None:
                    is_turn_complete = bool(getattr(sc, "turn_complete", False))
                    if is_turn_complete:
                        last_turn_complete_at = message_count
                        logger.info("Gemini Live TURN_COMPLETE at message #%d", message_count)
                    await _send_server_content(sc)

                # Some SDK versions expose a convenience text field.
                msg_text = getattr(message, "text", None)
                if msg_text:
                    await ws.send_json({"type": "model_text", "text": msg_text})

                # Handle tool calls - this is where RAG happens for voice chat!
                tool_call = getattr(message, "tool_call", None)
                if tool_call is not None:
                    function_calls = getattr(tool_call, "function_calls", None) or []
                    if function_calls:
                        logger.info("Gemini Live TOOL_CALL received with %d function(s)", len(function_calls))
                        # Notify client that we're doing a wiki lookup
                        with contextlib.suppress(Exception):
                            await ws.send_json({"type": "tool_call", "detail": "Searching OSRS Wiki..."})
                        
                        function_responses = []
                        for fc in function_calls:
                            fc_name = getattr(fc, "name", None) or ""
                            fc_args = getattr(fc, "args", None) or {}
                            fc_id = getattr(fc, "id", None)
                            
                            logger.info("Tool call: name=%s args=%s", fc_name, fc_args)
                            
                            if fc_name == "search_osrs_wiki":
                                query = fc_args.get("query", "")
                                result_text = await _execute_wiki_search(query)
                                function_responses.append(
                                    types.FunctionResponse(
                                        id=fc_id,
                                        name=fc_name,
                                        response={"result": result_text},
                                    )
                                )
                            else:
                                # Unknown function - return error
                                function_responses.append(
                                    types.FunctionResponse(
                                        id=fc_id,
                                        name=fc_name,
                                        response={"error": f"Unknown function: {fc_name}"},
                                    )
                                )
                        
                        # Send tool responses back to Gemini
                        if function_responses:
                            logger.info("Sending %d tool response(s) to Gemini Live", len(function_responses))
                            await session.send_tool_response(function_responses=function_responses)
            
            # If we reach here, the Gemini session ended normally (timeout/disconnect)
            logger.warning("Gemini Live session ENDED after %d messages (last turn_complete at #%d)", 
                          message_count, last_turn_complete_at)
            with contextlib.suppress(Exception):
                await ws.send_json({"type": "session_ended", "detail": f"Session ended after {message_count} messages"})
        except asyncio.CancelledError:
            logger.info("Gemini Live pump task cancelled after %d messages", message_count)
            raise
        except Exception as exc:
            logger.exception("Gemini Live pump task error after %d messages: %s", message_count, exc)
            # Send error to client but don't re-raise - let the main loop handle cleanup
            with contextlib.suppress(Exception):
                await ws.send_json({"type": "error", "detail": f"Gemini Live stream error: {exc}"})
            raise

    def _friendly_live_error_detail(exc: Exception) -> str:
        raw = str(exc) if exc is not None else ""
        low = raw.lower()
        if "policy violation" in low or "publisher model" in low:
            return (
                "Gemini Live rejected the request (policy/permissions). "
                "This usually means the live model isn't available to this project/region, "
                "or the model id is misconfigured. "
                "Try setting GEMINI_LIVE_LOCATION=us-central1 and/or GEMINI_LIVE_MODEL=gemini-live-2.5-flash-exp-native-audio."
            )
        if "permission" in low or "unauth" in low or "forbidden" in low:
            return (
                "Gemini Live request was denied (permissions). "
                "Check that Vertex AI is enabled and the Cloud Run service account has access."
            )
        return raw or "Gemini Live connection failed."

    def _should_try_fallback(exc: Exception) -> bool:
        msg = (str(exc) or "").lower()
        return any(
            s in msg
            for s in (
                "publisher model",
                "policy violation",
                "not found",
                "resource not found",
                "invalid argument",
            )
        )

    def _candidate_attempts(*, primary_model: str, primary_location: str) -> list[tuple[str, str, str]]:
        """Returns (model, location, label) attempts in priority order."""
        attempts: list[tuple[str, str, str]] = [(primary_model, primary_location, "primary")]

        if auth_mode != "vertex":
            # In API-key mode, location is not used by the SDK; only try model fallbacks.
            if primary_model == "gemini-live-2.5-flash-native-audio":
                attempts.append(("gemini-live-2.5-flash-exp-native-audio", primary_location, "fallback_model"))
            return attempts

        # If someone configured global (common assumption), also try the Vertex region.
        # Vertex-backed Gemini endpoints are typically regional.
        if primary_location.lower() == "global":
            attempts.append((primary_model, default_live_location, "fallback_location"))

        # If the stable model is blocked for this project/region, try the exp variant.
        if primary_model == "gemini-live-2.5-flash-native-audio":
            attempts.append(("gemini-live-2.5-flash-exp-native-audio", primary_location, "fallback_model"))
            if primary_location.lower() == "global":
                attempts.append(("gemini-live-2.5-flash-exp-native-audio", default_live_location, "fallback_both"))

        # De-dupe while preserving order.
        seen: set[tuple[str, str]] = set()
        out: list[tuple[str, str, str]] = []
        for m, loc, label in attempts:
            key = (m, loc)
            if key in seen:
                continue
            seen.add(key)
            out.append((m, loc, label))
        return out

    from contextlib import AsyncExitStack

    try:
        async with AsyncExitStack() as stack:
            session = None
            used_model = model
            used_location = location
            used_label = "primary"

            last_exc: Exception | None = None
            for cand_model, cand_location, cand_label in _candidate_attempts(
                primary_model=model, primary_location=location
            ):
                if auth_mode == "vertex":
                    cand_client = genai.Client(
                        vertexai=True,
                        project=project,
                        location=cand_location,
                        http_options=types.HttpOptions(api_version="v1beta1"),
                    )
                else:
                    cand_client = genai.Client(
                        api_key=api_key,
                        http_options=types.HttpOptions(api_version="v1beta1"),
                    )
                try:
                    session = await stack.enter_async_context(
                        cand_client.aio.live.connect(model=cand_model, config=connect_config)
                    )
                    used_model, used_location, used_label = cand_model, cand_location, cand_label
                    break
                except Exception as exc:
                    last_exc = exc
                    logger.exception(
                        "Gemini Live connect failed (attempt=%s auth=%s model=%s location=%s)",
                        cand_label,
                        auth_mode,
                        cand_model,
                        cand_location,
                    )
                    if not _should_try_fallback(exc):
                        break

            if session is None:
                raise last_exc or RuntimeError("Gemini Live connect failed")

            await ws.send_json(
                {
                    "type": "ready",
                    "model": used_model,
                    "location": used_location,
                    "attempt": used_label,
                    "auth": auth_mode,
                }
            )

            pump_task = asyncio.create_task(_pump_from_gemini(session))
            try:
                while True:
                    # Check if pump task ended (Gemini session closed)
                    if pump_task.done():
                        logger.warning("Gemini Live pump task ended - session closed by server")
                        # The pump task already sent session_ended message
                        break

                    # Use wait_for with a timeout so we can check pump_task periodically
                    try:
                        incoming = await asyncio.wait_for(ws.receive(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue  # Loop back to check pump_task

                    if incoming.get("type") == "websocket.disconnect":
                        break

                    if incoming.get("bytes") is not None:
                        audio_bytes = incoming["bytes"]
                        if audio_bytes:
                            logger.info("Received audio from client: %d bytes", len(audio_bytes))
                            await session.send_realtime_input(
                                audio={"mime_type": "audio/pcm;rate=16000", "data": audio_bytes}
                            )
                        continue

                    text_payload = incoming.get("text")
                    if not text_payload:
                        continue
                    try:
                        evt = json.loads(text_payload)
                    except Exception:
                        continue

                    evt_type = (evt.get("type") or "").strip()
                    if evt_type == "audio_stream_end":
                        logger.info("Received audio_stream_end from client, sending to Gemini Live")
                        await session.send_realtime_input(audio_stream_end=True)
                    elif evt_type == "text":
                        t = (evt.get("text") or "").strip()
                        if t:
                            await session.send_client_content(
                                turns=types.Content(role="user", parts=[types.Part(text=t)]),
                                turn_complete=True,
                            )
                    elif evt_type == "close":
                        break
            finally:
                pump_task.cancel()
                with contextlib.suppress(Exception):
                    await pump_task
    except WebSocketDisconnect:
        return
    except Exception as exc:
        # Never let Live API errors bubble out; browsers will show a confusing 1008 close.
        logger.exception("Gemini Live session failed (auth=%s model=%s location=%s)", auth_mode, model, location)
        with contextlib.suppress(Exception):
            await ws.send_json(
                {
                    "type": "error",
                    "detail": _friendly_live_error_detail(exc),
                    "raw": str(exc)[:500],
                    "model": model,
                    "location": location,
                    "auth": auth_mode,
                }
            )
        with contextlib.suppress(Exception):
            await ws.close(code=1011)
        return


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


@app.get("/api/visits/total", response_model=VisitsTotalResponse)
def visits_total():
    store = get_visits_store()
    return VisitsTotalResponse(total_visits=store.get_total())


@app.post("/api/visits/increment", response_model=VisitsTotalResponse)
def visits_increment():
    store = get_visits_store()
    try:
        total = store.increment(1)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return VisitsTotalResponse(total_visits=total)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    return await _chat_impl(req)


async def _chat_impl(
    req: ChatRequest,
    status_cb: Callable[[str], Awaitable[None]] | None = None,
) -> ChatResponse:
    prefixes = allowed_url_prefixes()
    if settings.osrsbox_enabled:
        p = (settings.osrsbox_base_url or "").strip()
        if p and not p.endswith("/"):
            p += "/"
        if p and p not in prefixes:
            prefixes.append(p)
    history_store = get_public_chat_store()

    actions: list[str] = []
    web_query: str | None = None
    web_results: list[WebSearchResult] = []
    used_web_snippets = False
    used_youtube_sources = False
    did_focus_followup = False

    async def _status(msg: str) -> None:
        if status_cb:
            await status_cb(msg)

    async def _maybe_retry_with_broader_sources(
        *,
        prompt_user_message: str,
        conversation_context: str,
        raw_user_message: str,
        user_message: str,
        pronoun_followup: bool,
        topic_hint: str,
        prefixes: list[str],
        allow_external_sources: bool,
        prompt_chunks: list[RetrievedChunk],
        res: Any,
        opinion_question: bool,
        difficulty_question: bool,
        strategy_followup: bool,
        quest_help: bool,
        new_player_intent: bool,
        skill_training: bool,
    ) -> tuple[Any, list[RetrievedChunk], str | None, list[WebSearchResult]]:
        """Optional second pass: if the judge is unconvinced, do broader retrieval and regenerate once."""

        nonlocal web_query, web_results

        if not settings.answer_judge_enabled:
            return res, prompt_chunks, web_query, web_results

        try:
            judge_sources = [
                {"title": getattr(c, "title", None), "url": getattr(c, "url", "")}
                for c in (prompt_chunks or [])
                if getattr(c, "url", None)
            ][:6]
            j = judge_answer_confidence(
                user_message=prompt_user_message,
                answer=res.text or "",
                sources=judge_sources,
            )
            actions.append(f"Judge confidence={j.confidence:.2f}; needs_web_search={bool(j.needs_web_search)}.")

            if j.confidence >= float(settings.answer_judge_threshold):
                return res, prompt_chunks, web_query, web_results

            await _status("My librarian looks doubtful; fetching better sources...")
            actions.append("Judge flagged low confidence; forced live retrieval/web search.")

            retry_seed = (user_message if not pronoun_followup else (topic_hint or user_message)).strip()
            retry_qs = derive_search_queries(retry_seed) or [retry_seed]
            retry_primary = retry_qs[0]

            # Wiki first.
            live_chunks = await live_query_chunks(retry_primary, allowed_url_prefixes=prefixes)

            # Community (Reddit) + YouTube (guide vids), best-effort.
            reddit_chunks: list[RetrievedChunk] = []
            try:
                reddit_chunks = await reddit_search_chunks(query=retry_primary, max_results=4)
                if reddit_chunks:
                    actions.append("Consulted Reddit community threads (snippets only).")
            except Exception:
                reddit_chunks = []

            yt_chunks: list[RetrievedChunk] = []
            if settings.youtube_api_key:
                try:
                    yt_chunks = await combat_youtube_insight_chunks(user_message=user_message, max_videos=2)
                    if yt_chunks:
                        actions.append("Consulted YouTube guide videos for extra tips.")
                except Exception:
                    yt_chunks = []

            # If wiki still empty, try CSE web search.
            if not live_chunks:
                try:
                    web_chunks, web_hits = await live_search_web_and_fetch_chunks(
                        retry_primary,
                        allowed_url_prefixes=prefixes,
                        max_results=5,
                        max_chunks_total=8,
                    )
                except Exception:
                    web_chunks, web_hits = ([], [])

                if (not web_chunks) and web_hits and settings.web_scrape_enabled:
                    try:
                        scraped_chunks, _ = await live_search_web_and_scrape_chunks(
                            retry_primary,
                            max_results=5,
                            max_pages=int(settings.web_scrape_max_pages),
                            max_chunks_total=int(settings.web_scrape_max_chunks_total),
                            skip_url_prefixes=prefixes,
                        )
                    except Exception:
                        scraped_chunks = []
                    if scraped_chunks:
                        actions.append("Scraped non-wiki pages returned by CSE (untrusted web).")
                        web_chunks = scraped_chunks

                web_query = retry_primary
                if web_hits:
                    web_results = [
                        WebSearchResult(
                            title=str((r.title or "")[:120]),
                            url=str(r.url),
                            snippet=(str(r.snippet)[:240] if r.snippet else None),
                        )
                        for r in (web_hits or [])
                        if getattr(r, "url", None)
                    ][:5]

                live_chunks = web_chunks

            merged_chunks = (live_chunks or []) + (reddit_chunks or []) + (yt_chunks or [])

            # If we add reddit/youtube, allow external citations for the retry prompt.
            retry_allow_external = bool(allow_external_sources or reddit_chunks or yt_chunks)
            retry_prompt_chunks = _build_prompt_chunks(
                chunks=merged_chunks,
                retrieval_seed=retry_seed,
                user_question=(raw_user_message or user_message or ""),
                topic_hint=topic_hint or "",
                prefixes=prefixes,
                allow_external_sources=retry_allow_external,
            )

            if retry_prompt_chunks:
                retry_prompt = build_rag_prompt(
                    user_message=prompt_user_message,
                    conversation_context=conversation_context,
                    chunks=retry_prompt_chunks,
                    allowed_url_prefixes=prefixes,
                    allow_external_sources=bool(retry_allow_external),
                    allow_best_effort=bool(
                        opinion_question
                        or difficulty_question
                        or strategy_followup
                        or quest_help
                        or new_player_intent
                        or skill_training
                    ),
                )
                res = _generate_with_retry(retry_prompt)
                prompt_chunks = retry_prompt_chunks
                actions.append("Regenerated answer after judge-triggered retrieval.")

        except Exception:
            # Best-effort; never break chat.
            return res, prompt_chunks, web_query, web_results

        return res, prompt_chunks, web_query, web_results

    def _looks_truncated_answer(text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return True

        # If the model stops mid-sentence on a short reply, retry once.
        terminal = ".!?…\"')}]}"
        if len(t) <= 260 and (t[-1] not in terminal):
            if t.endswith((":", ",", ";", " -", "–", "—")):
                return True
            last = (t.split()[-1] if t.split() else "").lower().strip("\"'`.,;:!?()[]{}")
            if last in {"of", "and", "or", "to", "for", "with", "in", "on", "at", "from", "by", "as", "but"}:
                return True

        # Unbalanced parens/quotes can also indicate a cutoff.
        if len(t) <= 400:
            if t.count("(") > t.count(")"):
                return True
            if t.count("\"") % 2 == 1 and t.count("\"") <= 7:
                return True

        return False

    def _generate_with_retry(prompt: str, *, temperature: float = 0.2, max_output_tokens: int = 2048):
        client = GeminiVertexClient()
        res = client.generate(prompt, temperature=temperature, max_output_tokens=max_output_tokens)
        if _looks_truncated_answer(res.text):
            actions.append("Model output looked cut off; retried generation.")
            repair_prompt = (
                prompt
                + "\n\nIMPORTANT: Your previous answer was cut off mid-thought. Answer again fully, ending with a complete sentence."
            )
            res = client.generate(repair_prompt, temperature=temperature, max_output_tokens=max(2560, max_output_tokens))
        return res

    def _has_source_citations(text: str) -> bool:
        import re

        return bool(re.search(r"\[Source\s+\d+\]", text or ""))

    def _is_low_confidence_answer(text: str) -> bool:
        ans_l = (text or "").strip().lower()
        if not ans_l:
            return True
        # Treat both plain and "medieval" refusals as low-confidence so we re-research.
        phrases = (
            "do not know",
            "don't know",
            "cannot tell",
            "can't tell",
            "naught of",
            "cant find",
            "can't find",
            "cannot find",
            "no mention",
            "not mentioned",
            "not in the sources",
            "not in my sources",
            "sources do not",
            "my sources do not",
            "my knowledge is silent",
            "knowledge is silent",
            "silent on",
            "speak not of",
            "speak not",
            "based on the wisdom i have",
            "wisdom i have at hand",
        )
        return any(p in ans_l for p in phrases)

    def _needs_pronoun_resolution(message: str) -> bool:
        m = (message or "").strip().lower()
        if not m:
            return False
        if len(m) > 140:
            return False

        # If the message already contains several specific keywords, don't treat it as
        # a context-dependent pronoun follow-up just because it contains words like "this".
        # This avoids anchoring to the previous topic for questions such as:
        # "what about helping edmond the slave? what's hard about this quest?"
        try:
            specific_terms = [t for t in extract_keywords(m, max_terms=10) if t and len(t) >= 5]
            if len(specific_terms) >= 3:
                return False
        except Exception:
            pass

        pronouns = (
            " it ",
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
        # Only treat "this" as a follow-up signal for very short prompts.
        if " this " in padded and len(m) <= 48:
            return True

        return any(p in padded for p in pronouns)

    def _looks_like_opinion_or_community_question(message: str) -> bool:
        m = (message or "").strip().lower()
        if not m:
            return False
        import re

        # Common "consensus" phrasing (including common typos).
        if re.search(r"\b(save|saved|saving|leave|leaving|left)\b.*\bla(?:st|t)\b", m):
            return True
        if re.search(r"\b(quest|quests)\b.*\b(la(?:st|t)|final)\b", m):
            return True
        if ("quest cape" in m or "quest point cape" in m) and any(w in m for w in ("people", "most", "usually", "common", "popular")):
            return True

        # Community / subjective questions usually benefit from wider-web sources.
        needles = (
            "hardest",
            "most difficult",
            "easiest",
            "best",
            "worst",
            "most annoying",
            "most fun",
            "favorite",
            "favourite",
            "overrated",
            "underrated",
            "tier list",
            "community",
            "people say",
            "reddit",
            "what does reddit",
            "what does the community",
            "what do people think",
            "what do most people",
            "what do people usually",
            "what do players usually",
            "most people",
            "most common",
            "commonly",
            "popular",
            "generally",
            "usually",
            "consensus",
            "what quest do people",
            "last quest",
            "final quest",
            "leave for last",
            "leave until last",
            "opinions",
            "opinion",
        )
        return any(n in m for n in needles)

    def _extract_json_object(text: str) -> dict | None:
        raw = (text or "").strip()
        if not raw:
            return None
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end < 0 or end <= start:
            return None
        try:
            obj = json.loads(raw[start : end + 1])
        except Exception:
            return None
        return obj if isinstance(obj, dict) else None

    def _llm_intent_is_community(message: str) -> bool | None:
        """Return True/False if the LLM can confidently classify; else None.

        Only runs when Vertex is configured.
        """

        if not settings.google_cloud_project:
            return None

        msg = (message or "").strip()
        if not msg:
            return None

        # Keep cost/latency bounded.
        msg = msg[:800]

        prompt = (
            "You are classifying an Old School RuneScape question.\n"
            "Return ONLY JSON with keys intent and confidence.\n"
            "intent MUST be one of: community, gameplay\n"
            "confidence MUST be a number from 0 to 1.\n\n"
            "Definitions:\n"
            "- community: asking what players think/do/say; popularity; what most people save for last; tier lists; reddit-style consensus.\n"
            "- gameplay: asking factual game info, mechanics, requirements, walkthroughs, how-to steps, or setup.\n\n"
            f"Question: {msg}\n"
        )

        try:
            res = GeminiVertexClient().generate(prompt).text
            obj = _extract_json_object(res)
            if not obj:
                return None
            intent = str(obj.get("intent") or "").strip().lower()
            try:
                conf_raw = obj.get("confidence")
                if conf_raw is None:
                    conf = 0.0
                else:
                    conf = float(str(conf_raw).strip())
            except Exception:
                conf = 0.0
            if conf < 0.70:
                return None
            if intent == "community":
                return True
            if intent == "gameplay":
                return False
            return None
        except Exception:
            return None

    def _topic_hint_from_text(text: str) -> str:
        t = (text or "").strip()
        if not t:
            return ""

        # Avoid latching onto sentence-starter capitalization (e.g., "There") or pronouns.
        # These tend to appear in follow-ups like "Look again. There must be..." and break retrieval.
        false_topics = {
            "there",
            "this",
            "that",
            "it",
            "here",
            "please",
            "thanks",
            "thank you",
            "ok",
            "okay",
            "yes",
            "no",
            "him",
            "her",
            "them",
            "they",
            "he",
            "she",
            "we",
            "i",
            "you",
            "me",
            "my",
            "your",
        }

        # Ignore common meta/instruction messages that are not actual topics.
        t_l = t.lower().strip()
        if t_l in {
            "more",
            "more info",
            "more information",
            "more details",
            "more answers",
            "another answer",
            "another",
            "continue",
            "keep going",
            "expand",
            "elaborate",
            "say more",
            "try again",
            "search again",
            "query for more answers",
        }:
            return ""

        # If the message contains a known quest title (even in lowercase), prefer it.
        qtitle = find_quest_title_in_text(t)
        if qtitle:
            return qtitle

        # Prefer explicit "about X" phrasing.
        import re

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

        m = re.search(r"\babout\s+(.+)$", t, flags=re.IGNORECASE)
        if m:
            topic = m.group(1).strip().strip("?!.\"")
            if 1 <= len(topic) <= 60:
                if topic.strip().lower() not in false_topics:
                    return topic

        # Lowercase-friendly topic: users sometimes just type an entity name (e.g. "quest cape").
        # If it looks like a short noun phrase (not a question), use it as the topic.
        if len(t) <= 60 and not re.match(r"^(how|what|when|where|who|why)\b", t, flags=re.IGNORECASE):
            if re.fullmatch(r"[a-z0-9][a-z0-9'\- ]{2,60}", t, flags=re.IGNORECASE):
                words = [w for w in re.split(r"\s+", t) if w]
                if 1 <= len(words) <= 5:
                    titled = _smart_title_case(" ".join(words))
                    if titled.strip().lower() not in false_topics:
                        return titled

        # Fall back: capture sequences of Title-Cased words, allowing Roman numerals and hyphenated subtitles.
        # Examples: "The Whisperer", "Desert Treasure II", "Desert Treasure II - The Fallen Empire".
        title_token = r"(?:\b[A-Z][a-z0-9']+\b|\b[IVX]{1,6}\b|\b\d+\b)"
        pattern = rf"{title_token}(?:\s+{title_token}){{0,7}}(?:\s*-\s*{title_token}(?:\s+{title_token}){{0,7}})?"

        caps = re.findall(pattern, t)
        if caps:
            # Prefer the latest non-noise capture.
            for raw in reversed(caps):
                cand = str(raw or "").strip().strip("?!.\"")
                cand = re.sub(r"\s+", " ", cand)
                if not cand:
                    continue
                if cand in {"I", "Me", "You"}:
                    continue
                # Filter obvious false topics like "There".
                if cand.strip().lower() in false_topics:
                    continue
                # Single-word captures are common (boss names), but avoid sentence-starters/noise.
                if 1 <= len(cand) <= 80:
                    return cand

        # Lowercase-friendly quest fallback: capture entity after quest verbs.
        # e.g. "how do I start dragon slayer?" -> "Dragon Slayer" (which should redirect).
        m_q = re.search(
            r"\b(?:start|begin|complete|finish)\s+(?:the\s+)?([a-z0-9][a-z0-9'\- ]{2,80})",
            t,
            flags=re.IGNORECASE,
        )
        if m_q:
            tail = (m_q.group(1) or "").strip()
            tail = re.sub(r"\s+", " ", tail).strip("?!.\"'")
            words = tail.split()
            tail = " ".join(words[:6])
            if tail:
                titled = _smart_title_case(tail)
                if 1 <= len(titled) <= 80:
                    return titled

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
                    if titled.strip().lower() not in false_topics:
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
                "survive",
                "survival",
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

    def _best_effort_fragment_of_seren_plan() -> str:
        # Best-effort, commonly used approach; not guaranteed and may vary by stats/gear.
        return (
            "(common player experience; not sourced) For the Fragment of Seren fight, most players win by "
            "building a lot of healing and timing big heals around her burst-damage specials.\n\n"
            "Practical plan:\n"
            "1) Go in with strong healing: Saradomin brews + super restores, plus high-heal food.\n"
            "2) Bring emergency heals: Phoenix necklaces are commonly used to survive her big hits; "
            "pair them with quick healing items so you don’t get combo’d out.\n"
            "3) Use reliable DPS you can sustain: many players use ranged or magic with consistent accuracy, "
            "and prioritize staying alive over max damage.\n"
            "4) Keep your HP high before any ‘big hit’ moments: if you get clipped while low, the fight snowballs fast.\n"
            "5) Consider Redemption timing (if you’re using prayer) but don’t rely on prayer alone as your only defense.\n\n"
            "If you tell me your combat stats, whether you’re using ranged or magic, and what supplies you have access to "
            "(brews? phoenix necklaces? blood spells?), I’ll tailor an exact inventory + step-by-step timing plan."
        )

    def _looks_like_quest_help_question(message: str) -> bool:
        m = (message or "").lower()
        return any(
            w in m
            for w in (
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

    def _looks_like_new_player_question(message: str) -> bool:
        m = (message or "").lower()
        return any(
            p in m
            for p in (
                "first character",
                "new character",
                "new player",
                "brand new",
                "just started",
                "starting out",
                "beginner",
                "tutorial island",
                "account type",
                "account types",
                "ironman",
                "hardcore ironman",
                "ultimate ironman",
                "group ironman",
            )
        )

    def _looks_like_skill_training_question(message: str) -> bool:
        m = (message or "").lower()
        if not m:
            return False

        if any(p in m for p in ("level", "lvl", "xp", "experience", "training", "train ", "grind")):
            return True

        skills = (
            "attack",
            "strength",
            "defence",
            "hitpoints",
            "ranged",
            "magic",
            "prayer",
            "runecraft",
            "runecrafting",
            "construction",
            "agility",
            "herblore",
            "thieving",
            "crafting",
            "fletching",
            "slayer",
            "hunter",
            "mining",
            "smithing",
            "fishing",
            "cooking",
            "firemaking",
            "woodcutting",
            "farming",
        )
        return any(s in m for s in skills) and any(p in m for p in ("train", "training", "level", "xp", "best way", "fastest", "afk"))

    def _apply_intent_hints(*, base_question: str, new_player_intent: bool, quest_help: bool, strategy_followup: bool, skill_training: bool) -> str:
        q = (base_question or "").strip()
        if not q:
            return q

        hints: list[str] = []

        if new_player_intent:
            hints.append(
                "New player/setup intent: list the main options a brand-new OSRS account can choose (and what each implies). "
                "Give 4-7 concrete options with short pros/cons and a suggested default. "
                "Examples of option categories: account mode (regular vs Ironman variants), onboarding path (Tutorial Island/Adventure Paths), "
                "membership vs F2P, early goals/questing direction."
            )

        if quest_help:
            hints.append(
                "Quest-help intent: provide requirements (stats/items/quests), key steps/checkpoints, and common pitfalls. "
                "If there's a /Quick guide or /Walkthrough, prefer it."
            )

        if strategy_followup:
            hints.append(
                "Combat/strategy intent: answer as a practical plan (setup → inventory/gear → mechanics/phases → survival tips). "
                "If the user didn't provide stats/gear, ask 1-2 quick questions at the end to tailor the plan."
            )

        if skill_training:
            hints.append(
                "Skill-training intent: give multiple training options (fast vs cheap vs AFK), include level brackets when possible, "
                "and call out key unlocks/quests that speed training. End with a simple recommended path based on the user's likely constraints."
            )

        if not hints:
            return q

        # Keep this short and directive; the UI will render markdown-like bullets.
        return q + "\n\nAnswering hints:\n- " + "\n- ".join(hints)

    def _is_followup_more_request(message: str) -> bool:
        """Return True when the user is asking for more detail, not a new topic."""
        import re

        m = (message or "").strip().lower()
        if not m:
            return False
        if len(m) > 80:
            return False

        # Normalize punctuation and spacing.
        m = re.sub(r"[\t\r\n]+", " ", m)
        m = re.sub(r"\s+", " ", m).strip()

        # Must contain some follow-up indicator.
        if not any(w in m for w in ("more", "another", "expand", "elaborate", "continue", "again")):
            return False

        # If the message contains a real topic keyword (beyond generic follow-up words), treat it as a normal query.
        noise = {
            "more",
            "another",
            "answers",
            "answer",
            "detail",
            "details",
            "info",
            "information",
            "explain",
            "explanation",
            "expand",
            "elaborate",
            "continue",
            "again",
            "please",
            "pls",
            "query",
            "search",
            "retry",
            "for",
            "me",
            "give",
            "some",
            "a",
            "an",
            "the",
            "to",
        }
        keys = extract_keywords(m, max_terms=8)
        meaningful = [k for k in keys if (k or "").strip().lower() not in noise]
        if meaningful:
            return False

        return True

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
    raw_user_message = (req.message or "").strip()

    # Interpret short meta prompts like "more answers" as a follow-up on the prior user question.
    followup_more = bool(prev_user_message and _is_followup_more_request(raw_user_message))

    # Use an effective question for retrieval + prompting.
    user_message = raw_user_message
    prompt_user_message = raw_user_message
    if followup_more:
        actions.append("Interpreted request as a follow-up; expanding the previous question.")
        user_message = (prev_user_message or "").strip()
        prompt_user_message = (
            f"{user_message}\n\nPlease provide additional detail, alternatives, and practical tips beyond the prior answer."
        )

    prev_topic_hint = _topic_hint_from_text(prev_user_message)
    cur_topic_hint = _topic_hint_from_text(user_message)

    # If the immediate prior turn is a pronoun/meta follow-up, keep the last explicit topic from
    # further back in the session so we don't lose context across multiple follow-ups.
    history_topic_hint = ""
    if prior_turns:
        for t in prior_turns:
            um = (t.user_message or "").strip()
            if not um:
                continue
            hint = _topic_hint_from_text(um)
            if hint:
                history_topic_hint = hint
                break

    pronoun_followup = bool(prev_user_message and _needs_pronoun_resolution(raw_user_message))

    # Prefer current explicit topic when present; otherwise fall back to prior turn, then session history.
    topic_hint = (cur_topic_hint or prev_topic_hint or history_topic_hint).strip()

    # If the user explicitly mentions Seren/Fragment, prefer that as the topic so retrieval
    # can pull the correct boss/strategies pages.
    _um_l = (user_message or "").lower()
    if topic_hint and ("song of the elves" in topic_hint.lower()):
        if "seren" in _um_l or "fragment" in _um_l:
            topic_hint = "Fragment of Seren"
    elif (not topic_hint) and ("seren" in _um_l or "fragment of seren" in _um_l):
        topic_hint = "Fragment of Seren"

    strategy_followup = bool(topic_hint and _looks_like_strategy_question(user_message))
    quest_help = bool(topic_hint and _looks_like_quest_help_question(user_message))
    new_player_intent = _looks_like_new_player_question(user_message)
    skill_training = _looks_like_skill_training_question(user_message)

    # Difficulty/"hardest" questions are often hybrid: part community consensus, part practical help.
    # Treat them as opinion/community even if an intent classifier leans "gameplay".
    msg_l = (user_message or "").lower()
    difficulty_question = any(
        w in msg_l
        for w in (
            "hardest",
            "most difficult",
            "most challenging",
            "difficulty",
            "hard part",
            "hardest part",
            "toughest",
            "toughest boss",
            "hardest boss",
            "toughest fight",
            "hardest fight",
        )
    )

    # If the user asks for "latest"/"current" info, skip the answer cache.
    force_fresh = any(w in msg_l for w in ("latest", "today", "current", "new", "update"))
    if followup_more:
        # Avoid serving the same cached answer; user explicitly asked for more.
        force_fresh = True

    if new_player_intent:
        force_fresh = True
        actions.append("New-player/setup question detected; favored onboarding sources.")
    if skill_training:
        force_fresh = True
        actions.append("Skill training/leveling question detected; favored training-method sources.")

    if not followup_more:
        prompt_user_message = _apply_intent_hints(
            base_question=prompt_user_message,
            new_player_intent=new_player_intent,
            quest_help=quest_help,
            strategy_followup=strategy_followup,
            skill_training=skill_training,
        )

    opinion_question = _looks_like_opinion_or_community_question(user_message)
    # LLM intent classification (best-effort): helps distinguish consensus questions from factual wiki questions.
    llm_intent = _llm_intent_is_community(user_message)
    if llm_intent is True:
        opinion_question = True
        actions.append("LLM judged this as a community/opinion question.")
    elif llm_intent is False and opinion_question and not difficulty_question:
        # If heuristic fired but LLM is confident it's factual/gameplay, prefer gameplay behavior.
        opinion_question = False
        actions.append("LLM judged this as a gameplay/factual question.")
    elif llm_intent is False and opinion_question and difficulty_question:
        actions.append("Detected a difficulty/\"hardest part\" question; treating as hybrid community + how-to.")
    if opinion_question:
        # Cached answers tend to overfit generic overlaps (e.g., "quest") on opinion prompts.
        force_fresh = True
        actions.append("Opinion/community question detected; favored fresh sources.")

    # Context-dependent follow-ups ("her", "that", "it") should not be served from cache.
    if pronoun_followup:
        force_fresh = True
        actions.append("Follow-up detected; refreshed my search.")

    # For quest walkthrough/requirements questions, prefer live retrieval so we pull the actual quest guide pages.
    if quest_help:
        force_fresh = True
        actions.append("Quest help detected; favored fresh wiki lookups.")

    # For combat/strategy questions, we still prefer fresh retrieval, but we also allow best-effort
    # guidance even if we can't fetch a perfect strategy page.
    if strategy_followup:
        force_fresh = True
        actions.append("Combat/strategy question detected; favored fresh lookups.")

    # Fast path: reuse a previously-generated, cited answer for similar questions.
    await _status("Leafing through my old scrolls for a matching answer...")
    cached = None if force_fresh else AnswerCacheStore().find_similar(
        question=user_message, allowed_url_prefixes=prefixes
    )
    if cached:
        # If the cached answer is explicitly low-confidence, force a fresh retrieval.
        if _is_low_confidence_answer(cached.answer or "") or not _has_source_citations(cached.answer or ""):
            cached = None
    if cached:
        actions.append("Returned a previously-prepared scroll (with citations).")
        videos = []
        if quest_help:
            try:
                await _status("Peering into the crystal screen for quest videos...")
                raw_videos = await quest_youtube_videos_with_summaries(user_message=user_message)
                videos = [VideoItem.model_validate(v) for v in (raw_videos or [])]
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
            user_message=raw_user_message,
            bot_answer=cached.answer,
            sources=[s.model_dump() for s in sources],
            videos=[v.model_dump() for v in videos],
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
    retrieval_seed = user_message
    if pronoun_followup and prev_topic_hint:
        retrieval_seed = f"{user_message} {prev_topic_hint}".strip()
    elif pronoun_followup:
        retrieval_seed = f"{user_message} {prev_user_message}".strip()

    # If we can infer a specific boss/topic from the prior turn and the user is
    # asking for tactics, search the boss's /Strategies page first.
    queries = derive_search_queries(retrieval_seed)

    def _primary_live_query(*, raw: str, topic_hint: str, queries: list[str], fallback: str) -> str:
        # For live retrieval (MediaWiki + CSE), we want a query that is specific enough to find the
        # right *subpage* (e.g., boss/strategies) rather than always searching the bare quest title.
        th = (topic_hint or "").strip()
        raw_q = (raw or "").strip()

        # If we've explicitly boosted derived queries (strategies, walkthrough, etc.), prefer those.
        if (strategy_followup or quest_help or opinion_question or difficulty_question) and (queries or []):
            for q in (queries or []):
                qq = (q or "").strip()
                if not qq:
                    continue
                if th and qq.lower() == th.lower():
                    continue
                # Prefer subpages and explicit modifiers.
                if "/" in qq or any(w in qq.lower() for w in ("strateg", "walkthrough", "quick guide", "boss", "fight", "hardest", "toughest", "challeng")):
                    return qq

        # If the user asked a richer question than just the topic title, prefer their raw phrasing.
        if raw_q and th and raw_q.lower() != th.lower() and len(raw_q) >= (len(th) + 6):
            return raw_q

        if th:
            return th
        if raw_q and len(raw_q) >= 8:
            return raw_q
        # Otherwise prefer the longest derived query (usually the least lossy).
        best = ""
        for q in (queries or []):
            qq = (q or "").strip()
            if len(qq) > len(best):
                best = qq
        return best or (fallback or "").strip()

    if difficulty_question and topic_hint:
        # Boost likely relevant subpages for "hardest" questions so we pull tactics pages instead of
        # just the quest overview.
        boosted = [
            f"{topic_hint} boss",
            f"{topic_hint} final boss",
            f"{topic_hint} fight",
            f"{topic_hint} hardest part",
            f"{topic_hint} tips",
        ]
        # A few high-impact quest -> boss mappings (kept small on purpose).
        boss_map = {
            "Song of the Elves": ["Fragment of Seren", "Fragment of Seren/Strategies"],
        }
        for extra in boss_map.get(topic_hint, []):
            boosted.insert(0, extra)
        for q in reversed(boosted):
            if q and q.lower() not in {x.lower() for x in queries}:
                queries.insert(0, q)
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
    for q in queries or [user_message]:
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

    # Ignore generic words that otherwise make unrelated quest pages look relevant.
    generic_weak_terms = {
        "osrs",
        "runescape",
        "quest",
        "quests",
        "guide",
        "walkthrough",
        "help",
        "helping",
        "hard",
        "hardest",
        "difficult",
        "difficulty",
        "complete",
        "completing",
    }
    filtered_key_terms = [t for t in key_terms if t and t.lower() not in generic_weak_terms]
    filtered_topic_terms = [t for t in topic_terms if t and t.lower() not in generic_weak_terms]
    if filtered_key_terms:
        key_terms = filtered_key_terms
    if filtered_topic_terms:
        topic_terms = filtered_topic_terms
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

        # If this looks like a short entity query, prefer pages whose *titles* match the topic.
        # This avoids false positives where a strategy page matches "cape" via gear lists.
        if topic_terms and topic_hint and len(topic_hint.split()) <= 4:
            title_hay = "\n".join(((c.title or "") for c in chunks)).lower()
            hits_title = sum(1 for t in topic_terms[:2] if t and t in title_hay)
            weak_local = weak_local or (hits_title == 0)

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
        primary_q = _primary_live_query(
            raw=raw_user_message,
            topic_hint=topic_hint,
            queries=queries,
            fallback=retrieval_seed,
        )
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

            # Even if we couldn't fetch chunks, still show a sample of what the web search returned.
            web_query = primary_q
            if web_hits:
                web_results = [
                    WebSearchResult(
                        title=str((r.title or "")[:120]),
                        url=str(r.url),
                        snippet=(str(r.snippet)[:240] if r.snippet else None),
                    )
                    for r in (web_hits or [])
                    if getattr(r, "url", None)
                ][:5]
                actions.append(f"Google CSE returned {len(web_results)} lead(s).")

            if web_chunks:
                actions.append("Used a web search fallback to find the right wiki page.")
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

            elif web_hits:
                actions.append("Web search found leads, but I couldn't fetch allowed wiki pages from them.")

                # If we can't fetch full wiki pages (e.g., results are Reddit/community links),
                # fall back to using the search result snippets as limited citations.
                # This allows the assistant to answer with sources even when the destination
                # site blocks scraping or isn't a MediaWiki page.
                if not chunks:
                    try:
                        import re

                        terms = extract_keywords(primary_q, max_terms=10)

                        def _snip_score(r) -> int:
                            hay = "\n".join([str(getattr(r, "title", "") or ""), str(getattr(r, "snippet", "") or ""), str(getattr(r, "url", "") or "")]).lower()
                            return sum(1 for t in (terms or []) if t and t.lower() in hay)

                        ranked_hits = sorted([r for r in (web_hits or []) if getattr(r, "url", None)], key=_snip_score, reverse=True)
                        snippet_chunks: list[RetrievedChunk] = []
                        for r in ranked_hits[:5]:
                            u = str(r.url)
                            snip = (str(getattr(r, "snippet", "") or "") or "").strip()
                            snip = re.sub(r"\s+", " ", snip)
                            if not u or len(snip) < 40:
                                continue
                            t = str(getattr(r, "title", "") or "(search result)")
                            snippet_chunks.append(
                                RetrievedChunk(
                                    text=f"Search snippet: {snip}",
                                    url=u,
                                    title=f"[Web Snippet] {t[:180]}",
                                )
                            )
                            if len(snippet_chunks) >= 4:
                                break

                        if snippet_chunks:
                            actions.append("Used search-result snippets as limited citations (couldn't fetch full pages).")
                            chunks = snippet_chunks
                            used_web_snippets = True
                    except Exception:
                        pass

                if settings.web_scrape_enabled:
                    await _status("Those leads are outside the wiki; carefully scraping for clues...")
                    try:
                        scraped_chunks, _ = await live_search_web_and_scrape_chunks(
                            primary_q,
                            max_results=5,
                            max_pages=int(settings.web_scrape_max_pages),
                            max_chunks_total=int(settings.web_scrape_max_chunks_total),
                            skip_url_prefixes=prefixes,
                        )
                    except Exception:
                        scraped_chunks = []

                    if scraped_chunks:
                        actions.append("Scraped non-wiki pages returned by CSE (untrusted web).")
                        chunks = scraped_chunks
            else:
                actions.append("Web search returned no usable leads (check CSE keys and site restriction).")

    # Even if we have wiki chunks, opinion/community questions benefit from wider web sources (Reddit, forums).
    if opinion_question:
        if settings.web_scrape_enabled:
            await _status("Listening for tavern gossip (community sources)...")

            if not settings.google_cse_api_key or not (settings.google_cse_community_cx or settings.google_cse_cx):
                actions.append(
                    "Google CSE is not configured; can't pull Reddit/community leads (set GOOGLE_CSE_API_KEY and a CX)."
                )

            def _build_community_queries(seed: str) -> list[str]:
                base = (seed or "").strip()
                if not base:
                    return []
                # Bias toward community discussion where consensus/"most people" answers live.
                candidates = [
                    # Reddit-first: force domain/r first so we don't waste budget on generic wiki pages.
                    f"{base} site:reddit.com/r/2007scape",
                    f"osrs {base} site:reddit.com/r/2007scape",
                    f"{base} site:reddit.com osrs",
                    f"osrs {base} reddit",
                    # Secondary community sources
                    f"osrs {base} forum",
                    f"osrs {base} discord",
                ]
                # De-dupe while preserving order.
                seen_q: set[str] = set()
                out: list[str] = []
                for q in candidates:
                    qn = q.strip().lower()
                    if not qn or qn in seen_q:
                        continue
                    seen_q.add(qn)
                    out.append(q)
                return out

            try:
                scraped_chunks: list = []
                web_hits: list = []
                used_query = ""

                seed = (raw_user_message or user_message or retrieval_seed)
                tried_queries: list[str] = []
                for q in _build_community_queries(seed)[:5]:
                    tried_queries.append(q)
                    sc, hits = await live_search_web_and_scrape_chunks(
                        q,
                        max_results=8,
                        max_pages=int(settings.web_scrape_max_pages),
                        max_chunks_total=int(settings.web_scrape_max_chunks_total),
                        skip_url_prefixes=prefixes,
                    )
                    # Keep the first query's hits for UI visibility; keep best chunks found.
                    if not used_query:
                        used_query = q
                        web_hits = hits
                    if sc:
                        scraped_chunks = sc
                        used_query = q
                        web_hits = hits
                        break
            except Exception:
                scraped_chunks, web_hits = ([], [])
                used_query = ""

            if not web_hits:
                # If the user's CX is site-restricted to wiki-only, Reddit will never appear.
                cx_used = settings.google_cse_community_cx or settings.google_cse_cx
                if cx_used:
                    actions.append(
                        "Google CSE returned 0 community leads (if your CX is wiki-restricted, add Reddit sites or set GOOGLE_CSE_COMMUNITY_CX)."
                    )

            if web_hits:
                web_query = used_query or _primary_live_query(
                    raw=raw_user_message,
                    topic_hint=topic_hint,
                    queries=queries,
                    fallback=retrieval_seed,
                )
                web_results = [
                    WebSearchResult(
                        title=str((r.title or "")[:120]),
                        url=str(r.url),
                        snippet=(str(r.snippet)[:240] if r.snippet else None),
                    )
                    for r in (web_hits or [])
                    if getattr(r, "url", None)
                ][:5]
                actions.append(f"Google CSE returned {len(web_results)} lead(s).")

                # Even if scraping fails (common for Reddit), use the search snippets as lightweight
                # evidence so the model can actually consider community consensus.
                try:
                    import re

                    terms = extract_keywords(web_query or (raw_user_message or user_message or retrieval_seed), max_terms=10)

                    def _snip_score(r) -> int:
                        hay = "\n".join([
                            str(getattr(r, "title", "") or ""),
                            str(getattr(r, "snippet", "") or ""),
                            str(getattr(r, "url", "") or ""),
                        ]).lower()
                        return sum(1 for t in (terms or []) if t and t.lower() in hay)

                    ranked_hits = sorted(
                        [r for r in (web_hits or []) if getattr(r, "url", None)],
                        key=_snip_score,
                        reverse=True,
                    )
                    snippet_chunks: list[RetrievedChunk] = []
                    for r in ranked_hits[:6]:
                        u = str(r.url)
                        snip = (str(getattr(r, "snippet", "") or "") or "").strip()
                        snip = re.sub(r"\s+", " ", snip)
                        if not u or len(snip) < 40:
                            continue
                        t = str(getattr(r, "title", "") or "(search result)")
                        snippet_chunks.append(
                            RetrievedChunk(
                                text=f"Search snippet: {snip}",
                                url=u,
                                title=f"[Community Snippet] {t[:180]}",
                            )
                        )
                        if len(snippet_chunks) >= 3:
                            break

                    if snippet_chunks:
                        used_web_snippets = True
                        actions.append("Included community search snippets in sources.")
                        chunks = (snippet_chunks + (chunks or []))
                except Exception:
                    pass

            if scraped_chunks:
                actions.append("Added community sources from the wider web (untrusted web).")
                # Prefer community sources over generic wiki pages for opinion prompts.
                chunks = scraped_chunks + (chunks or [])
            elif web_hits and not chunks:
                # As a last resort (e.g., Reddit blocks scraping), use snippets so we can still cite community leads.
                try:
                    import re

                    terms = extract_keywords(used_query or (raw_user_message or user_message or retrieval_seed), max_terms=10)

                    def _snip_score(r) -> int:
                        hay = "\n".join([str(getattr(r, "title", "") or ""), str(getattr(r, "snippet", "") or ""), str(getattr(r, "url", "") or "")]).lower()
                        return sum(1 for t in (terms or []) if t and t.lower() in hay)

                    ranked_hits = sorted([r for r in (web_hits or []) if getattr(r, "url", None)], key=_snip_score, reverse=True)
                    snippet_chunks: list[RetrievedChunk] = []
                    for r in ranked_hits[:6]:
                        u = str(r.url)
                        snip = (str(getattr(r, "snippet", "") or "") or "").strip()
                        snip = re.sub(r"\s+", " ", snip)
                        if not u or len(snip) < 40:
                            continue
                        t = str(getattr(r, "title", "") or "(search result)")
                        snippet_chunks.append(
                            RetrievedChunk(
                                text=f"Search snippet: {snip}",
                                url=u,
                                title=f"[Web Snippet] {t[:180]}",
                            )
                        )
                        if len(snippet_chunks) >= 4:
                            break

                    if snippet_chunks:
                        actions.append("Used community search snippets as citations (couldn't scrape full pages).")
                        chunks = snippet_chunks
                        used_web_snippets = True
                except Exception:
                    pass
        else:
            actions.append("Tip: enable WEB_SCRAPE_ENABLED=true to cite community sources like Reddit.")

    def _build_prompt_chunks(
        *,
        chunks: list,
        retrieval_seed: str,
        user_question: str,
        topic_hint: str,
        prefixes: list[str],
        allow_external_sources: bool,
    ) -> list:
        # De-dupe by URL before prompting so citation numbers match what we display.
        # BUT: include multiple relevant excerpts from the same page so we don't miss terms
        # that appear later in the article.
        key_terms_for_prompt: list[str] = []
        # Always include user-question terms so we don't overfit on a generic wiki page title
        # (e.g., "Quest point cape") for community/opinion prompts.
        if user_question:
            key_terms_for_prompt.extend(extract_keywords(user_question, max_terms=14))
        if topic_hint:
            key_terms_for_prompt.extend(extract_keywords(topic_hint, max_terms=6))
        key_terms_for_prompt.extend(extract_keywords(retrieval_seed, max_terms=10))

        # Avoid generic terms dominating ranking (they cause irrelevant generic pages like
        # "Quest point cape" to win when the user asked about a specific NPC/quest).
        generic_terms = {
            "osrs",
            "runescape",
            "quest",
            "quests",
            "guide",
            "walkthrough",
            "help",
            "helping",
            "hard",
            "hardest",
            "difficult",
            "difficulty",
        }
        key_terms_for_prompt = [t for t in key_terms_for_prompt if t and t.lower() not in generic_terms]

        seen_k: set[str] = set()
        key_terms_for_prompt = [t for t in key_terms_for_prompt if t and not (t in seen_k or seen_k.add(t))]

        prefixes_t = tuple(p for p in (prefixes or []) if p)
        url_to_chunks: dict[str, list] = {}
        for c in chunks:
            u = (getattr(c, "url", "") or "").strip()
            if not u:
                continue
            # Normally, keep citations on the configured wiki domains.
            # If external sources are allowed, allow external URLs too.
            if prefixes_t and not u.startswith(prefixes_t) and not bool(allow_external_sources):
                continue
            url_to_chunks.setdefault(u, []).append(c)

        def _text_score(text: str) -> int:
            hay = (text or "").lower()
            return sum(1 for t in key_terms_for_prompt if t and t.lower() in hay)

        def _title_score(title: str | None) -> int:
            hay = (title or "").lower()
            return sum(1 for t in key_terms_for_prompt[:6] if t and t.lower() in hay)

        ranked: list[tuple[str, int, int]] = []
        for u, cs in url_to_chunks.items():
            best_text = max((_text_score((getattr(c, "text", "") or "")) for c in (cs or [])), default=0)
            # Prefer titles that actually look like the topic.
            best_title = max((_title_score(getattr(c, "title", None)) for c in (cs or [])), default=0)
            ranked.append((u, int(best_title), int(best_text)))

        # Rank by title match first, then content match.
        ranked.sort(key=lambda p: (p[1], p[2]), reverse=True)

        title_nonzero = [u for (u, ts, _cs) in ranked if ts > 0]
        if title_nonzero:
            ranked_urls = title_nonzero
        else:
            text_nonzero = [u for (u, _ts, cs) in ranked if cs > 0]
            ranked_urls = text_nonzero if text_nonzero else [u for (u, _ts, _cs) in ranked]

        prompt_chunks: list = []
        ranked_urls = ranked_urls[:6]
        for u in ranked_urls:
            cs = url_to_chunks.get(u) or []
            if not cs:
                continue

            # If we have any meaningful key terms, require at least one to match either
            # title or body before allowing this URL into the prompt.
            if key_terms_for_prompt:
                best_text = max((_text_score((getattr(c, "text", "") or "")) for c in (cs or [])), default=0)
                best_title = max((_title_score(getattr(c, "title", None)) for c in (cs or [])), default=0)
                if (best_text + best_title) == 0:
                    continue

            ranked_chunks = sorted(cs, key=lambda c: _text_score((getattr(c, "text", "") or "")), reverse=True)
            picked = ranked_chunks[:3]
            merged_text = "\n\n".join(
                ((getattr(c, "text", "") or "").strip()) for c in picked if (getattr(c, "text", "") or "").strip()
            ).strip()
            if len(merged_text) > 2200:
                merged_text = merged_text[:2200] + "..."

            prompt_chunks.append(
                type(cs[0])(
                    text=merged_text or (getattr(cs[0], "text", "") or ""),
                    url=u,
                    title=getattr(cs[0], "title", None),
                )
            )

        # If external sources are allowed and we have any, ensure at least one makes it
        # into the prompt. This prevents a single generic wiki page from crowding out
        # community sources for questions like "which quests do people save for last".
        if allow_external_sources and prefixes_t:
            has_external_anywhere = any(
                (getattr(c, "url", "") or "").strip() and not (getattr(c, "url", "") or "").strip().startswith(prefixes_t)
                for c in (chunks or [])
            )
            has_external_in_prompt = any(
                (getattr(c, "url", "") or "").strip() and not (getattr(c, "url", "") or "").strip().startswith(prefixes_t)
                for c in (prompt_chunks or [])
            )
            if has_external_anywhere and (not has_external_in_prompt):
                ext_urls = [u for u in ranked_urls if u and not u.startswith(prefixes_t)]
                if not ext_urls:
                    ext_urls = [u for u in url_to_chunks.keys() if u and not u.startswith(prefixes_t)]
                if ext_urls:
                    u = ext_urls[0]
                    cs = url_to_chunks.get(u) or []
                    if cs:
                        ranked_chunks = sorted(cs, key=lambda c: _text_score((getattr(c, "text", "") or "")), reverse=True)
                        picked = ranked_chunks[:3]
                        merged_text = "\n\n".join(
                            ((getattr(c, "text", "") or "").strip())
                            for c in picked
                            if (getattr(c, "text", "") or "").strip()
                        ).strip()
                        if len(merged_text) > 2200:
                            merged_text = merged_text[:2200] + "..."
                        ext_chunk = type(cs[0])(
                            text=merged_text or (getattr(cs[0], "text", "") or ""),
                            url=u,
                            title=getattr(cs[0], "title", None),
                        )
                        if prompt_chunks:
                            prompt_chunks[-1] = ext_chunk
                        else:
                            prompt_chunks.append(ext_chunk)

            # If we have non-YouTube external sources (e.g., Reddit/forums), ensure at least
            # one of those makes it into the prompt too (YouTube alone shouldn't crowd them out).
            try:
                from urllib.parse import urlparse

                def _is_external(u: str) -> bool:
                    uu = (u or "").strip()
                    return bool(uu) and (not uu.startswith(prefixes_t))

                def _is_youtube(u: str) -> bool:
                    try:
                        h = (urlparse(u).hostname or "").lower()
                    except Exception:
                        h = ""
                    return bool(h) and (h.endswith("youtube.com") or h == "youtu.be")

                has_non_yt_external_anywhere = any(
                    _is_external((getattr(c, "url", "") or "").strip()) and (not _is_youtube((getattr(c, "url", "") or "").strip()))
                    for c in (chunks or [])
                )
                has_non_yt_external_in_prompt = any(
                    _is_external((getattr(c, "url", "") or "").strip()) and (not _is_youtube((getattr(c, "url", "") or "").strip()))
                    for c in (prompt_chunks or [])
                )

                if has_non_yt_external_anywhere and (not has_non_yt_external_in_prompt):
                    # Pick the first non-YouTube external URL we have.
                    cand_urls = [u for u in url_to_chunks.keys() if _is_external(u) and (not _is_youtube(u))]
                    if cand_urls:
                        u = cand_urls[0]
                        cs = url_to_chunks.get(u) or []
                        if cs:
                            ranked_chunks = sorted(cs, key=lambda c: _text_score((getattr(c, "text", "") or "")), reverse=True)
                            picked = ranked_chunks[:3]
                            merged_text = "\n\n".join(
                                ((getattr(c, "text", "") or "").strip()) for c in picked if (getattr(c, "text", "") or "").strip()
                            ).strip()
                            if len(merged_text) > 2200:
                                merged_text = merged_text[:2200] + "..."
                            ext_chunk = type(cs[0])(
                                text=merged_text or (getattr(cs[0], "text", "") or ""),
                                url=u,
                                title=getattr(cs[0], "title", None),
                            )
                            if prompt_chunks:
                                prompt_chunks[-1] = ext_chunk
                            else:
                                prompt_chunks.append(ext_chunk)
            except Exception:
                pass

        return prompt_chunks

    allow_external_sources = bool(settings.web_scrape_enabled or (settings.youtube_api_key and quest_help) or used_web_snippets)

    # We'll allow external sources only if we actually add some (web snippets, scraped pages, or YouTube).
    allow_external_sources = bool(settings.web_scrape_enabled or used_web_snippets or used_youtube_sources)

    prompt_chunks = _build_prompt_chunks(
        chunks=chunks,
        retrieval_seed=retrieval_seed,
        user_question=(raw_user_message or user_message or ""),
        topic_hint=topic_hint,
        prefixes=prefixes,
        allow_external_sources=allow_external_sources,
    )

    # Add YouTube-derived insights as additional citable sources (best-effort).
    # This runs only when a YouTube API key is configured.
    if quest_help and settings.youtube_api_key:
        try:
            await _status("Listening to a few quest guide videos for extra tips...")
            yt_chunks = await quest_youtube_insight_chunks(
                user_message=user_message,
                max_videos=3,
            )
            if yt_chunks:
                used_youtube_sources = True
                chunks = yt_chunks + (chunks or [])
                allow_external_sources = bool(settings.web_scrape_enabled or used_web_snippets or used_youtube_sources)
                prompt_chunks = _build_prompt_chunks(
                    chunks=chunks,
                    retrieval_seed=retrieval_seed,
                    user_question=(raw_user_message or user_message or ""),
                    topic_hint=topic_hint,
                    prefixes=prefixes,
                    allow_external_sources=allow_external_sources,
                )
                actions.append("Added YouTube guide insights as sources.")
        except Exception:
            pass

    # Opinion/community questions: YouTube creators often have useful takes.
    if opinion_question and settings.youtube_api_key:
        try:
            await _status("Consulting a few OSRS videos for community takes...")
            yt_chunks = await opinion_youtube_insight_chunks(
                user_message=user_message,
                max_videos=2,
            )
            if yt_chunks:
                used_youtube_sources = True
                chunks = yt_chunks + (chunks or [])
                allow_external_sources = bool(settings.web_scrape_enabled or used_web_snippets or used_youtube_sources)
                prompt_chunks = _build_prompt_chunks(
                    chunks=chunks,
                    retrieval_seed=retrieval_seed,
                    user_question=(raw_user_message or user_message or ""),
                    topic_hint=topic_hint,
                    prefixes=prefixes,
                    allow_external_sources=allow_external_sources,
                )
                actions.append("Added YouTube community takes as sources.")
        except Exception:
            pass

    # If our stricter prompt filtering produced nothing but we do have some local chunks,
    # treat that as a weak retrieval and attempt a fresh wiki lookup.
    if (not prompt_chunks) and chunks and (not force_fresh):
        try:
            await _status("Those pages seemed irrelevant; searching anew...")
            retry_q = _primary_live_query(
                raw=raw_user_message,
                topic_hint=topic_hint,
                queries=queries,
                fallback=retrieval_seed,
            )
            live_chunks = await live_query_chunks(retry_q, allowed_url_prefixes=prefixes)
            if live_chunks:
                actions.append("Local sources looked irrelevant; refreshed wiki search.")
                chunks = live_chunks
                prompt_chunks = _build_prompt_chunks(
                    chunks=chunks,
                    retrieval_seed=retrieval_seed,
                    user_question=(raw_user_message or user_message or ""),
                    topic_hint=topic_hint,
                    prefixes=prefixes,
                    allow_external_sources=allow_external_sources,
                )
        except Exception:
            pass

    if not prompt_chunks:
        # Auto-retry once with a more targeted wiki query. This prevents users from needing
        # to re-ask questions when retrieval is flaky or the first query was too broad.
        await _status("Trying a more targeted lookup...")
        retry_chunks, retry_web_query, retry_web_results, retry_actions = await _auto_retry_targeted_retrieval(
            raw_user_message=(raw_user_message or ""),
            user_message=(user_message or ""),
            retrieval_seed=(retrieval_seed or ""),
            topic_hint=(topic_hint or ""),
            msg_l=(msg_l or ""),
            difficulty_question=bool(difficulty_question),
            queries=(queries or []),
            prefixes=prefixes,
            existing_web_query=web_query,
            existing_web_results=web_results,
        )
        if retry_chunks:
            actions.extend(retry_actions)
            chunks = retry_chunks
            if retry_web_query:
                web_query = retry_web_query
            if retry_web_results is not None:
                web_results = retry_web_results
            prompt_chunks = _build_prompt_chunks(
                chunks=chunks,
                retrieval_seed=retrieval_seed,
                user_question=(raw_user_message or user_message or ""),
                topic_hint=topic_hint,
                prefixes=prefixes,
                allow_external_sources=allow_external_sources,
            )

    # ── YouTube fallback ───────────────────────────────────────────────────────
    # If we still have no chunks, try YouTube as a last resort.
    if not prompt_chunks and settings.youtube_api_key:
        await _status("Checking the tavern's crystal ball (YouTube)...")
        try:
            yt_chunks = await general_youtube_fallback_chunks(
                user_message=(raw_user_message or user_message or retrieval_seed),
                max_videos=3,
            )
            if yt_chunks:
                actions.append("Found YouTube videos as fallback sources.")
                chunks = yt_chunks
                allow_external_sources = True
                prompt_chunks = _build_prompt_chunks(
                    chunks=chunks,
                    retrieval_seed=retrieval_seed,
                    user_question=(raw_user_message or user_message or ""),
                    topic_hint=topic_hint,
                    prefixes=prefixes,
                    allow_external_sources=True,
                )
        except Exception:
            pass

    if not prompt_chunks:
        await _status("My shelves come up empty; no citable pages found.")
        if strategy_followup and ("seren" in msg_l or "fragment" in msg_l or "song of the elves" in msg_l):
            fallback = _best_effort_fragment_of_seren_plan()
        elif new_player_intent:
            fallback = (
                "(best effort; not sourced) If you’re setting up a brand-new OSRS account, here are the main choices that matter up front:\n\n"
                "- Account mode: Regular (recommended) vs Ironman/HCIM/UIM (self-sufficient challenge modes).\n"
                "- Onboarding path: do Tutorial Island, then consider Adventure Paths / early quests for direction.\n"
                "- F2P vs members: members unlocks faster training, travel, and early quest rewards.\n"
                "- Early goal: questing unlocks (teleports, stamina quality-of-life) vs pure skilling vs combat.\n\n"
                "If you tell me (1) F2P or members and (2) whether you want Ironman or regular, I’ll suggest a simple first-week plan."
            )
        elif skill_training:
            fallback = (
                "(best effort; not sourced) I couldn’t fetch citable training pages just now, but I can still help.\n\n"
                "Tell me which skill you’re leveling, your current level, and whether you prefer fast XP, cheap, or AFK. "
                "I’ll give you a level-bracketed path (fast/cheap/AFK options) plus key unlocks/quests to speed it up."
            )
        elif opinion_question or difficulty_question:
            fallback = (
                "I couldn't fetch citable pages for that just now, my friend — but I can still help. "
                "When players ask for the 'hardest part' of a quest, it's often subjective, and the best answer depends on what step you're stuck on. "
                "Tell me which step/room/boss you're at (or what the game message says), and what your stats/gear are, and I'll give a practical plan."
            )
        elif strategy_followup:
            fallback = (
                "(common player experience; not sourced) I couldn't fetch citable strategy pages just now, "
                "but I can still help you win this fight.\n\n"
                "Tell me the boss/monster name, your combat stats, and what style you're using (melee/range/mage), "
                "and I’ll give you an inventory + step-by-step plan."
            )
        elif bool(settings.web_scrape_enabled) or (web_results and len(web_results) > 0):
            fallback = (
                "I couldn't fetch any citable sources for that question just now (wiki or community). "
                "Please try rephrasing your question or ask about a different topic."
            )
        else:
            fallback = (
                "I couldn't find any citable OSRS Wiki sources for that question just now. "
                "Please try rephrasing your question or ask about a different topic."
            )
        videos: list[VideoItem] = []
        if quest_help:
            try:
                await _status("Scrying the crystal screen for quest videos...")
                raw_videos = await quest_youtube_videos_with_summaries(user_message=req.message)
                videos = [VideoItem.model_validate(v) for v in (raw_videos or [])]
            except Exception:
                videos = []

        history_id = history_store.add(
            session_id=req.session_id,
            user_message=raw_user_message,
            bot_answer=fallback,
            sources=[],
            videos=[v.model_dump() for v in videos],
        )
        return ChatResponse(
            answer=fallback,
            sources=[],
            history_id=str(history_id),
            videos=videos,
            actions=actions,
            web_query=web_query,
            web_results=web_results,
        )

    await _status("Arranging citations and weaving your answer...")
    prompt = build_rag_prompt(
        user_message=prompt_user_message,
        conversation_context=conversation_context,
        chunks=prompt_chunks,
        allowed_url_prefixes=prefixes,
        allow_external_sources=bool(allow_external_sources),
        allow_best_effort=bool(
            opinion_question
            or difficulty_question
            or strategy_followup
            or quest_help
            or new_player_intent
            or skill_training
        ),
    )

    try:
        await _status("Consulting my crystal ball (Gemini)...")
        res = _generate_with_retry(prompt)
    except Exception as exc:
        # For local development, keep the endpoint usable even if Vertex isn't configured.
        fallback = (
            "Gemini (Vertex AI) is not configured yet. "
            "Set GOOGLE_CLOUD_PROJECT (and authenticate with gcloud) to enable LLM answers."
        )

        videos: list[VideoItem] = []
        if quest_help:
            try:
                raw_videos = await quest_youtube_videos_with_summaries(user_message=user_message)
                videos = [VideoItem.model_validate(v) for v in (raw_videos or [])]
            except Exception:
                videos = []

        sources = []
        for c in prompt_chunks[:5]:
            if not c.url:
                continue
            sources.append(SourceChunk(title=c.title, url=c.url, text=c.text[:500]))
        history_id = history_store.add(
            session_id=req.session_id,
            user_message=raw_user_message,
            bot_answer=f"{fallback}\n\nError: {exc}",
            sources=[s.model_dump() for s in sources],
            videos=[v.model_dump() for v in videos],
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

    res, prompt_chunks, web_query, web_results = await _maybe_retry_with_broader_sources(
        prompt_user_message=prompt_user_message,
        conversation_context=conversation_context,
        raw_user_message=raw_user_message,
        user_message=user_message,
        pronoun_followup=pronoun_followup,
        topic_hint=topic_hint,
        prefixes=prefixes,
        allow_external_sources=bool(allow_external_sources),
        prompt_chunks=prompt_chunks,
        res=res,
        opinion_question=bool(opinion_question),
        difficulty_question=bool(difficulty_question),
        strategy_followup=bool(strategy_followup),
        quest_help=bool(quest_help),
        new_player_intent=bool(new_player_intent),
        skill_training=bool(skill_training),
    )

    # If the model effectively said "I don't know" (including the roleplay variants), do one retry
    # with a fresh, focused wiki lookup to avoid returning random sources.
    if _is_low_confidence_answer(res.text or "") or not _has_source_citations(res.text or ""):
        try:
            await _status("That answer felt uncertain; double-checking the wiki...")
            actions.append("Answer looked uncertain; re-queried the wiki for better sources.")
            retry_seed = (user_message if not pronoun_followup else (topic_hint or user_message)).strip()
            retry_qs = derive_search_queries(retry_seed) or [retry_seed]
            retry_primary = retry_qs[0]
            retry_chunks = await live_query_chunks(retry_primary, allowed_url_prefixes=prefixes)
            retry_prompt_chunks = _build_prompt_chunks(
                chunks=retry_chunks,
                retrieval_seed=retry_seed,
                user_question=(raw_user_message or user_message or ""),
                topic_hint=topic_hint,
                prefixes=prefixes,
                allow_external_sources=allow_external_sources,
            )
            if retry_prompt_chunks:
                retry_prompt = build_rag_prompt(
                    user_message=prompt_user_message,
                    conversation_context=conversation_context,
                    chunks=retry_prompt_chunks,
                    allowed_url_prefixes=prefixes,
                    allow_external_sources=bool(allow_external_sources),
                    allow_best_effort=bool(
                        opinion_question
                        or difficulty_question
                        or strategy_followup
                        or quest_help
                        or new_player_intent
                        or skill_training
                    ),
                )
                res = _generate_with_retry(retry_prompt)
                prompt_chunks = retry_prompt_chunks
                actions.append("Rewrote the answer using refreshed sources.")
        except Exception:
            # Retry is best-effort.
            pass

    # If the model picked a specific quest title (common for "hardest quest" community questions),
    # do a focused wiki retrieval on that quest and regenerate once. This helps enrich the answer
    # with factual details (requirements/boss/mechanics) even when the initial sources were mostly
    # community/YouTube.
    if not did_focus_followup:
        try:
            picked_quest = find_quest_title_in_text(res.text or "")
            if picked_quest:
                already_has = any(
                    picked_quest.lower() in (str(getattr(c, "title", "") or "").lower())
                    for c in (prompt_chunks or [])
                )
                if not already_has:
                    await _status(f"Double-checking the wiki for {picked_quest}...")
                    focus_chunks = await live_query_chunks(picked_quest, allowed_url_prefixes=prefixes)
                    if focus_chunks:
                        focus_all = (focus_chunks or []) + (prompt_chunks or [])
                        focus_prompt_chunks = _build_prompt_chunks(
                            chunks=focus_all,
                            retrieval_seed=picked_quest,
                            user_question=(raw_user_message or user_message or ""),
                            topic_hint=topic_hint,
                            prefixes=prefixes,
                            allow_external_sources=allow_external_sources,
                        )
                        if focus_prompt_chunks:
                            focus_prompt = build_rag_prompt(
                                user_message=(prompt_user_message + f"\n\nIf you name a specific quest (like '{picked_quest}'), include 2-4 factual details about it from the wiki sources (requirements, boss/mechanics, or notable difficulty reasons), then summarize the community consensus.")
                                if prompt_user_message else prompt_user_message,
                                conversation_context=conversation_context,
                                chunks=focus_prompt_chunks,
                                allowed_url_prefixes=prefixes,
                                allow_external_sources=bool(allow_external_sources),
                                allow_best_effort=bool(
                                    opinion_question
                                    or difficulty_question
                                    or strategy_followup
                                    or quest_help
                                    or new_player_intent
                                    or skill_training
                                ),
                            )
                            res = _generate_with_retry(focus_prompt)
                            prompt_chunks = focus_prompt_chunks
                            actions.append(f"Follow-up retrieval: added focused wiki excerpts for {picked_quest}.")
                            did_focus_followup = True
        except Exception:
            # Best-effort; don't break chat.
            pass

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

    def _is_youtube_url(u: str) -> bool:
        ul = (u or "").lower()
        return "youtube.com/" in ul or "youtu.be/" in ul

    def _is_reddit_url(u: str) -> bool:
        ul = (u or "").lower()
        return "reddit.com/" in ul

    prefix_tuple = tuple(prefixes or [])

    # De-dupe by URL first, preserving order.
    unique_chunks: list[RetrievedChunk] = []
    seen_urls: set[str] = set()
    for c in (prompt_chunks or []):
        u = (getattr(c, "url", None) or "").strip()
        if not u:
            continue
        if u in seen_urls:
            continue
        seen_urls.add(u)
        unique_chunks.append(c)

    wiki_chunks = [c for c in unique_chunks if (getattr(c, "url", "") or "").startswith(prefix_tuple)]
    yt_chunks = [c for c in unique_chunks if _is_youtube_url(getattr(c, "url", "") or "")]
    reddit_chunks = [c for c in unique_chunks if _is_reddit_url(getattr(c, "url", "") or "")]
    other_chunks = [c for c in unique_chunks if c not in wiki_chunks and c not in yt_chunks and c not in reddit_chunks]

    picked: list[RetrievedChunk] = []

    # Prefer wiki sources for factual grounding.
    picked.extend(wiki_chunks[:4])

    # If we have YouTube insights, ensure at least one shows up in citations.
    if yt_chunks and all(not _is_youtube_url(getattr(c, "url", "") or "") for c in picked):
        picked.append(yt_chunks[0])

    # If we have Reddit, include at most one (community context).
    if reddit_chunks and all(not _is_reddit_url(getattr(c, "url", "") or "") for c in picked) and len(picked) < 5:
        picked.append(reddit_chunks[0])

    # Fill remaining slots.
    for c in other_chunks + wiki_chunks[4:] + yt_chunks[1:] + reddit_chunks[1:]:
        if len(picked) >= 5:
            break
        if c in picked:
            continue
        picked.append(c)

    for c in picked[:5]:
        u = (c.url or "").strip()
        if not u:
            continue
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
            if (not _is_low_confidence_answer(res.text or "")) and _has_source_citations(res.text or ""):
                AnswerCacheStore().add(
                    question=user_message,
                    answer=res.text,
                    sources=[s.model_dump() for s in sources],
                )
        except Exception:
            pass

    # Quest-only YouTube results (best-effort; doesn't block the core answer).
    # Only show these for explicit quest-help prompts (walkthrough/requirements), not meta/community questions.
    videos = []
    if quest_help:
        try:
            await _status("Scrying the crystal screen for quest videos...")
            raw_videos = await quest_youtube_videos_with_summaries(user_message=user_message)
            videos = [VideoItem.model_validate(v) for v in (raw_videos or [])]
            if videos:
                actions.append("Found quest videos and summarized them.")
        except Exception:
            videos = []

    history_id = history_store.add(
        session_id=req.session_id,
        user_message=raw_user_message,
        bot_answer=res.text,
        sources=[s.model_dump() for s in sources],
        videos=[v.model_dump() for v in videos],
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
