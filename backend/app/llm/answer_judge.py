from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..core.config import settings
from .gemini_vertex import GeminiVertexClient


@dataclass(frozen=True)
class Judgement:
    confidence: float
    needs_web_search: bool
    reason: str


_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")


def _coerce_float(x: object, *, default: float = 0.0) -> float:
    try:
        v = float(x)  # type: ignore[arg-type]
    except Exception:
        return float(default)
    if v != v:  # NaN
        return float(default)
    return v


def _parse_judgement(text: str) -> Judgement:
    raw = (text or "").strip()
    m = _JSON_OBJ_RE.search(raw)
    if not m:
        return Judgement(confidence=0.0, needs_web_search=True, reason="Judge returned no JSON")

    try:
        obj = json.loads(m.group(0))
    except Exception:
        return Judgement(confidence=0.0, needs_web_search=True, reason="Judge JSON parse failed")

    confidence = _coerce_float(obj.get("confidence"), default=0.0)
    confidence = max(0.0, min(1.0, confidence))

    needs_web_search = bool(obj.get("needs_web_search"))
    reason = str(obj.get("reason") or "").strip()[:240]
    if not reason:
        reason = "(no reason)"

    return Judgement(confidence=confidence, needs_web_search=needs_web_search, reason=reason)


def judge_answer_confidence(*, user_message: str, answer: str, sources: list[dict]) -> Judgement:
    """Ask a small LLM to judge if the draft answer is well-supported.

    This is intentionally strict: if sources look unrelated, missing, or the answer is hedgy,
    it should recommend doing a web/wiki fetch.
    """

    model = settings.answer_judge_model or settings.gemini_model

    prompt = (
        "You are a strict QA judge for a RAG assistant.\n"
        "Given: (1) user question, (2) assistant draft answer, (3) a list of sources (title/url).\n\n"
        "Task: Decide whether the draft answer is well-supported by the sources AND relevant to the question.\n"
        "If sources are missing, unrelated, or the answer contains uncertainty/refusal language, mark low confidence.\n"
        "If the answer seems off-topic relative to the question, mark low confidence.\n\n"
        "Return ONLY a JSON object with these keys:\n"
        "- confidence: number from 0 to 1 (0 = not reliable, 1 = reliable)\n"
        "- needs_web_search: boolean (true if we should do a live fetch / new retrieval)\n"
        "- reason: short string\n\n"
        f"USER_QUESTION:\n{(user_message or '').strip()}\n\n"
        f"DRAFT_ANSWER:\n{(answer or '').strip()}\n\n"
        "SOURCES:\n"
        f"{json.dumps(sources or [], ensure_ascii=False)[:4000]}\n"
    )

    client = GeminiVertexClient(model_name=model)
    res = client.generate(prompt, temperature=0.0, max_output_tokens=256)
    return _parse_judgement(res.text)
