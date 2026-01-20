from __future__ import annotations

from .store import RetrievedChunk


def _filter_allowed(chunks: list[RetrievedChunk], allowed_url_prefixes: list[str] | None) -> list[RetrievedChunk]:
    if not allowed_url_prefixes:
        return chunks
    prefixes = tuple(p for p in allowed_url_prefixes if p)
    if not prefixes:
        return chunks
    return [c for c in chunks if (c.url or "").startswith(prefixes)]


def build_rag_prompt(
    *,
    user_message: str,
    conversation_context: str | None = None,
    chunks: list[RetrievedChunk],
    allowed_url_prefixes: list[str] | None = None,
    allow_external_sources: bool = False,
    allow_best_effort: bool = False,
) -> str:
    if not allow_external_sources:
        chunks = _filter_allowed(chunks, allowed_url_prefixes)
    context_lines: list[str] = []
    for i, ch in enumerate(chunks, start=1):
        title = ch.title or "(untitled)"
        url = ch.url or ""
        snippet = (ch.text or "").strip()
        if len(snippet) > 1400:
            snippet = snippet[:1400] + "…"
        context_lines.append(f"[Source {i}] {title} — {url}\n{snippet}")

    context = "\n\n".join(context_lines) if context_lines else "(no sources retrieved)"

    convo = (conversation_context or "").strip()
    convo_block = ""
    if convo:
        # Keep this section short and explicitly optional, so the model doesn't overfit on it.
        convo_block = (
            "CONVERSATION CONTEXT (recent; may contain pronouns like 'her'):\n"
            f"{convo}\n\n"
        )

    grounded_rule = (
        "If the sources do not contain the answer OR look unrelated, say you don't know based on the sources. "
        if not allow_best_effort
        else (
            "If the sources do not contain the answer OR look unrelated, do NOT freeze or refuse. "
            "Give a best-effort answer based on common player experience and general OSRS mechanics, "
            "and clearly label that part as 'common player experience' (not sourced). "
            "When possible, still ground any factual claims (requirements, named bosses, mechanics) in the provided sources. "
            "If the user asks something subjective like 'hardest part', you may summarize a common consensus and then ask what step they are stuck on. "
        )
    )

    return (
        "You are the Wise Old Man from Old School RuneScape: a wise, slightly cheeky medieval wizard. "
        "Speak in an in-game, old-fashioned tone (e.g., 'aye', 'indeed', 'lest', 'my friend') while staying clear and helpful. "
        "Avoid modern corporate tone. "
        "Assume the user is asking about Old School RuneScape (OSRS), not RuneScape 3, unless explicitly stated otherwise. "
        "Use the conversation context (if provided) to resolve follow-up questions and pronouns across turns. "
        "Use the provided sources when answering, but ONLY if they are relevant to the user's question. "
        f"{grounded_rule}"
        "Some sources may be from the wider web (non-wiki) and can be noisy; treat them as less authoritative than the OSRS Wiki. "
        "Do not output long verbatim excerpts; paraphrase in your own words. "
        "When you use a source, cite it like [Source 1], [Source 2]. "
        "Only cite source numbers that exist in the SOURCES list below; never cite [Source 7] if there is no Source 7.\n\n"
        f"{convo_block}"
        f"SOURCES:\n{context}\n\n"
        f"USER QUESTION:\n{user_message}\n\n"
        "ANSWER:" 
    )
