from __future__ import annotations

import re


_WORD_RE = re.compile(r"[a-zA-Z0-9_']+")

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "can",
    "do",
    "does",
    "for",
    "from",
    "get",
    "how",
    "i",
    "in",
    "is",
    "it",
    "know",
    "me",
    "of",
    "on",
    "or",
    "tell",
    "that",
    "the",
    "to",
    "what",
    "when",
    "where",
    "who",
    "why",
    "with",
    "you",
    "your",

    # Combat / guide filler terms (keep retrieval focused on the entity name)
    "beat",
    "defeat",
    "kill",
    "fight",
    "boss",
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
    "quick",
    "walkthrough",
    "requirements",
    "reqs",
}


def _tokens(text: str) -> list[str]:
    return [m.group(0) for m in _WORD_RE.finditer(text or "")]


def _keywords(text: str) -> list[str]:
    toks = [t.lower() for t in _tokens(text)]
    toks = [t for t in toks if t and t not in _STOPWORDS]
    # de-dupe while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for t in toks:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def extract_keywords(user_message: str, *, max_terms: int = 8) -> list[str]:
    """Extract a compact list of keywords from a user message."""
    keys = _keywords(user_message)
    return keys[:max_terms]


def derive_search_queries(user_message: str) -> list[str]:
    """Derive better search queries from a conversational question.

    This is intentionally heuristic (no extra LLM call):
    - Try to extract the topic after "about".
    - Try a compact keyword query.
    - Try an OSRS-qualified query.
    """

    msg = (user_message or "").strip()
    if not msg:
        return []

    queries: list[str] = []

    msg_l = msg.lower()

    # Common OSRS naming: "Quest cape" is the "Quest point cape" page on the wiki.
    # Adding this early prevents unrelated boss/strategy pages from matching on generic terms like "quest"/"cape".
    if "quest cape" in msg_l and "quest point cape" not in msg_l:
        queries.append("Quest point cape")
        queries.append("Quest point cape requirements")
        queries.append("Quest point cape osrs")

    combat_intent = any(
        w in msg_l
        for w in (
            "beat",
            "defeat",
            "kill",
            "fight",
            "strategy",
            "strategies",
            "guide",
            "gear",
            "inventory",
            "mechanic",
            "mechanics",
            "pray",
            "prayer",
        )
    )

    m = re.search(r"\babout\s+(.+)$", msg, flags=re.IGNORECASE)
    if m:
        topic = m.group(1).strip().strip("?!.\"")
        if topic:
            queries.append(topic)
            queries.append(f"{topic} osrs")
            if combat_intent:
                queries.append(f"{topic} strategies")
                queries.append(f"{topic}/Strategies")
                queries.append(f"{topic} boss fight")

    keys = _keywords(msg)
    if keys:
        key_q = " ".join(keys[:8])
        queries.append(key_q)
        queries.append(f"{key_q} osrs")

        # If the user is asking how to beat/handle something, bias towards strategy pages.
        if combat_intent and len(keys) >= 1:
            queries.append(f"{key_q} strategies")

    # Always include the raw message last.
    queries.append(msg)

    # De-dupe while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        qn = re.sub(r"\s+", " ", q).strip()
        if not qn:
            continue
        if qn.lower() in seen:
            continue
        seen.add(qn.lower())
        out.append(qn)

    return out[:4]
