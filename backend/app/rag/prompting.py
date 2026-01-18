from __future__ import annotations

from .store import RetrievedChunk


def build_rag_prompt(*, user_message: str, chunks: list[RetrievedChunk]) -> str:
    context_lines: list[str] = []
    for i, ch in enumerate(chunks, start=1):
        title = ch.title or "(untitled)"
        url = ch.url or ""
        context_lines.append(f"[Source {i}] {title} — {url}\n{ch.text}")

    context = "\n\n".join(context_lines) if context_lines else "(no sources retrieved)"

    return (
        "You are a helpful assistant for Old School RuneScape questions. "
        "Use the provided sources when answering. If the sources do not contain the answer, say you don't know. "
        "When you use a source, cite it like [Source 1], [Source 2].\n\n"
        f"SOURCES:\n{context}\n\n"
        f"USER QUESTION:\n{user_message}\n\n"
        "ANSWER:" 
    )
