from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import settings


@dataclass(frozen=True)
class RAGSource:
    id: str
    name: str
    mediawiki_api: str | None
    allowed_url_prefixes: list[str]


def _default_config_path() -> Path:
    # repo root /rag_sources.json
    return Path(__file__).resolve().parents[3] / "rag_sources.json"


def load_rag_sources(config_path: str | None = None) -> list[RAGSource]:
    path = Path(config_path or settings.rag_sources_path or _default_config_path())
    if not path.exists():
        return []

    data = json.loads(path.read_text(encoding="utf-8"))
    raw_sources = (data or {}).get("sources") or []

    out: list[RAGSource] = []
    for s in raw_sources:
        sid = (s or {}).get("id")
        name = (s or {}).get("name")
        api = (s or {}).get("mediawiki_api")
        prefixes = (s or {}).get("allowed_url_prefixes") or []

        if not sid:
            continue
        if not isinstance(prefixes, list):
            prefixes = []

        out.append(
            RAGSource(
                id=str(sid),
                name=str(name or sid),
                mediawiki_api=str(api) if api else None,
                allowed_url_prefixes=[str(p) for p in prefixes if p],
            )
        )

    return out


def allowed_url_prefixes(config_path: str | None = None) -> list[str]:
    prefixes: list[str] = []
    for s in load_rag_sources(config_path=config_path):
        if s.id == "osrsbox" and not bool(settings.osrsbox_enabled):
            continue
        prefixes.extend(s.allowed_url_prefixes)
    # de-dupe while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for p in prefixes:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def get_source_by_id(source_id: str, config_path: str | None = None) -> RAGSource | None:
    for s in load_rag_sources(config_path=config_path):
        if s.id == source_id:
            return s
    return None
