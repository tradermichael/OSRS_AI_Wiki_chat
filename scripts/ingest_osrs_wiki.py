from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure repo root is on sys.path when running this script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.rag.ingest_mediawiki import chunk_page, search_and_fetch
from backend.app.rag.store import RAGStore, make_chunk_id


OSRS_WIKI_API = "https://oldschool.runescape.wiki/api.php"
FANDOM_WIKI_API = "https://oldschoolrunescape.fandom.com/api.php"


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wiki", choices=["osrs", "fandom"], default="osrs")
    p.add_argument("--query", required=True)
    p.add_argument("--limit", type=int, default=3)
    args = p.parse_args()

    api = OSRS_WIKI_API if args.wiki == "osrs" else FANDOM_WIKI_API

    pages = await search_and_fetch(api, args.query, limit=args.limit)

    store = RAGStore()

    texts: list[str] = []
    metas: list[dict] = []
    ids: list[str] = []

    for page in pages:
        for idx, (chunk_text, meta) in enumerate(chunk_page(page)):
            if not chunk_text.strip():
                continue
            texts.append(chunk_text)
            metas.append(meta)
            ids.append(make_chunk_id(meta.get("url") or page.url, idx))

    if texts:
        store.add_documents(texts=texts, metadatas=metas, ids=ids)
        print(f"Ingested {len(texts)} chunks from {len(pages)} pages")
    else:
        print("No text ingested")


if __name__ == "__main__":
    asyncio.run(main())
