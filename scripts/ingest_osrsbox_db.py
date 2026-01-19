from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure repo root is on sys.path when running this script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.core.config import settings
from backend.app.rag.osrsbox_db import doc_to_chunks, osrsbox_fetch_items_by_query, osrsbox_fetch_monsters_by_query
from backend.app.rag.store import RAGStore, make_chunk_id


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["items", "monsters"], default="items")
    p.add_argument("--query", required=True)
    p.add_argument("--limit", type=int, default=5)
    p.add_argument(
        "--base-url",
        default=None,
        help="OSRSBox static JSON base URL (default from settings: https://www.osrsbox.com/osrsbox-db/)",
    )
    args = p.parse_args()

    base_url = (args.base_url or settings.osrsbox_base_url or "https://www.osrsbox.com/osrsbox-db/").strip()

    if args.kind == "items":
        docs = await osrsbox_fetch_items_by_query(args.query, base_url=base_url, limit=args.limit)
    else:
        docs = await osrsbox_fetch_monsters_by_query(args.query, base_url=base_url, limit=max(1, min(args.limit, 5)))

    store = RAGStore()

    texts: list[str] = []
    metas: list[dict] = []
    ids: list[str] = []

    for doc in docs:
        chunks = doc_to_chunks(doc, max_chunks=3)
        for idx, ch in enumerate(chunks):
            if not (ch.text or "").strip():
                continue
            texts.append(ch.text)
            metas.append({"url": ch.url, "title": ch.title})
            ids.append(make_chunk_id(ch.url, idx))

    if texts:
        store.add_documents(texts=texts, metadatas=metas, ids=ids)
        print(f"Ingested {len(texts)} chunk(s) from {len(docs)} OSRSBox {args.kind} doc(s).")
        print("Tip: set OSRSBOX_ENABLED=true so these citations are allowed in chat responses.")
    else:
        print("No text ingested (no matches or fetch failed).")


if __name__ == "__main__":
    asyncio.run(main())
