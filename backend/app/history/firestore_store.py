from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from ..core.config import settings


@dataclass(frozen=True)
class PublicChatRecord:
    id: str
    created_at: str
    session_id: str | None
    user_message: str
    bot_answer: str
    sources: list[dict]
    videos: list[dict]


class FirestorePublicChatStore:
    def __init__(self) -> None:
        # Import lazily so local dev without Firestore deps can still run with sqlite.
        from google.cloud import firestore  # type: ignore

        project = settings.google_cloud_project or None
        self._client = firestore.Client(project=project)
        self._col = self._client.collection(settings.firestore_collection or "public_chat")
        # Store per-session turn logs in a subcollection to avoid composite-index requirements
        # for querying recent turns by session.
        self._sessions_col = self._client.collection((settings.firestore_collection or "public_chat") + "_sessions")

    def _session_turns_collection(self, session_id: str):
        sid = str(session_id or "").strip()
        return self._sessions_col.document(sid).collection("turns")

    def add(
        self,
        *,
        session_id: str | None,
        user_message: str,
        bot_answer: str,
        sources: list[dict],
        videos: list[dict] | None = None,
    ) -> str:
        created_at_dt = datetime.now(timezone.utc)
        doc_ref = self._col.document()  # auto id
        doc_ref.set(
            {
                "created_at": created_at_dt,
                "session_id": session_id,
                "user_message": user_message,
                "bot_answer": bot_answer,
                "sources": sources or [],
                "videos": videos or [],
            }
        )

        # Also persist to a per-session subcollection so lookups don't require Firestore
        # composite indexes (which can otherwise cause "no memory" symptoms).
        sid = str(session_id or "").strip()
        if sid:
            try:
                self._session_turns_collection(sid).document(doc_ref.id).set(
                    {
                        "created_at": created_at_dt,
                        "global_id": doc_ref.id,
                        "session_id": sid,
                        "user_message": user_message,
                        "bot_answer": bot_answer,
                        "sources": sources or [],
                        "videos": videos or [],
                    }
                )
            except Exception:
                # Best-effort: failure here should not break chat.
                pass
        return doc_ref.id

    def list(self, *, limit: int = 50, offset: int = 0) -> list[PublicChatRecord]:
        from google.cloud import firestore  # type: ignore

        limit = max(1, min(int(limit), 200))
        offset = max(0, int(offset))

        q = (
            self._col.order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .offset(offset)
        )
        out: list[PublicChatRecord] = []
        for doc in q.stream():
            d = doc.to_dict() or {}
            created_at = d.get("created_at")
            if isinstance(created_at, datetime):
                created_at_str = created_at.astimezone(timezone.utc).isoformat()
            else:
                created_at_str = str(created_at or "")

            out.append(
                PublicChatRecord(
                    id=doc.id,
                    created_at=created_at_str,
                    session_id=d.get("session_id"),
                    user_message=str(d.get("user_message") or ""),
                    bot_answer=str(d.get("bot_answer") or ""),
                    sources=list(d.get("sources") or []),
                    videos=list(d.get("videos") or []),
                )
            )
        return out

    def list_by_session(self, *, session_id: str, limit: int = 10) -> list[PublicChatRecord]:
        from google.cloud import firestore  # type: ignore

        session_id = str(session_id or "").strip()
        if not session_id:
            return []

        limit = max(1, min(int(limit), 50))

        def _to_record(doc_id: str, d: dict) -> PublicChatRecord:
            created_at = d.get("created_at")
            if isinstance(created_at, datetime):
                created_at_str = created_at.astimezone(timezone.utc).isoformat()
            else:
                created_at_str = str(created_at or "")

            rid = str(d.get("global_id") or doc_id)
            return PublicChatRecord(
                id=rid,
                created_at=created_at_str,
                session_id=d.get("session_id"),
                user_message=str(d.get("user_message") or ""),
                bot_answer=str(d.get("bot_answer") or ""),
                sources=list(d.get("sources") or []),
                videos=list(d.get("videos") or []),
            )

        # Preferred path: per-session turns subcollection (no composite index required).
        try:
            q = (
                self._session_turns_collection(session_id)
                .order_by("created_at", direction=firestore.Query.DESCENDING)
                .limit(limit)
            )
            out: list[PublicChatRecord] = []
            for doc in q.stream():
                d = doc.to_dict() or {}
                out.append(_to_record(doc.id, d))
            if out:
                return out
        except Exception:
            pass

        # Fallback: scan recent global history and filter in-memory (also index-free).
        # This keeps conversation context working even if older data predates the session subcollection.
        scan_limit = max(50, min(500, limit * 25))
        out: list[PublicChatRecord] = []
        try:
            q = self._col.order_by("created_at", direction=firestore.Query.DESCENDING).limit(scan_limit)
            for doc in q.stream():
                d = doc.to_dict() or {}
                if str(d.get("session_id") or "").strip() != session_id:
                    continue
                out.append(_to_record(doc.id, d))
                if len(out) >= limit:
                    break
        except Exception:
            return []

        return out

    def get(self, record_id: str) -> PublicChatRecord | None:
        doc = self._col.document(str(record_id)).get()
        if not doc.exists:
            return None
        d = doc.to_dict() or {}
        created_at = d.get("created_at")
        if isinstance(created_at, datetime):
            created_at_str = created_at.astimezone(timezone.utc).isoformat()
        else:
            created_at_str = str(created_at or "")

        return PublicChatRecord(
            id=doc.id,
            created_at=created_at_str,
            session_id=d.get("session_id"),
            user_message=str(d.get("user_message") or ""),
            bot_answer=str(d.get("bot_answer") or ""),
            sources=list(d.get("sources") or []),
            videos=list(d.get("videos") or []),
        )
