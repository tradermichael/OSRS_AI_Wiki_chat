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
        q = (
            self._col.where("session_id", "==", session_id)
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(limit)
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
