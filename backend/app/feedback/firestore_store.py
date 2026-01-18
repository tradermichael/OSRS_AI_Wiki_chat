from __future__ import annotations

from datetime import datetime, timezone

from ..core.config import settings


class FirestoreFeedbackStore:
    def __init__(self) -> None:
        from google.cloud import firestore  # type: ignore

        project = settings.google_cloud_project or None
        self._client = firestore.Client(project=project)
        # Store feedback separately so we can query/aggregate.
        self._col = self._client.collection("feedback")

    def add(self, *, history_id: str, rating: int, session_id: str | None) -> str:
        rating = int(rating)
        if rating not in (-1, 1):
            raise ValueError("rating must be -1 or 1")

        created_at_dt = datetime.now(timezone.utc)
        doc_ref = self._col.document()
        doc_ref.set(
            {
                "created_at": created_at_dt,
                "history_id": str(history_id),
                "rating": rating,
                "session_id": session_id,
            }
        )
        return doc_ref.id

    def summary(self, *, history_id: str) -> dict:
        from google.cloud import firestore  # type: ignore

        q = self._col.where(filter=firestore.FieldFilter("history_id", "==", str(history_id)))
        up = 0
        down = 0
        for doc in q.stream():
            d = doc.to_dict() or {}
            r = int(d.get("rating") or 0)
            if r == 1:
                up += 1
            elif r == -1:
                down += 1
        return {"thumbs_up": up, "thumbs_down": down, "net": up - down}
