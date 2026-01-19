from __future__ import annotations

from datetime import datetime, timezone

from ..core.config import settings


class FirestoreGoldStore:
    """Firestore-backed global gold total.

    Stores a single document with an integer total so the counter persists across
    Cloud Run deploys/instances and across all user sessions.
    """

    def __init__(self) -> None:
        # Import lazily so local dev without Firestore deps can still run with sqlite.
        from google.cloud import firestore  # type: ignore

        project = settings.google_cloud_project or None
        self._client = firestore.Client(project=project)
        col = settings.firestore_site_collection or "site_state"
        doc = settings.firestore_gold_doc or "gold_total"
        self._doc_ref = self._client.collection(col).document(doc)

    def get_total(self) -> int:
        snap = self._doc_ref.get()
        if not snap.exists:
            # Initialize lazily.
            self._doc_ref.set({"total": 0, "updated_at": datetime.now(timezone.utc)})
            return 0
        d = snap.to_dict() or {}
        try:
            return int(d.get("total") or 0)
        except Exception:
            return 0

    def add(self, amount_gold: int) -> int:
        if amount_gold <= 0:
            raise ValueError("amount_gold must be > 0")

        from google.cloud import firestore  # type: ignore

        @firestore.transactional
        def _tx_update(transaction: firestore.Transaction) -> int:
            snap = self._doc_ref.get(transaction=transaction)
            if snap.exists:
                d = snap.to_dict() or {}
                current = int(d.get("total") or 0)
            else:
                current = 0

            new_total = current + int(amount_gold)
            transaction.set(
                self._doc_ref,
                {"total": int(new_total), "updated_at": datetime.now(timezone.utc)},
                merge=True,
            )
            return int(new_total)

        tx = self._client.transaction()
        return _tx_update(tx)
