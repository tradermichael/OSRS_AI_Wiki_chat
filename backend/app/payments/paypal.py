from __future__ import annotations

import base64

import httpx

from ..core.config import settings


class PayPalClient:
    def __init__(self) -> None:
        env = (settings.paypal_env or "sandbox").strip().lower()
        if env not in {"sandbox", "live"}:
            raise ValueError("PAYPAL_ENV must be sandbox or live")

        self._base = "https://api-m.sandbox.paypal.com" if env == "sandbox" else "https://api-m.paypal.com"
        self._client_id = settings.paypal_client_id
        self._client_secret = settings.paypal_client_secret

    async def _get_access_token(self) -> str:
        if not self._client_id or not self._client_secret:
            raise RuntimeError("PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET must be set")

        basic = base64.b64encode(f"{self._client_id}:{self._client_secret}".encode("utf-8")).decode("utf-8")
        headers = {"Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded"}
        data = {"grant_type": "client_credentials"}

        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{self._base}/v1/oauth2/token", headers=headers, data=data)
            r.raise_for_status()
            token = r.json().get("access_token")
            if not token:
                raise RuntimeError("PayPal OAuth did not return access_token")
            return token

    async def create_order(self, *, amount_usd: str, note: str | None = None) -> tuple[str, str]:
        token = await self._get_access_token()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        payload = {
            "intent": "CAPTURE",
            "purchase_units": [
                {
                    "amount": {"currency_code": "USD", "value": amount_usd},
                    "description": (note or "Donation")[0:127],
                }
            ],
            "application_context": {
                "return_url": settings.paypal_return_url,
                "cancel_url": settings.paypal_cancel_url,
                "shipping_preference": "NO_SHIPPING",
                "user_action": "PAY_NOW",
            },
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{self._base}/v2/checkout/orders", headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()

        order_id = data.get("id")
        links = data.get("links") or []
        approve = next((l.get("href") for l in links if l.get("rel") == "approve"), None)
        if not order_id or not approve:
            raise RuntimeError("PayPal order response missing id/approve link")

        return order_id, approve

    async def capture_order(self, *, order_id: str) -> dict:
        token = await self._get_access_token()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{self._base}/v2/checkout/orders/{order_id}/capture", headers=headers)
            r.raise_for_status()
            return r.json()
