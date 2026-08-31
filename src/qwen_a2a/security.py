"""Optional request signature verification defined by the Qwen gateway spec."""

import hashlib
import hmac
import time
from collections.abc import Iterable

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class QwenSignatureMiddleware:
    """Verify SHA-256 signatures only when credentials are configured."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        client_id: str | None,
        client_secret: str | None,
        protected_paths: Iterable[str],
        max_skew_seconds: int = 300,
    ) -> None:
        self._app = app
        self._client_id = client_id
        self._client_secret = client_secret
        self._protected_paths = frozenset(protected_paths)
        self._max_skew_ms = max_skew_seconds * 1000

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not self._requires_signature(scope):
            await self._app(scope, receive, send)
            return
        headers = _headers(scope)
        if not self._is_valid(scope, headers):
            response = JSONResponse(
                {"detail": "invalid request signature"}, status_code=401
            )
            await response(scope, receive, send)
            return
        await self._app(scope, receive, send)

    def _requires_signature(self, scope: Scope) -> bool:
        return bool(
            self._client_id
            and self._client_secret
            and scope["type"] == "http"
            and scope.get("path") in self._protected_paths
        )

    def _is_valid(self, scope: Scope, headers: dict[str, str]) -> bool:
        client_id = headers.get("x-client-id")
        timestamp = headers.get("x-tm")
        nonce = headers.get("x-nonce")
        token = headers.get("x-token")
        if not all((client_id, timestamp, nonce, token)):
            return False
        if client_id != self._client_id:
            return False
        try:
            timestamp_ms = int(timestamp)
        except ValueError:
            return False
        if abs(int(time.time() * 1000) - timestamp_ms) > self._max_skew_ms:
            return False

        source = "&".join(
            (
                str(scope.get("method", "")).upper(),
                str(scope.get("path", "")),
                timestamp,
                client_id,
                self._client_secret or "",
                nonce,
            )
        )
        expected = hashlib.sha256(source.encode()).hexdigest()
        return hmac.compare_digest(expected, token)


def _headers(scope: Scope) -> dict[str, str]:
    return {
        key.decode("latin-1").lower(): value.decode("latin-1")
        for key, value in scope.get("headers", [])
    }
