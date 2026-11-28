"""Request auth, two ways in:

1. Service token: Blistro's Next.js server (which already authenticated the user
   with Kinde) sends `X-Service-Token: <SERVICE_TOKEN>` and `X-User-Id`.
2. Kinde bearer token: a client sends the user's Kinde access token; we verify it
   against Kinde's JWKS and take the user id from `sub`.

AUTH_MODE=dev (local + evals only, refused when ENV=production) accepts an
`X-Dev-User` header instead."""

from __future__ import annotations

import hmac
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache

import jwt
from fastapi import Header, HTTPException, status

from app.config import get_settings


@dataclass(frozen=True)
class User:
    id: str


@lru_cache
def _jwks_client(issuer: str) -> jwt.PyJWKClient:
    return jwt.PyJWKClient(
        f"{issuer.rstrip('/')}/.well-known/jwks.json", cache_keys=True, lifespan=3600
    )


def verify_kinde_token(token: str) -> User:
    s = get_settings()
    if not s.kinde_issuer_url:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "KINDE_ISSUER_URL not set")
    try:
        key = _jwks_client(s.kinde_issuer_url).get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            key.key,
            algorithms=["RS256"],
            issuer=s.kinde_issuer_url.rstrip("/"),
            audience=s.kinde_audience,
            options={"require": ["exp", "sub", "iss"], "verify_aud": bool(s.kinde_audience)},
            leeway=30,
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"invalid token: {exc}") from exc
    return User(id=str(claims["sub"]))


_USER_ID = re.compile(r"^[A-Za-z0-9_:.\-]{1,128}$")


async def current_user(
    authorization: str | None = Header(default=None),
    x_dev_user: str | None = Header(default=None),
    x_service_token: str | None = Header(default=None),
    x_user_id: str | None = Header(default=None),
) -> User:
    s = get_settings()
    if s.auth_mode == "dev":
        if not x_dev_user:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "X-Dev-User header required")
        return User(id=x_dev_user)
    if x_service_token is not None:
        if not s.service_token or not hmac.compare_digest(
            x_service_token.encode(), s.service_token.encode()
        ):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid service token")
        if not x_user_id or not _USER_ID.match(x_user_id):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "X-User-Id header required")
        _rate_limit(x_user_id)
        return User(id=x_user_id)
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
    user = verify_kinde_token(authorization.split(" ", 1)[1].strip())
    _rate_limit(user.id)
    return user


# Sliding-window limit per user. In-process, so it's per instance; fine for one
# API instance. With several, move it to Postgres or Redis.
_hits: dict[str, deque[float]] = defaultdict(deque)


def _rate_limit(user_id: str) -> None:
    limit = get_settings().rate_limit_per_minute
    now = time.monotonic()
    window = _hits[user_id]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= limit:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "rate limit exceeded")
    window.append(now)
