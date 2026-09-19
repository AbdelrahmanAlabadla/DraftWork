from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from clerk_backend_api import AuthenticateRequestOptions, Clerk
from clerk_backend_api.security.types import AuthStatus
from fastapi import Request

from app import config


@dataclass(frozen=True)
class ClerkIdentity:
    clerk_user_id: str


def is_configured() -> bool:
    return bool(config.CLERK_PUBLISHABLE_KEY and config.CLERK_SECRET_KEY)


async def authenticate(request: Request) -> ClerkIdentity | None:
    """Verify an optional Clerk bearer token and return its immutable subject."""
    authorization = request.headers.get("Authorization", "")
    if not authorization:
        return None
    if not is_configured():
        raise ValueError("Clerk authentication is not configured")

    client = Clerk(bearer_auth=config.CLERK_SECRET_KEY)
    state = await client.authenticate_request_async(
        request,
        AuthenticateRequestOptions(
            secret_key=config.CLERK_SECRET_KEY,
            jwt_key=config.CLERK_JWT_KEY,
            authorized_parties=list(config.CLERK_AUTHORIZED_PARTIES),
            accepts_token=["session_token"],
        ),
    )
    if state.status is not AuthStatus.SIGNED_IN or not state.payload:
        reason = getattr(state.reason, "value", state.reason)
        raise ValueError(str(reason or "invalid session token"))
    subject = str(state.payload.get("sub") or "").strip()
    if not subject:
        raise ValueError("Clerk token has no subject")
    return ClerkIdentity(clerk_user_id=subject)


async def fetch_profile(clerk_user_id: str) -> dict[str, Any]:
    """Fetch trusted profile fields from Clerk's Backend API."""
    client = Clerk(bearer_auth=config.CLERK_SECRET_KEY)
    user = await client.users.get_async(user_id=clerk_user_id)
    primary_email = None
    primary_id = getattr(user, "primary_email_address_id", None)
    for address in getattr(user, "email_addresses", []) or []:
        if getattr(address, "id", None) == primary_id:
            primary_email = getattr(address, "email_address", None)
            break
    first = str(getattr(user, "first_name", None) or "").strip()
    last = str(getattr(user, "last_name", None) or "").strip()
    display_name = " ".join(part for part in (first, last) if part) or None
    return {
        "primary_email": primary_email,
        "display_name": display_name,
        "image_url": getattr(user, "image_url", None),
    }
