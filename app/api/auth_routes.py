from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from app import config
from app.api import repositories
from app.api.session_middleware import require_session_id, require_user_id

router = APIRouter(prefix="/api/v1", tags=["authentication"])


@router.get("/auth/config")
def auth_config() -> dict[str, str | bool | None]:
    return {
        "enabled": bool(config.CLERK_PUBLISHABLE_KEY and config.CLERK_SECRET_KEY),
        "publishable_key": config.CLERK_PUBLISHABLE_KEY,
        "frontend_api_url": config.CLERK_FRONTEND_API_URL,
    }


@router.post("/auth/claim")
def claim_current_session(
    session_id: str = Depends(require_session_id),
    user_id: str = Depends(require_user_id),
) -> dict[str, str]:
    # SessionMiddleware performs the locked, idempotent claim before routing.
    return {"status": "claimed", "session_id": session_id, "user_id": user_id}


@router.post("/auth/logout")
def logout(request: Request, _user_id: str = Depends(require_user_id)) -> dict[str, str]:
    request.state.rotate_session_after_response = True
    return {"status": "signed_out"}


@router.get("/me")
def me(user_id: str = Depends(require_user_id)) -> dict:
    user = repositories.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "id": str(user["id"]),
        "primary_email": user.get("primary_email"),
        "display_name": user.get("display_name"),
        "image_url": user.get("image_url"),
        "created_at": user["created_at"],
        "last_login_at": user.get("last_login_at"),
    }


@router.get("/me/exams")
def my_exams(user_id: str = Depends(require_user_id)) -> dict[str, list[dict]]:
    return {"exams": repositories.list_exams_for_user(user_id)}
