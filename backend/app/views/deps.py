from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..config import settings
from ..services.security import decode_jwt

bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    """Resolve and validate JWT Bearer token."""
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")
    token = credentials.credentials
    try:
        payload = decode_jwt(token, settings.jwt_secret)
        return payload  # contains sub/email/type
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token") from exc


def get_optional_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> Optional[dict]:
    """Optionally resolve JWT Bearer token. Returns None if not authenticated."""
    if credentials is None:
        return None
    token = credentials.credentials
    try:
        payload = decode_jwt(token, settings.jwt_secret)
        return payload
    except ValueError:
        return None

