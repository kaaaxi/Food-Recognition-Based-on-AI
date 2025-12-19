from datetime import datetime
from typing import Optional, Tuple

from ..config import settings
from ..models.schemas import TokenResponse
from .security import decode_jwt, encode_jwt, hash_password, verify_password
from .supabase_service import SupabaseService


class AuthService:
    def __init__(self, supabase: SupabaseService) -> None:
        self.supabase = supabase

    def register(self, email: str, password: str, name: str) -> Tuple[TokenResponse, Optional[str]]:
        existing = self.supabase.find_user(email)
        if existing:
            raise ValueError("User already exists")
        salt, password_hash = hash_password(password)
        user = self.supabase.upsert_user(email, name, salt, password_hash)
        user_id = str(user.get("id")) if user else None
        tokens = self._issue_tokens(email=email, user_id=user_id)
        return tokens, user_id

    def login(self, email: str, password: str) -> Tuple[TokenResponse, Optional[str]]:
        user = self.supabase.find_user(email)
        if not user:
            raise ValueError("Invalid credentials")
        salt = user.get("salt")
        password_hash = user.get("password_hash")
        if not (salt and password_hash and verify_password(password, salt, password_hash)):
            raise ValueError("Invalid credentials")
        tokens = self._issue_tokens(email=email, user_id=str(user.get("id")))
        return tokens, str(user.get("id"))

    def refresh(self, refresh_token: str) -> TokenResponse:
        payload = decode_jwt(refresh_token, settings.jwt_secret)
        return self._issue_tokens(email=payload.get("email"), user_id=payload.get("sub"))

    def _issue_tokens(self, email: str, user_id: Optional[str]) -> TokenResponse:
        access = encode_jwt(
            {"sub": user_id or email, "email": email, "type": "access"},
            settings.jwt_secret,
            settings.access_token_ttl,
        )
        refresh = encode_jwt(
            {"sub": user_id or email, "email": email, "type": "refresh"},
            settings.jwt_secret,
            settings.refresh_token_ttl,
        )
        return TokenResponse(
            access_token=access,
            refresh_token=refresh,
            expires_in=settings.access_token_ttl,
        )

    def delete_user_data(self, user_id: str) -> None:
        self.supabase.delete_user_data(user_id)

