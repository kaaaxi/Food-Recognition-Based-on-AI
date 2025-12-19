from fastapi import HTTPException, status

from ..models.schemas import LoginRequest, RefreshRequest, RegisterRequest, TokenResponse
from ..services.auth_service import AuthService


class AuthController:
    def __init__(self, auth_service: AuthService) -> None:
        self.auth_service = auth_service

    def register(self, payload: RegisterRequest) -> TokenResponse:
        try:
            tokens, _ = self.auth_service.register(payload.email, payload.password, payload.name)
            return tokens
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    def login(self, payload: LoginRequest) -> TokenResponse:
        try:
            tokens, _ = self.auth_service.login(payload.email, payload.password)
            return tokens
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc

    def refresh(self, payload: RefreshRequest) -> TokenResponse:
        try:
            return self.auth_service.refresh(payload.refresh_token)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc

    def delete_user_data(self, user_id: str) -> None:
        self.auth_service.delete_user_data(user_id)

