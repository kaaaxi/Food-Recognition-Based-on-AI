import os
from pathlib import Path


class Settings:
    """Centralized configuration pulled from environment variables."""

    project_name: str = "AI Food Recognition"
    api_prefix: str = "/api"
    supabase_url: str = os.getenv("SUPABASE_URL", "https://bjdgqkraswfosujcspmg.supabase.co")
    supabase_key: str | None = os.getenv("SUPABASE_KEY")
    jwt_secret: str = os.getenv("JWT_SECRET", "change-me-in-production")
    jwt_algorithm: str = "HS256"
    access_token_ttl: int = 60 * 30  # 30 minutes
    refresh_token_ttl: int = 60 * 60 * 24 * 7  # 7 days
    models_root: Path = Path(__file__).resolve().parent.parent / "models"
    qwen_model: str = os.getenv("QWEN_MODEL", "qwen-plus")
    qwen_base_url: str = os.getenv(
        "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    dascope_api_key_env: str = "DASHSCOPE_API_KEY"


settings = Settings()

