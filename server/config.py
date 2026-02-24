"""
Configuration management using pydantic-settings.
All secrets loaded from environment variables.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- Retell AI ---
    retell_api_key: str
    retell_agent_id: str = ""  # populated after setup script runs

    # --- OpenAI ---
    openai_api_key: str

    # --- Google Calendar ---
    google_credentials_json: str = ""  # base64-encoded service account JSON
    google_calendar_id: str = "primary"

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000
    environment: str = "development"  # development | production
    log_level: str = "INFO"

    # --- CORS ---
    allowed_origins: str = "*"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
