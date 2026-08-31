"""Environment-backed application configuration."""

from functools import lru_cache

from pydantic import HttpUrl, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    qwen_api_url: HttpUrl
    qwen_api_key: SecretStr
    qwen_model: str = "qwen3.8-flash"
    qwen_enable_thinking: bool = True
    qwen_enable_web_search: bool = True
    qwen_request_timeout_seconds: float = 120.0

    a2a_public_url: HttpUrl = HttpUrl("http://127.0.0.1:8000")
    a2a_host: str = "0.0.0.0"
    a2a_port: int = 8000
    a2a_rpc_path: str = "/a2a"
    a2a_event_callback_path: str = "/agent/event/callback"

    qwen_client_id: str | None = None
    qwen_client_secret: SecretStr | None = None
    qwen_signature_max_skew_seconds: int = 300
    log_level: str = "INFO"

    @field_validator("a2a_rpc_path", "a2a_event_callback_path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError("path must start with '/'")
        if len(value) > 1 and value.endswith("/"):
            return value.rstrip("/")
        return value

    @field_validator("qwen_client_id", mode="before")
    @classmethod
    def normalize_optional_string(cls, value: object) -> object:
        return None if value == "" else value

    @field_validator("qwen_client_secret", mode="before")
    @classmethod
    def normalize_optional_secret(cls, value: object) -> object:
        return None if value == "" else value

    @field_validator("qwen_api_key")
    @classmethod
    def validate_api_key(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value():
            raise ValueError("QWEN_API_KEY must not be empty")
        return value

    @model_validator(mode="after")
    def validate_signature_pair(self) -> "Settings":
        if bool(self.qwen_client_id) != bool(self.qwen_client_secret):
            raise ValueError(
                "QWEN_CLIENT_ID and QWEN_CLIENT_SECRET must be configured together"
            )
        if self.qwen_request_timeout_seconds <= 0:
            raise ValueError("QWEN_REQUEST_TIMEOUT_SECONDS must be positive")
        if self.qwen_signature_max_skew_seconds < 0:
            raise ValueError("QWEN_SIGNATURE_MAX_SKEW_SECONDS cannot be negative")
        return self

    @property
    def agent_endpoint(self) -> str:
        return f"{str(self.a2a_public_url).rstrip('/')}{self.a2a_rpc_path}"


@lru_cache
def get_settings() -> Settings:
    """Return one validated settings instance per process."""

    return Settings()  # type: ignore[call-arg]
