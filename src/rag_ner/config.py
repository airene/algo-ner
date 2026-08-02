"""Application settings loaded from environment variables and .env."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )

    app_host: str = "0.0.0.0"
    app_port: int = Field(default=8002, ge=1, le=65_535)
    api_key: str = ""

    device: str = "cpu"
    model_cache_dir: str = "./models"
    model_id: str = "iic/nlp_raner_named-entity-recognition_chinese-base-generic"
    model_revision: str = "master"

    # RaNER uses a 512-position encoder and needs two positions for special tokens.
    max_input_characters: int = Field(default=480, ge=1, le=510)
    max_concurrent_requests: int = Field(default=4, ge=1)


settings = Settings()
